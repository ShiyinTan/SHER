import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset
from torch_geometric.data import HeteroData, Batch

import numpy as np
import os
import json
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm

from rouge_score import rouge_scorer

from models import (SRLAttentionEncoder,
                    HGTRankingModel,
                    DocAttentionPooling,
                    bpr_loss_with_sampling,
                    sent_srl_consistency)
from utils import (
    aggregate_metrics,
    compute_sentence_labels,
    build_srl_srl_sim_edges,
    build_sent_srl_edges,
    build_doc_contain_sent_edges,
    build_sent_next_edges,
    build_doc_fully_connected,
    preprocess_docs,
    _fmt_metrics,
    compute_ranking_metrics
)

import re
import spacy
from sentence_transformers import SentenceTransformer
from scipy.stats import spearmanr



import argparse




# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 – helpers that are identical to the original (kept for reference)
# ═══════════════════════════════════════════════════════════════════════════════


def sentence_split_with_span(text, nlp):
    doc = nlp(text)
    return [{"text": s.text, "start": s.start_char, "end": s.end_char} for s in doc.sents]

def get_docs_sentences(docs, nlp, max_char=40000):
    return [sentence_split_with_span(doc[:max_char], nlp) for doc in docs]

def find_sentence_ids(c_s, c_e, sentences):
    return [i for i, s in enumerate(sentences)
            if not (c_e <= s['start'] or c_s >= s['end'])]

def char_span_to_token_indices(char_span, offsets):
    s, e = char_span
    return [i for i, (cs, ce) in enumerate(offsets) if not (ce <= s or cs >= e)]

def get_srl_embedding(token_embeddings, token_indices, pooling="mean"):
    if len(token_indices) == 0:
        return None
    span = token_embeddings[token_indices]
    return span.mean(dim=0) if pooling == "mean" else span.max(dim=0).values



def merge_sliding_window_outputs(token_embeddings, offset_mapping, attention_mask=None, merge_strategy="mean"):
    num_chunks, max_len, hidden_dim = token_embeddings.shape
    span2embeds = defaultdict(list)
    # print(offset_mapping)
    for chunk_id in range(num_chunks):
        for tok_id in range(max_len):
            s, e = offset_mapping[chunk_id][tok_id]
            if s == 0 and e == 0:
                continue
            if attention_mask is not None and attention_mask[chunk_id, tok_id].item() == 0:
                continue
            span2embeds[(s, e)].append(token_embeddings[chunk_id, tok_id])
    sorted_spans = sorted(span2embeds.keys(), key=lambda x: x[0])
    final_embeddings, final_offsets = [], []
    # print(sorted_spans)
    for span in sorted_spans:
        embeds = span2embeds[span]
        # print(f"Span {span} has {len(embeds)} embeddings from {num_chunks} chunks.")
        # if embeds[0].shape[0] == 0:
        #     # edge case: empty sentence with one chunk and two special tokens (CLS, SEP)
        #     print(f"Empty sentence edge case hit. span={span}, num_chunks={num_chunks}, max_len={max_len}")
        #     embeds = [torch.zeros(hidden_dim)]
        if merge_strategy == "mean":
            merged = torch.stack(embeds).mean(0)
        elif merge_strategy == "max":
            merged = torch.stack(embeds).max(0).values
        else:
            merged = embeds[0]
        final_embeddings.append(merged)
        final_offsets.append(span)
    if len(final_embeddings) == 0:
        # edge case: no valid tokens found across all chunks (should not happen with proper tokenization)
        # print("No valid tokens found across all chunks. Returning zero embedding.")
        final_embeddings = [torch.zeros(hidden_dim)]
        final_offsets = [(0, 0)]
    return torch.stack(final_embeddings), final_offsets


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 – BATCHED sentence / SRL embedding extraction
#
#  Key idea
#  --------
#  The bottleneck is running `transformer` one sentence at a time.  We instead
#  collect ALL sentences across ALL docs in the batch, pad them to a common
#  length, and run a single forward pass.  After that we reassemble per-doc,
#  per-sentence token embeddings exactly as before.
#
#  Why this is tricky
#  ------------------
#  Each sentence may overflow into multiple chunks (sliding-window tokenisation).
#  We therefore pad at the *chunk* level, not the sentence level.  Concretely:
#
#    1. Tokenise every sentence independently -> variable number of chunks, each
#       of length <= 512.
#    2. Flatten all chunks into one list -> pad to a single batch -> one GPU call.
#    3. Re-split by sentence and call merge_sliding_window_outputs as before.
# ═══════════════════════════════════════════════════════════════════════════════

def _tokenize_sentence(sent_text: str, tokenizer) -> dict:
    """Tokenise one sentence with sliding-window overflow. Returns CPU tensors."""
    enc = tokenizer(
        sent_text,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
        stride=128,
        return_overflowing_tokens=True,
    )
    return enc   # keys: input_ids, attention_mask, offset_mapping, (overflow_to_sample_mapping)


def get_docs_embedding_enc_offsets_batched(
    docs_sentences: List[List[dict]],
    transformer,
    tokenizer,
    batch_size: int = 64,
    device: str = "cpu",
) -> Tuple[List[List[torch.Tensor]], List[List[dict]], List[List[list]]]:
    """
    Batched replacement for get_docs_embedding_enc_offsets_by_sentences.

    Returns the exact same data structure:
        docs_embedding_by_sentences  : List[doc] of List[sent] of Tensor(seq_len, d)
        docs_enc_by_sentences        : List[doc] of List[sent] of enc-dict
        docs_offsets_by_sentences    : List[doc] of List[sent] of List[(cs,ce)]
    """

    # ── Step A: tokenise every sentence, collect chunk metadata ───────────────
    # flat_encs[i]   = enc dict (CPU) for sentence i across all docs/sents
    # flat_info[i]   = (doc_id, sent_id_within_doc, num_chunks)
    flat_encs: List[dict] = []
    flat_info: List[Tuple[int, int, int]] = []

    for doc_id, sentences in enumerate(docs_sentences):
        for sent_id, sent in enumerate(sentences):
            enc = _tokenize_sentence(sent['text'], tokenizer)
            num_chunks = enc['input_ids'].shape[0]
            flat_encs.append(enc)
            flat_info.append((doc_id, sent_id, num_chunks))

    # print(flat_encs, flat_info)

    # ── Step B: flatten all chunks into one big list ───────────────────────────
    # chunk_to_sent[chunk_global_id] = (sent_flat_id, chunk_local_id)
    all_input_ids   : List[torch.Tensor] = []
    all_attn_masks  : List[torch.Tensor] = []
    all_offset_maps : List[torch.Tensor] = []
    chunk_to_sent   : List[Tuple[int, int]] = []

    for sent_flat_id, enc in enumerate(flat_encs):
        n = enc['input_ids'].shape[0]
        for chunk_local_id in range(n):
            all_input_ids.append(enc['input_ids'][chunk_local_id])
            all_attn_masks.append(enc['attention_mask'][chunk_local_id])
            all_offset_maps.append(enc['offset_mapping'][chunk_local_id])
            chunk_to_sent.append((sent_flat_id, chunk_local_id))

    total_chunks = len(all_input_ids)

    # ── Step C: GPU forward pass in micro-batches ─────────────────────────────
    # We pad within each micro-batch because sequence lengths may differ.
    all_token_embeddings: List[Optional[torch.Tensor]] = [None] * total_chunks

    for start in range(0, total_chunks, batch_size):
        end = min(start + batch_size, total_chunks)

        batch_ids   = torch.nn.utils.rnn.pad_sequence(
            all_input_ids[start:end], batch_first=True,
            padding_value=tokenizer.pad_token_id
        ).to(device)
        batch_masks = torch.nn.utils.rnn.pad_sequence(
            all_attn_masks[start:end], batch_first=True, padding_value=0
        ).to(device)

        with torch.no_grad():
            out = transformer(input_ids=batch_ids, attention_mask=batch_masks)

        token_embs = out.last_hidden_state  # (B, seq_len, d)

        for b_i, global_chunk_id in enumerate(range(start, end)):
            # trim to real length so downstream merge is not confused by padding
            real_len = batch_masks[b_i].sum().item()
            all_token_embeddings[global_chunk_id] = token_embs[b_i, :real_len].cpu()

    # ── Step D: reassemble per-sentence, re-pad chunks, merge sliding windows ──
    # sent_chunks[sent_flat_id] = list of (chunk_local_id, token_emb, offset_map)
    sent_chunks: List[List] = [[] for _ in range(len(flat_encs))]
    for global_chunk_id, (sent_flat_id, chunk_local_id) in enumerate(chunk_to_sent):
        sent_chunks[sent_flat_id].append(
            (chunk_local_id,
             all_token_embeddings[global_chunk_id],
             all_offset_maps[chunk_local_id])
        )

    # ── Step E: build the nested output structure ──────────────────────────────
    docs_embedding_by_sentences = [[None] * len(sents) for sents in docs_sentences]
    docs_enc_by_sentences       = [[None] * len(sents) for sents in docs_sentences]
    docs_offsets_by_sentences   = [[None] * len(sents) for sents in docs_sentences]

    for sent_flat_id, (doc_id, sent_id, num_chunks) in enumerate(flat_info):
        chunks = sorted(sent_chunks[sent_flat_id], key=lambda x: x[0])

        chunk_embs    = [c[1] for c in chunks]           # list of (seq_len_i, d)
        chunk_offsets = [c[2].tolist() for c in chunks]  # list of offset lists

        # Re-pad chunks from the same sentence so we can call merge_sliding_window_outputs
        max_len = max(e.shape[0] for e in chunk_embs)
        d       = chunk_embs[0].shape[1]
        stacked = torch.zeros(len(chunk_embs), max_len, d)
        attn_m  = torch.zeros(len(chunk_embs), max_len, dtype=torch.long)
        for ci, emb in enumerate(chunk_embs):
            l = emb.shape[0]
            stacked[ci, :l] = emb
            attn_m[ci, :l]  = 1
        padded_offsets = [
            o + [(0, 0)] * (max_len - len(o)) for o in chunk_offsets
        ]

        if stacked.shape[0] == 1 and stacked.shape[1] == 2:
            # empty sentence edge case (matches original behaviour)
            print(f"doc {doc_id} sent {sent_id} empty. Zeroing.")
            final_embs    = torch.zeros(1, d, device=device)
            final_offsets = [(0, 0)]
        else:
            final_embs, final_offsets = merge_sliding_window_outputs(
                stacked, padded_offsets, attention_mask=attn_m)
            final_embs = final_embs.to(device)

        docs_embedding_by_sentences[doc_id][sent_id] = final_embs
        docs_enc_by_sentences[doc_id][sent_id]       = flat_encs[sent_flat_id]
        docs_offsets_by_sentences[doc_id][sent_id]   = final_offsets

    return docs_embedding_by_sentences, docs_enc_by_sentences, docs_offsets_by_sentences


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 – BATCHED SRL attention encoder
#
#  Original:  [srl_encoder(srl_dict) for srl_dict in srl_embs_with_roles]
#             -> one Python loop per SRL, each doing one tiny Linear.
#
#  Batched:   Collect role embeddings for ALL SRLs into one [N, R, d] tensor,
#             run the attention scorer once, scatter-sum -> [N, d].
# ═══════════════════════════════════════════════════════════════════════════════

def batch_srl_encoder(srl_encoder: SRLAttentionEncoder,
                      srl_embs_with_roles: dict) -> torch.Tensor:
    """
    Vectorised replacement for:
        [srl_encoder(srl_dict) for _, srl_dict in srl_embs_with_roles.items()]

    Returns: Tensor [num_srls, hidden_dim]
    """
    roles    = srl_encoder.roles
    role_ids = torch.arange(len(roles), device=srl_encoder.device)

    role_emb_mat = srl_encoder.role_emb(role_ids)   # [R, d]
    null_emb_mat = srl_encoder.null_emb(role_ids)   # [R, d]

    srl_keys = list(srl_embs_with_roles.keys())
    N = len(srl_keys)
    R = len(roles)
    d = srl_encoder.hidden_dim

    combined = torch.zeros(N, R, d, device=srl_encoder.device)
    mask     = torch.zeros(N, R,    device=srl_encoder.device)

    for n, key in enumerate(srl_keys):
        srl_dict = srl_embs_with_roles[key]
        for r, role in enumerate(roles):
            if role in srl_dict and srl_dict[role] is not None:
                combined[n, r] = srl_dict[role].to(srl_encoder.device) + role_emb_mat[r]
                mask[n, r]     = 1.0
            else:
                combined[n, r] = null_emb_mat[r] + role_emb_mat[r]

    # [N, R, d] -> [N, R, 1] -> [N, R]
    scores = srl_encoder.attn(combined).squeeze(-1)
    scores = scores.masked_fill(mask == 0, float('-inf'))
    attn_weights = F.softmax(scores, dim=1)   # [N, R]

    # weighted sum over roles: [N, d]
    srl_embs = (attn_weights.unsqueeze(-1) * combined).sum(dim=1)
    return srl_embs


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 – get_overall_srl_embeddings (unchanged logic)
# ═══════════════════════════════════════════════════════════════════════════════

def get_overall_srl_embeddings(srls, docs_sentences,
                               docs_embedding_by_sentences,
                               docs_offsets_by_sentences,
                               embedding_dim=768, device="cpu"):
    overall_srl_embeddings = {}
    for doc_k in srls['document'].keys():
        doc_id = int(doc_k)
        for k in srls['document'][doc_k].keys():
            srls_in_sentence = srls['document'][doc_k][k]
            for srl_id in srls_in_sentence.keys():
                srl = srls_in_sentence[srl_id]
                overall_srl_embeddings[srl_id] = {}
                srl_embedding = None
                for argument in srl:
                    arg_char_span = argument[0]
                    arg_role      = argument[1]
                    sentence_ids  = find_sentence_ids(
                        arg_char_span[0], arg_char_span[1], docs_sentences[doc_id])
                    for arg_sent_id in sentence_ids:
                        local_span = (
                            arg_char_span[0] - docs_sentences[doc_id][arg_sent_id]['start'],
                            arg_char_span[1] - docs_sentences[doc_id][arg_sent_id]['start'],
                        )
                        token_indices = char_span_to_token_indices(
                            local_span, docs_offsets_by_sentences[doc_id][arg_sent_id])
                        srl_embedding = get_srl_embedding(
                            docs_embedding_by_sentences[doc_id][arg_sent_id], token_indices)
                        if srl_embedding is None:
                            continue
                    if srl_embedding is None:
                        # print(f'srl embedding is None, doc: {doc_id}, sent: {k}')
                        srl_embedding = torch.zeros(embedding_dim, device=device)
                    overall_srl_embeddings[srl_id][arg_role] = srl_embedding
    return overall_srl_embeddings


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 – graph builder (encapsulated for clarity)
# ═══════════════════════════════════════════════════════════════════════════════

def _get_leaf_keys(d):
    keys = []
    for k, v in d.items():
        if isinstance(v, dict) and v:  # 继续往下
            keys.extend(_get_leaf_keys(v))
        else:
            keys.append(k)
    return keys

def build_sample_graph(doc_embeddings, sentence_embeddings, srl_embs,
                       sample_srl_sim, docs_sentences, srls, device="cpu"):

    srl_keys = _get_leaf_keys(srls['document'])
    srl2idx = {s: i for i, s in enumerate(srl_keys)}
    sent_keys = [f"doc_{doc_i}_sent_{sent_i}" for doc_i in range(len(docs_sentences)) for sent_i in range(len(docs_sentences[doc_i]))]
    sent2idx = {s: i for i, s in enumerate(sent_keys)}
    doc_keys = [f"doc_{doc_i}" for doc_i in range(len(docs_sentences))]
    doc2idx = {d: i for i, d in enumerate(doc_keys)}

    # print('docs: ', len(srl_keys), len(sent_keys), len(doc_keys), 'summary', len(srls['summary']))
    
    idx = torch.arange(len(srl_keys))
    self_loop = torch.stack([idx, idx], dim=0) # self-loop only if no SRLs in summary (edge case)
    if len(srls['summary'])!=0:
        edge_srl2srl_sim      = build_srl_srl_sim_edges(
            sample_srl_sim['srl_relation_S'], sample_srl_sim['srl_argument_S'], 0.5, 0.9)
        edge_srl2srl_sim = torch.cat([edge_srl2srl_sim, self_loop], dim=1)
    else:
        edge_srl2srl_sim = self_loop
    edge_sent2srl_contain, sent2srl_dict = build_sent_srl_edges(srl_keys, sent2idx, srl2idx)
    idx = torch.arange(len(sent_keys))
    self_loop = torch.stack([idx, idx], dim=0) # self-loop only if no SRLs in summary (edge case)
    edge_sent2sent_next = build_sent_next_edges(sent_keys, sent2idx)
    edge_sent2sent_next = torch.cat([edge_sent2sent_next, self_loop], dim=1)
    edge_doc2sent_contain = build_doc_contain_sent_edges(sent_keys, doc2idx, sent2idx)
    edge_doc2doc = build_doc_fully_connected(len(docs_sentences))

    x_dict = {
        "document": doc_embeddings.to(device),
        "sentence": sentence_embeddings.to(device),
        "srl":      srl_embs.to(device),
    }
    edge_index_dict = {
        ("document", "connect",  "document"): edge_doc2doc.to(device),
        ("document", "contain",  "sentence"): edge_doc2sent_contain.to(device),
        ("sentence", "in",       "document"): edge_doc2sent_contain.flip(0).to(device),
        ("sentence", "next",     "sentence"): edge_sent2sent_next.to(device),
        ("sentence", "prev",     "sentence"): edge_sent2sent_next.flip(0).to(device),
        ("sentence", "contain",  "srl"):      edge_sent2srl_contain.to(device),
        ("srl",      "in",       "sentence"): edge_sent2srl_contain.flip(0).to(device),
        ("srl",      "sim",      "srl"):      edge_srl2srl_sim.to(device),
        ("srl",      "rev_sim",  "srl"):      edge_srl2srl_sim.flip(0).to(device),
    }
    return x_dict, edge_index_dict, sent2srl_dict


def build_hetero_data(x_dict, edge_index_dict) -> HeteroData:
    data = HeteroData()
    for ntype, feat in x_dict.items():
        data[ntype].x = feat
    for etype, ei in edge_index_dict.items():
        data[etype].edge_index = ei
    return data





# ── Shard cache to avoid reloading from disk every step ───────────────────────
_cached_shard_idx  = -1
_cached_shard_data = None

def load_shard(sample_i, data_dir, data_name, split="train", shard_size=2000):
    global _cached_shard_idx, _cached_shard_data
    shard_idx = sample_i // shard_size
    # include split in cache key so train/val shards don't collide
    cache_key = (split, shard_idx)
    if cache_key != _cached_shard_idx:
        path = os.path.join(
            data_dir,
            f"{data_name}_overall_similarities_{split}_shard_{shard_idx}.pt")
        _cached_shard_data = torch.load(path)
        _cached_shard_idx  = cache_key
    return _cached_shard_data[sample_i % shard_size]






@torch.no_grad()
def run_validation(
    val_dataset,
    srl_encoder: SRLAttentionEncoder,
    hgt_model:   HGTRankingModel,
    num_samples: int,
    nlp,
    transformer,
    tokenizer,
    split_name:  str = "validation",
    shard_size:  int = 2000,
    device: str = "cpu",
) -> dict:
    """
    Evaluate on `num_samples` examples from val_dataset.
 
    Returns a dict with:
        val_loss          – mean total loss
        sent_spearman     – mean Spearman ρ between sent_scores and sent labels
        srl_spearman      – mean Spearman ρ between srl_scores  and srl  labels
    """
    srl_encoder.eval()
    hgt_model.eval()
 
    indices = list(range(min(num_samples, len(val_dataset))))
 
    total_loss      = 0.0
    sent_rhos       = []
    srl_rhos        = []
    sent_metrics_list: List[dict] = []
    srl_metrics_list:  List[dict] = []

 
    for val_i in tqdm(indices, desc=f"[{split_name}]", leave=False):
        # print(val_i)
        # try:
        # if val_i < 4400: continue # 测试用
        sample_coref_srl = val_dataset[val_i]
        sample_srl_sim   = load_shard(val_i, data_dir, data_name, split=split_name, shard_size=shard_size)
        srls             = json.loads(sample_coref_srl['srl'])
        docs             = preprocess_docs(sample_coref_srl['document'])
        summary          = sample_coref_srl['summary']
        docs_sentences   = get_docs_sentences(docs, nlp)

        docs_emb, _, docs_off = get_docs_embedding_enc_offsets_batched(
            docs_sentences, transformer=transformer, tokenizer=tokenizer, batch_size=EMBED_BATCH, device=device)

        if len(docs) == 0 or len(summary.strip()) == 0:
            print(f"Sample {val_i} skipped due to empty docs or summary.")
            continue

        try:
            sentence_embeddings_list = []
            doc_embeddings_list      = []
            for doc_emb in docs_emb:
                doc_sent_emb = [s.mean(0) for s in doc_emb]
                sentence_embeddings_list.extend(doc_sent_emb)
                doc_embeddings_list.append(torch.stack(doc_sent_emb).mean(0))

            sentence_embeddings = torch.stack(sentence_embeddings_list)
            doc_embeddings      = torch.stack(doc_embeddings_list)
        
            srl_embs_with_roles = get_overall_srl_embeddings(
                srls, docs_sentences, docs_emb, docs_off, device=device)
            srl_embs = batch_srl_encoder(srl_encoder, srl_embs_with_roles)

            all_sentences   = [s['text'] for ds in docs_sentences for s in ds]
            sent2sum_sim    = compute_sentence_labels(all_sentences, summary, scorer)
            if len(srls['summary'])!=0:
                srl2sum_sim_rel = torch.max(sample_srl_sim['srl_relation_S_to_summary'], dim=1).values
                srl2sum_sim_arg = torch.max(sample_srl_sim['srl_argument_S_summary'],    dim=1).values
                srl2sum_sim     = alpha_srl_label * srl2sum_sim_rel + (1 - alpha_srl_label) * srl2sum_sim_arg
            else:
                srl2sum_sim_rel = srl2sum_sim_arg = srl2sum_sim = None

            x_dict, edge_index_dict, sent2srl_dict = build_sample_graph(
                doc_embeddings, sentence_embeddings, srl_embs, sample_srl_sim, docs_sentences, srls, 
                device=device)
        except Exception as e:
            print(f"Error stacking embeddings for sample {val_i}: {e}")
            # sentence_embeddings = torch.zeros((len(sentence_embeddings_list), transformer.config.hidden_size), device=device)
            # doc_embeddings      = torch.zeros((len(doc_embeddings_list), transformer.config.hidden_size), device=device)


        sent_scores, srl_scores = hgt_model(x_dict, edge_index_dict)

        l1 = bpr_loss_with_sampling(sent_scores, sent2sum_sim.to(device))
        if len(srls['summary'])!=0:
            l2 = bpr_loss_with_sampling(srl_scores,  srl2sum_sim.to(device))
        else:
            l2 = torch.tensor(0.0, device=device)
        l3 = sent_srl_consistency(sent_scores, srl_scores, sent2srl_dict)
        loss = w1 * l1 + w2 * l2 + w3 * l3
        total_loss += loss.item()

        # ── All ranking metrics for sentences ──────────────────────────────
        s_m = compute_ranking_metrics(
            sent_scores.detach().cpu().float().numpy(),
            sent2sum_sim.numpy(),
        )
        sent_metrics_list.append(s_m)

        # ── All ranking metrics for SRLs ───────────────────────────────────
        if len(srls['summary'])!=0 or srl2sum_sim is not None:
            r_m = compute_ranking_metrics(
                srl_scores.detach().cpu().float().numpy(),
                srl2sum_sim.numpy(),
            )
            srl_metrics_list.append(r_m)
 
        # except Exception as e:
        #     print(f"[val] sample {val_i} skipped: {e}")
        #     continue
 
    n = max(len(sent_metrics_list), 1)
    # Aggregate and prefix keys
    sent_agg = {f"sent_{k}": v for k, v in aggregate_metrics(sent_metrics_list).items()}
    srl_agg  = {f"srl_{k}":  v for k, v in aggregate_metrics(srl_metrics_list).items()}
 
    metrics = {"val_loss": total_loss / n}
    metrics.update(sent_agg)
    metrics.update(srl_agg)
    # Composite model-selection score:
    #   Spearman captures global order; NDCG@10 captures top-k quality.
    #   Equal weight across both heads.
    metrics["val_score"] = (
        metrics.get("sent_spearman", 0.0) + metrics.get("srl_spearman",  0.0) +
        metrics.get("sent_ndcg@10",  0.0) + metrics.get("srl_ndcg@10",   0.0)
    )
 
    srl_encoder.train()
    hgt_model.train()
    return metrics



def save_checkpoint(args, step: int, metrics: dict, split="test", is_best: bool = False):
    """Save both models + optimizer state. Copies to best.pt when improved."""
    ckpt = {
        "args":            vars(args),
        "step":              step,
        "metrics":           metrics,
        "srl_encoder":       srl_encoder.state_dict(),
        "hgt_ranking_model": hgt_ranking_model.state_dict(),
        "optimizer":         optimizer.state_dict(),
    }
    ckpt_dir = os.path.join(CKPT_DIR, data_name, f"alpha_{args.alpha_srl_label}-w_{args.w1}_{args.w2}_{args.w3}_d_{args.hgt_hidden_dim}")
    os.makedirs(ckpt_dir, exist_ok=True)
    step_path = os.path.join(ckpt_dir, f"ckpt_step_{step}_{split}.pt")
    torch.save(ckpt, step_path)
    print(f"[ckpt] saved  → {step_path}")

    metrics["step"] = step
    with open(os.path.join(ckpt_dir, "val_results.jsonl"), "a") as f:
        f.write(json.dumps(metrics) + "\n")
 
    if is_best:
        best_path = os.path.join(CKPT_DIR, "best.pt")
        torch.save(ckpt, best_path)
        print(f"[ckpt] *** new best → {best_path}  val_score={metrics['val_score']:.4f}")




# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 – TRAINING LOOP
#
#  Two levels of batching:
#    EMBED_BATCH  – number of sentence-chunks per transformer forward pass
#    GRAPH_BATCH  – number of samples whose HeteroData graphs are batched for HGT
#
#  Effective batch size = GRAPH_BATCH * ACCUM_STEPS.
# ═══════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name",        type=str, default="multi_news")
    parser.add_argument("--embed_batch_size", type=int, default=64,
                        help="Sentence chunks per GPU forward (transformer)")
    parser.add_argument("--graph_batch_size", type=int, default=4,
                        help="Samples batched together for HGT")
    parser.add_argument("--grad_accum",       type=int, default=2,
                        help="Gradient accumulation steps")
    parser.add_argument("--val_freq",         type=int, default=600,
                        help="Run validation every N optimizer steps")
    parser.add_argument("--val_samples",      type=int, default=400,
                        help="Number of validation samples to evaluate per run")
    parser.add_argument("--ckpt_dir",         type=str, default="checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--num_epochs",      type=int, default=3)
    parser.add_argument("--w1",              type=float, default=0.5, help="Weight for sentence loss")
    parser.add_argument("--w2",              type=float, default=0.5, help="Weight for SRL loss")
    parser.add_argument("--w3",              type=float, default=0.5, help="Weight for consistency loss")
    parser.add_argument("--alpha_srl_label", type=float, default=0.5, help="Alpha for combining SRL relation and argument similarity into SRL labels")
    parser.add_argument("--hgt_hidden_dim", type=int, default=128, help="Hidden dimension for HGT")
    parser.add_argument("--hgt_num_heads", type=int, default=4, help="Number of attention heads for HGT")
    parser.add_argument("--hgt_num_layers", type=int, default=2, help="Number of layers for HGT")
    args = parser.parse_args()

    print(args)

    # ─────────────────────────────────────────────
    # Global model / tokenizer setup (unchanged)
    # ─────────────────────────────────────────────
    nlp = spacy.load("en_core_web_sm")
    st_model = SentenceTransformer("all-mpnet-base-v2")
    tokenizer  = st_model.tokenizer
    transformer = st_model._first_module().auto_model
    device = 'cuda:0'
    transformer = transformer.to(device)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)


    EMBED_BATCH  = args.embed_batch_size
    GRAPH_BATCH  = args.graph_batch_size
    ACCUM_STEPS  = args.grad_accum
    VAL_FREQ     = args.val_freq
    VAL_SAMPLES  = args.val_samples
    CKPT_DIR     = args.ckpt_dir
    data_name    = args.data_name
    
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(os.path.join(CKPT_DIR, data_name), exist_ok=True)

    home_dir = os.path.expanduser('~')
    data_dir = os.path.join(home_dir, "Code/NLP/dataset_MDS/MDS-RA2_data2")

    dataset_with_coref_srl_new = load_dataset(f"{data_name}_with_srl_new")

    if data_name in ["bigsurvey", "peersum"]:
        shard_size = 1000
    else:
        shard_size = 2000

    alpha_srl_label = args.alpha_srl_label
    dim = 768
    w1, w2, w3 = args.w1, args.w2, args.w3


    metadata = (
        ['document', 'sentence', 'srl'],
        [('document', 'connect',  'document'),
        ('document', 'contain',  'sentence'),
        ('sentence', 'in',       'document'),
        ('sentence', 'next',     'sentence'),
        ('sentence', 'prev',     'sentence'),
        ('sentence', 'contain',  'srl'),
        ('srl',      'in',       'sentence'),
        ('srl',      'sim',      'srl'),
        ('srl',      'rev_sim',  'srl')]
    )
    in_dim_dict = {"document": dim, "sentence": dim, "srl": dim}

    srl_encoder       = SRLAttentionEncoder(hidden_dim=dim, device=device)
    hgt_ranking_model = HGTRankingModel(metadata, in_dim_dict, hidden_dim=args.hgt_hidden_dim,
            num_heads=args.hgt_num_heads,
            num_layers=args.hgt_num_layers, device=device)

    optimizer = torch.optim.Adam([
        {"params": srl_encoder.parameters(),       "lr": 1e-4},
        {"params": hgt_ranking_model.parameters(), "lr": 1e-4},
    ])

    train_data = dataset_with_coref_srl_new['train']
    val_data   = dataset_with_coref_srl_new['validation']
    optimizer.zero_grad()





    # ── Main loop ──────────────────────────────────────────────────────────────────
    global_step = 0
    hetero_batch_buffer: List[dict] = []

    best_val_score = 0


    ## final test
    test_data = dataset_with_coref_srl_new['test']
    test_metrics = run_validation(
        val_dataset=test_data,
        srl_encoder=srl_encoder,
        hgt_model=hgt_ranking_model,
        num_samples=len(test_data),
        nlp=nlp,
        transformer=transformer,
        tokenizer=tokenizer,
        shard_size=shard_size,
        device=device,
        split_name="test",
    )
    print(_fmt_metrics(test_metrics, "First Test", "test"))
    # save_checkpoint(args, "final_test", test_metrics, split="test", is_best=False)

    for epoch in range(args.num_epochs):

        for sample_i in tqdm(range(len(train_data))):
            # if sample_i < 6143: # 用于测试异常项
            #     continue

            # ── 1. Load raw sample ────────────────────────────────────────────────────
            sample_coref_srl = train_data[sample_i]
            sample_srl_sim   = load_shard(sample_i, data_dir, data_name, split="train", shard_size=shard_size)
            srls             = json.loads(sample_coref_srl['srl'])
            docs             = preprocess_docs(sample_coref_srl['document'])
            summary          = sample_coref_srl['summary']
            docs_sentences   = get_docs_sentences(docs, nlp)

            if len(docs) == 0 or len(summary.strip()) == 0:
                print(f"Sample {sample_i} skipped due to empty docs or summary.")
                continue

            # ── 2. Batched transformer encoding (Section 2) ───────────────────────────
            #    All sentences in this sample are encoded in a single batched call.
            docs_emb, docs_enc, docs_off = get_docs_embedding_enc_offsets_batched(
                docs_sentences, transformer=transformer, tokenizer=tokenizer, batch_size=EMBED_BATCH, device=device)

            # ── 3. Sentence / doc embeddings ──────────────────────────────────────────
            sentence_embeddings_list = []
            doc_embeddings_list      = []
            for doc_emb in docs_emb:
                doc_sent_emb = [s.mean(0) for s in doc_emb]
                sentence_embeddings_list.extend(doc_sent_emb)
                doc_embeddings_list.append(torch.stack(doc_sent_emb).mean(0))
            sentence_embeddings = torch.stack(sentence_embeddings_list)
            doc_embeddings      = torch.stack(doc_embeddings_list)

            # ── 4. SRL embeddings with batched attention encoder (Section 3) ──────────
            srl_embs_with_roles = get_overall_srl_embeddings(
                srls, docs_sentences, docs_emb, docs_off, device=device)
            srl_embs = batch_srl_encoder(srl_encoder, srl_embs_with_roles)  # [N_srl, d]

            # ── 5. Labels ─────────────────────────────────────────────────────────────
            all_sentences    = [s['text'] for ds in docs_sentences for s in ds]
            sent2sum_sim     = compute_sentence_labels(all_sentences, summary, scorer)
            if len(srls['summary']) == 0:
                print("None SRL for summary")
                srl2sum_sim_relation = None
                srl2sum_sim_argument = None
                srl2sum_sim = None
            else:
                srl2sum_sim_rel  = torch.max(sample_srl_sim['srl_relation_S_to_summary'], dim=1).values
                srl2sum_sim_arg  = torch.max(sample_srl_sim['srl_argument_S_summary'],    dim=1).values
                srl2sum_sim      = alpha_srl_label * srl2sum_sim_rel + (1 - alpha_srl_label) * srl2sum_sim_arg

            # ── 6. Build per-sample HeteroData ────────────────────────────────────────
            x_dict, edge_index_dict, sent2srl_dict = build_sample_graph(
                doc_embeddings, sentence_embeddings, srl_embs, sample_srl_sim, docs_sentences, srls, 
                device=device)

            hetero_batch_buffer.append({
                "x_dict":           x_dict,
                "edge_index_dict":  edge_index_dict,
                "sent2srl_dict":    sent2srl_dict,
                "sent2sum_sim":     sent2sum_sim,
                "srl2sum_sim":      srl2sum_sim,
                "n_sent":           sentence_embeddings.shape[0],
                "n_srl":            srl_embs.shape[0],
            })

            # ── 7. HGT forward on GRAPH_BATCH samples at once ─────────────────────────
            is_last_sample = (sample_i == len(train_data) - 1)
            if len(hetero_batch_buffer) < GRAPH_BATCH and not is_last_sample:
                continue

            # Build a batched heterogeneous graph via PyG
            # Batch.from_data_list increments node indices in edge_index automatically.
            hetero_data_list = [
                build_hetero_data(buf["x_dict"], buf["edge_index_dict"])
                for buf in hetero_batch_buffer
            ]
            batched_graph = Batch.from_data_list(hetero_data_list)

            batched_x  = {ntype: batched_graph[ntype].x         for ntype in metadata[0]}
            batched_ei = {etype: batched_graph[etype].edge_index for etype in metadata[1]}
            sent_scores_all, srl_scores_all = hgt_ranking_model(batched_x, batched_ei)

            # ── 8. Split scores back and compute per-sample loss ──────────────────────
            sent_ptr   = 0
            srl_ptr    = 0
            total_loss = torch.tensor(0.0, device=device)

            for buf in hetero_batch_buffer:
                ns = buf["n_sent"]
                nr = buf["n_srl"]

                sent_scores = sent_scores_all[sent_ptr: sent_ptr + ns]
                srl_scores  = srl_scores_all[srl_ptr:  srl_ptr  + nr]
                sent_ptr += ns
                srl_ptr  += nr

                l1 = bpr_loss_with_sampling(sent_scores, buf["sent2sum_sim"].to(device))
                if len(srls['summary']) == 0 or buf["srl2sum_sim"] is None:
                    l2 = torch.tensor(0.0, device=device)
                else:
                    l2 = bpr_loss_with_sampling(srl_scores,  buf["srl2sum_sim"].to(device))
                l3 = sent_srl_consistency(sent_scores, srl_scores, buf["sent2srl_dict"])
                total_loss += w1 * l1 + w2 * l2 + w3 * l3

            # normalise by effective batch size
            total_loss = total_loss / (len(hetero_batch_buffer) * ACCUM_STEPS)
            total_loss.backward()

            hetero_batch_buffer.clear()
            global_step += 1

            if global_step % ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

                opt_step = global_step // ACCUM_STEPS  # number of completed optimizer steps
                if opt_step % VAL_FREQ == 0:
                    metrics = run_validation(
                        val_dataset=val_data,
                        srl_encoder=srl_encoder,
                        hgt_model=hgt_ranking_model,
                        num_samples=VAL_SAMPLES,
                        nlp=nlp,
                        transformer=transformer,
                        tokenizer=tokenizer,
                        shard_size=shard_size,
                        device=device,
                        split_name="valid",
                    )

                    print(_fmt_metrics(metrics, opt_step, "valid"))
        
                    is_best = metrics["val_score"] > best_val_score
                    if is_best:
                        best_val_score = metrics["val_score"]
        
                    save_checkpoint(args, opt_step, metrics, split="valid", is_best=is_best)



        ## final test
        test_data = dataset_with_coref_srl_new['test']
        test_metrics = run_validation(
            val_dataset=test_data,
            srl_encoder=srl_encoder,
            hgt_model=hgt_ranking_model,
            num_samples=len(test_data),
            nlp=nlp,
            transformer=transformer,
            tokenizer=tokenizer,
            shard_size=shard_size,
            device=device,
            split_name="test",
        )
        print(_fmt_metrics(test_metrics, "final_test", "test"))
        save_checkpoint(args, "final_test", test_metrics, split="test", is_best=False)