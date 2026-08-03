import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset, load_from_disk, concatenate_datasets, DatasetDict

import numpy as np
import pandas as pd
import os
import json
from collections import defaultdict
from typing import List, Tuple, Optional
from tqdm import tqdm


from rouge_score import rouge_scorer

from models import (SRLAttentionEncoder, 
                    HGTRankingModel, 
                    DocAttentionPooling, 
                    bpr_loss_with_sampling, 
                    sent_srl_consistency)
from utils import (
    compute_sentence_labels,
    build_srl_srl_sim_edges,
    build_sent_srl_edges,
    build_doc_contain_sent_edges,
    build_sent_next_edges,
    build_doc_fully_connected,
    preprocess_docs,
)


import re




## 模型和tokenizer加载放在外面，避免重复加载
import spacy

nlp = spacy.load("en_core_web_sm")

from sentence_transformers import SentenceTransformer
import torch

st_model = SentenceTransformer("all-mpnet-base-v2")

device = 'cuda:0'
tokenizer = st_model.tokenizer
transformer = st_model._first_module().auto_model
transformer = transformer.to(device)


def sentence_split_with_span(text):
    doc = nlp(text)
    return [{"text": s.text, "start": s.start_char, "end": s.end_char} for s in doc.sents]


def get_docs_sentences(docs, max_char=40000):
    return [sentence_split_with_span(doc[:max_char]) for doc in docs]

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



def merge_sliding_window_outputs(
    token_embeddings,
    offset_mapping: List[List[Tuple[int, int]]],
    attention_mask = None,
    merge_strategy = "mean",
):
    """
    Merge token embeddings from sliding windows back to the original
    non-chunked token sequence.

    Args:
        token_embeddings: Tensor of shape [num_chunks, max_len, hidden_dim]
        offset_mapping: List[List[(char_start, char_end)]], global char spans
        attention_mask: Optional Tensor of shape [num_chunks, max_len]
        merge_strategy: One of {"mean", "max", "first"}

    Returns:
        final_embeddings:
            Tensor [num_original_tokens, hidden_dim]
        final_offsets:
            List[(char_start, char_end)]
    """
    assert merge_strategy in {"mean", "max", "first"}
    num_chunks, max_len, hidden_dim = token_embeddings.shape
    # (char_start, char_end) -> list of embeddings
    span2embeds = defaultdict(list)

    for chunk_id in range(num_chunks):
        for tok_id in range(max_len):
            s, e = offset_mapping[chunk_id][tok_id]

            # padding or special tokens
            if s == 0 and e == 0:
                continue
            if attention_mask is not None:
                if attention_mask[chunk_id, tok_id].item() == 0:
                    continue
            emb = token_embeddings[chunk_id, tok_id]
            span2embeds[(s, e)].append(emb)
    
    # sort by original token order
    sorted_spans = sorted(span2embeds.keys(), key=lambda x: x[0])
    final_embeddings, final_offsets = [], []
    for span in sorted_spans:
        embeds = span2embeds[span]
        if merge_strategy == "mean":
            merged = torch.stack(embeds, dim=0).mean(dim=0)
        elif merge_strategy == "max":
            merged = torch.stack(embeds, dim=0).max(dim=0).values
        elif merge_strategy == "first":
            merged = embeds[0]
        final_embeddings.append(merged)
        final_offsets.append(span)
    final_embeddings = torch.stack(final_embeddings, dim=0)

    return final_embeddings, final_offsets


def get_docs_embedding_enc_offsets_by_sentences(docs_sentences):
    docs_embedding_by_sentences = []
    docs_enc_by_sentences = []
    docs_offsets_by_sentences = []
    
    for sentences in docs_sentences:
        embedding_by_sentences = []
        enc_by_sentences = []
        offsets_by_sentences = []
        
        for sent_i, sent in enumerate(sentences):
            # enc = tokenizer(
            #     sent['text'],
            #     return_offsets_mapping=True,
            #     return_tensors="pt"
            # )
            # 使用sliding window+overflow，保证过长sentence也能正常encoding
            enc = tokenizer(
                sent['text'],
                return_offsets_mapping=True,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
                stride=128,
                return_overflowing_tokens=True
            )
        
            enc = {k: v.to(device) for k, v in enc.items()}
            
            offsets = enc["offset_mapping"].cpu().tolist()

            # print(enc["offset_mapping"].shape)
            
            with torch.no_grad():
                outputs = transformer(**enc)
            
            token_embeddings = outputs.last_hidden_state  # (num_chunk_per_sent, seq_len, hidden_dim)

            ## 使用sliding window的tokenizer后，需要merging还原回非sliding的状态，让后续操作感知不到该chunk过程
            ### 如果token_embeddings的shape第二维为2，意味着这个chunk或句子为空，该怎么办
            ### 对于整个句子为空的情况，我们直接一个0向量，offset设置为[0,0]
            if token_embeddings.shape[0]==1 and token_embeddings.shape[1]==2:
                print(f"sentence {sent_i} is empty: {sent['text']}. Set to zero tensor, offset [(0,0)].")
                final_embeddings = torch.zeros(1, token_embeddings.shape[2], device=device)
                final_offsets = [(0,0)]
            else:
                final_embeddings, final_offsets = merge_sliding_window_outputs(token_embeddings, offsets) # (seq_len, hidden_dim)
            seq_len = final_embeddings.shape[0]
            
            if seq_len > 384:
                print(f"sentence_id: {sent_i} has long sequence: {seq_len}")
            
            embedding_by_sentences.append(final_embeddings)
            enc_by_sentences.append(enc)
            offsets_by_sentences.append(final_offsets)
    
        docs_embedding_by_sentences.append(embedding_by_sentences)
        docs_enc_by_sentences.append(enc_by_sentences)
        docs_offsets_by_sentences.append(offsets_by_sentences)

    return docs_embedding_by_sentences, docs_enc_by_sentences, docs_offsets_by_sentences




def get_overall_srl_embeddings(srls, docs_sentences, docs_embedding_by_sentences, docs_offsets_by_sentences, embedding_dim=768):
    overall_srl_embeddings = {}

    for doc_k in srls['document'].keys():
        doc_id = int(doc_k)
        for k,v in srls['document'][doc_k].items():
            srls_in_sentence = srls['document'][doc_k][k]
            for srl_id in srls_in_sentence.keys():
                srl = srls_in_sentence[srl_id]
                overall_srl_embeddings[srl_id] = {}
                for argument in srl:
                    arg_char_span = argument[0]
                    arg_role = argument[1]
                    arg_text = argument[2]
        
                    sentence_ids = find_sentence_ids(arg_char_span[0], arg_char_span[1], docs_sentences[doc_id])

                    for arg_sent_id in sentence_ids:
                        # arg_sent_id = sentence_ids[-1]
                        arg_sent_local_char_span = (arg_char_span[0] - docs_sentences[doc_id][arg_sent_id]['start'], arg_char_span[1] - docs_sentences[doc_id][arg_sent_id]['start'])
                        arg_sent_local_token_indices = char_span_to_token_indices(arg_sent_local_char_span, docs_offsets_by_sentences[doc_id][arg_sent_id])
                        srl_embedding = get_srl_embedding(docs_embedding_by_sentences[doc_id][arg_sent_id], arg_sent_local_token_indices)
                        if srl_embedding is None:
                            # print(f'srl embedding is None, doc: {doc_id}, sent: {k}')
                            continue

                    if srl_embedding is None:
                        print(f'srl embedding is None, doc: {doc_id}, sent: {k}')
                        srl_embedding = torch.zeros(embedding_dim, device=device)
                    
                    overall_srl_embeddings[srl_id][arg_role] = srl_embedding
    return overall_srl_embeddings

def get_summary_sentencecs(summary):
    summary_sentences = sentence_split_with_span(summary)
    return summary_sentences

def get_summary_embedding_enc_offsets_by_sentences(summary_sentences):
    summary_embedding_by_sentences = []
    summary_enc_by_sentences = []
    summary_offsets_by_sentences = []

    for sent in summary_sentences:
        # enc = tokenizer(
        #         sent['text'],
        #         return_offsets_mapping=True,
        #         return_tensors="pt"
        #     )
        # 使用sliding window+overflow，保证过长sentence也能正常encoding
        enc = tokenizer(
            sent['text'],
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
            stride=128,
            return_overflowing_tokens=True
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        offsets = enc["offset_mapping"].cpu().tolist()

        with torch.no_grad():
            outputs = transformer(**enc)

        token_embeddings = outputs.last_hidden_state  # (1, seq_len, hidden_dim)

        final_embeddings, final_offsets = merge_sliding_window_outputs(token_embeddings, offsets) # (seq_len, hidden_dim)
        
        summary_embedding_by_sentences.append(final_embeddings)
        summary_enc_by_sentences.append(enc)
        summary_offsets_by_sentences.append(final_offsets)

    return summary_embedding_by_sentences, summary_enc_by_sentences, summary_offsets_by_sentences



def get_overall_srl_embeddings_summary(srls, summary_sentences, summary_embedding_by_sentences, summary_offsets_by_sentences, embedding_dim=768):
    overall_srl_embeddings_summary = {}

    for k,v in srls['summary'].items():
        srls_in_sentence = srls['summary'][k]
        for srl_id in srls_in_sentence.keys():
            srl = srls_in_sentence[srl_id]
            overall_srl_embeddings_summary[srl_id] = {}
            for argument in srl:
                arg_char_span = argument[0]
                arg_role = argument[1]
                arg_text = argument[2]

                # print(arg_char_span)
                sentence_ids = find_sentence_ids(arg_char_span[0], arg_char_span[1], summary_sentences)

                for arg_sent_id in sentence_ids:
                    # arg_sent_id = sentence_ids[-1]
                    arg_sent_local_char_span = (arg_char_span[0] - summary_sentences[arg_sent_id]['start'], arg_char_span[1] - summary_sentences[arg_sent_id]['start'])
                    arg_sent_local_token_indices = char_span_to_token_indices(arg_sent_local_char_span, summary_offsets_by_sentences[arg_sent_id])
                    srl_embedding = get_srl_embedding(summary_embedding_by_sentences[arg_sent_id], arg_sent_local_token_indices)
                    if srl_embedding is None:
                        continue
                if srl_embedding is None:
                    print(f'srl embedding is None, summary, sent {k}')
                    srl_embedding = torch.zeros(embedding_dim, device=device)
                overall_srl_embeddings_summary[srl_id][arg_role] = srl_embedding
    return overall_srl_embeddings_summary








# 初始化 ROUGE
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)


home_dir = os.path.expanduser('~')
data_dir = os.path.join(home_dir, "Code/NLP/dataset_MDS/MDS-RA2_data2")


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, default="multi_news") # wcep, multi_x_science_sum, wikisum, multi_news
args = parser.parse_args()

data_name = args.data_name # wcep, multi_x_science_sum, wikisum, multi_news

dataset_with_coref_srl_new = load_dataset(f"{data_name}_with_srl_new")


alpha_srl_label = 0.5
dim = 768

device = 'cuda:0'

# 损失参数
w1 = 0.5
w2 = 0.5
w3 = 0.5

metadata = (['document', 'sentence', 'srl'],
 [('document', 'connect', 'document'),
  ('document', 'contain', 'sentence'),
  ('sentence', 'in', 'document'),
  ('sentence', 'next', 'sentence'),
  ('sentence', 'prev', 'sentence'),
  ('sentence', 'contain', 'srl'),
  ('srl', 'in', 'sentence'),
  ('srl', 'sim', 'srl'),
  ('srl', 'rev_sim', 'srl')])

in_dim_dict = {
    "document": dim,
    "sentence": dim,
    "srl": dim
}


# model create

srl_encoder = SRLAttentionEncoder(hidden_dim=dim, device=device)
hgt_ranking_model = HGTRankingModel(metadata, in_dim_dict, device=device)


# optimizer create

accum_steps = 8  # 等价 batch size = 8

optimizer = torch.optim.Adam([
    {"params": srl_encoder.parameters(), "lr": 1e-4},
    {"params": hgt_ranking_model.parameters(), "lr": 1e-4},
])



## TODO：validation和test验证函数+save model

# for epoch in range(epoch)
optimizer.zero_grad()

for step, sample_i in enumerate(tqdm(range(len(dataset_with_coref_srl_new['train'])))):

    # Load overall similarity shard data every shard
    shard_idx = sample_i // 2000
    idx_in_shard = sample_i % 2000
    shard_file = os.path.join(data_dir, f"{data_name}_overall_similarities_train_shard_{shard_idx}.pt")
    shard_data = torch.load(shard_file)
    
    ## load sample
    sample_coref_srl = dataset_with_coref_srl_new['train'][sample_i]
    sample_srl_sim = shard_data[idx_in_shard]
    srls = json.loads(sample_coref_srl['srl'])

    ## process sample
    docs = preprocess_docs(sample_coref_srl['document'])
    summary = sample_coref_srl['summary']
    docs_sentences = get_docs_sentences(docs)

    ## get embeddings
    docs_embedding_by_sentences, docs_enc_by_sentences, docs_offsets_by_sentences = get_docs_embedding_enc_offsets_by_sentences(docs_sentences)
    # =========================
    # 1. sentence/doc embedding 获取
    # =========================
    sentence_embeddings = []
    doc_embeddings = []
    for doc in docs_embedding_by_sentences:
        doc_sent_emb = []
        for sent in doc:
            # sent: (num_srl, dim)
            sent_emb = sent.mean(dim=0)
            doc_sent_emb.append(sent_emb)
            sentence_embeddings.append(sent_emb)
    
        doc_embedding = torch.stack(doc_sent_emb, dim=0).mean(0)
        doc_embeddings.append(doc_embedding)
    sentence_embeddings = torch.stack(sentence_embeddings, dim=0)
    doc_embeddings = torch.stack(doc_embeddings, dim=0)

    # =========================
    # 2. srl embedding 获取
    # =========================
    srl_embs_with_roles = get_overall_srl_embeddings(srls, docs_sentences, docs_embedding_by_sentences, docs_offsets_by_sentences)
    srl_embs = [srl_encoder(srl_dict) for srl_key, srl_dict in srl_embs_with_roles.items()]
    srl_embs = torch.stack(srl_embs, dim=0)

    ## 构建label
    # =========================
    # 1. Sentence Label
    # =========================
    all_sentences = [sent['text'] for doc_sentences in docs_sentences for sent in doc_sentences]
    sent2sum_sim = compute_sentence_labels(all_sentences, summary, scorer)
    # =========================
    # 2. SRL Label
    # =========================
    srl2sum_sim_relation = torch.max(sample_srl_sim['srl_relation_S_to_summary'], dim=1).values
    srl2sum_sim_argument = torch.max(sample_srl_sim['srl_argument_S_summary'], dim=1).values
    srl2sum_sim = alpha_srl_label * srl2sum_sim_relation + (1-alpha_srl_label) * srl2sum_sim_argument

    ## 构建图
    sent2idx = {s: i for i, s in enumerate(sample_srl_sim['sent_keys'])}
    doc_set = sorted(set([s.split('_sent_')[0] for s in sample_srl_sim['sent_keys']]))
    doc2idx = {d: i for i, d in enumerate(doc_set)}
    sent_num = sentence_embeddings.shape[0]
    srl_num = srl_embs.shape[0]
    # =========================
    # 1. SRL SRL
    # =========================
    edge_srl2srl_sim = build_srl_srl_sim_edges(sample_srl_sim['srl_relation_S'], sample_srl_sim['srl_argument_S'], 0.5, 0.9)
    # =========================
    # 2. Sentence SRL
    # =========================
    edge_sent2srl_contain = torch.stack([torch.randint(0, sent_num, (srl_num,)), torch.arange(srl_num)]) ## TODO, sent2srl修改
    sent2srl_dict = defaultdict(list)
    for i in range(edge_sent2srl_contain.shape[1]):
        sent2srl_dict[edge_sent2srl_contain[0][i].item()].append(edge_sent2srl_contain[1][i].item())
    # =========================
    # 3. Sentence Sentence
    # =========================
    edge_sent2sent_next = build_sent_next_edges(sample_srl_sim['sent_keys'])
    edge_doc2sent_contain = build_doc_contain_sent_edges(sample_srl_sim['sent_keys'], sent2idx=sent2idx, doc2idx=doc2idx)
    # =========================
    # 4. Doc Doc
    # =========================
    edge_doc2doc_fullconnect = build_doc_fully_connected(len(docs))

    ## 构建数据
    x_dict = {
        "document": doc_embeddings,
        "sentence": sentence_embeddings,
        "srl": srl_embs
    }
    edge_index_dict = {
        # 🔥 Document ↔ Document
        ("document", "connect", "document"): edge_doc2doc_fullconnect.to(device),
    
        # Document ↔ Sentence
        ("document", "contain", "sentence"): edge_doc2sent_contain.to(device),
        ("sentence", "in", "document"): edge_doc2sent_contain.flip(0).to(device),
    
        # Sentence ↔ Sentence
        ("sentence", "next", "sentence"): edge_sent2sent_next.to(device),
        ("sentence", "prev", "sentence"): edge_sent2sent_next.flip(0).to(device),
        
        # Sentence ↔ SRL
        ("sentence", "contain", "srl"): edge_sent2srl_contain.to(device),
        ("srl", "in", "sentence"): edge_sent2srl_contain.flip(0).to(device),
    
        # SRL ↔ SRL
        ("srl", "sim", "srl"): edge_srl2srl_sim.to(device),
        ("srl", "rev_sim", "srl"): edge_srl2srl_sim.flip(0).to(device),
    }

    ## 前向过程
    sent_scores, srl_scores = hgt_ranking_model(x_dict, edge_index_dict)

    ## 损失计算
    loss_sent_importance = bpr_loss_with_sampling(sent_scores, sent2sum_sim.to(device))
    loss_srl_importance = bpr_loss_with_sampling(srl_scores, srl2sum_sim.to(device))
    loss_sent_srl_consistency = sent_srl_consistency(sent_scores, srl_scores, sent2srl_dict)

    loss = w1 * loss_sent_importance + w2 * loss_srl_importance + w3 * loss_sent_srl_consistency

    loss = loss / accum_steps

    ## BP 过程
    loss.backward()
    if (step + 1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()