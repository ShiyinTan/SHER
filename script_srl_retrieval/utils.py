from collections import defaultdict
import re
from typing import List, Tuple

import torch
from rouge_score import rouge_scorer
import numpy as np


# =========================
# 1. Sentence Label
# =========================
def compute_sentence_labels(sentences, summary, scorer):
    """
    sentences: List[str]
    summary: str
    scorer: RougeScorer

    return: Tensor [num_sentence]
    """
    scores = []
    for sent in sentences:
        score = scorer.score(sent, summary)
        rouge_l = score['rougeL'].fmeasure
        scores.append(rouge_l)

    scores = torch.tensor(scores, dtype=torch.float)

    # normalize（很重要）
    if len(scores) > 1:
        scores = (scores - scores.mean()) / (scores.std() + 1e-8)

    return scores


#####=========================
# 2. build labels
# ==========================

def build_srl_srl_sim_edges(srl_relation_S, srl_argument_S, alpha, quantile_thresh):
    """
    quantile_thresh = 0.9: 前10%的similarity作为edges
    """
    srl_similarity = (alpha * srl_relation_S + (1-alpha) * srl_argument_S)

    srl_similarity = srl_similarity.fill_diagonal_(0)

    k = int(srl_similarity.numel() * quantile_thresh)

    threshold = torch.kthvalue(srl_similarity.view(-1), k).values

    # threshold = torch.quantile(srl_similarity, quantile_thresh)

    # 只保留上三角（k=1 表示去掉对角线）
    # tri_mask = torch.triu(torch.ones_like(srl_similarity, dtype=torch.bool), diagonal=1)
    # 同时满足：大于阈值 + 在上三角
    # mask = (srl_similarity > threshold) & tri_mask
    mask = srl_similarity > threshold
    
    top20_values = srl_similarity[mask]
    top20_indices = mask.nonzero(as_tuple=False)

    edge_srl2srl = top20_indices.T
    return edge_srl2srl

def build_sent_srl_edges(srl_list, sent2idx, srl2idx):
    """
    srl_list: List[str]
    """
    sent2srl_dict = defaultdict(list)

    src = []  # sentence
    dst = []  # srl

    for srl in srl_list:
        # 解析：doc_1_sent_10_srl_0
        parts = srl.split('_srl_')
        sent_str = parts[0]   # doc_1_sent_10

        # 映射
        sent_id = sent2idx[sent_str]
        srl_id = srl2idx[srl]

        src.append(sent_id)
        dst.append(srl_id)

        sent2srl_dict[sent_id].append(srl_id)

    edge_index = torch.tensor([src, dst], dtype=torch.long)

    return edge_index, sent2srl_dict

def build_doc_contain_sent_edges(sent_list, doc2idx, sent2idx):
    doc_src = []
    doc_dst = []
    
    for s in sent_list:
        doc = s.split('_sent_')[0]
        
        doc_id = doc2idx[doc]
        sent_id = sent2idx[s]
        
        doc_src.append(doc_id)
        doc_dst.append(sent_id)
    
    edge_doc2sent = torch.tensor([doc_src, doc_dst])
    return edge_doc2sent

def build_sent_next_edges(sent_list, sent2idx):
    """
    sent_list: List[str]
    e.g. "doc_1_sent_3"
    """

    doc2sents = defaultdict(list)
    for s in sent_list:
        # 解析
        doc_part, sent_part = s.split('_sent_')
        sent_id = int(sent_part)

        doc2sents[doc_part].append((sent_id, s))

    src, dst = [], []
    for doc, sents in doc2sents.items():
        sents_sorted = sorted(sents, key=lambda x: x[0])
        for i in range(len(sents_sorted) - 1):
            s1 = sents_sorted[i][1]
            s2 = sents_sorted[i + 1][1]
            src.append(sent2idx[s1])
            dst.append(sent2idx[s2])
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    return edge_index


def build_doc_fully_connected(num_docs, self_loop=False):
    src, dst = [], []

    for i in range(num_docs):
        for j in range(num_docs):
            if not self_loop and i == j:
                continue
            src.append(i)
            dst.append(j)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return edge_index


def preprocess_docs(docs):
    all_docs = docs.split("|||||")
    new_docs = []
    for i, doc in enumerate(all_docs):
        doc = re.sub(r'\s*\n\s*\n', ' ', str(doc))
        doc = re.sub(r'\s+', ' ', doc) # 替换多个空格为1个
        doc = doc.strip() # 新加入，使得前后空格消失
        
        if doc.strip() == '' or len(doc) < 50: # if empty
            continue
        new_docs.append(doc)

    return new_docs




# ── Pretty-print helper ────────────────────────────────────────────────────────
 
def _fmt_metrics(metrics: dict, step: int, split: str) -> str:
    """Build a compact multi-line string for console output."""
    lines = [f"\n{'─'*60}", f"  {split.capitalize()} @ step {step}", f"{'─'*60}"]
 
    groups = [
        ("Loss",      ["val_loss"]),
        ("Sentence",  ["sent_spearman", "sent_kendall",
                       "sent_ndcg@5", "sent_ndcg@10", "sent_ndcg@20", "sent_ndcg@50",
                       "sent_map", "sent_mrr",
                       "sent_p@5", "sent_p@10", "sent_p@20", "sent_p@50"]),
        ("SRL",       ["srl_spearman",  "srl_kendall",
                       "srl_ndcg@5",  "srl_ndcg@10", "srl_ndcg@20", "srl_ndcg@50",
                       "srl_map",  "srl_mrr",
                       "srl_p@5",  "srl_p@10", "srl_p@20", "srl_p@50"]),
        ("Composite", ["val_score"]),
    ]
 
    for group_name, keys in groups:
        lines.append(f"  {group_name}:")
        for k in keys:
            if k in metrics:
                lines.append(f"    {k:<20s} {metrics[k]:+.4f}")
    lines.append(f"{'─'*60}")
    return "\n".join(lines)
 
 


from scipy.stats import spearmanr, kendalltau
 
 
# ── Pure-numpy metric helpers (no torch dependency) ───────────────────────────
 
def _ndcg_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    """
    Normalised Discounted Cumulative Gain at k.
    labels are treated as relevance gains (continuous or binary).
    """
    n = len(scores)
    k = min(k, n)
    if k == 0:
        return 0.0
 
    order      = np.argsort(scores)[::-1]          # descending by score
    ideal_order = np.argsort(labels)[::-1]          # descending by label
 
    discounts = 1.0 / np.log2(np.arange(2, k + 2))  # 1/log2(2), 1/log2(3), …
 
    dcg  = np.sum(labels[order[:k]]  * discounts)
    idcg = np.sum(labels[ideal_order[:k]] * discounts)
 
    return float(dcg / idcg) if idcg > 0 else 0.0
 
 
def _map_score(scores: np.ndarray, binary_labels: np.ndarray) -> float:
    """
    Mean Average Precision.
    binary_labels: 1 for relevant, 0 otherwise.
    """
    order = np.argsort(scores)[::-1]
    rel   = binary_labels[order]
 
    if rel.sum() == 0:
        return 0.0
 
    precisions = []
    hits = 0
    for i, r in enumerate(rel):
        if r == 1:
            hits += 1
            precisions.append(hits / (i + 1))
 
    return float(np.mean(precisions))
 
 
def _mrr_score(scores: np.ndarray, binary_labels: np.ndarray) -> float:
    """
    Mean Reciprocal Rank – rank of the first relevant item.
    """
    order = np.argsort(scores)[::-1]
    for rank, idx in enumerate(order):
        if binary_labels[idx] == 1:
            return 1.0 / (rank + 1)
    return 0.0
 
 
def _precision_at_k(scores: np.ndarray, binary_labels: np.ndarray, k: int) -> float:
    """Fraction of top-k predictions that are relevant."""
    k = min(k, len(scores))
    order = np.argsort(scores)[::-1]
    return float(binary_labels[order[:k]].mean())
 

def compute_ranking_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    ndcg_ks: Tuple[int, ...] = (5, 10, 20, 50),
    prec_ks: Tuple[int, ...] = (5, 10, 20, 50),
) -> dict:
    """
    Compute all ranking metrics for a single sample.
 
    Args:
        scores : model output, shape [N]
        labels : continuous relevance labels (e.g. ROUGE-L), shape [N]
        ndcg_ks: k values for NDCG
        prec_ks: k values for Precision@k
 
    Returns:
        dict of metric_name -> float
    """
    result: dict = {}
    n = len(scores)
 
    if n < 2:
        # degenerate list – return zeros so the sample is skipped gracefully
        for k in ndcg_ks:
            result[f"ndcg@{k}"] = 0.0
        for k in prec_ks:
            result[f"p@{k}"] = 0.0
        result.update({"spearman": 0.0, "kendall": 0.0, "map": 0.0, "mrr": 0.0})
        return result
 
    # ── Spearman ρ ────────────────────────────────────────────────────────────
    rho, _ = spearmanr(scores, labels)
    result["spearman"] = 0.0 if np.isnan(rho) else float(rho)
 
    # ── Kendall τ ─────────────────────────────────────────────────────────────
    tau, _ = kendalltau(scores, labels)
    result["kendall"] = 0.0 if np.isnan(tau) else float(tau)
 
    # ── Binarise labels for threshold-based metrics ───────────────────────────
    # After z-score normalisation in compute_sentence_labels, 0 is the mean.
    # Items above the mean are "relevant".
    binary = (labels > 0).astype(float)
 
    # ── NDCG@k (continuous labels as gains) ───────────────────────────────────
    # Shift labels to [0, ∞) so negative gains don't hurt DCG artificially.
    shifted = labels - labels.min()
    for k in ndcg_ks:
        result[f"ndcg@{k}"] = _ndcg_at_k(scores, shifted, k)
 
    # ── MAP ───────────────────────────────────────────────────────────────────
    result["map"] = _map_score(scores, binary)
 
    # ── MRR ───────────────────────────────────────────────────────────────────
    result["mrr"] = _mrr_score(scores, binary)
 
    # ── Precision@k ───────────────────────────────────────────────────────────
    for k in prec_ks:
        result[f"p@{k}"] = _precision_at_k(scores, binary, k)
 
    return result

def aggregate_metrics(per_sample: List[dict]) -> dict:
    """Average each metric key over all samples. Returns a flat dict."""
    if not per_sample:
        return {}
    keys = per_sample[0].keys()
    return {k: float(np.mean([s[k] for s in per_sample])) for k in keys}