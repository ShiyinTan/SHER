
import os
import spacy
from fastcoref import spacy_component
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict


import re

from collections import defaultdict
import random
import numpy as np
import pandas as pd

from datasets import load_dataset, load_from_disk, concatenate_datasets, DatasetDict

import glob # 用于搜索shard的文件

import json


from tqdm import tqdm
import torch.nn.functional as F
from typing import List, Tuple, Optional

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

import re
import spacy

nlp = spacy.load("en_core_web_sm")

def sentence_split_with_span(text):
    doc = nlp(text)
    sentences = []
    for sent in doc.sents:
        sentences.append({
            "text": sent.text,
            "start": sent.start_char,
            "end": sent.end_char
        })
    return sentences

def get_docs_sentences(docs, max_char):
    docs_sentences = []
    
    for doc in docs:
        sentences = sentence_split_with_span(doc[:max_char])
        docs_sentences.append(sentences)
        
    return docs_sentences

def get_summary_sentencecs(summary):
    summary_sentences = sentence_split_with_span(summary)
    return summary_sentences


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
        token_embeddings:
            Tensor of shape [num_chunks, max_len, hidden_dim]
        offset_mapping:
            List[List[(char_start, char_end)]], global char spans
        attention_mask:
            Optional Tensor of shape [num_chunks, max_len]
        merge_strategy:
            One of {"mean", "max", "first"}

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

    final_embeddings = []
    final_offsets = []

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

def get_summary_embedding_enc_offsets_by_sentences(summary_sentences):
    summary_embedding_by_sentences = []
    summary_enc_by_sentences = []
    summary_offsets_by_sentences = []

    for sent_i, sent in enumerate(summary_sentences):
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

        if token_embeddings.shape[0]==1 and token_embeddings.shape[1]==2:
            print(f"sentence {sent_i} is empty: {sent['text']}. Set to zero tensor, offset [(0,0)].")
            final_embeddings = torch.zeros(1, token_embeddings.shape[2], device=device)
            final_offsets = [(0,0)]
        else:
            final_embeddings, final_offsets = merge_sliding_window_outputs(token_embeddings, offsets) # (seq_len, hidden_dim)
        
        summary_embedding_by_sentences.append(final_embeddings)
        summary_enc_by_sentences.append(enc)
        summary_offsets_by_sentences.append(final_offsets)

    return summary_embedding_by_sentences, summary_enc_by_sentences, summary_offsets_by_sentences


def find_sentence_ids(c_s, c_e, sentences):
    sent_ids = []
    for i, sent in enumerate(sentences):
        s_s, s_e = sent['start'], sent['end']
        # 有 overlap 就算
        if not (c_e <= s_s or c_s >= s_e):
            sent_ids.append(i)
    return sent_ids

def char_span_to_token_indices(char_span, offsets):
    s, e = char_span
    return [
        i for i, (cs, ce) in enumerate(offsets)
        if not (ce <= s or cs >= e)
    ]

def get_srl_embedding(token_embeddings, token_indices, pooling="mean"):
    if len(token_indices) == 0:
        return None
    
    span_tokens = token_embeddings[token_indices]
    
    if pooling == "mean":
        return span_tokens.mean(dim=0)
    elif pooling == "max":
        return span_tokens.max(dim=0).values
    else:
        raise ValueError("Unknown pooling")
        

def get_overall_srl_embeddings(srls, docs_sentences, docs_embedding_by_sentences, docs_offsets_by_sentences):
    overall_srl_embeddings = {}

    embedding_dim = 768
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

def get_overall_srl_embeddings_summary(srls, summary_sentences, summary_embedding_by_sentences, summary_offsets_by_sentences):
    overall_srl_embeddings_summary = {}

    embedding_dim = 768
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
                    print(f'srl embedding is None, summary, sent {k}, srl {srl_id}, role {arg_role}: {arg_text}')
                    srl_embedding = torch.zeros(embedding_dim, device=device)
                overall_srl_embeddings_summary[srl_id][arg_role] = srl_embedding
    return overall_srl_embeddings_summary

def get_overall_sent_embeddings(docs_embedding_by_sentences):
    overall_sent_embeddings = {}

    for doc_id, doc_embeddinigs in enumerate(docs_embedding_by_sentences):
        for sent_id, sentence_embedding in enumerate(doc_embeddinigs):
            sent_emb = sentence_embedding.mean(0)
            overall_sent_embeddings[f'doc_{doc_id}_sent_{sent_id}'] = sent_emb

    return overall_sent_embeddings

## 分relation embeddings和分arguments embeddings
def get_srl_relation_argument_S(overall_srl_embeddings):
    srl_keys = []
    srl_relation_embeddings = []
    srl_arguments_embeddings = []

    if len(overall_srl_embeddings) == 0:
        print("overall_srl_embeddings is empty")
        return srl_keys, None, None
    
    for srl_k in overall_srl_embeddings.keys():
        srl_embeddings = overall_srl_embeddings[srl_k]
        srl_keys.append(srl_k)
        srl_relation_embeddings.append(srl_embeddings['relation'])
        srl_args_embeddings = []
        for srl_arg_role in srl_embeddings.keys():
            if srl_arg_role != 'relation':
                srl_args_embeddings.append(srl_embeddings[srl_arg_role])
        for i in range(len(srl_args_embeddings), 7):
            srl_args_embeddings.append(torch.zeros(srl_embeddings['relation'].shape[0], device=device))
        srl_arguments_embeddings.append(srl_args_embeddings)

    srl_relation_embeddings = torch.stack(srl_relation_embeddings, dim=0)

    srl_relation_embeddings = F.normalize(srl_relation_embeddings, dim=-1) # pairwise similarity
    srl_relation_S = srl_relation_embeddings @ srl_relation_embeddings.T
    
    srl_relation_S = srl_relation_S - torch.eye(srl_relation_S.shape[0], device=device)


    srl_arguments_embeddings = torch.stack(
        [torch.stack(inner, dim=0) for inner in srl_arguments_embeddings],
        dim=0
    )
    
    srl_arguments_embeddings = F.normalize(srl_arguments_embeddings, dim=-1) # pairwise similarity
    
    out = torch.einsum("bkd,Bld->bBkl", srl_arguments_embeddings, srl_arguments_embeddings)
    
    B, _, K, _ = out.shape
    
    idx = torch.arange(K, device=out.device)
    out[:, :, idx, idx] = 0
    
    A2B_sim = out.max(dim=-1).values.mean(dim=-1)
    B2A_sim = out.max(dim=-2).values.mean(dim=-1)
    
    srl_argument_S = A2B_sim

    # doc_sample = {
    #     "srl_keys": srl_keys,
    #     "srl_relation_S": srl_relation_S.cpu(), 
    #     "srl_argument_S": srl_argument_S.cpu()
    # }
    return srl_keys, srl_relation_S.cpu(), srl_argument_S.cpu()


# each srl in documents to summary's similarity
def get_srl_relation_argument_S_docsrl_to_summary(overall_srl_embeddings, overall_srl_embeddings_summary):
    srl_keys = []
    srl_keys_summary = []
    srl_relation_embeddings = []
    srl_arguments_embeddings = []
    srl_relation_embeddings_summary = []
    srl_arguments_embeddings_summary = []

    if len(overall_srl_embeddings) == 0 or len(overall_srl_embeddings_summary)==0:
        print("overall_srl_embeddings or overall_srl_embeddings_summary is empty")
        return srl_keys, srl_keys_summary, None, None
    
    for srl_k in overall_srl_embeddings.keys():
        srl_embeddings = overall_srl_embeddings[srl_k]
        srl_keys.append(srl_k)
        srl_relation_embeddings.append(srl_embeddings['relation'])
        srl_args_embeddings = []
        for srl_arg_role in srl_embeddings.keys():
            if srl_arg_role != 'relation':
                srl_args_embeddings.append(srl_embeddings[srl_arg_role])
        for i in range(len(srl_args_embeddings), 7):
            srl_args_embeddings.append(torch.zeros(srl_embeddings['relation'].shape[0], device=device))        
        srl_arguments_embeddings.append(srl_args_embeddings)

    for srl_k in overall_srl_embeddings_summary.keys():
        srl_embeddings = overall_srl_embeddings_summary[srl_k]
        srl_keys_summary.append(srl_k)
        srl_relation_embeddings_summary.append(srl_embeddings['relation'])
        srl_args_embeddings = []
        for srl_arg_role in srl_embeddings.keys():
            if srl_arg_role != 'relation':
                srl_args_embeddings.append(srl_embeddings[srl_arg_role])
        for i in range(len(srl_args_embeddings), 7):
            srl_args_embeddings.append(torch.zeros(srl_embeddings['relation'].shape[0], device=device))        
        srl_arguments_embeddings_summary.append(srl_args_embeddings)

    srl_relation_embeddings = torch.stack(srl_relation_embeddings, dim=0)
    srl_relation_embeddings = F.normalize(srl_relation_embeddings, dim=-1) # pairwise similarity
    srl_relation_embeddings_summary = torch.stack(srl_relation_embeddings_summary, dim=0)
    srl_relation_embeddings_summary = F.normalize(srl_relation_embeddings_summary, dim=-1) # pairwise similarity

    # print(srl_relation_embeddings.shape, srl_relation_embeddings_summary.shape)
    
    srl_relation_S_to_summary = srl_relation_embeddings @ srl_relation_embeddings_summary.T

    # srl_relation_S_to_summary = np.fill_diagonal(srl_relation_S_to_summary.cpu().numpy(), 0)
    # srl_relation_S_to_summary = srl_relation_S_to_summary - torch.eye(srl_relation_S_to_summary.shape[0], device=device)

    
    srl_arguments_embeddings = torch.stack(
        [torch.stack(inner, dim=0) for inner in srl_arguments_embeddings],
        dim=0
    )
    srl_arguments_embeddings = F.normalize(srl_arguments_embeddings, dim=-1) # pairwise similarity
    srl_arguments_embeddings_summary = torch.stack(
        [torch.stack(inner, dim=0) for inner in srl_arguments_embeddings_summary],
        dim=0
    )
    srl_arguments_embeddings_summary = F.normalize(srl_arguments_embeddings_summary, dim=-1) # pairwise similarity
    
    out = torch.einsum("bkd,Bld->bBkl", srl_arguments_embeddings, srl_arguments_embeddings_summary)
    # print(out.shape)
    # # out = np.fill_diagonal(out, 0)
    # print(out)
    
    A2B_sim = out.max(dim=-1).values.mean(dim=-1)
    B2A_sim = out.max(dim=-2).values.mean(dim=-1)
    
    srl_argument_S_summary = A2B_sim

    return srl_keys, srl_keys_summary, srl_relation_S_to_summary.cpu(), srl_argument_S_summary.cpu()


def get_sent_S(overall_sent_embeddings):
    sent_keys = []
    sent_embeddings = []
    
    for sent_key, sent_embs in overall_sent_embeddings.items():
        sent_keys.append(sent_key)
        sent_embeddings.append(sent_embs)

    sent_embeddings = torch.stack(sent_embeddings)

    sent_embeddings = F.normalize(sent_embeddings, dim=-1) # pairwise similarity
    sent_S = sent_embeddings @ sent_embeddings.T
    
    sent_S = sent_S - torch.eye(sent_S.shape[0], device=device)

    return sent_keys, sent_S.cpu()




## Load dataset

import argparse

parser = argparse.ArgumentParser()

# 添加参数
parser.add_argument("--data_name", type=str, default="multi_news") # wcep, multi_x_science_sum, wikisum, multi_news

args = parser.parse_args()

data_name = args.data_name

if data_name in ["multi_news"]:
    dataset_with_srl_new = load_dataset(f"{data_name}_with_srl_new") # saved dataset with srls
    print("load dataset: multi_news")
elif data_name in ["multi_x_science_sum", "wikisum", "arxiv"]:
    dataset_with_srl_new = load_dataset(f"{data_name}_with_srl_new")
    print(f"load dataset: {data_name}")
else:
    dataset_with_srl_new = load_dataset(f"{data_name}_with_srl_new")
    print(f"load dataset: {data_name}")
print(dataset_with_srl_new)


from sentence_transformers import SentenceTransformer
import torch

st_model = SentenceTransformer("all-mpnet-base-v2")

tokenizer = st_model.tokenizer
transformer = st_model._first_module().auto_model

device = transformer.device

max_char = 40000

# 设置每个 shard 的大小
SHARD_SIZE = 1000 # bigsurvey: 1000 peersum: 1000

shard_data = []
for idx, sample in enumerate(tqdm(dataset_with_srl_new['train'])):
    
    docs = preprocess_docs(sample['document'])
    srls = json.loads(sample['srl'])
    summary = sample['summary']

    shard_idx = idx // SHARD_SIZE
    shard_file = f"../dataset_MDS/MDS-RA2_data2/{data_name}_overall_similarities_train_shard_{shard_idx}.pt"
    if os.path.exists(shard_file):
        print(f"Shard {shard_idx} 已存在，跳过")
        continue
    
    if len(docs) == 0:
        shard_data.append({})
    else:
        if len(srls['summary'])==0:
            doc_sample = {
                "srl_keys": None,
                "srl_keys_summary": None,
                "srl_relation_S": None, 
                "srl_argument_S": None,
                "srl_relation_S_to_summary": None,
                "srl_argument_S_summary": None,
                "sent_keys": None,
                "sent_S": None
            }
        else:
            docs_sentences = get_docs_sentences(docs, max_char)
            summary_sentences = get_summary_sentencecs(summary)
            
            docs_embedding_by_sentences, docs_enc_by_sentences, docs_offsets_by_sentences = get_docs_embedding_enc_offsets_by_sentences(docs_sentences)
            summary_embedding_by_sentences, summary_enc_by_sentences, summary_offsets_by_sentences = get_summary_embedding_enc_offsets_by_sentences(summary_sentences)
            
            overall_srl_embeddings = get_overall_srl_embeddings(srls, docs_sentences, docs_embedding_by_sentences, docs_offsets_by_sentences)
            overall_srl_embeddings_summary = get_overall_srl_embeddings_summary(srls, summary_sentences, summary_embedding_by_sentences, summary_offsets_by_sentences)
            overall_sent_embeddings = get_overall_sent_embeddings(docs_embedding_by_sentences)
            sent_keys, sent_S = get_sent_S(overall_sent_embeddings)
            
            srl_keys, srl_relation_S, srl_argument_S  = get_srl_relation_argument_S(overall_srl_embeddings)
            srl_keys, srl_keys_summary, srl_relation_S_to_summary, srl_argument_S_summary = get_srl_relation_argument_S_docsrl_to_summary(overall_srl_embeddings, overall_srl_embeddings_summary)
            
            
            doc_sample = {
                "srl_keys": srl_keys,
                "srl_keys_summary": srl_keys_summary,
                "srl_relation_S": srl_relation_S, 
                "srl_argument_S": srl_argument_S,
                "srl_relation_S_to_summary": srl_relation_S_to_summary,
                "srl_argument_S_summary": srl_argument_S_summary,
                "sent_keys": sent_keys,
                "sent_S": sent_S.cpu()
            }

        shard_data.append(doc_sample)

    # 达到 shard 大小就保存
    if (idx + 1) % SHARD_SIZE == 0:
        torch.save(shard_data, shard_file)
        print(f"Saved {shard_file} with {len(shard_data)} samples")
        shard_data = []  # 清空当前 shard
    
# 保存最后一个 shard（不足 SHARD_SIZE 的部分）
if len(shard_data) > 0:
    shard_idx = len(dataset_with_srl_new['train']) // SHARD_SIZE
    shard_file = f"../dataset_MDS/MDS-RA2_data2/{data_name}_overall_similarities_train_shard_{shard_idx}.pt"
    torch.save(shard_data, shard_file)
    print(f"Saved {shard_file} with {len(shard_data)} samples")

print("train数据完成!!!!!!!!!!!!!!!!!!!!!!!")

# 读取：
# import torch
# import glob
# # 获取所有 shard 文件
# shard_files = sorted(glob.glob("overall_similarities_train_shard_*.pt"))
# # 用于存放全部数据（如果内存够）
# overall_similarities_train = []
# for shard_file in shard_files:
#     shard_data = torch.load(shard_file)  # 读取一个 shard
#     overall_similarities_train.extend(shard_data)  # 合并到总列表
#     print(f"Loaded {shard_file} with {len(shard_data)} samples")
# print(f"Total samples loaded: {len(overall_similarities_train)}")



shard_data = []
for idx, sample in enumerate(tqdm(dataset_with_srl_new['validation'])):
    docs = preprocess_docs(sample['document'])
    srls = json.loads(sample['srl'])
    summary = sample['summary']

    shard_idx = idx // SHARD_SIZE
    shard_file = f"../dataset_MDS/MDS-RA2_data2/{data_name}_overall_similarities_valid_shard_{shard_idx}.pt"
    if os.path.exists(shard_file):
        print(f"Shard {shard_idx} 已存在，跳过")
        continue
    
    if len(docs) == 0:
        shard_data.append({})
    else:
        if len(srls['summary'])==0:
            doc_sample = {
                "srl_keys": None,
                "srl_keys_summary": None,
                "srl_relation_S": None, 
                "srl_argument_S": None,
                "srl_relation_S_to_summary": None,
                "srl_argument_S_summary": None,
                "sent_keys": None,
                "sent_S": None
            }
        else:
            docs_sentences = get_docs_sentences(docs, max_char)
            summary_sentences = get_summary_sentencecs(summary)
            
            docs_embedding_by_sentences, docs_enc_by_sentences, docs_offsets_by_sentences = get_docs_embedding_enc_offsets_by_sentences(docs_sentences)
            summary_embedding_by_sentences, summary_enc_by_sentences, summary_offsets_by_sentences = get_summary_embedding_enc_offsets_by_sentences(summary_sentences)
            
            overall_srl_embeddings = get_overall_srl_embeddings(srls, docs_sentences, docs_embedding_by_sentences, docs_offsets_by_sentences)
            overall_srl_embeddings_summary = get_overall_srl_embeddings_summary(srls, summary_sentences, summary_embedding_by_sentences, summary_offsets_by_sentences)
            overall_sent_embeddings = get_overall_sent_embeddings(docs_embedding_by_sentences)
            sent_keys, sent_S = get_sent_S(overall_sent_embeddings)
            
            srl_keys, srl_relation_S, srl_argument_S  = get_srl_relation_argument_S(overall_srl_embeddings)
            srl_keys, srl_keys_summary, srl_relation_S_to_summary, srl_argument_S_summary = get_srl_relation_argument_S_docsrl_to_summary(overall_srl_embeddings, overall_srl_embeddings_summary)
            
            doc_sample = {
                "srl_keys": srl_keys,
                "srl_keys_summary": srl_keys_summary,
                "srl_relation_S": srl_relation_S, 
                "srl_argument_S": srl_argument_S,
                "srl_relation_S_to_summary": srl_relation_S_to_summary,
                "srl_argument_S_summary": srl_argument_S_summary,
                "sent_keys": sent_keys,
                "sent_S": sent_S.cpu()
            }
        shard_data.append(doc_sample)
    
    # 达到 shard 大小就保存
    if (idx + 1) % SHARD_SIZE == 0:
        torch.save(shard_data, shard_file)
        print(f"Saved {shard_file} with {len(shard_data)} samples")
        shard_data = []  # 清空当前 shard
    
    
# 保存最后一个 shard（不足 SHARD_SIZE 的部分）
if len(shard_data) > 0:
    shard_idx = len(dataset_with_srl_new['validation']) // SHARD_SIZE
    shard_file = f"../dataset_MDS/MDS-RA2_data2/{data_name}_overall_similarities_valid_shard_{shard_idx}.pt"
    torch.save(shard_data, shard_file)
    print(f"Saved {shard_file} with {len(shard_data)} samples")


print("valid数据完成!!!!!!!!!!!!!!!!!!!!!!!")




shard_data = []
for idx, sample in enumerate(tqdm(dataset_with_srl_new['test'])):
    
    docs = preprocess_docs(sample['document'])
    srls = json.loads(sample['srl'])
    summary = sample['summary']

    shard_idx = idx // SHARD_SIZE
    shard_file = f"../dataset_MDS/MDS-RA2_data2/{data_name}_overall_similarities_test_shard_{shard_idx}.pt"
    if os.path.exists(shard_file):
        print(f"Shard {shard_idx} 已存在，跳过")
        continue

    if len(docs) == 0:
        shard_data.append({})
    else:
        if len(srls['summary'])==0:
            doc_sample = {
                "srl_keys": None,
                "srl_keys_summary": None,
                "srl_relation_S": None, 
                "srl_argument_S": None,
                "srl_relation_S_to_summary": None,
                "srl_argument_S_summary": None,
                "sent_keys": None,
                "sent_S": None
            }
        else:
            docs_sentences = get_docs_sentences(docs, max_char)
            summary_sentences = get_summary_sentencecs(summary)
            
            docs_embedding_by_sentences, docs_enc_by_sentences, docs_offsets_by_sentences = get_docs_embedding_enc_offsets_by_sentences(docs_sentences)
            summary_embedding_by_sentences, summary_enc_by_sentences, summary_offsets_by_sentences = get_summary_embedding_enc_offsets_by_sentences(summary_sentences)
            
            overall_srl_embeddings = get_overall_srl_embeddings(srls, docs_sentences, docs_embedding_by_sentences, docs_offsets_by_sentences)
            overall_srl_embeddings_summary = get_overall_srl_embeddings_summary(srls, summary_sentences, summary_embedding_by_sentences, summary_offsets_by_sentences)
            overall_sent_embeddings = get_overall_sent_embeddings(docs_embedding_by_sentences)
            sent_keys, sent_S = get_sent_S(overall_sent_embeddings)
            
            srl_keys, srl_relation_S, srl_argument_S  = get_srl_relation_argument_S(overall_srl_embeddings)
            srl_keys, srl_keys_summary, srl_relation_S_to_summary, srl_argument_S_summary = get_srl_relation_argument_S_docsrl_to_summary(overall_srl_embeddings, overall_srl_embeddings_summary)
            
            doc_sample = {
                "srl_keys": srl_keys,
                "srl_keys_summary": srl_keys_summary,
                "srl_relation_S": srl_relation_S, 
                "srl_argument_S": srl_argument_S,
                "srl_relation_S_to_summary": srl_relation_S_to_summary,
                "srl_argument_S_summary": srl_argument_S_summary,
                "sent_keys": sent_keys,
                "sent_S": sent_S.cpu()
            }
        shard_data.append(doc_sample)
        
    # 达到 shard 大小就保存
    if (idx + 1) % SHARD_SIZE == 0:
        torch.save(shard_data, shard_file)
        print(f"Saved {shard_file} with {len(shard_data)} samples")
        shard_data = []  # 清空当前 shard
        

# 保存最后一个 shard（不足 SHARD_SIZE 的部分）
if len(shard_data) > 0:
    shard_idx = len(dataset_with_srl_new['test']) // SHARD_SIZE
    shard_file = f"../dataset_MDS/MDS-RA2_data2/{data_name}_overall_similarities_test_shard_{shard_idx}.pt"
    torch.save(shard_data, shard_file)
    print(f"Saved {shard_file} with {len(shard_data)} samples")

print("test数据完成!!!!!!!!!!!!!!!!!!!!!!!")

# overall_similarities = {
#     "train": overall_similarities_train,
#     "validation": overall_similarities_valid,
#     "test": overall_similarities_test
# }

# print("保存数据")
# # torch.save(overall_similarities, "../dataset_MDS/MDS-RA2_data/small_dataset_overall_similarities.pt")
# torch.save(overall_similarities, "../dataset_MDS/MDS-RA2_data/multi_news_overall_similarities.pt")

