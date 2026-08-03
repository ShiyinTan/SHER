import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TORCH_USE_CUDA_DSA'] = '1'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

cores = os.cpu_count()
cores

from tqdm import tqdm
import json
import pickle

from collections import defaultdict

import re


from itertools import islice



import sys

# 添加父目录到 sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname('__file__'), "../SRLScore"))
parent_dir
sys.path.append(parent_dir)

import nltk
nltk.download('wordnet')

from functools import lru_cache
from allennlp.predictors.predictor import Predictor

from nltk.stem.wordnet import WordNetLemmatizer
from tqdm import tqdm
import spacy

import re

from typing import Dict, Tuple, List, Union

from spacy.language import Doc
from allennlp_models.coref import CorefPredictor
from allennlp_models.structured_prediction import SemanticRoleLabelerPredictor
from nltk.stem.wordnet import WordNetLemmatizer

from SRLScore import SRLScore

import numpy as np
import pandas as pd

from datasets import load_dataset

from nltk.tokenize import sent_tokenize

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


hf_datasets = hf_datasets = load_dataset(
    "multi_news"
)
hf_datasets


with open("./dataset/multi_news/empty_doc_indices_dict.pkl", "rb") as f:
    empty_doc_indices_dict = pickle.load(f)

with open("./dataset/multi_news/single_doc_indices_dict.pkl", "rb") as f:
    single_doc_indices_dict = pickle.load(f)

with open("./dataset/multi_news/normal_doc_indices_dict.pkl", "rb") as f:
    normal_doc_indices_dict = pickle.load(f)

train_single_doc_list = single_doc_indices_dict['train']
valid_single_doc_list = single_doc_indices_dict['valid']
test_single_doc_list = single_doc_indices_dict['test']

normal_dataset_indices_train = normal_doc_indices_dict['train']
normal_dataset_indices_valid = normal_doc_indices_dict['valid']
normal_dataset_indices_test = normal_doc_indices_dict['test']


single_doc_dataset_train = hf_datasets['train'].select(train_single_doc_list)
normal_dataset_train = hf_datasets['train'].select(normal_dataset_indices_train)

single_doc_dataset_valid = hf_datasets['validation'].select(valid_single_doc_list)
normal_dataset_valid = hf_datasets['validation'].select(normal_dataset_indices_valid)

single_doc_dataset_test = hf_datasets['test'].select(test_single_doc_list)
normal_dataset_test = hf_datasets['test'].select(normal_dataset_indices_test)



def custom_key(s): # 字符串为"train"时最大，test最小，valid次之，其他值报错
    assert s in ["train", "test", "valid"]
    if s == "train":
        return 2
    elif s == "test":
        return 0
    elif s == "valid":
        return 1  # 介于 test 和 train 中间


def doc_name_convert(name): # convert 'train: {(i,j)}' to 'train', i, j
    split_ij_str = name.split(":")
    split_str = split_ij_str[0].strip() # "trian", "valid", "test"
    ij_str = split_ij_str[1].strip()
    i, j = eval(ij_str)
    return split_str, i, j

def summary_name_convert(name): # convert 'train: {i}' to 'train', i
    split_i_str = name.split(":")
    split_str = split_i_str[0].strip() # "trian", "valid", "test"
    i_str = split_i_str[1].strip()
    i = int(i_str)
    return split_str, i

# 需要有normal_dataset_xxx在全局空间内
def print_doc_by_name(name, print_name=True, return_doc=False):
    """
    name: 'train: {(i,j)}'
    """
    split, i, j = doc_name_convert(name)
    assert split in ["train", "valid", "test"]
    if split == "train":
        doc = preprocess_docs(normal_dataset_train[i]['document'])[j]
        summary = normal_dataset_train[i]['summary']
        if print_name:
            print(name, doc)
        else:
            print(doc)
    elif split == "valid":
        doc = preprocess_docs(normal_dataset_valid[i]['document'])[j]
        summary = normal_dataset_valid[i]['summary']
        if print_name:
            print(name, doc)
        else:
            print(doc)
    elif split == "test":
        doc = preprocess_docs(normal_dataset_test[i]['document'])[j]
        summary = normal_dataset_test[i]['summary']
        if print_name:
            print(name, doc)
        else:
            print(doc)
    if return_doc:
        return doc
    else:
        return None

def return_doc_summary_by_name(name):
    """
    name: 'train: {i}'
    """
    split, i = summary_name_convert(name)
    assert split in ["train", "valid", "test"]
    docs = None
    summary = None
    if split == "train":
        docs = preprocess_docs(normal_dataset_train[i]['document'])
        summary = normal_dataset_train[i]['summary']
    elif split == "valid":
        docs = preprocess_docs(normal_dataset_valid[i]['document'])
        summary = normal_dataset_valid[i]['summary']
    elif split == "test":
        docs = preprocess_docs(normal_dataset_test[i]['document'])
        summary = normal_dataset_test[i]['summary']
    
    return docs, summary


def rank_strs_docs(name1, name2): # rank 保证train在前，i小的在前，j小的在前
    split_str_1, i_1, j_1 = doc_name_convert(name1)
    split_str_2, i_2, j_2 = doc_name_convert(name2)
    
    if custom_key(split_str_1) > custom_key(split_str_2): # train在前
        return (name1,name2)
    elif custom_key(split_str_1) < custom_key(split_str_2): # train在前
        return (name2,name1)
    else: # split相同时
        if i_1 < i_2: 
            return (name1,name2)
        elif i_1 > i_2:
            return (name2,name1)
        else: # i相同时
            if j_1 < j_2:
                return (name1,name2)
            else:
                return (name2,name1)


def rank_strs_summary(name1, name2): # rank 保证train在前，i小的在前，j小的在前
    split_str_1, i_1 = summary_name_convert(name1)
    split_str_2, i_2 = summary_name_convert(name2)
    
    if custom_key(split_str_1) > custom_key(split_str_2): # train在前
        return (name1,name2)
    elif custom_key(split_str_1) < custom_key(split_str_2): # train在前
        return (name2,name1)
    else: # split相同时
        if i_1 < i_2: 
            return (name1,name2)
        else:
            return (name2,name1)

        
def truncate_text_by_sentence(text, max_length):
    sentences = sent_tokenize(text)
    truncated = ''
    total_len = 0
    for sent in sentences:
        if total_len + len(sent) <= max_length:
            truncated += sent + ' '
            total_len += len(sent)
        else:
            break
    return truncated.strip()
        




"""使用SRLScore生成两个文本的factuality"""

scorer = SRLScore()

SRLscore_trian = defaultdict(list)
SRLscore_valid = defaultdict(list)
SRLscore_test = defaultdict(list)


for i in tqdm(range(normal_dataset_train.num_rows)):
    docs = normal_dataset_train[i]['document']
    docs_list = preprocess_docs(docs)
    summary = normal_dataset_train[i]['summary']
    
    if i%1000==0 and i!=0:
        with open("./dataset/multi_news/SRLscore_trian.pkl", "wb") as f:
            print("dump the SRLscore file")
            pickle.dump(SRLscore_trian, f)

    for j, doc in enumerate(docs_list):
        if len(doc) > 500000:
            print(f"doc {(i,j)} has length larger than 500000. ")
            doc = truncate_text_by_sentence(doc, 500000)
        srl_score = scorer.score(doc, summary)
        SRLscore_trian[i].append(srl_score)

        
# for i in tqdm(range(normal_dataset_valid.num_rows)):
#     docs = normal_dataset_valid[i]['document']
#     docs_list = preprocess_docs(docs)
#     summary = normal_dataset_valid[i]['summary']

#     if i%1000==0 and i!=0:
#         with open("./dataset/multi_news/SRLscore_valid.pkl", "wb") as f:
#             print("dump the SRLscore file")
#             pickle.dump(SRLscore_valid, f)
    
#     for j, doc in enumerate(docs_list):
#         if len(doc) > 500000:
#             print(f"doc {(i,j)} has length larger than 500000. ")
#             doc = truncate_text_by_sentence(doc, 500000)
#         srl_score = scorer.score(doc, summary)
#         SRLscore_valid[i].append(srl_score)
        

# for i in tqdm(range(normal_dataset_test.num_rows)):
#     docs = normal_dataset_test[i]['document']
#     docs_list = preprocess_docs(docs)
#     summary = normal_dataset_test[i]['summary']
    
#     if i%1000==0 and i!=0:
#         with open("./dataset/multi_news/SRLscore_test.pkl", "wb") as f:
#             print("dump the SRLscore file")
#             pickle.dump(SRLscore_test, f)

#     for j, doc in enumerate(docs_list):
#         if len(doc) > 500000:
#             print(f"doc {(i,j)} has length larger than 500000. ")
#             doc = truncate_text_by_sentence(doc, 500000)
#         srl_score = scorer.score(doc, summary)
#         SRLscore_test[i].append(srl_score)





with open("./dataset/multi_news/SRLscore_trian.pkl", "wb") as f:
    print("final dump the SRLscore file")
    pickle.dump(SRLscore_trian, f)
