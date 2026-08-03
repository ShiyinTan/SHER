import os

from scipy.fftpack import dst
import spacy
from fastcoref import spacy_component
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict

import re
import json

from collections import defaultdict

from datasets import load_dataset, DatasetDict

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


data_name = "wikisum" # wcep, multi_x_science_sum, wikisum, multi_news


if data_name in ["multi_news"]:
    original_datasets = load_dataset(data_name)
    max_char = 8000 # multi_news: 8000
    print("load dataset: multi_news, max_char: {max_char}")
elif data_name in ["multi_x_science_sum", "wikisum", "arxiv"]:
    original_datasets = load_dataset(f"{data_name}_original") # original dataset
    original_datasets = original_datasets.select_columns(['summary', 'document'])
    max_char = 20000 # multi_news: 8000
    print(f"load dataset: {data_name}, max_char: {max_char}")
elif data_name in ["wcep"]:
    original_datasets = load_dataset(f"{data_name}_original")
    original_datasets = original_datasets.select_columns(['summary', 'document'])
    max_char = 40000 # multi_news: 8000
    print(f"load dataset: {data_name}, max_char: {max_char}")
print(original_datasets)

from fastcoref import spacy_component

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("fastcoref", config={"enable_progress_bar": False})



def extract_coref(text):
    doc = nlp(text[:max_char])
    coref_clusters = []
    for cluster in doc._.coref_clusters:
        mentions_list = []
        
        if not cluster:  # cluster 为 None 或空列表时跳过
            coref_clusters.append(mentions_list)
            continue
        if None in cluster:
            coref_clusters.append(mentions_list)
            continue
            
        main = doc.text[cluster[0][0]:cluster[0][1]]  # 取 cluster 第一个 mention 作为主实体
        
        for coref_span in cluster:
            if coref_span is None: # 避免cluster这个list中存在None值.
                continue
            start, end = coref_span
            mention_text = doc.text[start:end]
            sent = doc.char_span(start, end).sent if doc.char_span(start,end) else doc.text[start:end]
            context = sent.text if hasattr(sent,'text') else str(sent)

            mentions_list.append({
                    "mention": mention_text,
                    # "context": context,
                    "char_span": (start, end)
                })

        coref_clusters.append(mentions_list)
    return coref_clusters



def add_coref(example):
    mention_to_cluster = {}
    cluster_info = {}
    cluster_counter = 0
    docs = preprocess_docs(example['document'])
    summary = example['summary']
    
    cluster_info['document'] = {}
    for doc_id, doc_str in enumerate(docs):
        cluster_info['document'][str(doc_id)] = extract_coref(doc_str)  # 假设文本列名为 "text"
    
    cluster_info['summary'] = extract_coref(summary)

    example["coref_clusters"] = cluster_info

    example["coref_clusters"] = json.dumps(example["coref_clusters"])
    
    return example




# -------------- 正式运行 -------------------
save_dir = "./MDS-RA2_data2/"
chunk_size = 2000 # chunk+断点保存 防止中途出现错误，不用每次从头重跑


print("train dataset process start")
ds = original_datasets['train']
for i in range(0, len(ds), chunk_size):
    chunk_id = i // chunk_size
    out_path = f"{save_dir}/{data_name}_with_coref_train_{chunk_id}"
    if os.path.exists(out_path):
        print(f"Chunk {chunk_id} 已存在，跳过")
        continue
    
    chunk = ds.select(range(i, min(i + chunk_size, len(ds))))
    processed = chunk.map(add_coref)

    processed.save_to_disk(out_path)
    print(f"Chunk {chunk_id}: range {i} ~ {min(i + chunk_size, len(ds))} 处理并保存完成")
print("train dataset process end")


print("validation dataset process start")
ds = original_datasets['validation']
if data_name != "wikisum":
    for i in range(0, len(ds), chunk_size):
        chunk_id = i // chunk_size
        out_path = f"{save_dir}/{data_name}_with_coref_validation_{chunk_id}"
        if os.path.exists(out_path):
            print(f"Chunk {chunk_id} 已存在，跳过")
            continue
        
        chunk = ds.select(range(i, min(i + chunk_size, len(ds))))
        processed = chunk.map(add_coref)
    
        processed.save_to_disk(out_path)
        print(f"Chunk {chunk_id}: range {i} ~ {min(i + chunk_size, len(ds))} 处理并保存完成")
else:
    print(f"data set {data_name} skipped!")
print("validation dataset process end")


print("test dataset process start")
ds = original_datasets['test']
for i in range(0, len(ds), chunk_size):
    chunk_id = i // chunk_size
    out_path = f"{save_dir}/{data_name}_with_coref_test_{chunk_id}"
    if os.path.exists(out_path):
        print(f"Chunk {chunk_id} 已存在，跳过")
        continue
    
    chunk = ds.select(range(i, min(i + chunk_size, len(ds))))
    processed = chunk.map(add_coref)

    processed.save_to_disk(out_path)
    print(f"Chunk {chunk_id}: range {i} ~ {min(i + chunk_size, len(ds))} 处理并保存完成")

print("test dataset process end")
