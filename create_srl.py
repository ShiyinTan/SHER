
## for SRL
import os

import sys

# 添加父目录到 sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname('__file__'), "../SRLScore"))
parent_dir
sys.path.append(parent_dir)

from tqdm import tqdm
import json
import pickle

import logging
# logging.basicConfig(level=logging.INFO)

from collections import defaultdict

import re
import json

from itertools import islice

import numpy as np
import pandas as pd
import spacy

from datasets import load_dataset, load_from_disk

from typing import Dict, Tuple, List, Union
from processor import Processor
from custom_datatypes import SRLTuple


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


class SRLTupleWithSpan(SRLTuple):
    """
    Extended SRLTuple that additionally stores the character span
    for each SRL attribute.
    """

    # char span，便于和Coref融合
    span_attributes = [
        "agent_span",
        "negation_span",
        "relation_span",
        "patient_span",
        "recipient_span",
        "time_span",
        "location_span",
    ]

    def __init__(
        self,
        agent=None,
        negation=None,
        relation=None,
        patient=None,
        recipient=None,
        time=None,
        location=None,
        agent_span=None,
        negation_span=None,
        relation_span=None,
        patient_span=None,
        recipient_span=None,
        time_span=None,
        location_span=None,
    ):
        super().__init__(
            agent=agent,
            negation=negation,
            relation=relation,
            patient=patient,
            recipient=recipient,
            time=time,
            location=location,
        )

        # Save span attributes
        self.agent_span = agent_span
        self.negation_span = negation_span
        self.relation_span = relation_span
        self.patient_span = patient_span
        self.recipient_span = recipient_span
        self.time_span = time_span
        self.location_span = location_span

        self.agent_coref_cluster = None
        self.negation_coref_cluster = None
        self.relation_coref_cluster = None
        self.patient_coref_cluster = None
        self.recipient_coref_cluster = None
        self.time_coref_cluster = None
        self.location_coref_cluster = None

    def format_tuple_with_span(self):
        """
        Returns a tuple combining values and spans:
        ((agent, agent_span), (negation, negation_span), ...)
        """
        result = []
        for attr, span_attr in zip(self.attributes, self.span_attributes):
            value = getattr(self, attr)
            span = getattr(self, span_attr)
            result.append((value, span))
        return tuple(result)

    def format_to_dict(self):
        """
        Returns a list of string combining values and spans:
        [{type: agent, value: agent value, span: agent_span}, {...}, ...)
        """
        result = []
        for attr, span_attr in zip(self.attributes, self.span_attributes):
            value = getattr(self, attr)
            span = getattr(self, span_attr)
            if value == None or span == None:
                continue
            sub_result = {"type": attr, "value": value, "span": span}
            result.append(sub_result)
        return result
    
    # def __repr__(self):
    #     """Pretty printing including spans."""
    #     parts = []
    #     for attr, span_attr in zip(self.attributes, self.span_attributes):
    #         val = getattr(self, attr)
    #         span = getattr(self, span_attr)
    #         parts.append(f"{attr}: {val} [{span}]")
    #     return "SRLTupleWithSpan(" + ", ".join(parts) + ")"


def token_span_to_char_span(doc, token_start, token_end):
    char_start = doc[token_start].idx
    last_token = doc[token_end - 1]
    char_end = last_token.idx + len(last_token.text)
    return (char_start, char_end)



import argparse

parser = argparse.ArgumentParser()

# 添加参数
# wcep, multi_x_science_sum, wikisum, multi_news, bigsurvey, peersum
parser.add_argument("--data_name", type=str, default="multi_news")

args = parser.parse_args()

data_name = args.data_name

proc = Processor(do_coref=False, use_cuda=True)

if data_name in ["multi_news"]:
    original_datasets = load_dataset(data_name)
    max_char = 40000
    print("load dataset: multi_news, max_char: {max_char}")
elif data_name in ["multi_x_science_sum", "wikisum", "arxiv"]:
    original_datasets = load_dataset(f"{data_name}_original") # original dataset
    original_datasets = original_datasets.select_columns(['summary', 'document'])
    max_char = 40000
    print(f"load dataset: {data_name}, max_char: {max_char}")
else:
    original_datasets = load_dataset(f"{data_name}_original")
    original_datasets = original_datasets.select_columns(['summary', 'document'])
    max_char = 40000
    print(f"load dataset: {data_name}, max_char: {max_char}")
print(original_datasets)



def extract_srl(text, doc_id):
    source_doc = proc.nlp(text[:max_char])
    srl = []
    
    for sent in source_doc.sents:
        try:
            result = proc.srl_model.predict(sent.text)
        except RuntimeError as e:
            # 记录错误但跳过
            print(sent.text)
            logging.warning(f"SRL failed at doc {doc_id} sent {i}: {e}")
            result = []
        # result = proc.srl_model.predict(sent.text)
        srl.append(result)


    all_tuples = defaultdict(list) # sent: list of SRL tuple
    for sent_idx, (srl_annotations, sentence) in enumerate(zip(srl, source_doc.sents)):
        if srl_annotations == []:
            continue
    
        for verb_idx, annotation in enumerate(srl_annotations["verbs"]):
            curr_tuple = SRLTupleWithSpan()
    
            spans = proc._convert_tags_to_spans(
                        annotation["tags"], offset=sentence.start
                    )
            if len(spans) < 2:
                continue
    
            # Assign SRL values based on the extracted spans
            for span, attribute in spans:
                attribute_tuple, attribute_value = proc._generate_attribute_tuple(
                    span, source_doc
                )

                if span.end > sent.end:
                    logging.warning(f"span.end is larger than sent.end. Doc: {doc_id}, sent: {sent_idx}. Value: {attribute_value}")
                    continue
    
                # Complicated way of assigning the correct attribute with the span value
                if attribute in proc.attribute_map.keys() and (
                    attribute_value != "" or attribute_tuple
                ):
                    if attribute_tuple:
                        curr_tuple.__setattr__(
                            proc.attribute_map[attribute], attribute_tuple
                        )
                        # convert token span to char span
                        char_span = token_span_to_char_span(source_doc, span.start, span.end)
                        curr_tuple.__setattr__(
                            f"{proc.attribute_map[attribute]}_span", char_span
                        )
                    else:  # implies attribute_value != ""
                        curr_tuple.__setattr__(
                            proc.attribute_map[attribute], attribute_value
                        )
                        # convert token span to char span
                        char_span = token_span_to_char_span(source_doc, span.start, span.end)
                        curr_tuple.__setattr__(
                            f"{proc.attribute_map[attribute]}_span", char_span
                        )
    
            all_tuples[str(sent_idx)].append(curr_tuple.format_to_dict())
    return all_tuples



def add_srl(example):
    srl_info = {}
    
    docs = preprocess_docs(example['document'])
    summary = example['summary']
    
    srl_info['document'] = {}
    for doc_id, doc_str in enumerate(docs):
        srl_info['document'][str(doc_id)] = extract_srl(doc_str, doc_id)  # 假设文本列名为 "text"
    
    srl_info['summary'] = extract_srl(summary, -1)

    example["srl"] = srl_info

    example["srl"] = json.dumps(example["srl"])
    
    return example




save_dir = "./MDS-RA2_data2/"
chunk_size = 2000 # chunk+断点保存 防止中途出现错误，不用每次从头重跑


# 仅保存新srl列

print("train dataset process start")
ds = original_datasets['train']
for i in range(0, len(ds), chunk_size):
    chunk_id = i // chunk_size
    out_path = f"{save_dir}/{data_name}_with_srl_train_{chunk_id}"
    if os.path.exists(out_path):
        print(f"Chunk {chunk_id} 已存在，跳过")
        continue
    
    chunk = ds.select(range(i, min(i + chunk_size, len(ds))))
    processed = chunk.map(add_srl)

    new_column_srl = processed.select_columns(["summary", "srl"])

    new_column_srl.save_to_disk(out_path)
    print(f"Chunk {chunk_id}: range {i} ~ {min(i + chunk_size, len(ds))} 处理并保存完成")
print("train dataset process end")


print("validation dataset process start")
ds = original_datasets['validation']
for i in range(0, len(ds), chunk_size):
    chunk_id = i // chunk_size
    out_path = f"{save_dir}/{data_name}_with_srl_validation_{chunk_id}"
    if os.path.exists(out_path):
        print(f"Chunk {chunk_id} 已存在，跳过")
        continue
    
    chunk = ds.select(range(i, min(i + chunk_size, len(ds))))
    processed = chunk.map(add_srl)

    new_column_srl = processed.select_columns(["summary", "srl"])

    new_column_srl.save_to_disk(out_path)
    print(f"Chunk {chunk_id}: range {i} ~ {min(i + chunk_size, len(ds))} 处理并保存完成")
print("validation dataset process end")


print("test dataset process start")
ds = original_datasets['test']
for i in range(0, len(ds), chunk_size):
    chunk_id = i // chunk_size
    out_path = f"{save_dir}/{data_name}_with_srl_test_{chunk_id}"
    if os.path.exists(out_path):
        print(f"Chunk {chunk_id} 已存在，跳过")
        continue
    
    chunk = ds.select(range(i, min(i + chunk_size, len(ds))))
    processed = chunk.map(add_srl)

    new_column_srl = processed.select_columns(["summary", "srl"])

    new_column_srl.save_to_disk(out_path)
    print(f"Chunk {chunk_id}: range {i} ~ {min(i + chunk_size, len(ds))} 处理并保存完成")

print("test dataset process end")