# SHER
The official code for SHER: Semantic-Role-Aware Heterogeneous Graph Retrieval for Multi-Document Summarization.

SHER is a **query-free sentence retriever** for multi-document summarization (MDS).  

It follows the *retrieve-then-summarize* paradigm: first select salient sentences from a document cluster, then feed them to a downstream summarizer under a length budget.

This repository currently provides:
- utilities used in experiments / complementary analyses:
  - `create_srl.sh`: Create SRL Frames.
  - `calculate_SRL_similarity.py`: Calculate the similarity between SRL frames. 





---

## Requirements
```python
pip install -r requirements.txt
```

---

## Method Overview
SHER models a document cluster as a **heterogeneous graph** over three node types:
| Node type | Role |
|-----------|------|
| Document | Global cluster context |
| Sentence | Retrieval candidate |
| SRL Frame | Fine-grained event / fact unit (*who did what to whom / when / where*) |
### Pipeline
1. **SRL Frame Creation**  
   Parse each sentence into SRL frames \((v, \mathcal{A})\) (predicate + arguments).  
   Apply coreference resolution so pronouns / mentions map to representative entities across sentences.
  ```
  create_srl.sh
  ```

2. **Initial Representation Encoding**  
   Encode the specific embeddings for documents, sentences, and SRL frames.

3. **Heterogeneous Graph Construction** 
  Build edges:
   - `Doc-Contain-Sent`, `Sent-Contain-Frame` (hierarchy)
   - `Sent-Next-Sent` (discourse order)
   - `Frame-Similar-Frame` (cross-document event correspondence)
  ```
  calculate_SRL_similarity.sh
  ```

4. **Traning, HGT Encoding + Multi-level Ranking**  
  A Heterogeneous Graph Transformer propagates information across node/relation types.  
  Two MLP heads predict sentence scores \(z_s\) and frame scores \(z_f\).

  ```
  python script_srl_retrieval/retrieval_train_batched.py
  ```

5. **Inference**  
   Rank sentences by \(z_s\), keep top content under the summarizer length budget (or top 80% for short inputs), restore original document order, then summarize.

---

