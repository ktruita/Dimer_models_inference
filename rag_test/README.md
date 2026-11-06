# Qwen3-Embedding + Qwen3-Reranker: retrieve-then-rerank demo

`retrieve_rerank.ipynb` shows how the two models work together as a two-stage search pipeline (the retrieval part of RAG):

1. **Qwen3-Embedding-0.6B** (bi-encoder) turns documents and queries into 1024-dim vectors, then finds the top 10 candidates by cosine similarity: fast, for recall.
2. **Qwen3-Reranker-0.6B** (cross-encoder) reads each (query, candidate) pair together and scores P("yes, relevant"): slow, for precision.

Sections:
- a bilingual English/Tagalog walkthrough (12 passages): similarity heatmap, cross-language retrieval, reranker scores vs cosine
- SciFact benchmark subset (30 claims, 300 abstracts): Recall@k / MRR / nDCG before vs after reranking
- Matryoshka embeddings: accuracy when vectors are cut to 512/256/128/64 dims
- examples of rankings the reranker fixed
- an end-to-end `search()` for your own questions

## CPU / memory

Both models are 0.6B parameters (~2.4 GB RAM each in float32). To fit a 7 GB laptop the notebook never holds both at once:
it embeds everything, caches the vectors (`outputs/scifact_doc_vectors_256.npy`), unloads the embedder, then loads the reranker.
Documents are cut to 256 tokens (`DOC_MAX_TOKENS`) to keep CPU time down.

On this laptop: ~1–3 s per document to embed (one-time, cached), ~0.3 s per query, ~3.3 s per reranked pair
(~33 s per query for 10 candidates). Full run: ~18 min with cached vectors.
On a GPU everything runs in float16 automatically and is far faster.

## Results

- SciFact subset: embedding only recall@1 0.86 / nDCG@10 0.95; + reranker recall@1 0.96 / nDCG@10 1.00 (3 claims fixed, none made worse)
- Cross-language (Tagalog ↔ English) retrieval: 6/6 walkthrough questions correct
- 512-dim vectors keep recall@10 = 1.00 at half the storage

## Environment

Uses the existing `ocr` env (kernel **Python (ocr)**): transformers ≥ 4.51, torch (CPU), pandas, matplotlib, jinja2.
No vector database is needed for this size; a numpy array holds the vectors. At scale, store them in FAISS, Qdrant, pgvector, etc.

## Data

- `data/scifact_*`: subset of BEIR SciFact (`BeIR/scifact`, `BeIR/scifact-qrels`): 30 test claims, their 33 relevant abstracts + 267 random distractors
- `data/mini_corpus.jsonl`: 12 short general-knowledge passages (English and Tagalog) written for the walkthrough

## Outputs

`outputs/metrics.csv` (before/after reranking), `outputs/rank_changes.csv`, `outputs/rerank_comparison.png`, cached document vectors
