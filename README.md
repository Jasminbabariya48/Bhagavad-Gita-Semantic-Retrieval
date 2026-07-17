# 🕉️ Bhagavad Gita Semantic Retrieval System

A fine-tuned, multilingual semantic retrieval system that takes a natural-language
question — in English or Sanskrit — and returns the most relevant Bhagavad Gita
verse, its Sanskrit text, and its English translation/commentary.

This is **not** a chatbot. There is no generation step and no hallucination
surface — every answer is a verse that actually exists in the corpus, ranked by
a fine-tuned embedding model plus a hybrid dense+lexical retrieval pipeline.

```
"How to control the mind?"  ─────►  Verse 6.35, Sanskrit text, English translation, similarity score
```

---

## 1. What this project actually does

| Stage | What happens | Where |
|---|---|---|
| Data acquisition | Pull 657 verse/translation pairs from a public HF dataset | `data.ipynb` |
| Preprocessing | Clean text, dedupe, build a unified `document` field | `Data_preprocessing.ipynb`, `src/preprocessing.py` |
| Training data generation | Turn each verse into ~6 (query, document) pairs, then triplets | `src/query_pairs.py`, `src/triplet_generation.py` |
| Fine-tuning | Fine-tune `intfloat/multilingual-e5-small` with contrastive loss | `src/trainer.py` |
| Indexing | Encode all 657 documents, build a FAISS index + BM25 index | `src/faiss_builder.py`, `src/bm25_search.py` |
| Retrieval | Dense + BM25 → Reciprocal Rank Fusion → cross-encoder rerank | `src/retrieval.py`, `src/hybrid_search.py`, `src/rrf.py`, `src/reranker.py` |
| Evaluation | Recall@K, MRR, nDCG, MAP, Hit Rate, latency, model comparison | `src/evaluator.py`, `scripts/run_compare_models.py` |
| Demo | Streamlit UI + CLI query tool | `app/streamlit_app.py`, `scripts/run_query.py` |

## 2. The headline result

Ten configurations were benchmarked end-to-end on the same 3,942-query
evaluation set (see `outputs/evaluation/model_comparison_leaderboard.csv` and
[`docs/EVALUATION.md`](docs/EVALUATION.md) for the full breakdown):

| Configuration | Recall@1 | Recall@5 | MRR | nDCG@10 |
|---|---|---|---|---|
| MiniLM, dense-only (baseline) | 0.131 | 0.684 | 0.334 | 0.452 |
| e5-small, dense-only, e5 prefixes | 0.276 | 0.789 | 0.463 | 0.573 |
| e5-small, hybrid (dense + BM25 + RRF) | 0.364 | 0.847 | 0.548 | 0.647 |
| e5-small, hybrid + reranker | 0.492 | 0.901 | 0.649 | 0.729 |
| **e5-small, fine-tuned, hybrid + reranker** | **0.558** | **0.927** | **0.703** | **0.774** |

Fine-tuning plus the full pipeline **more than quadrupled Recall@1** (0.131 →
0.558) over the naive single-stage baseline the project started from. Every
stage in that chain — e5 prefixing, hybrid search, reranking, fine-tuning —
contributed a measurable improvement; none of it was redundant. The full
reasoning behind this is in [`docs/EVALUATION.md`](docs/EVALUATION.md).

## 3. Why the pipeline looks the way it does

A single bi-encoder similarity score has a ceiling: it has to compress lexical
overlap and semantic meaning into one number, which is exactly why the
baseline's Recall@5 was reasonable (0.684 — the right verse is *usually*
nearby) but Recall@1 was weak (0.131 — it rarely won outright). The fix isn't
a better single model, it's separating concerns:

```
User Query
    │
    ▼
Preprocessing (unicode / whitespace normalize)
    │
    ▼
Query Expansion  (config/query_synonyms.yaml)
    │
    ├─────────────────────┬─────────────────────┐
    ▼                      ▼
Dense Search (FAISS)   BM25 Search (rank_bm25)
top 50                 top 50
    │                      │
    └──────────┬───────────┘
               ▼
    Reciprocal Rank Fusion (RRF, k=60)
               │
               ▼
             top 20
               │
               ▼
    Cross-Encoder Re-ranking
               │
               ▼
             top 5
               │
               ▼
   Verse ID · Similarity · Sanskrit · Translation
```

Dense retrieval catches semantic paraphrase ("control the mind" ≈ "restrain
the senses"); BM25 catches exact-term matches dense embeddings sometimes blur
together (Sanskrit proper nouns, repeated key terms). RRF fuses the two
rankings by *position*, not raw score, so neither retriever needs score
calibration against the other. The cross-encoder then reads each
(query, candidate) pair jointly — far more accurate than cosine similarity,
but too slow to run over the whole corpus, which is why it only ever sees the
fused top-20, not the full index.

Full architectural reasoning: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## 4. The dataset

657 Bhagavad Gita verses pulled from the `Mercity/bhagavad_gita-embeddings`
dataset on Hugging Face, each with the original Sanskrit and an English
field that — this matters — is not a literal one-line translation but a
full Prabhupada-style purport/commentary, averaging **~1,620 characters**
per verse (up to 15,600). That single fact shapes several downstream
decisions (chunking, document construction, embedding truncation risk) —
see [`docs/DATASET.md`](docs/DATASET.md) for the full data-quality writeup.

## 5. Project structure

```
Retrival_Sys/
├── config/
│   ├── config.yaml            # every path, model name, hyperparameter
│   └── query_synonyms.yaml    # static thesaurus for query expansion
├── data/
│   ├── raw/                   # bhagavad_gita.csv / gita_clean.csv/json — HF export
│   ├── processed/             # cleaned_dataset.csv, documents.csv
│   └── training/              # query_pairs.csv (3,942 rows), triplets.csv
├── models/                    # fine-tuned SentenceTransformer checkpoint (multilingual-e5-small base)
├── checkpoints/                # training checkpoints
├── faiss_store/                # faiss.index, bm25.pkl, metadata.pkl, documents.pkl
├── notebooks/                  # 7 exploratory notebooks — kept for provenance
├── src/                        # production modules (see docs/ARCHITECTURE.md)
├── app/streamlit_app.py        # interactive demo UI
├── scripts/                    # one CLI entrypoint per pipeline stage
├── tests/                      # pytest unit tests (pure logic, no model downloads)
├── outputs/evaluation/         # model_comparison_leaderboard.csv and friends
├── data.ipynb                  # dataset download from Hugging Face
├── Data_preprocessing.ipynb    # cleaning + document construction (prototype)
└── requirements.txt
```

## 6. Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Clean the raw dataset
python scripts/run_preprocessing.py

# 2. Generate training pairs and triplets
python scripts/run_generate_query_pairs.py
python scripts/run_generate_triplets.py

# 3. Fine-tune the embedding model
python scripts/run_train.py --model intfloat/multilingual-e5-small --epochs 2

# 4. Build the FAISS + BM25 indexes
python scripts/run_build_index.py

# 5. Evaluate
python scripts/run_evaluate.py

# 6. Compare configurations (dense-only / hybrid / hybrid+rerank / fine-tuned)
python scripts/run_compare_models.py

# 7. Query it
python scripts/run_query.py
# or
streamlit run app/streamlit_app.py
```

## 7. Further reading

- [`docs/DATASET.md`](docs/DATASET.md) — where the data came from, what
  cleaning was applied, and the data-quality quirks that shaped later
  decisions (the "translation is actually a commentary" issue, in
  particular).
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — module-by-module
  breakdown of the retrieval pipeline and the reasoning behind each design
  choice.
- [`docs/EVALUATION.md`](docs/EVALUATION.md) — the full ten-run leaderboard,
  what each stage bought in isolation, and an honest discussion of where the
  system still fails (Recall@1 sits at 0.558, not 1.0 — that gap is
  analyzed, not hidden).
