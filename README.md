# immverse_ai — Bhagavad Gita Semantic Retrieval

A production-ready semantic retrieval system for Bhagavad Gita verses. Given a
natural-language query like *"How to control the mind?"*, the system returns
the most relevant Sanskrit verse and its English translation — **not** a
chatbot, a retrieval engine.

This project is a from-scratch, modular rebuild of seven exploratory Jupyter
notebooks (kept in `notebooks/` for reference/provenance). Every notebook's
logic was preserved and moved into `src/`, then extended with the retrieval
quality work the notebooks hadn't done yet: hybrid search, RRF, cross-encoder
re-ranking, query expansion, and a fuller evaluation suite.

## Why the pipeline changed

The original pipeline was single-stage: embed with a fine-tuned
SentenceTransformer, search a FAISS `IndexFlatIP`, return top-k. That's fast
but has a ceiling — bi-encoder similarity alone gets the *right neighborhood*
reliably (Recall@5 was healthy) but rank-1 was inconsistent, because a single
dense embedding has to compress lexical and semantic signal into one score.

The new pipeline widens the net and then narrows it precisely:

```
User Query
    │
    ▼
Preprocessing (unicode/whitespace normalize)
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

Dense catches semantic paraphrases ("control the mind" ≈ "restrain the
senses"); BM25 catches exact-term matches dense embeddings sometimes blur
together (proper nouns, repeated key terms). RRF fuses the two rankings by
*position*, not raw score, so the retrievers never need score calibration
against each other. The cross-encoder then reads each (query, candidate)
pair jointly — much more accurate than bi-encoder cosine similarity, but too
slow to run over the whole corpus, which is why it only sees the fused
top-20, not the entire index.

## Project structure

```
immverse_ai/
├── config/
│   ├── config.yaml            # every path, model name, hyperparameter — see below
│   └── query_synonyms.yaml    # static thesaurus for query expansion
├── data/
│   ├── raw/                   # raw dataset CSV (not committed)
│   ├── processed/             # cleaned_dataset.csv (output of preprocessing.py)
│   └── training/              # query_pairs.csv, triplets.csv
├── models/                    # fine-tuned SentenceTransformer checkpoints
├── faiss_store/                # faiss.index, metadata.pkl, documents.pkl, bm25.pkl
├── notebooks/                 # original exploratory notebooks (unmodified, for provenance)
├── src/
│   ├── config.py               # YAML config loader (attribute-style access, path resolution)
│   ├── logging_utils.py        # shared logger + PIPELINE_VERSION stamp
│   ├── preprocessing.py        # raw -> cleaned dataset (fills the gap left by the missing notebook)
│   ├── query_pairs.py          # cleaned dataset -> (query, document, label) pairs
│   ├── triplet_generation.py   # query pairs -> (anchor, positive, negative) triplets
│   ├── embedding.py             # SentenceTransformer wrapper (e5 prefixing, normalization)
│   ├── trainer.py               # fine-tunes the embedding model, MLflow-instrumented
│   ├── evaluator.py             # Recall@K, MRR, nDCG, MAP, Hit Rate, latency
│   ├── faiss_builder.py         # builds/saves/loads the FAISS index + metadata
│   ├── bm25_search.py           # sparse BM25 retrieval (rank_bm25)
│   ├── dense_search.py          # dense retrieval (embedding model + FAISS)
│   ├── rrf.py                   # Reciprocal Rank Fusion (pure function, unit-tested)
│   ├── hybrid_search.py         # dense + BM25 combined via RRF
│   ├── query_expansion.py       # thesaurus-based query expansion
│   ├── reranker.py              # CrossEncoder re-ranking of fused candidates
│   ├── retrieval.py             # RetrievalPipeline — wires every stage together
│   ├── mlflow_utils.py          # MLflow run/logging helpers
│   └── utils.py                 # small shared helpers (timer decorator, etc.)
├── app/
│   └── streamlit_app.py        # interactive demo UI
├── scripts/                    # CLI entrypoints, one per pipeline stage
│   ├── run_preprocessing.py
│   ├── run_generate_query_pairs.py
│   ├── run_generate_triplets.py
│   ├── run_train.py
│   ├── run_build_index.py
│   ├── run_evaluate.py
│   ├── run_compare_models.py    # compares all embedding.candidates, logs each to MLflow
│   ├── run_error_analysis.py
│   └── run_query.py             # interactive CLI retrieval
├── tests/                      # pytest unit tests (31 tests, no model downloads required)
├── requirements.txt
└── README.md
```

## Configuration

Everything that used to be a hardcoded path or magic number in the notebooks
now lives in `config/config.yaml`: dataset paths, the active embedding model,
the list of candidate models to compare, hybrid-search top-k values, the RRF
damping constant, reranker settings, MLflow tracking URI, and a
`pipeline_version` string that gets logged at the start of every script run
(cheap insurance against a stale deployment silently running old code).

No module in `src/` hardcodes a path — everything goes through
`src.config.get_config()`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Place your raw dataset (columns: `unique_key`, `sanskrit_data`,
`translation`) at `data/raw/gita_dataset.csv`, or point
`paths.raw_dataset` in `config/config.yaml` at wherever it lives.

## Running the pipeline end-to-end

```bash
# 1. Clean the raw dataset
python scripts/run_preprocessing.py

# 2. Generate training pairs and triplets
python scripts/run_generate_query_pairs.py
python scripts/run_generate_triplets.py

# 3. (Optional) fine-tune the embedding model
python scripts/run_train.py --model intfloat/multilingual-e5-small --epochs 2

# 4. Build the FAISS + BM25 indexes
python scripts/run_build_index.py

# 5. Evaluate
python scripts/run_evaluate.py

# 6. Compare embedding-model candidates (logs each run to MLflow)
python scripts/run_compare_models.py

# 7. Error analysis (correct vs. wrong retrievals, similarity scores)
python scripts/run_error_analysis.py --sample 100

# 8. Query interactively
python scripts/run_query.py

# 9. Launch the Streamlit demo
streamlit run app/streamlit_app.py
```

MLflow-logging scripts (`run_train.py`, `run_evaluate.py`,
`run_compare_models.py`) expect an MLflow tracking server at the URI in
`config.yaml` (`mlflow.tracking_uri`, default `http://127.0.0.1:5000`).
Start one locally with:

```bash
mlflow server --host 127.0.0.1 --port 5000
```

If the server isn't reachable, these scripts log a warning and continue
without tracking rather than failing the run.

## Retrieval quality levers

| Lever | Where | What it does |
|---|---|---|
| Embedding model | `config.embedding.active_model` / `.candidates` | Swap or compare `multilingual-e5-small`, `bge-m3`, `gte-multilingual-base`, `paraphrase-multilingual-MiniLM-L12-v2` |
| Hybrid search | `config.retrieval.dense_top_k` / `.bm25_top_k` | How wide the initial net is cast before fusion |
| RRF | `config.retrieval.rrf_k` | Damping constant; higher = flatter fusion weighting |
| Query expansion | `config.query_expansion.*`, `config/query_synonyms.yaml` | Toggle on/off, tune the thesaurus, cap expansion terms |
| Re-ranking | `config.reranker.*` | Toggle on/off, swap the CrossEncoder model |

## Testing

```bash
pytest tests/ -v
```

The test suite (31 tests) covers config loading, preprocessing, query-pair
and triplet generation, RRF fusion math, BM25 search, query expansion, hybrid
search (with fake rankers), the reranker's fallback path, and all evaluation
metrics — all pure-Python/numpy, so no model downloads are required to run
`pytest`.

## Notes on the original notebooks

- `01_create_triplets.ipynb` and `02_create_query_pairs.ipynb` → refactored
  into `src/triplet_generation.py` and `src/query_pairs.py`, logic preserved
  exactly (same query templates, same seeded random-negative sampling).
- `03_mlflow_practice.ipynb` → generalized into `src/mlflow_utils.py`, used
  by every training/evaluation/comparison script instead of being a one-off
  demo.
- `04_train_embedding.ipynb` → `src/trainer.py`, same
  `MultipleNegativesRankingLoss` training loop, now config-driven and
  MLflow-instrumented (the notebook used `TripletLoss`-shaped data but never
  actually applied `TripletLoss` in training — this is preserved as a config
  option in `training.loss`).
- `05_evaluate_model.ipynb` → `src/evaluator.py`, same Recall@K/MRR/nDCG
  formulas, extended with MAP, Hit Rate, and latency per the new
  requirements.
- `06_build_faiss_index.ipynb` → `src/faiss_builder.py`, same
  `IndexFlatIP` over L2-normalized embeddings.
- `07_rag_demo.ipynb` → `src/retrieval.py` (pipeline orchestration) +
  `scripts/run_query.py` (CLI) + `app/streamlit_app.py` (UI), all sharing
  one `RetrievalPipeline` class instead of duplicating retrieval logic per
  entrypoint.
- The notebook's referenced `01_data_preprocessing.ipynb` was not present in
  the export — `src/preprocessing.py` reconstructs that step from the
  `cleaned_dataset.csv` schema the later notebooks assume.
#   B h a g a v a d - G i t a - S e m a n t i c - R e t r i e v a l 
 
 #   B h a g a v a d - G i t a - S e m a n t i c - R e t r i e v a l 
 
 