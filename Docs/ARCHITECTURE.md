# Architecture

## 1. Design principle: config-driven, no hardcoded paths

Every path, model name, and hyperparameter lives in `config/config.yaml`.
No module under `src/` hardcodes a path — everything is read through
`src.config.get_config()`, which loads the YAML once (cached via
`functools.lru_cache`) and exposes attribute-style access:

```python
cfg = get_config()
cfg.embedding.active_model     # "models"  (points at the fine-tuned checkpoint)
cfg.retrieval.dense_top_k      # 50
cfg.path("processed_dataset")  # absolute Path, resolved against project root
```

This matters in practice, not just as style: it's the reason
`scripts/run_compare_models.py` can sweep ten different
model/retrieval-config combinations without any of `src/` changing, and the
reason the same codebase runs identically whether invoked from the project
root, from `scripts/`, or from a notebook.

## 2. Module map

```
src/
├── config.py            YAML config loader (Config wrapper, path resolution)
├── logging_utils.py      shared logger + pipeline-version stamp on every run
├── preprocessing.py       raw -> cleaned dataset (dedup, null-drop, document build)
├── query_pairs.py         cleaned dataset -> (query, document, label) pairs
├── triplet_generation.py  query pairs -> (anchor, positive, negative) triplets
├── embedding.py            SentenceTransformer wrapper: e5 prefixing, normalization
├── trainer.py               fine-tuning loop (MultipleNegativesRankingLoss)
├── evaluator.py             Recall@K / MRR / nDCG / MAP / Hit Rate / latency
├── faiss_builder.py         builds/saves/loads the FAISS IndexFlatIP + metadata
├── bm25_search.py            sparse BM25 retrieval (rank_bm25)
├── dense_search.py           dense retrieval (embedding model + FAISS)
├── rrf.py                     Reciprocal Rank Fusion (pure function)
├── hybrid_search.py           dense + BM25 combined via RRF
├── query_expansion.py         thesaurus-based query expansion
├── reranker.py                 CrossEncoder re-ranking of fused candidates
├── retrieval.py                RetrievalPipeline — wires every stage together
├── mlflow_utils.py              MLflow run/logging helpers
└── utils.py                     small shared helpers
```

`src/retrieval.py`'s `RetrievalPipeline` is the single object every
consumer (Streamlit app, CLI, evaluation scripts) goes through — retrieval
logic is assembled once, not duplicated per entrypoint.

## 3. The embedding model: why `multilingual-e5-small`, and why prefixes matter

`multilingual-e5-small` is a 12-layer, 384-hidden-dim XLM-R-tokenizer BERT
model (confirmed from the shipped checkpoint's `config.json`:
`hidden_size: 384`, `num_hidden_layers: 12`, `vocab_size: 250037`), wrapped
by SentenceTransformer with mean pooling + L2 normalization
(`modules.json`: Transformer → Pooling → Normalize).

The e5 model family was trained with an asymmetric convention: queries get a
`"query: "` prefix, documents get a `"passage: "` prefix, at both training
and inference time. Skipping this is a silent accuracy bug — the model still
runs and still produces embeddings, it just performs meaningfully worse,
because it was never trained to treat unprefixed text the way its own
training distribution expected.

`src/embedding.py` handles this centrally:

```python
def model_needs_e5_prefix(model_name: str) -> bool:
    return "e5" in model_name.lower()
```

`EmbeddingModel.encode_queries()` / `.encode_documents()` apply the correct
prefix automatically based on this detection, so no calling code has to
remember to do it. The leaderboard makes the cost of forgetting this
concrete: e5-small *without* prefixes scores Recall@1 = 0.158; the same
model *with* prefixes scores 0.276 — a 75% relative improvement from string
formatting alone, before any fine-tuning or hybrid search. See
`docs/EVALUATION.md` §2 for the full comparison.

## 4. Fine-tuning (`src/trainer.py`)

- **Loss:** `MultipleNegativesRankingLoss` over (anchor, positive, negative)
  triplets — despite the data being triplet-shaped, MNR loss treats every
  other in-batch positive as an additional implicit negative, which is a
  stronger training signal per batch than `TripletLoss`'s one-explicit-negative
  setup. `TripletLoss` remains available via `config.training.loss` for
  parity with the exploratory notebook, but MNR is the default.
- **Base model:** `intfloat/multilingual-e5-small`.
  Training was run on Google Colab (confirmed via the notebook's Drive-mount
  cell) with the full 3,942-triplet training set.
- **Negative sampling:** uniform random — for every (anchor, positive) pair,
  a random *different* document is sampled as the negative
  (`src/triplet_generation.py::build_triplets`, seeded for reproducibility).
  This is a deliberate baseline choice, not an oversight, but it's also the
  most obvious lever left on the table: with 657 verses covering overlapping
  themes (karma, dharma, detachment recur across many verses), a random
  negative is very often *topically unrelated* to the anchor, which teaches
  the model coarse "this is obviously wrong" contrasts rather than the
  fine-grained distinctions that determine whether the *correct* verse beats
  three or four topically-similar ones for rank #1. Mining hard negatives —
  the wrong document the *current* embedding model already ranks most
  similar to the anchor, excluding the single nearest match to avoid
  penalizing genuine near-duplicates — is the highest-leverage next
  experiment for closing the Recall@1 gap further. See
  `docs/EVALUATION.md` §5 for the reasoning tied to actual numbers.
- **Output:** saved as a full SentenceTransformer checkpoint under `models/`
  (Transformer + Pooling + Normalize modules), loadable directly by
  `EmbeddingModel` since `config.embedding.active_model` points at the
  checkpoint path rather than a Hugging Face model ID once fine-tuning is
  done.

## 5. Indexing (`src/faiss_builder.py`, `src/bm25_search.py`)

- **Dense:** every document is encoded (with the `passage:` prefix), then
  `faiss.normalize_L2` is applied before building `IndexFlatIP` — i.e. an
  exact (not approximate) cosine-similarity index over 657 vectors. Exact
  search is deliberately not swapped for an ANN index (HNSW/IVF): at 657
  documents, brute-force inner product is sub-millisecond, so there is no
  accuracy/speed tradeoff to make yet. This should be revisited if the
  corpus grows past roughly tens of thousands of documents.
- **Sparse:** `rank_bm25.BM25Okapi` over a simple regex tokenizer
  (lowercase + `\w+` split) applied to the same document text. No
  language-specific tokenizer is used for the Sanskrit portion — this is a
  known simplification (see `docs/DATASET.md` §7 on script normalization).
- Both indexes, plus `metadata.pkl` (verse ID / Sanskrit / translation per
  row) and `documents.pkl` (raw document text list), are saved to
  `faiss_store/` so retrieval never needs the original CSV at query time.

## 6. Hybrid retrieval and fusion (`src/hybrid_search.py`, `src/rrf.py`)

Dense search returns its top 50 by cosine similarity; BM25 returns its top
50 by BM25 score. These two score scales are not comparable (cosine
similarity lives in [-1, 1]; BM25 scores are unbounded and corpus-dependent),
so instead of trying to calibrate/normalize them against each other, RRF
fuses by **rank position only**:

```
RRF_score(doc) = Σ over rankers r of  1 / (k + rank_r(doc))
```

with `k = 60` (the standard constant from the original RRF paper). A
document that both retrievers agree on — even at different ranks — reliably
outranks a document only one retriever likes, without either retriever's
raw score ever needing to be trusted numerically. This is a pure function
(`reciprocal_rank_fusion` in `src/rrf.py`) and is unit-tested independent of
any model — the fusion math is correct whether or not the embedding model
downloads correctly, which decouples "is the fusion logic right" from "is
the model any good" as debugging questions.

## 7. Re-ranking (`src/reranker.py`)

The fused top-20 candidates go through a `CrossEncoder`
(`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`), which scores each
(query, candidate) pair *jointly* — the two texts attend to each other
through the same transformer, rather than being independently embedded and
compared by cosine similarity. This is strictly more expressive than a
bi-encoder score, at the cost of not being scalable to the full corpus
(hence: only the fused top-20 gets reranked, not all 657 documents). The
leaderboard shows this stage alone lifting Recall@1 from 0.364 (hybrid, no
rerank) to 0.492 (hybrid + rerank) on the un-fine-tuned e5-small — the
single largest jump of any individual stage. See `docs/EVALUATION.md`.

## 8. Query expansion (`src/query_expansion.py`)

A static YAML thesaurus (`config/query_synonyms.yaml`) maps trigger words to
a short list of related terms, appended to the query before retrieval
(e.g. "anger" → adds "mind", "self control", "discipline", "emotion",
capped at `config.query_expansion.max_expansion_terms`). This is
deliberately not an LLM-backed expansion — no external call, fully
deterministic, and cheap enough to run on every query. The tradeoff is
coverage: it only helps queries whose key terms are already in the
thesaurus. Swapping in an LLM-backed expander later is a drop-in change
(same `expand(query) -> str` interface), not a pipeline redesign.

## 9. End-to-end request flow (`src/retrieval.py`)

```python
RetrievalPipeline.retrieve(query, use_query_expansion=True, use_reranker=True, top_k=5)
```

1. `clean_text()` — unicode NFC normalize, collapse whitespace (same
   normalization used at indexing time, so a query and a document that are
   "the same text" hit the same matching surface).
2. Query expansion (optional, on by default).
3. `HybridSearch.search()` — dense top-50 + BM25 top-50 → RRF → top-20.
4. `CrossEncoderReranker.rerank()` (optional, on by default) → top-5.
5. Results assembled from `metadata.pkl` by row index, returned as
   `RetrievalResult` objects carrying verse ID, Sanskrit, translation,
   final score, rank, and the RRF stage score for debuggability.

Every stage is individually toggleable at call time (`use_query_expansion`,
`use_reranker`, `top_k`), which is what the Streamlit app's sidebar exposes
directly — a reviewer can turn reranking off mid-session and watch Recall@1
visibly degrade, which is a more convincing demonstration of a stage's value
than a static leaderboard row.
