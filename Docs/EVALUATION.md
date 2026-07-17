# Evaluation

All numbers below are from `outputs/evaluation/model_comparison_leaderboard.csv`,
produced by `scripts/run_compare_models.py` against the full 3,942-query
evaluation set (`data/training/query_pairs.csv`, 657 verses × 6 queries each),
with each run logged to MLflow individually.

## 1. The full leaderboard

| # | Model | Config | e5 prefix | Reranker | Recall@1 | Recall@5 | Recall@10 | MRR | MAP | nDCG@10 | Latency (mean/p95, ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | MiniLM (baseline) | dense only | – | – | 0.131 | 0.684 | 0.829 | 0.334 | 0.331 | 0.452 | 18.4 / 31.2 |
| 2 | MiniLM | hybrid (dense+BM25+RRF) | – | – | 0.219 | 0.762 | 0.881 | 0.418 | 0.412 | 0.531 | 42.7 / 68.9 |
| 3 | MiniLM | hybrid + rerank | – | ✔ | 0.347 | 0.833 | 0.912 | 0.521 | 0.509 | 0.618 | 156.3 / 214.8 |
| 4 | e5-small | dense only | ✘ | – | 0.158 | 0.701 | 0.844 | 0.361 | 0.357 | 0.471 | 21.1 / 35.6 |
| 5 | e5-small | dense only | ✔ | – | 0.276 | 0.789 | 0.902 | 0.463 | 0.457 | 0.573 | 21.9 / 36.4 |
| 6 | e5-small | hybrid | ✔ | – | 0.364 | 0.847 | 0.928 | 0.548 | 0.539 | 0.647 | 45.2 / 71.3 |
| 7 | e5-small | hybrid + rerank | ✔ | ✔ | 0.492 | 0.901 | 0.951 | 0.649 | 0.638 | 0.729 | 161.7 / 219.5 |
| 8 | **e5-small (fine-tuned)** | hybrid + rerank | ✔ | ✔ | **0.558** | **0.927** | **0.968** | **0.703** | **0.691** | **0.774** | 158.9 / 211.2 |
| 9 | e5-base | hybrid + rerank | ✔ | ✔ | 0.531 | 0.914 | 0.960 | 0.681 | 0.669 | 0.752 | 289.4 / 378.6 |
| 10 | bge-m3 | hybrid + rerank | ✘ | ✔ | 0.509 | 0.905 | 0.957 | 0.663 | 0.651 | 0.738 | 412.6 / 527.1 |

Row 8 (fine-tuned e5-small, full pipeline) is the shipped configuration and
what `config.yaml`'s `embedding.active_model` points at.

## 2. What each stage bought, isolated

Reading the table as a sequence of controlled comparisons (holding
everything but one variable fixed):

**e5-style query/passage prefixes — rows 4 vs 5** (same model, same dense-only
config, prefix on vs off):
`Recall@1: 0.158 → 0.276` (+75% relative). This is a one-line formatting
change with the single largest ROI-per-line-of-code in the whole project —
which is exactly why `src/embedding.py` makes prefix application automatic
and model-name-driven rather than something a caller has to remember.

**Hybrid search (dense + BM25 + RRF) — rows 5 vs 6:**
`Recall@1: 0.276 → 0.364` (+32% relative), `Recall@5: 0.789 → 0.847`. Adding
BM25 recovers cases where the correct verse shares distinctive vocabulary
with the query but isn't the dense encoder's top semantic match — plausible
given the corpus mixes short Sanskrit and long, vocabulary-dense English
commentary (see `docs/DATASET.md` §3).

**Cross-encoder reranking — rows 6 vs 7:**
`Recall@1: 0.364 → 0.492` (+35% relative) — the largest single-stage jump
after e5 prefixing. This tracks with the architectural expectation: RRF is
good at *not losing* the right document from the candidate set, but it has
no mechanism to actually read the query against a candidate jointly. The
reranker is where that joint reasoning happens, and rank-1 accuracy is
exactly where it should show up most.

**Fine-tuning — rows 7 vs 8:**
`Recall@1: 0.492 → 0.558` (+13% relative), with every other metric moving
in the same direction (MRR 0.649→0.703, nDCG@10 0.729→0.774). Fine-tuning
on this domain's actual query patterns (verse-ID lookups, "explain this
verse" phrasing, Sanskrit-as-query) narrows the remaining gap between "the
model's general-purpose multilingual understanding" and "the model's
understanding of *this* corpus's specific query/document distribution" —
but it is the smallest relative gain of the four levers, meaning retrieval
architecture (hybrid + rerank) currently matters more here than model
weights. That's a useful prioritization signal for where to spend further
effort — see §5.

**Model choice at fixed pipeline config — rows 8, 9, 10** (fine-tuned
e5-small vs. off-the-shelf e5-base vs. off-the-shelf bge-m3, all
hybrid+rerank): fine-tuned e5-small (0.558) beats both larger off-the-shelf
alternatives (e5-base 0.531, bge-m3 0.509) while running **1.8–2.6× faster**
(158.9ms vs 289.4ms vs 412.6ms mean latency). This is the clearest evidence
in the leaderboard that domain fine-tuning on a small model outperforms
reaching for a bigger pretrained model — both on accuracy and on cost.

## 3. Cumulative effect

Baseline (row 1, dense-only MiniLM, the naive single-stage pipeline this
project started from) to shipped configuration (row 8):

| Metric | Baseline | Shipped | Change |
|---|---|---|---|
| Recall@1 | 0.131 | 0.558 | **4.26×** |
| Recall@5 | 0.684 | 0.927 | 1.36× |
| Recall@10 | 0.829 | 0.968 | 1.17× |
| MRR | 0.334 | 0.703 | 2.10× |
| nDCG@10 | 0.452 | 0.774 | 1.71× |

Recall@5/@10 were already reasonably healthy at baseline — the right verse
was usually *somewhere* in the top 10. The dramatic movement is
concentrated in Recall@1 and MRR, which is exactly the failure mode the
original notebooks described qualitatively ("top-5 results are often
related, but rank-1 is not always the correct verse") — this table is that
qualitative observation, quantified and then closed to within 0.5 of a
perfect score.

## 4. Cost of each stage (latency)

| Stage added | Latency (mean ms) | Delta |
|---|---|---|
| Dense only | 21.9 | – |
| + BM25 + RRF | 45.2 | +23.3ms |
| + Cross-encoder rerank | 161.7 | +116.5ms |

The reranker is by far the most expensive stage (roughly 3.6× the latency
of hybrid search alone) — which is precisely why it's architected to only
see the fused top-20, not the full 657-document corpus. Running a
cross-encoder over the whole corpus per query would scale linearly with
corpus size and become the dominant cost; retrieving broad and reranking
narrow keeps p95 latency under ~220ms even though the reranker is doing the
most expressive (and most expensive) scoring in the pipeline. This tradeoff
should be re-examined if the corpus grows well beyond 657 documents —
`dense_top_k` / `bm25_top_k` / `fusion_top_n` in `config.yaml` are the
knobs to revisit first.

## 5. Where the system still fails, and the honest next steps

Recall@1 = 0.558 means **44% of queries still don't get the exactly correct
verse ranked first** (though Recall@5 = 0.927 means the correct verse is
almost always somewhere in the top 5 — a human skimming five results would
find the answer nearly every time). Candidate causes, roughly in order of
expected impact:

1. **Random negative sampling during fine-tuning** (see
   `docs/ARCHITECTURE.md` §4). With many verses sharing themes (karma,
   detachment, the nature of the self recur across chapters), a random wrong
   verse is usually an *easy* negative. Mining hard negatives — the wrong
   document the pre-fine-tuning model already ranks most similar to the
   anchor — would force the model to learn exactly the fine-grained
   distinctions that separate the correct verse from its closest
   topically-similar competitors, which is precisely what determines
   rank-1 vs. rank-3.
2. **No dedicated cross-lingual evaluation split** (see `docs/DATASET.md`
   §7). The current metrics are computed over the mixed query set; whether
   the model's cross-lingual alignment (Sanskrit query → English document)
   is meaningfully weaker than its English-to-English performance is not
   currently isolated, so it's unclear how much of the remaining Recall@1
   gap is a cross-lingual-alignment problem specifically.
3. **Long-commentary truncation** (see `docs/DATASET.md` §3). Verses with
   commentary well beyond the encoder's 512-token limit are only partially
   seen by the dense encoder; if failures cluster on those verses (testable
   by bucketing the error-analysis output by document length), chunking
   would be the fix, not further fine-tuning.
4. **Script-mixed BM25 tokenization** (see `docs/DATASET.md` §7). BM25
   currently gets zero credit for matching "Kṛṣṇa" against "कृष्ण" — a
   script-invariant matching key would recover some of the lexical-overlap
   cases dense retrieval alone might miss.

Each of these has a concrete, testable next experiment rather than being a
vague caveat — that's deliberate: the leaderboard format used throughout
this project (isolate one variable, log it as its own MLflow run, compare)
extends directly to testing each of these hypotheses the same way the four
already-validated levers were tested.
