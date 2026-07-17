# Dataset

## 1. Source

The raw data comes from the [`Mercity/bhagavad_gita-embeddings`](https://huggingface.co/datasets/Mercity/bhagavad_gita-embeddings)
dataset on Hugging Face, pulled via `datasets.load_dataset` in `data.ipynb`:

```python
from datasets import load_dataset
ds = load_dataset("Mercity/bhagavad_gita-embeddings")
df = ds["train"].to_pandas()
```

The upstream dataset ships five columns: `unique_key`, `sanskrit_data`,
`translation`, and two precomputed embedding columns (`minilm_embedding`,
`bge_embedding`) from the dataset's own publisher. This project **does not
use those precomputed embeddings** — every embedding here is produced by
this project's own encoding pipeline, so retrieval quality reflects this
project's modeling choices, not the upstream author's. The raw export is
still kept in `data/raw/bhagavad_gita.csv` for reference/reproducibility.

Two working subsets are cut from the raw export and saved to
`data/raw/gita_clean.csv` / `.json`:

```python
clean_df = df[["unique_key", "sanskrit_data", "translation"]]
```

## 2. Scale

| | |
|---|---|
| Total verses | **657** |
| Avg. Sanskrit length | 98 characters |
| Avg. "translation" length | **1,621 characters** |
| Max "translation" length | 15,627 characters |
| Min "translation" length | 96 characters |
| Avg. combined document length | 1,720 characters |

## 3. The most important data-quality finding: the `translation` field is not a translation

This is worth stating plainly because it shapes several downstream design
decisions: the `translation` column is not a literal, verse-length English
rendering of the Sanskrit. It's a **Prabhupada-style purport** — a
paragraphs-long theological and historical commentary that happens to also
contain the literal translation somewhere inside it. Verse 1's Sanskrit is
98 characters; its "translation" is a multi-paragraph essay about
Dhṛtarāṣṭra's psychology, the significance of Kurukṣetra as a pilgrimage
site, and the disciplic succession through which the Gita should be read.

Consequences this had for the pipeline:

- **Document construction.** Concatenating Sanskrit + full commentary into
  one `document` field (see §5) means each indexed unit is long and
  topic-dense, not a tight one-to-one paraphrase pair. This is *good* for
  BM25 (more lexical surface to match against) and *risky* for the dense
  encoder, which truncates at its max sequence length — `multilingual-e5-small`
  caps at 512 tokens, well under the ~15,600-character (roughly
  3,000+ token) outlier commentaries. In practice this means the encoder is
  reading a prefix of the longest commentaries, not the whole thing — a
  real chunking strategy (splitting long purports into overlapping windows,
  embedding each, and aggregating) is the natural next step if those long
  verses turn out to be systematically under-retrieved. See §7.
- **Query-pair generation.** Because `translation` is commentary-length, the
  query template that uses a translation snippet (`translation[:120]`, see
  `src/query_pairs.py`) captures the *opening* of the commentary, not
  necessarily its most retrieval-relevant sentence. This is a reasonable
  first-pass heuristic, not a guarantee of query quality — flagged here
  rather than silently assumed correct.
- **Evaluation fairness.** Recall@K here measures "did the system return
  the correct *verse*", where a verse's document includes a long commentary.
  A system could in principle get partial credit for matching commentary
  topic drift rather than verse-specific content. This is one reason the
  cross-lingual and Sanskrit-only evaluation angle (§7) matters: it isolates
  whether retrieval is actually anchored to the verse itself, or riding on
  commentary length/topic overlap.

## 4. Cleaning (`Data_preprocessing.ipynb`)

Applied in order:

1. **Column selection** — keep only `unique_key`, `sanskrit_data`,
   `translation`; drop the precomputed embedding columns.
2. **Deduplication** — `df.drop_duplicates()` on exact row equality.
3. **Null removal** — `df.dropna()`.
4. **Text normalization** — both Sanskrit and translation fields go through
   the same cleaning function: strip `\n`/`\t`, collapse repeated whitespace
   with `re.sub(r"\s+", " ", text)`, strip leading/trailing whitespace.
   Notably no diacritic stripping, no case folding, no script conversion —
   raw Devanagari and IAST-style romanized Sanskrit (e.g. `Śrī`, `Kṛṣṇa`) are
   both preserved as-is in the corpus. This is intentional (destroying
   diacritics loses information a multilingual encoder can actually use)
   but it does mean the corpus is not script-normalized — see §7.
5. **Document construction** — the exploratory notebook builds a
   labeled template:

   ```python
   document = f"""
   Verse ID: {row.unique_key}

   Sanskrit:
   {row.sanskrit_data}

   Translation:
   {row.translation}
   """
   ```

   The production path (`src/preprocessing.py`, used to actually generate
   `data/processed/cleaned_dataset.csv`) instead concatenates
   `sanskrit_data + "\n" + translation` directly, without the
   `Verse ID: / Sanskrit: / Translation:` labels. **This is a real
   divergence between the prototype notebook and the shipped pipeline**,
   worth being explicit about rather than letting the notebook imply the
   labeled format is what's actually indexed. The unlabeled format is
   slightly more token-efficient (no label overhead eating into the
   512-token budget) at the cost of losing an explicit field boundary the
   encoder could otherwise exploit — a plausible follow-up experiment is
   comparing retrieval quality with the labeled template reinstated.

## 5. Outputs

| File | Rows | Columns | Purpose |
|---|---|---|---|
| `data/raw/bhagavad_gita.csv` | 657 | 5 (incl. precomputed embeddings, unused) | Full HF export, kept for reproducibility |
| `data/raw/gita_clean.csv` / `.json` | 657 | 3 | Raw text columns only |
| `data/processed/cleaned_dataset.csv` | 657 | 4 (+ `document`) | What the indexing/training pipeline actually reads |
| `data/processed/documents.csv` | 657 | 2 (`unique_key`, `document`) | Document-only view, used by the notebook's FAISS-building step |
| `data/training/query_pairs.csv` | 3,942 | 3 (`query`, `document`, `label`) | 6 queries generated per verse (657 × 6) |
| `data/training/triplets.csv` | 3,942 | 3 (`anchor`, `positive`, `negative`) | One triplet per query pair, random-negative sampled |

## 6. Query pair generation (6 per verse)

`src/query_pairs.py` generates, for every verse:

1. `"What is explained in verse {id}?"`
2. `"What does this verse say?"`
3. `"Explain verse {id}"`
4. `"Meaning of verse {id}"`
5. `translation[:120]` — the first 120 characters of the commentary
6. The Sanskrit text itself, used as a Sanskrit-language query against the
   same (Sanskrit + English) document

That last one is what makes the training data genuinely cross-lingual, not
just English-to-English: the model sees Sanskrit text as an anchor and the
combined document as the positive, which is what teaches it that Devanagari
input and English input for the same verse should land near each other in
embedding space.

## 7. Open data-quality considerations (not fully solved here)

Flagging these honestly rather than treating them as solved, since they're
exactly the kind of judgment calls worth being explicit about:

- **Chunking.** Long commentaries risk truncation at the encoder's 512-token
  limit. Not currently chunked/windowed — a natural extension is splitting
  purports over some length threshold into overlapping passages, embedding
  each, and either indexing them as separate retrievable units (with a
  parent-verse pointer) or aggregating (e.g. max-pooling) their embeddings
  back to one verse-level vector.
- **Script normalization.** Sanskrit text is a mix of Devanagari and, within
  the commentary text, IAST-diacritic romanization (`Kṛṣṇa`, `dharma-kṣetra`)
  — but these two scripts are never unified into a single matching key. Two
  strings that are "the same word" in different scripts currently get no
  credit from BM25 (zero token overlap) and only whatever cross-script
  generalization the multilingual encoder learned unsupervised. A normalized
  matching key (script-invariant, used as an auxiliary BM25 field without
  replacing the display text) would close part of that gap.
- **Translation length imbalance.** The 96-to-15,627-character range in the
  `translation` field means document embeddings are not length-normalized in
  any content sense — a short verse's document and a long verse's document
  compete in the same similarity space despite very different information
  density. Whether this systematically biases retrieval toward
  longer/shorter documents is worth a dedicated ablation (bucket verses by
  document length, compare Recall@1 across buckets).
- **No dedicated cross-lingual evaluation split.** Recall/MRR/nDCG are
  currently computed over the mixed query set (English templates + Sanskrit
  query, combined together), not decomposed by query language. A
  Sanskrit-only vs. English-only breakdown would show whether the fine-tuned
  model is actually better at cross-lingual alignment, or whether the
  aggregate metric is being carried by the (easier) English-to-English
  pairs.
