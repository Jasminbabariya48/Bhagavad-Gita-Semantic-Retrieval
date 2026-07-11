# immverse_ai — Bhagavad Gita Semantic Retrieval

A production-ready semantic retrieval system for Bhagavad Gita verses.

Given a natural-language query like:

> "How to control the mind?"

the system returns the most relevant Sanskrit verse and its English translation.

---

## Why the pipeline changed

The original pipeline was single-stage:

1. Embed query using SentenceTransformer
2. Search FAISS IndexFlatIP
3. Return top-k results

That's fast but has a ceiling.

---

## User Query Pipeline

### 1. Preprocessing

- Unicode normalization
- Whitespace normalization

### 2. Query Expansion

- Synonym expansion
- Context enrichment