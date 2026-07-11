# 🕉️ Bhagavad Gita Semantic Retrieval System

A production-ready **semantic search and retrieval system for Bhagavad Gita verses** that retrieves the most relevant Sanskrit verses along with their English translations based on natural language queries.

The system understands the **semantic meaning** behind user questions rather than relying only on keyword matching. Users can ask philosophical or life-related questions, and the system retrieves the most contextually relevant verses from the Bhagavad Gita.

Example:

**User Query:**

> How can I control my mind?

**System Response:**

Returns the most relevant Bhagavad Gita verse with:
- Sanskrit verse
- English translation
- Similarity score
- Contextual information


---

# 🚀 Project Highlights

- Built a semantic retrieval engine from scratch
- Converted Bhagavad Gita verses into meaningful vector embeddings
- Implemented similarity-based retrieval using FAISS
- Improved retrieval quality using advanced NLP techniques
- Modular pipeline design for scalability
- Designed for future integration with RAG-based conversational AI systems


---

# 🎯 Motivation

Traditional keyword-based search systems fail when users ask questions using different words but with the same meaning.

Example:

Keyword Search:

```
"control mind"
```

may not retrieve:

```
"How can one achieve mastery over thoughts?"
```

because the words are different.

Semantic retrieval solves this problem by understanding the meaning of the query.


---

# 🏗️ System Architecture


```
                 User Query
                     |
                     ↓
          Query Preprocessing
                     |
                     ↓
          Query Expansion
                     |
                     ↓
        Sentence Transformer Model
                     |
                     ↓
          Query Embedding Vector
                     |
                     ↓
              FAISS Search
                     |
                     ↓
          Similarity Ranking
                     |
                     ↓
        Top Relevant Bhagavad Gita
               Verses Retrieved
                     |
                     ↓
        Sanskrit Verse + Translation
```


---

# 🔥 Features

## 1. Semantic Search

Uses transformer-based embeddings to understand the meaning of queries.

Example:

Query:

```
How to overcome fear?
```

can retrieve verses related to:

```
Courage, detachment, self-control, and mental strength
```


---

## 2. Vector-Based Retrieval

Each Bhagavad Gita verse is converted into a numerical vector representation.

The system compares:

```
Query Embedding
        |
        |
        ↓
Verse Embeddings
```

using similarity metrics.


---

## 3. FAISS Similarity Search

Implemented Facebook AI Similarity Search (FAISS) for efficient nearest-neighbor retrieval.

Advantages:

- Fast vector search
- Scalable to large datasets
- Low latency retrieval


---

## 4. Query Enhancement

The retrieval pipeline improves user queries through:

- Text normalization
- Query expansion
- Synonym handling
- Context enrichment


---

# 🔄 Retrieval Pipeline


## Step 1: Data Processing

The Bhagavad Gita dataset contains:

- Chapter number
- Verse number
- Sanskrit text
- Transliteration
- English translation


Processing steps:

- Data cleaning
- Text normalization
- Removing unnecessary characters
- Preparing searchable documents


---

## Step 2: Text Embedding Generation

Each verse is converted into embeddings using:

```
Sentence Transformer Model
```

Example:

```
Verse Text
     |
     ↓
Embedding Model
     |
     ↓
Dense Vector Representation
```


---

## Step 3: Index Creation

Generated embeddings are stored in FAISS index.

Example:

```
Verse 2.47
     |
     ↓
[0.234, 0.541, 0.123 ...]
```


---

## Step 4: Query Retrieval

When a user enters a question:

```
User Question
      |
      ↓
Embedding Generation
      |
      ↓
FAISS Similarity Search
      |
      ↓
Top-K Relevant Verses
```


---

# 🧠 Technologies Used


## Programming Language

- Python


## NLP / Deep Learning

- Sentence Transformers
- Transformers
- Natural Language Processing


## Vector Database

- FAISS


## Data Processing

- Pandas
- NumPy


## Development Tools

- Jupyter Notebook
- Git
- GitHub


---

# 📂 Project Structure


```
Bhagavad-Gita-Semantic-Retrieval/

│
├── data/
│   └── bhagavad_gita_dataset.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_embedding_generation.ipynb
│   ├── 03_vector_search.ipynb
│   └── 04_evaluation.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── embedding.py
│   ├── retrieval.py
│   └── evaluation.py
│
├── tests/
│
├── requirements.txt
│
├── README.md
│
└── .gitignore

```


---

# ⚙️ Installation


Clone the repository:

```bash
git clone https://github.com/Jasminbabariya48/Bhagavad-Gita-Semantic-Retrieval.git
```


Move into project directory:

```bash
cd Bhagavad-Gita-Semantic-Retrieval
```


Create virtual environment:

```bash
python -m venv .venv
```


Activate environment:


Windows:

```bash
.venv\Scripts\activate
```


Linux/Mac:

```bash
source .venv/bin/activate
```


Install dependencies:

```bash
pip install -r requirements.txt
```


---

# ▶️ Usage


Run the retrieval pipeline:

```bash
python src/retrieval.py
```


Example:


Input:

```
What is the importance of karma?
```


Output:

```
Chapter: 2
Verse: 47

Sanskrit:
कर्मण्येवाधिकारस्ते...

Translation:
You have the right to perform your duties...
```


---

# 📊 Evaluation

The retrieval system can be evaluated using:


## Retrieval Metrics

- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)


## Similarity Evaluation

- Cosine Similarity
- Ranking Accuracy


Example:

```
Query:
How to achieve peace?

Retrieved:
Chapter 2 Verse 70

Similarity Score:
0.89
```


---

# 🚀 Future Improvements


## 1. Retrieval-Augmented Generation (RAG)

Integrate Large Language Models to generate human-like answers using retrieved verses.


## 2. Hybrid Search

Combine:

- Keyword search
- Semantic search


for better retrieval accuracy.


## 3. Re-ranking Model

Add cross-encoder based ranking:

```
Bi-Encoder Retrieval
        |
        ↓
Cross Encoder Re-ranking
        |
        ↓
Final Results
```


## 4. Web Application

Deploy using:

- Flask
- FastAPI
- Streamlit


---

# 🌟 Learning Outcomes

Through this project:

- Understanding of semantic search systems
- Practical implementation of embeddings
- Vector database handling
- NLP pipeline development
- Information retrieval techniques
- Foundation for RAG applications


---

# 👨‍💻 Author

**Jasmin Babariya**

Data Scientist | AI/ML Engineer

GitHub:
https://github.com/Jasminbabariya48


---

# 📜 License

This project is developed for educational and research purposes.