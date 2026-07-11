"""
Streamlit demo for the Bhagavad Gita semantic retrieval pipeline.

Run with:
    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.config import get_config
from src.logging_utils import get_logger, log_pipeline_version
from src.retrieval import RetrievalPipeline

logger = get_logger(__name__)

st.set_page_config(page_title="Bhagavad Gita Semantic Retrieval", page_icon="🕉️", layout="wide")


@st.cache_resource(show_spinner="Loading models and indexes...")
def load_pipeline() -> RetrievalPipeline:
    log_pipeline_version(logger)
    return RetrievalPipeline()


def main():
    cfg = get_config()

    st.title("🕉️ Bhagavad Gita Semantic Retrieval")
    st.caption(
        "Ask a question in natural language and retrieve the most relevant "
        "Sanskrit verse with its English translation."
    )

    with st.sidebar:
        st.header("Pipeline Settings")
        use_query_expansion = st.checkbox("Query expansion", value=cfg.query_expansion.enabled)
        use_reranker = st.checkbox("Cross-encoder re-ranking", value=cfg.reranker.enabled)
        top_k = st.slider("Results to show", min_value=1, max_value=10, value=cfg.retrieval.final_top_k)
        st.divider()
        st.caption(f"Active embedding model: `{cfg.embedding.active_model}`")
        st.caption(f"Pipeline version: `{cfg.pipeline_version}`")

    try:
        pipeline = load_pipeline()
    except Exception as exc:
        st.error(
            "Could not load the retrieval pipeline. Have you run "
            "`scripts/run_build_index.py` yet? "
            f"\n\nDetails: {exc}"
        )
        return

    query = st.text_input("Your question", placeholder="e.g. How to control the mind?")
    example_cols = st.columns(4)
    examples = ["What is karma?", "Who is Krishna?", "What is Dharma?", "Explain detachment"]
    for col, ex in zip(example_cols, examples):
        if col.button(ex):
            query = ex

    if not query:
        st.info("Enter a question above, or click one of the examples.")
        return

    with st.spinner("Retrieving relevant verses..."):
        results = pipeline.retrieve(
            query,
            use_query_expansion=use_query_expansion,
            use_reranker=use_reranker,
            top_k=top_k,
        )

    if not results:
        st.warning("No results found.")
        return

    for r in results:
        with st.container(border=True):
            header_col, score_col = st.columns([4, 1])
            header_col.subheader(f"#{r.rank} — Verse {r.verse_id}")
            score_col.metric("Score", f"{r.similarity:.3f}")

            sk_col, tr_col = st.columns(2)
            sk_col.markdown("**Sanskrit**")
            sk_col.write(r.sanskrit)
            tr_col.markdown("**Translation**")
            tr_col.write(r.translation)


if __name__ == "__main__":
    main()
