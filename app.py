"""
app.py — Free EDA Agent (no AI API needed)
-------------------------------------------
100% free. No API key. No rate limits. No credit card.
Just upload a dataset and get a full analysis instantly.

Run:  python3 -m streamlit run app.py
"""

import uuid
import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from eda_engine import EDAEngine

load_dotenv()

DAILY_LIMIT = int(os.getenv("DAILY_LIMIT", "999"))  # effectively unlimited
MAX_FILE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "20"))
APP_TITLE   = os.getenv("APP_TITLE", "🔍 Free EDA Agent")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Free EDA Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session ───────────────────────────────────────────────────────────────────
if "result"   not in st.session_state: st.session_state.result   = None
if "filename" not in st.session_state: st.session_state.filename = ""

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔍 Free EDA Agent")
    st.markdown(
        "A **completely free** data analysis tool — "
        "no API key, no subscription, no limits."
    )
    st.markdown("---")
    st.markdown("**✅ Completely free — forever**")
    st.markdown("No AI API · No credit card · No login")
    st.markdown("---")
    st.markdown("**Supported formats**")
    st.markdown("CSV · Excel (.xlsx) · TSV · JSON")
    st.markdown("---")
    st.markdown("**What it analyses**")
    st.markdown(
        "- 📋 Dataset overview\n"
        "- 🧹 Missing values & duplicates\n"
        "- 📊 Descriptive statistics\n"
        "- 📈 Distributions & skewness\n"
        "- ⚠️ Outlier detection\n"
        "- 🔗 Correlations\n"
        "- 🏷️ Categorical breakdowns\n"
        "- 💡 Smart recommendations"
    )

# ── Header ────────────────────────────────────────────────────────────────────
st.title(APP_TITLE)
st.markdown(
    "Upload your dataset and get a **full data analysis report instantly** — "
    "completely free, no account or API key needed."
)
st.info("⚡ Powered by Python + Pandas — no AI API, no cost, no limits.", icon="🆓")

# ── Upload ────────────────────────────────────────────────────────────────────
col_upload, col_question = st.columns([2, 1], gap="large")

with col_upload:
    uploaded_file = st.file_uploader(
        "📂 Upload your dataset",
        type=["csv", "xlsx", "xls", "tsv", "json"],
        help=f"Max file size: {MAX_FILE_MB} MB",
    )

with col_question:
    st.markdown("**Ask a specific question** *(optional)*")
    user_question = st.text_area(
        "question",
        placeholder=(
            "e.g. Which columns have the most missing data?\n"
            "or: Are there strong correlations between columns?"
        ),
        height=130,
        label_visibility="collapsed",
    )

# ── Load & preview ────────────────────────────────────────────────────────────
df = None
if uploaded_file:
    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        st.error(f"❌ File is {size_mb:.1f} MB — max allowed is {MAX_FILE_MB} MB.")
        st.stop()

    try:
        ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
        if ext == "csv":            df = pd.read_csv(uploaded_file)
        elif ext in ("xlsx","xls"): df = pd.read_excel(uploaded_file)
        elif ext == "tsv":          df = pd.read_csv(uploaded_file, sep="\t")
        elif ext == "json":         df = pd.read_json(uploaded_file)
        else:
            st.error("Unsupported format."); st.stop()
    except Exception as e:
        st.error(f"❌ Could not read file: {e}"); st.stop()

    st.success(f"✅ **{uploaded_file.name}** — {df.shape[0]:,} rows × {df.shape[1]} columns")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows",            f"{df.shape[0]:,}")
    m2.metric("Columns",         df.shape[1])
    m3.metric("Missing values",  f"{df.isnull().sum().sum():,}")
    m4.metric("Numeric columns", len(df.select_dtypes(include="number").columns))

    with st.expander("👀 Preview first 10 rows", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)

    st.markdown("")
    if st.button("🚀 Run EDA Analysis — Free!", type="primary", use_container_width=True):
        st.session_state.result   = None
        st.session_state.filename = uploaded_file.name
        msgs = []

        with st.status("⚙️ Analysing your data...", expanded=True) as status_box:
            engine = EDAEngine()
            result = engine.analyze(
                df,
                user_question=user_question,
                filename=uploaded_file.name,
                progress_callback=lambda m: msgs.append(m),
            )
            for m in msgs:
                st.write(m)
            status_box.update(label="✅ Done!", state="complete", expanded=False)

        st.session_state.result = result
        st.rerun()

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.result:
    result   = st.session_state.result
    filename = st.session_state.filename

    st.markdown("---")
    st.header(f"📊 EDA Report — {filename}")

    with st.expander("📄 Full Analysis Report", expanded=True):
        st.markdown(result["report"])

    charts = result.get("charts", [])
    if charts:
        st.markdown("---")
        st.header("📈 Visualisations")
        for i in range(0, len(charts), 2):
            cols = st.columns(2, gap="medium")
            for j, col_widget in enumerate(cols):
                if i + j < len(charts):
                    with col_widget:
                        st.subheader(charts[i+j]["title"])
                        st.plotly_chart(charts[i+j]["fig"], use_container_width=True)

    st.markdown("---")
    col_r, col_d = st.columns(2)
    with col_r:
        if st.button("🔄 Analyse another dataset", use_container_width=True):
            st.session_state.result = None
            st.rerun()
    with col_d:
        st.download_button(
            "⬇️ Download report (.md)",
            data=result["report"],
            file_name=f"eda_{filename.rsplit('.',1)[0]}.md",
            mime="text/markdown",
            use_container_width=True,
        )

st.markdown("---")
st.caption("100% free · No API · Built with Python + Pandas + Plotly + Streamlit")
