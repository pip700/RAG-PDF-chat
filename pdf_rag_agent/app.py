"""
PDF RAG Agent - Multi-Provider Support
Uses FREE HuggingFace embeddings (no API key needed for embeddings!)
"""

import os
import streamlit as st
from dotenv import load_dotenv

from utils.pdf_processor import process_pdfs
from utils.agent_builder import build_agent, run_agent, PROVIDER_MODELS

load_dotenv()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title="PDF RAG Agent",
    page_icon="🔬",
    layout="wide",
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
    .free-badge {
        background: #d4edda;
        color: #155724;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .paid-badge {
        background: #fff3cd;
        color: #856404;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SESSION STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
defaults = {
    "vector_store": None,
    "agent": None,
    "chat_history": [],
    "pdf_loaded": False,
    "processing_result": None,
    "tool_call_log": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIDEBAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.sidebar:
    st.markdown("## 🔬 PDF RAG Agent")
    st.divider()

    # Provider Selection
    st.markdown("### 🤖 LLM Provider")

    provider = st.selectbox(
        "Provider",
        list(PROVIDER_MODELS.keys()),
        index=0,  # Default: Groq (FREE)
    )

    # Badge
    if "FREE" in provider:
        st.markdown('<span class="free-badge">✅ FREE</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="paid-badge">💰 PAID</span>', unsafe_allow_html=True)

    # Model
    model_name = st.selectbox("Model", PROVIDER_MODELS[provider])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)

    st.divider()

    # API Key Check
    st.markdown("### 🔑 API Key")

    if provider == "Groq (FREE)":
        key = os.getenv("GROQ_API_KEY")
        if key and len(key) > 10:
            st.success(f"✅ Groq: ...{key[-8:]}")
        else:
            st.error("❌ GROQ_API_KEY missing")
            st.markdown("👉 [Get FREE key](https://console.groq.com/keys)")
            st.code("GROQ_API_KEY=gsk_xxx")

    elif provider == "Google Gemini (FREE)":
        key = os.getenv("GOOGLE_API_KEY")
        if key and len(key) > 10:
            st.success(f"✅ Google: ...{key[-8:]}")
        else:
            st.error("❌ GOOGLE_API_KEY missing")
            st.markdown("👉 [Get FREE key](https://aistudio.google.com/app/apikey)")
            st.code("GOOGLE_API_KEY=AIza_xxx")

    elif provider == "Ollama (FREE Local)":
        st.info("🏠 No API key needed")
        st.code("ollama serve")

    elif provider == "OpenAI":
        key = os.getenv("OPENAI_API_KEY")
        if key and len(key) > 10:
            st.success(f"✅ OpenAI: ...{key[-8:]}")
        else:
            st.error("❌ OPENAI_API_KEY missing")

    st.divider()

    # Chunking
    st.markdown("### 📐 Chunking")
    chunk_size = st.slider("Size", 200, 2000, 800, 100)
    chunk_overlap = st.slider("Overlap", 0, 500, 200, 50)

    st.divider()

    # Upload
    st.markdown("### 📁 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Upload",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button("⚡ Process", use_container_width=True, type="primary"):
            with st.status("Processing...", expanded=True) as status:
                try:
                    st.write("📖 Loading PDFs...")
                    result = process_pdfs(
                        uploaded_files, chunk_size, chunk_overlap, provider
                    )
                    st.session_state.processing_result = result
                    st.session_state.vector_store = result.vector_store

                    st.write("🧠 Building RAG Chain...")
                    agent = build_agent(
                        result.vector_store, provider, model_name, temperature
                    )
                    st.session_state.agent = agent
                    st.session_state.pdf_loaded = True
                    st.session_state.chat_history = []
                    st.session_state.tool_call_log = []

                    status.update(label="✅ Done!", state="complete")
                except Exception as e:
                    status.update(label="❌ Error", state="error")
                    st.error(f"{e}")

    st.divider()

    # Stats
    if st.session_state.pdf_loaded:
        res = st.session_state.processing_result
        c1, c2 = st.columns(2)
        c1.metric("Pages", res.total_pages)
        c2.metric("Chunks", res.total_chunks)

    st.divider()

    c1, c2 = st.columns(2)
    if c1.button("🗑️ Clear"):
        st.session_state.chat_history = []
        st.rerun()
    if c2.button("♻️ Reset"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown('<p class="main-title">🔬 PDF RAG Agent</p>', unsafe_allow_html=True)

if not st.session_state.pdf_loaded:
    st.info("👈 Select provider, upload PDFs, click Process")

    st.markdown("""
    ### 🆓 FREE Providers

    | Provider | Get API Key | Speed |
    |----------|-------------|-------|
    | **Groq** | [console.groq.com](https://console.groq.com/keys) | ⚡⚡⚡ Fastest |
    | **Google Gemini** | [aistudio.google.com](https://aistudio.google.com/app/apikey) | ⚡⚡ Fast |
    | **Ollama** | No key needed (local) | ⚡ Depends on hardware |

    📝 **Embeddings are FREE** - Uses HuggingFace (no API key needed!)
    """)
    st.stop()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TABS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
tab_chat, tab_extract, tab_summary = st.tabs(["💬 Chat", "🎯 Extract", "📝 Summary"])


def query(q: str) -> str:
    return run_agent(st.session_state.agent, q)


# Chat Tab
with tab_chat:
    st.markdown("### 💬 Chat with Documents")

    # Quick buttons
    cols = st.columns(4)
    quick = [
        ("👤 People", "List all people mentioned with their roles"),
        ("📅 Dates", "List all dates with context"),
        ("🔢 Values", "List all numeric values and measurements"),
        ("📋 Summary", "Summarize the key points"),
    ]
    quick_q = None
    for i, (label, prompt) in enumerate(quick):
        if cols[i].button(label, use_container_width=True):
            quick_q = prompt

    st.divider()

    # History
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    user_input = st.chat_input("Ask anything...")
    active = quick_q or user_input

    if active:
        st.chat_message("user").markdown(active)
        st.session_state.chat_history.append({"role": "user", "content": active})

        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching..."):
                try:
                    answer = query(active)
                    st.markdown(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"❌ {e}")


# Extract Tab
with tab_extract:
    st.markdown("### 🎯 Custom Extraction")

    templates = {
        "🩺 Patient Info": "Extract patient name, age, gender, diagnosis",
        "💊 Medications": "List all medications with dosage",
        "🧪 Lab Results": "Extract all test results with values",
        "📋 Tables": "Find and format all tabular data",
    }

    cols = st.columns(4)
    selected = None
    for i, (name, prompt) in enumerate(templates.items()):
        if cols[i].button(name, use_container_width=True, key=f"ext_{i}"):
            selected = prompt

    custom = st.text_area("Your request:", value=selected or "", height=80)

    if st.button("🎯 Extract", type="primary") and custom:
        with st.spinner("Extracting..."):
            try:
                answer = query(custom + " Format clearly with markdown.")
                st.markdown(answer)
                st.download_button("📥 Download", answer, "extraction.txt")
            except Exception as e:
                st.error(f"❌ {e}")


# Summary Tab
with tab_summary:
    st.markdown("### 📝 Document Summary")

    summary_type = st.radio("Type:", ["Full Summary", "Key Points", "Brief"], horizontal=True)

    if st.button("📝 Generate", type="primary"):
        with st.spinner("Summarizing..."):
            try:
                answer = query(f"Create a {summary_type} of the document with key findings and page references.")
                st.markdown(answer)
                st.download_button("📥 Download", answer, "summary.txt")
            except Exception as e:
                st.error(f"❌ {e}")
