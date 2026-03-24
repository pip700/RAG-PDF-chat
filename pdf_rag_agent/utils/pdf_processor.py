"""
PDF processing with FREE HuggingFace embeddings (no API key needed)
"""

import os
import tempfile
from dataclasses import dataclass, field

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

# Use HuggingFace embeddings by default (FREE, no API key)
from langchain_community.embeddings import HuggingFaceEmbeddings


@dataclass
class ProcessingResult:
    vector_store: InMemoryVectorStore = None
    total_pages: int = 0
    total_chunks: int = 0
    file_details: list = field(default_factory=list)
    all_documents: list = field(default_factory=list)


def get_embeddings(provider: str):
    """Get embeddings - uses FREE HuggingFace by default."""

    # For OpenAI, use their embeddings if key is available
    if provider == "OpenAI" and os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(model="text-embedding-3-small")
        except Exception:
            pass  # Fall back to HuggingFace

    # For Google, use their embeddings if key is available
    if provider == "Google Gemini (FREE)" and os.getenv("GOOGLE_API_KEY"):
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            return GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
        except Exception:
            pass  # Fall back to HuggingFace

    # DEFAULT: HuggingFace embeddings (FREE, no API key needed!)
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


def load_pdfs(uploaded_files: list) -> tuple[list[Document], list[dict]]:
    """Load PDFs from uploaded files."""
    all_docs = []
    file_details = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        for doc in docs:
            doc.metadata["source_filename"] = uploaded_file.name
            doc.metadata["file_size_kb"] = round(uploaded_file.size / 1024, 1)

        all_docs.extend(docs)
        file_details.append({
            "name": uploaded_file.name,
            "pages": len(docs),
            "size_kb": round(uploaded_file.size / 1024, 1),
        })

        os.unlink(tmp_path)

    return all_docs, file_details


def chunk_documents(
    docs: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Split documents into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def process_pdfs(
    uploaded_files: list,
    chunk_size: int = 800,
    chunk_overlap: int = 200,
    provider: str = "Groq (FREE)",
) -> ProcessingResult:
    """Full pipeline: load → chunk → embed → store."""

    docs, file_details = load_pdfs(uploaded_files)
    chunks = chunk_documents(docs, chunk_size, chunk_overlap)

    # Get FREE embeddings
    embeddings = get_embeddings(provider)

    vector_store = InMemoryVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    return ProcessingResult(
        vector_store=vector_store,
        total_pages=len(docs),
        total_chunks=len(chunks),
        file_details=file_details,
        all_documents=chunks,
    )
