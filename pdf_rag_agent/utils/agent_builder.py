"""
RAG Chain with Multiple FREE LLM Providers
NO OpenAI dependency for basic usage
"""

import os
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import InMemoryVectorStore


def get_llm(provider: str, model_name: str, temperature: float = 0.0):
    """Get LLM based on selected provider."""

    if provider == "OpenAI":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
        )

    elif provider == "Groq (FREE)":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set. Get FREE key at: https://console.groq.com/keys")
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
        )

    elif provider == "Google Gemini (FREE)":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set. Get FREE key at: https://aistudio.google.com/app/apikey")
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=api_key,
        )

    elif provider == "Ollama (FREE Local)":
        from langchain_community.chat_models import ChatOllama
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(
            model=model_name,
            temperature=temperature,
            base_url=base_url,
        )

    else:
        raise ValueError(f"Unknown provider: {provider}")


# Available models per provider
PROVIDER_MODELS = {
    "Groq (FREE)": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"],
    "Google Gemini (FREE)": ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"],
    "Ollama (FREE Local)": ["llama3", "llama3:8b", "mistral", "phi3", "gemma:7b"],
    "OpenAI": ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
}


SYSTEM_PROMPT = """You are a PDF Document Analysis Assistant.

Use the following context from PDF documents to answer the question.
If the information is not in the context, say "Not found in the provided documents."

Context:
{context}

Question: {question}

Instructions:
- Cite the source file and page number when possible
- Use markdown formatting (headers, bullets, tables, bold)
- Quote exact values with units
- Be precise and thorough
"""


class RAGChain:
    """Simple RAG chain with multi-provider support."""

    def __init__(
        self,
        vector_store: InMemoryVectorStore,
        provider: str = "Groq (FREE)",
        model_name: str = "llama-3.3-70b-versatile",
        temperature: float = 0.0,
    ):
        self.vector_store = vector_store
        self.llm = get_llm(provider, model_name, temperature)
        self.prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
        self.chain = self.prompt | self.llm | StrOutputParser()

    def search(self, query: str, k: int = 6) -> str:
        """Search vector store."""
        if "tool_call_log" in st.session_state:
            st.session_state.tool_call_log.append({
                "tool": "search",
                "query": query,
            })

        results = self.vector_store.similarity_search(query=query, k=k)

        if not results:
            return "No relevant information found."

        context_parts = []
        seen = set()
        for doc in results:
            content_hash = hash(doc.page_content[:100])
            if content_hash in seen:
                continue
            seen.add(content_hash)

            source = doc.metadata.get("source_filename", "Unknown")
            page = doc.metadata.get("page", "?")
            context_parts.append(
                f"[Source: {source} | Page: {page}]\n{doc.page_content}"
            )

        return "\n\n---\n\n".join(context_parts)

    def invoke(self, query: str) -> str:
        """Process query and return response."""
        context = self.search(query, k=8)
        response = self.chain.invoke({
            "context": context,
            "question": query,
        })
        return response


def build_agent(
    vector_store: InMemoryVectorStore,
    provider: str = "Groq (FREE)",
    model_name: str = "llama-3.3-70b-versatile",
    temperature: float = 0.0,
) -> RAGChain:
    """Build RAG chain."""
    return RAGChain(vector_store, provider, model_name, temperature)


def run_agent(agent: RAGChain, query: str) -> str:
    """Run the RAG chain."""
    return agent.invoke(query)
