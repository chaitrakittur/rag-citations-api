# 🔎 rag-citations-api
**RAG (Retrieval-Augmented Generation) API with citations + refusal guardrails**  
Built with **FastAPI + OpenAI embeddings** and a lightweight vector store.

## ✨ What it does
- Ingest documents (`/ingest`)
- Ask questions (`/ask`)
- Returns **answer + citations**
- Refuses with **“I don’t know”** if context is insufficient

## 🧠 Architecture
```text
Client → FastAPI → (Embeddings) → Vector Store → Top-K Chunks → LLM Answer (with citations)
