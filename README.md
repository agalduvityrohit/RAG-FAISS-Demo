# 🔍 RAG FAISS Demo  
### Retrieval-Augmented Generation using SentenceTransformers + FAISS

This repository contains a complete RAG (Retrieval-Augmented Generation) demo written in Python.  
It demonstrates how to ingest documents, chunk them, create embeddings using **SentenceTransformers**,  
index them using **FAISS**, and retrieve relevant passages to answer a query.

---

## 🚀 Features
- Automatic document ingestion from `./docs/`
- Text chunking with overlapping windows
- Dense embedding generation using `all-MiniLM-L6-v2`
- FAISS inner-product index for ultra-fast similarity search
- Metadata tracking for transparent source citations
- CLI for indexing and querying
- Easy extension to add an LLM synthesizer (OpenAI, Llama, HF)


