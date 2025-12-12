
import os
import sys
import json
import math
import argparse
from pathlib import Path
from typing import List, Dict

# Optional imports with friendly messages
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    print("ERROR: sentence_transformers not installed. Install with: pip install sentence-transformers")
    raise

try:
    import faiss
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False
    print("WARNING: faiss not available. The script will attempt to use a fallback brute-force search (slower).")

import numpy as np
from tqdm import tqdm


EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # small & fast SBERT model
DOCS_DIR = Path('docs')
INDEX_PATH = Path('index.faiss')
META_PATH = Path('meta.json')
README_PATH = Path('README_rag.md')
CHUNK_SIZE = 200  # tokens ~ words; here we chunk by words for simplicity
CHUNK_OVERLAP = 50

# ---- Utilities ------

def ensure_sample_docs():
    """If ./docs is empty, create a few small sample text files."""
    DOCS_DIR.mkdir(exist_ok=True)
    if any(DOCS_DIR.iterdir()):
        return
    samples = {
        'install.txt': (
            "To install the app, download the installer from the official site and run the installer. "
            "On Linux, use the tarball, extract, and run ./install.sh. For Windows, run the .exe with admin rights."
        ),
        'usage.txt': (
            "Starting the service: run `service start myapp` or use the systemd unit `systemctl start myapp`. "
            "Default port is 8080. Logs are written to /var/log/myapp/. To run in foreground, use `myapp --debug`."
        ),
        'troubleshoot.txt': (
            "If you encounter 'permission denied', check the file ownership and permissions. Use `chmod`/`chown` or run with sudo. "
            "For network timeouts, verify firewall rules and that the remote host is reachable."
        ),
        'faq.txt': (
            "Q: How to reset password? A: Use the admin UI under Settings -> Account -> Reset Password. "
            "Q: Where can I find the config file? A: /etc/myapp/config.yaml"
        )
    }
    for name, text in samples.items():
        p = DOCS_DIR / name
        p.write_text(text)
    print(f"Created sample docs in {DOCS_DIR}/")


def chunk_text(text: str, chunk_size: int=CHUNK_SIZE, overlap: int=CHUNK_OVERLAP) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    n = len(words)
    while i < n:
        chunk_words = words[i:i+chunk_size]
        chunks.append(' '.join(chunk_words))
        i += chunk_size - overlap
    return chunks

# --------------------------- Indexing ---------------------------

def build_index(model_name: str = EMBEDDING_MODEL):
    """Read docs, chunk them, embed with SBERT, build FAISS index, and save metadata."""
    ensure_sample_docs()
    model = SentenceTransformer(model_name)
    docs = []  # list of (doc_id, filename, text)
    for i, p in enumerate(sorted(DOCS_DIR.iterdir())):
        if p.is_file():
            docs.append({'id': i, 'path': str(p), 'text': p.read_text(encoding='utf-8')})
    # chunk docs
    passages = []  # each passage: {'doc_id', 'text', 'meta'}
    for d in docs:
        chunks = chunk_text(d['text'])
        for j, c in enumerate(chunks):
            passages.append({'doc_id': d['id'], 'doc_path': d['path'], 'chunk_id': j, 'text': c})
    print(f"Prepared {len(passages)} passages from {len(docs)} documents.")

    # compute embeddings in batches
    texts = [p['text'] for p in passages]
    batch_size = 32
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc='Embedding'):
        batch = texts[i:i+batch_size]
        embs = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        embeddings.append(embs)
    embeddings = np.vstack(embeddings).astype('float32')
    dim = embeddings.shape[1]
    print(f"Embeddings shape: {embeddings.shape}")

    # build FAISS index
    if _HAS_FAISS:
        index = faiss.IndexFlatIP(dim)
        # normalize embeddings for cosine similarity via inner product
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        faiss.write_index(index, str(INDEX_PATH))
        print(f"FAISS index saved to {INDEX_PATH}")
    else:
        # fallback: save embeddings & use brute force search later
        np.save('embeddings.npy', embeddings)
        print("Saved embeddings.npy for fallback search (no faiss).")

    # save metadata
    meta = {'passages': passages, 'dim': dim}
    with open(META_PATH, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {META_PATH}")

    # write a short README
    README_PATH.write_text("""
RAG FAISS Demo README

To reproduce:
  1. python rag_faiss_demo.py --index
  2. python rag_faiss_demo.py --query "your question here" --k 3
""")
    print("Indexing finished.")

# --------------------------- Querying ---------------------------

def load_index_and_meta():
    if not META_PATH.exists():
        raise FileNotFoundError("meta.json not found. Run --index first to build the index.")
    meta = json.loads(META_PATH.read_text(encoding='utf-8'))
    passages = meta['passages']
    if _HAS_FAISS and INDEX_PATH.exists():
        index = faiss.read_index(str(INDEX_PATH))
        return index, passages
    elif Path('embeddings.npy').exists():
        embeddings = np.load('embeddings.npy')
        return embeddings, passages
    else:
        raise FileNotFoundError("No index found. Run --index first.")


def retrieve(query: str, k: int=3, model_name: str=EMBEDDING_MODEL):
    model = SentenceTransformer(model_name)
    q_emb = model.encode([query], convert_to_numpy=True).astype('float32')
    if _HAS_FAISS and INDEX_PATH.exists():
        faiss.normalize_L2(q_emb)
        index = faiss.read_index(str(INDEX_PATH))
        D, I = index.search(q_emb, k)
        # D are inner products (cosine) because we normalized
        idxs = I[0].tolist()
        scores = D[0].tolist()
        results = []
        for score, idx in zip(scores, idxs):
            p = json.loads(META_PATH.read_text())[ 'passages'][idx]
            results.append({'score': float(score), 'text': p['text'], 'doc_path': p['doc_path'], 'chunk_id': p['chunk_id']})
        return results
    else:
        embeddings = np.load('embeddings.npy')
        # cosine similarity via normalized dot
        q = q_emb[0]
        # normalize
        q = q / np.linalg.norm(q)
        embs = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        sims = embs.dot(q)
        topk = sims.argsort()[-k:][::-1]
        results = []
        meta = json.loads(META_PATH.read_text())
        for idx in topk:
            results.append({'score': float(sims[idx]), 'text': meta['passages'][idx]['text'], 'doc_path': meta['passages'][idx]['doc_path'], 'chunk_id': meta['passages'][idx]['chunk_id']})
        return results


def synthesize_answer(query: str, retrieved: List[Dict]) -> str:
    # Simple synthesizer: concatenates top passages and marks sources; can be replaced by LLM prompt.
    parts = []
    for i, r in enumerate(retrieved, 1):
        src = f"[source {i}: {Path(r['doc_path']).name}#chunk{r['chunk_id']}]"
        parts.append(f"{src} {r['text']}")
    answer = "\n\n".join(parts)
    header = f"Answer (aggregated from top {len(retrieved)} passages):\n"
    return header + answer

# --------------------------- CLI ---------------------------

def main():
    parser = argparse.ArgumentParser(description='RAG FAISS Demo')
    parser.add_argument('--index', action='store_true', help='Build index from docs/')
    parser.add_argument('--query', type=str, help='Query string to retrieve')
    parser.add_argument('--k', type=int, default=3, help='Top-K retrieval')
    parser.add_argument('--model', type=str, default=EMBEDDING_MODEL, help='SentenceTransformer model')
    args = parser.parse_args()

    if args.index:
        build_index(model_name=args.model)
        return

    if args.query:
        if not META_PATH.exists():
            print('Index not found. Run with --index first to create the index from ./docs/')
            return
        print(f"Running retrieval for: {args.query}")
        retrieved = retrieve(args.query, k=args.k, model_name=args.model)
        print("\nRetrieved passages:")
        for i, r in enumerate(retrieved, 1):
            print(f"[{i}] score={r['score']:.4f} source={Path(r['doc_path']).name}#chunk{r['chunk_id']}")
            print(r['text'][:300].strip() + ('...' if len(r['text'])>300 else ''))
            print('---')
        ans = synthesize_answer(args.query, retrieved)
        print('\n' + ans)
        return

    parser.print_help()

if __name__ == '__main__':
    main()
