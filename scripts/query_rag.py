import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def load_index(index_dir="models/rag_index"):
    idx_path = os.path.join(index_dir, "faiss_index.bin")
    docs_path = os.path.join(index_dir, "docs.json")
    if not os.path.exists(idx_path) or not os.path.exists(docs_path):
        raise RuntimeError("Index or metadata not found. Run scripts/build_rag_index.py first.")
    index = faiss.read_index(idx_path)
    with open(docs_path, "r", encoding="utf-8") as f:
        docs = json.load(f)
    return index, docs


def query(query_text, top_k=3, index_dir="models/rag_index", model_name="sentence-transformers/all-MiniLM-L6-v2"):
    index, docs = load_index(index_dir)
    model = SentenceTransformer(model_name)
    q_emb = model.encode([query_text], convert_to_numpy=True)
    q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9)
    D, I = index.search(q_emb.astype('float32'), top_k)
    results = []
    for score, iid in zip(D[0], I[0]):
        results.append({"score": float(score), "text": docs[iid]["text"]})
    return results


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("query", type=str)
    p.add_argument("--top-k", type=int, default=3)
    args = p.parse_args()
    res = query(args.query, top_k=args.top_k)
    for r in res:
        print(f"score={r['score']:.4f}\n{r['text']}\n---\n")
