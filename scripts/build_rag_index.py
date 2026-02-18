import os
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss


def read_docs(docs_dir):
    texts = []
    for p in Path(docs_dir).rglob("*.txt"):
        with open(p, "r", encoding="utf-8") as f:
            texts.append(f.read())
    return texts


def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def build_index(docs_dir="data/docs", out_dir="models/rag_index", model_name="sentence-transformers/all-MiniLM-L6-v2"):
    os.makedirs(out_dir, exist_ok=True)
    texts = read_docs(docs_dir)
    if not texts:
        raise RuntimeError(f"No .txt docs found in {docs_dir}")

    chunks = []
    for t in texts:
        chunks.extend(chunk_text(t, chunk_size=500, overlap=50))

    print(f"Total chunks: {len(chunks)}")

    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)

    # normalize for cosine similarity with inner product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    embeddings = embeddings / norms

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype('float32'))

    faiss.write_index(index, os.path.join(out_dir, "faiss_index.bin"))
    np.save(os.path.join(out_dir, "embeddings.npy"), embeddings)

    docs_meta = [{"id": i, "text": chunks[i]} for i in range(len(chunks))]
    with open(os.path.join(out_dir, "docs.json"), "w", encoding="utf-8") as f:
        json.dump(docs_meta, f, ensure_ascii=False, indent=2)

    print(f"Saved FAISS index and metadata to {out_dir}")


if __name__ == '__main__':
    build_index()
