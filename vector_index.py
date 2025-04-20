import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from chunkAndLoad import load_text_chunks, load_table_documents, build_embeddings

# ─── Configuration ────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K           = 5
# ──────────────────────────────────────────────────────────────────────────────

# 1) Initialize embedder
embedder = SentenceTransformer(EMBEDDING_MODEL)

def build_index():
    """
    Load and embed all documents (text chunks + table previews),
    then build and return a FAISS IndexFlatIP index plus the metadata list.
    """
    # load and chunk text
    text_docs = load_text_chunks("extracted/all_text.txt")
    # load table previews
    table_docs = load_table_documents("extracted/tables")
    all_docs = text_docs + table_docs

    # compute embeddings
    docs, embeddings = build_embeddings(all_docs)

    # normalize vectors for cosine similarity
    faiss.normalize_L2(embeddings)

    # create index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return docs, index

def retrieve(query: str, docs: list[dict], index: faiss.IndexFlatIP, k: int = TOP_K) -> list[dict]:
    """
    Given a query string, returns the top-k most similar documents
    from the FAISS index, including their source and similarity score.
    """
    # embed and normalize query
    qvec = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(qvec)

    # search
    distances, indices = index.search(qvec, k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        meta = docs[idx]
        results.append({
            "source":  meta["source"],
            "type":    meta["type"],
            "content": meta["content"],
            "score":   float(dist),
        })
    return results

if __name__ == "__main__":
    # build the index
    print("Building FAISS index…")
    docs, index = build_index()
    print(f"Index built with {len(docs)} vectors (dim={index.d}).")

    # example query
    query = "What was the return on equity for FY24?"
    print(f"\nRetrieving top {TOP_K} chunks for query: “{query}”\n")
    hits = retrieve(query, docs, index, k=TOP_K)
    for i, hit in enumerate(hits, start=1):
        print(f"{i}. [{hit['source']}] (score={hit['score']:.3f})")
        print(f"   {hit['content'][:200].replace(chr(10), ' ')}…\n")