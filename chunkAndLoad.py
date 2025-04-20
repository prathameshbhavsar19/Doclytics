import os
import glob
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# ─── Configuration ────────────────────────────────────────────────────────────
TEXT_FILE        = "extracted/all_text.txt"
TABLE_DIR        = "extracted/tables"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"
CHUNK_SIZE       = 500      # words per chunk
CHUNK_OVERLAP    = 100      # words overlap
# ──────────────────────────────────────────────────────────────────────────────

embedder = SentenceTransformer(EMBEDDING_MODEL)

def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    chunks = []
    step = size - overlap
    for start in range(0, len(words), step):
        end = start + size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
    return chunks

def load_text_chunks(text_file: str) -> list[dict]:
    with open(text_file, "r", encoding="utf-8") as f:
        full_text = f.read()
    pages = full_text.split("--- PAGE ")[1:]
    text_docs = []
    for page_block in pages:
        page_num_str, body = page_block.split("---", 1)
        page_num = int(page_num_str.strip())
        for idx, chunk in enumerate(chunk_text(body.strip(), CHUNK_SIZE, CHUNK_OVERLAP), start=1):
            text_docs.append({
                "type":    "text",
                "content": chunk,
                "source":  f"page_{page_num}_chunk_{idx}"
            })
    return text_docs

def load_table_documents(table_dir: str) -> list[dict]:
    table_docs = []
    for path in glob.glob(os.path.join(table_dir, "*.csv")):
        df = pd.read_csv(path)
        header = list(df.columns)
        preview = df.head(2).to_dict(orient="records") + df.tail(2).to_dict(orient="records")
        content = (
            f"Table {os.path.basename(path)}\n"
            f"Columns: {header}\n"
            f"Rows: {preview}"
        )
        table_docs.append({
            "type":    "table",
            "content": content,
            "source":  os.path.basename(path)
        })
    return table_docs

def build_embeddings(documents: list[dict]) -> tuple[list[dict], "np.ndarray"]:
    contents = [doc["content"] for doc in documents]
    embeddings = embedder.encode(contents, convert_to_numpy=True)
    return documents, embeddings

def main():
    # Load and chunk text
    text_docs = load_text_chunks(TEXT_FILE)
    print(f"Loaded and chunked text: {len(text_docs)} chunks")

    # Load table previews
    table_docs = load_table_documents(TABLE_DIR)
    print(f"Loaded tables: {len(table_docs)} items")

    # Combine and embed
    all_docs = text_docs + table_docs
    docs, embeddings = build_embeddings(all_docs)
    print(f"Total documents to embed: {len(docs)}")
    print(f"Embeddings shape: {embeddings.shape}")

    # Show first few sources
    print("First 5 document sources:")
    for doc in docs[:5]:
        print(f" - {doc['source']} ({doc['type']})")

    # Now you can pass `embeddings` and `docs` into your FAISS index builder, e.g.:
    # ... (faiss.IndexFlatIP build step) ...

if __name__ == "__main__":
    main()