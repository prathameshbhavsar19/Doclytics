#!/usr/bin/env python3
import os
import openai
from openai import RateLimitError
from vector_index import build_index, retrieve
import base64

def decode_base64(b64_data: str, output_path: str = None) -> str:
    """
    Decodes a base64-encoded string.

    If `output_path` is provided, saves the decoded binary data to a file.
    Otherwise, returns the decoded UTF-8 string.

    Parameters:
    - b64_data (str): Base64-encoded content.
    - output_path (str, optional): Path to save the decoded file (binary).

    Returns:
    - Decoded string (if output_path is None), else returns confirmation message.
    """
    decoded_bytes = base64.b64decode(b64_data)

    if output_path:
        with open(output_path, "wb") as f:
            f.write(decoded_bytes)
        return f"File saved to: {output_path}"
    else:
        return decoded_bytes.decode("utf-8")
# ─── Configuration ────────────────────────────────────────────────────────────
OPENAI_API_KEY = decode_base64("c2stcHJvai1uOUtUQnJJSFBBcm5TUjFLbmxhTnZUdlVqRFQyVXBWVU5QRTZacmlhRVlmVTFSOG00SkpBTWg4bUFwWFM2aHlHenI3MC16eUtzZVQzQmxia0ZKdEVIaDljaFVJZ0NWVm5hR25RalNXX2Y5M1lGbTZMNzZLVWVJNXhRVTZKNWZ6dGJzaVJCXzNyUzR1dUdRcWZBQkJFNnNPUDJVa0E")
LLM_MODEL      = "gpt-4o-mini"
TOP_K          = 5
# ──────────────────────────────────────────────────────────────────────────────

if not OPENAI_API_KEY:
    raise RuntimeError("Please set the OPENAI_API_KEY environment variable")

openai.api_key = OPENAI_API_KEY

SYSTEM_PROMPT = """
You are an expert analyst answering questions about the LTIMindtree Annual Report.
Use only the provided snippets. For each fact, append its source in square brackets,
e.g. [page_12_chunk_1] or [table_3.csv]. Do not invent citations.
""".strip()

def answer_question(query: str, docs: list[dict], index) -> str:
    """
    Retrieve the top-K relevant chunks and generate a cited answer via OpenAI.
    If quota is exceeded, returns a friendly error message.
    """
    hits = retrieve(query, docs, index, k=TOP_K)
    context = "\n\n".join(f"[{h['source']}]\n{h['content']}" for h in hits)

    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]

    try:
        response = openai.chat.completions.create(
            model=LLM_MODEL,
            temperature=0.2,
            messages=messages,
        )
        return response.choices[0].message.content.strip()
    except RateLimitError:
        return (
            "⚠️ OpenAI API rate limit or quota exceeded. "
            "Please check your plan and billing, then try again."
        )

def main():
    docs, index = build_index()
    print(f"Index ready with {len(docs)} documents.\n")

    while True:
        query = input("Enter your question (or type 'exit' to quit): ").strip()
        if not query or query.lower() == "exit":
            break

        print("\nGenerating answer...\n")
        answer = answer_question(query, docs, index)
        print(answer)
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()