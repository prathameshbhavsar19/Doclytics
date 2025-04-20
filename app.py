import os

import json

import streamlit as st

import openai

from vector_index import build_index, retrieve

from openai import AzureOpenAI

import matplotlib.pyplot as plt

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
 
# Initialize Azure OpenAI client

client = AzureOpenAI(

    api_key = decode_base64("Nk1UOUlWWUJVQkhNVEUzbUFrYklXSU5DbEhiWTlJVk1oalhQeERiVE5vanBmNGI1RTJhWUpRUUo5OUJBQUNIWUh2NlhKM3czQUFBQkFDT0dNenBV"),
    azure_endpoint = "https://virat-kumar-openai.openai.azure.com/",
    api_version    = "2024-02-15-preview"

)

# ─── Configuration ────────────────────────────────────────────────────────────

OPENAI_API_KEY = decode_base64("c2stcHJvai1uOUtUQnJJSFBBcm5TUjFLbmxhTnZUdlVqRFQyVXBWVU5QRTZacmlhRVlmVTFSOG00SkpBTWg4bUFwWFM2aHlHenI3MC16eUtzZVQzQmxia0ZKdEVIaDljaFVJZ0NWVm5hR25RalNXX2Y5M1lGbTZMNzZLVWVJNXhRVTZKNWZ6dGJzaVJCXzNyUzR1dUdRcWZBQkJFNnNPUDJVa0E")

LLM_MODEL      = "gpt-4o-mini"

TOP_K          = 3

# ──────────────────────────────────────────────────────────────────────────────
 
# Ensure the OpenAI API key is set

if not OPENAI_API_KEY:

    st.error("Please set the OPENAI_API_KEY environment variable")

    st.stop()
 
# Assign the key for SDK usage

openai.api_key = OPENAI_API_KEY
 
# System prompt instructing when to output chart data

SYSTEM_PROMPT = """

You are an expert analyst answering questions about the LTIMindtree Annual Report.

Use only the provided snippets. For each fact, append its source in square brackets,

e.g. [page_12_chunk_1] or [table_3.csv]. Do not invent citations.
 
**If your answer can be better visualized as a chart, append at the very end a

JSON blob under the heading ChartData:. For example:**

ChartData:

{"type": "bar", "labels": ["Region A","Region B"], "values": [120, 95]}
 
Otherwise, do not include any ChartData section.

""".strip()
 
@st.cache_data(show_spinner=False)

def initialize_index():

    """Load and cache the FAISS/vector index."""

    return build_index()
 
 
def generate_answer(query: str, docs, index):

    """Retrieve context, call the LLM, and parse any ChartData."""

    hits = retrieve(query, docs, index, k=TOP_K)

    context = "\n\n".join(f"[{h['source']}]\n{h['content']}" for h in hits)
 
    messages = [

        {"role": "system", "content": SYSTEM_PROMPT},

        {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {query}"}

    ]
 
    response = client.chat.completions.create(

        model="gpt-4o",

        messages=messages

    )
 
    full_text = response.choices[0].message.content.strip()
 
    # Extract any ChartData JSON

    chart_data = None

    if "ChartData:" in full_text:

        answer_text, json_blob = full_text.split("ChartData:", 1)

        answer = answer_text.strip()

        try:

            chart_data = json.loads(json_blob.strip())

        except json.JSONDecodeError:

            chart_data = None

    else:

        answer = full_text
 
    return answer, hits, chart_data
 
 
def main():

    st.set_page_config(page_title="Annual Report Q&A", layout="wide")

    st.title("LTIMindtree Annual Report Q&A")
 
    docs, index = initialize_index()

    query = st.text_input("Ask a question about the Annual Report")

    if st.button("Get Answer") and query:

        with st.spinner("Retrieving answer..."):

            answer, hits, chart_data = generate_answer(query, docs, index)
 
        # Display the textual answer

        st.subheader("Answer")

        st.write(answer)
 
        # List out the sources

        st.subheader("Sources")

        for h in hits:

            st.write(f"- **{h['source']}** (score: {h['score']:.3f})")
 
        # If chart data was returned, render it

        if chart_data:

            fig, ax = plt.subplots()

            typ = chart_data.get("type", "bar")

            labels = chart_data.get("labels", [])

            values = chart_data.get("values", [])
 
            if typ == "bar":

                ax.bar(labels, values)

            elif typ == "line":

                ax.plot(labels, values, marker='o')

            else:

                ax.bar(labels, values)
 
            ax.set_title(chart_data.get("title", "Comparison Chart"))

            ax.set_xlabel(chart_data.get("xlabel", ""))

            ax.set_ylabel(chart_data.get("ylabel", ""))
 
            st.pyplot(fig)
 
 
if __name__ == "__main__":

    main()

 