# Doclytics : LTIMindtree Annual Report Q&A Platform

This project is a Retrieval-Augmented Generation (RAG)-based question-answering system designed for analyzing corporate annual reports such as those from LTIMindtree. It allows users to upload PDF reports, extract structured and unstructured data, build searchable vector indexes using embeddings, and interact with the data using natural language questions powered by an LLM.

⸻

Overview

The solution consists of two main processes:

![image](https://github.com/user-attachments/assets/5d7ba574-6082-453d-a572-5d6fc90d376c)


Process 1: Data Upload and Preprocessing
	•	Admin uploads one or more annual report PDFs.
	•	The system extracts:
	•	Unstructured Data: Free-form text using PyMuPDF.
	•	Structured Data: Tabular content using tabula-py.
	•	Extracted data is saved in the extracted/ directory for permanent storage.

Process 2: RAG-Based Question Answering
	•	Both text chunks and table previews are embedded using a SentenceTransformer model.
	•	FAISS is used to build a high-dimensional vector index over the document embeddings.
	•	At runtime:
	•	The user submits a question.
	•	Top-K similar chunks are retrieved.
	•	These chunks, along with the query, are sent to Azure OpenAI (gpt-4o).
	•	A cited answer is returned, optionally with chart data.
	•	If ChartData is detected, a chart is rendered using matplotlib.

⸻

Features
	•	Upload and parse PDF reports (text and tables).
	•	Extract structured tables into .csv files.
	•	Embed document content and build a FAISS vector space.
	•	Query the system with natural language questions.
	•	Streamlit frontend for interactive use.
	•	LLM-powered answers with source citation.
	•	Dynamic chart rendering based on model response.

⸻

Project Structure

project-root/
├── app.py                      # Streamlit app interface
├── vector_index.py             # Embedding and FAISS vector search logic
├── chunkAndLoad.py             # Chunking and preprocessing utilities
├── extract_pdf.py              # PDF parsing (text + tables)
├── requirements.txt            # Dependency list
├── extracted/
│   ├── all_text.txt            # Combined unstructured text
│   └── tables/
│       └── *.csv               # Extracted structured tables

Requirements

Make sure you have:
	•	Python 3.10 or newer
	•	Java Runtime Environment (JRE) for tabula-py
	•	Azure OpenAI deployment for gpt-4o

⸻

Sample Prompt with Chart

When appropriate, the LLM can return a chart directive like:

ChartData:
{"type": "bar", "labels": ["FY23", "FY24"], "values": [160, 200]}

Credits
	•	Built with FAISS, PyMuPDF, tabula-py, SentenceTransformers, Azure OpenAI, and Streamlit.
	•	Inspired by modern Retrieval-Augmented Generation pipelines.






