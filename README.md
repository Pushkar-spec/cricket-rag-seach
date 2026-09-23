# Cricket-RAG-Seach

A beginner RAG (Retrieval-Augmented Generation) project built with **LangChain** — ask a question in the terminal, and it finds relevant passages from a cricket document using semantic search, then uses an LLM to generate a direct answer grounded in that context.

## How it works

1. **Load & split** — a `.txt` document is loaded and chunked (`TextLoader` + `RecursiveCharacterTextSplitter`)
2. **Embed & store** — chunks are embedded (`sentence-transformers/all-MiniLM-L6-v2`) and stored in **FAISS**
3. **Retrieve** — user's query is embedded and matched against stored chunks via similarity search
4. **Generate** — retrieved chunks + query are passed to an LLM (`Llama-3.1-8B-Instruct` via Hugging Face) to produce a natural-language answer

## Example

```
Enter your query (or type 'exit' to quit): Who is known as the God of Cricket?

Answer: Sachin Tendulkar is known as the "God of Cricket."
```

## Tech stack

LangChain · Hugging Face (embeddings + hosted LLM) · FAISS · Python

## Setup

```bash
git clone <your-repo-url>
cd <repo-folder>
python -m venv venv
venv\Scripts\activate        # Windows
pip install langchain langchain-community langchain-huggingface langchain-text-splitters faiss-cpu sentence-transformers torch python-dotenv
```

Add a `.env` file with your Hugging Face token:
```
HUGGINGFACEHUB_API_TOKEN=hf_your_token_here
```

Run:
```bash
python EmbeddedModels/document_similarity.py
```

## What I learned

Document chunking, embeddings, vector similarity search, and the RAG pattern — grounding an LLM's answers in retrieved context instead of its own training data.

---

Built as a hands-on learning project while studying LangChain and RAG fundamentals.
