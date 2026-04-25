# 🩺 Medical RAG Chatbot

An AI-powered medical assistant chatbot that answers user questions using retrieval-augmented generation (RAG) over medical PDF knowledge sources.  
This project uses LangChain, Pinecone vector search, Ollama-based embeddings/LLM, and a Flask chat UI.

## ✨ Features

### 🎯 Core Functionality
- **Medical Question Answering**: Answers health-related questions using indexed medical documents.
- **RAG Pipeline**: Retrieves relevant context before generation for grounded responses.
- **Prompt-Guided Responses**: Uses a focused system prompt to keep answers concise and structured.
- **Simple Chat API**: Flask endpoint handles user queries and returns model responses.

### 🧠 AI / NLP Pipeline
- **PDF Ingestion**: Loads medical PDFs from the `data/` directory.
- **Text Chunking**: Uses recursive character splitting for efficient retrieval.
- **Embeddings**: Uses Ollama embeddings (`nomic-embed-text-v2-moe:latest`).
- **Vector Store**: Uses Pinecone index (`medical-chatbot`) for semantic search.
- **LLM**: Uses local Ollama chat model (`medgemma1.5:4b-it-bf16`).

### 🎨 User Interface
- **Flask Web App** with chat-style interface
- **Responsive frontend** with modern layout
- **Typing indicator** for better UX
- **Clear chat button** to reset the current conversation view

---

## 🏗️ Project Structure

```text
End to End Medical Chatbot/
├── app.py                  # Flask app entry and RAG orchestration
├── store_index.py          # PDF processing + Pinecone index population
├── requirements.txt
├── pyproject.toml
├── setup.py
├── src/
│   ├── helper.py           # PDF loading, chunking, embeddings
│   └── prompt.py           # System prompt for answer behavior
├── templates/
│   └── chat.html           # Chat UI template
├── static/
│   └── style.css           # Frontend styles
└── data/                   # Put medical PDF files here
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- `pip` or `uv`
- [Ollama](https://ollama.com/) installed and running locally
- Pinecone account + API key

### 1) Clone the repository

```bash
git clone https://github.com/deep1305/End-to-End-Medical-Chatbot.git
cd "End to End Medical Chatbot"
```

### 2) Install dependencies

Using pip:

```bash
pip install -r requirements.txt
```

Using uv (optional):

```bash
uv sync
```

### 3) Configure environment

Create a `.env` file in project root:

```env
PINECONE_API_KEY=your_pinecone_api_key
```

### 4) Build/update vector index from PDFs

Add your PDFs to `data/`, then run:

```bash
python store_index.py
```

### 5) Run the chatbot app

```bash
python app.py
```

Open: [http://localhost:8080](http://localhost:8080)

---

## 🔧 Configuration Notes

Current defaults in code:

- **Pinecone index name**: `medical-chatbot` (see `store_index.py` and `app.py`)
- **Retriever top-k**: `3` (see `app.py`)
- **Embedding model**: `nomic-embed-text-v2-moe:latest` (see `src/helper.py`)
- **Chat model**: `medgemma1.5:4b-it-bf16` (see `app.py`)
- **Chunking**: `chunk_size=500`, `chunk_overlap=20` (see `src/helper.py`)

---

## 🧪 Example Queries

- "What are common symptoms of hypertension?"
- "Explain type 2 diabetes in simple terms."
- "What are treatment options for migraine?"
- "What are warning signs of dehydration?"
- "How is anemia diagnosed?"

---

## 🔐 Disclaimer

This project is for educational and informational purposes only.  
It is **not** a substitute for professional medical advice, diagnosis, or treatment.

---

## 🙌 Acknowledgments

- [LangChain](https://www.langchain.com/)
- [Pinecone](https://www.pinecone.io/)
- [Ollama](https://ollama.com/)
- [Flask](https://flask.palletsprojects.com/)
