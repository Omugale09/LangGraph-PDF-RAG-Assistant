# LangGraph PDF RAG Assistant

A local AI-powered PDF question-answering application built with **LangGraph, LangChain, FastAPI, Streamlit, FAISS, and LM Studio**.

The application allows users to upload PDF documents, extract and index their content, and ask questions based on the uploaded documents. It uses a locally hosted language model through LM Studio to generate answers from retrieved document context.

## Features

- **PDF Upload:** Upload PDF documents through a Streamlit interface.
- **Document Processing:** Extract text from PDFs and split it into smaller chunks.
- **Vector Search:** Store and retrieve document embeddings using FAISS.
- **Local LLM Integration:** Use LM Studio to run a locally hosted chat model.
- **LangGraph Workflow:** Organize document retrieval and answer generation into graph nodes.
- **REST API:** Use FastAPI endpoints for document ingestion and question answering.
- **Source References:** Return source information for retrieved document chunks.
- **Persistent Index:** Save the FAISS index locally for reuse.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application development |
| Streamlit | Frontend user interface |
| FastAPI | Backend REST API |
| LangChain | LLM, embeddings, and document processing |
| LangGraph | Retrieval and answer-generation workflow |
| FAISS | Vector storage and similarity search |
| LM Studio | Local model inference |
| PyPDF | PDF text extraction |
| Uvicorn | ASGI server |
| python-dotenv | Environment configuration |

## Project Architecture

```text
PDF Documents
     |
     v
Streamlit Frontend
     |
     v
FastAPI Backend
     |
     v
PDF Text Extraction
     |
     v
Text Chunking
     |
     v
LM Studio Embedding Model
     |
     v
FAISS Vector Index
     |
     v
User Question
     |
     v
LangGraph Workflow
     |
     +---- Retrieve relevant chunks from FAISS
     |
     +---- Generate answer using local LLM
     |
     v
Answer and Source References
     |
     v
Streamlit Frontend
```

## Project Structure

```text
langgraph_pdf_rag/
│
├── .venv/                    # Python virtual environment
├── main.py                   # Streamlit frontend
├── api.py                    # FastAPI backend and LangGraph workflow
├── .env                      # Local environment configuration
├── .gitignore                # Files excluded from Git
├── requirements.txt          # Python dependencies
│
└── faiss_index/              # Persisted FAISS vector index
```

The structure above assumes your frontend file is named `main.py`. Adjust the filename if your project uses a different name.

## Prerequisites

Install or configure the following before running the application:

- Python 3.12
- LM Studio
- A compatible chat model
- A compatible embedding model
- Git (optional, for version control)

Make sure the selected models are available in LM Studio and that its API server is running.

## Installation

### 1. Clone the repository

If your project is hosted on GitHub:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd langgraph_pdf_rag
```

Alternatively, navigate to your existing local project:

```bash
cd ~/Downloads/langgraph_pdf_rag
```

### 2. Create a virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

### 3. Install dependencies

If `requirements.txt` is available:

```bash
pip install -r requirements.txt
```

Otherwise, install the main dependencies:

```bash
pip install fastapi "uvicorn[standard]" streamlit \
  langchain langchain-community langchain-openai \
  langchain-text-splitters langgraph faiss-cpu \
  pypdf python-dotenv python-multipart httpx
```

## LM Studio Configuration

1. Open LM Studio.
2. Download or select a compatible chat model and embedding model.
3. Load the models as required.
4. Start the local server from LM Studio's Developer/Local Server interface.
5. Confirm the server address, port, and model identifiers.

The example below uses the configured server address for this project. Replace it if your LM Studio server uses another IP address.

Create a `.env` file in the project root:

```dotenv
# FastAPI backend
RAG_API_URL=http://127.0.0.1:8000

# LM Studio OpenAI-compatible API
LM_STUDIO_BASE_URL=http://192.168.10.3:1234/v1
LM_STUDIO_API_KEY=lm-studio

# Model identifiers must match the models available in LM Studio
LM_STUDIO_CHAT_MODEL=qwen2.5-coder-14b
LM_STUDIO_EMBEDDING_MODEL=text-embedding-nomic-embed-text-v1.5

# Local vector index
FAISS_INDEX_DIR=faiss_index
```

**Important:** Verify the LM Studio IP address and exact model identifiers. If the server runs on another computer, enable network access as appropriate and ensure port `1234` is reachable. The API key value shown above is a local placeholder; use the authentication configuration required by your server.

## Running the Application

Run the backend and frontend in separate terminal windows.

### 1. Start the FastAPI backend

Open the first terminal:

```bash
cd ~/Downloads/langgraph_pdf_rag
source .venv/bin/activate
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

The backend should be available at:

- API base URL: `http://127.0.0.1:8000`
- Interactive API documentation: `http://127.0.0.1:8000/docs`

### 2. Start the Streamlit frontend

Open a second terminal:

```bash
cd ~/Downloads/langgraph_pdf_rag
source .venv/bin/activate
streamlit run main.py
```

Open the local URL displayed in the terminal, usually:

`http://localhost:8501`

## API Endpoints

### POST `/ingest`

Uploads PDF documents, extracts text, splits it into chunks, creates embeddings, and stores them in the FAISS index.

Example using `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/ingest" \
  -F "files=@sample.pdf"
```

Replace `sample.pdf` with the path to your PDF. The multipart field name must match the backend parameter, expected here to be `files`.

### POST `/retrieve`

Submits a question to the backend for retrieval and answer generation.

Example request:

```bash
curl -X POST "http://127.0.0.1:8000/retrieve" \
  -H "Content-Type: application/json" \
  -d '{"question":"Summarize the uploaded document."}'
```

The request body shown assumes your `/retrieve` endpoint accepts a JSON field named `question`. Check the API documentation if your implementation uses a different request schema.

## How the RAG Workflow Works

1. **Ingestion:** The user uploads one or more PDFs.
2. **Extraction:** PDF loaders extract readable text from each document.
3. **Chunking:** The text splitter creates smaller overlapping chunks.
4. **Embedding:** LM Studio converts text chunks into vectors.
5. **Indexing:** FAISS stores the vectors for similarity search.
6. **Retrieval:** The LangGraph retrieval node searches for relevant chunks.
7. **Generation:** The generation node sends the retrieved context and question to the local chat model.
8. **Response:** The backend returns the generated answer and available source references.

## Troubleshooting

### 1. Connection refused by LM Studio

Check that the LM Studio server is running and that `LM_STUDIO_BASE_URL` contains the correct host, port, and `/v1` path.

For a remote LM Studio server, verify network connectivity and firewall settings.

### 2. HTTP 400 during embedding generation

Check the embedding model identifier and the request format accepted by the LM Studio server. Confirm that the embedding model is loaded and supports the embeddings endpoint.

If your LangChain integration sends token arrays instead of text strings, verify whether the installed `langchain-openai` version supports:

```python
check_embedding_ctx_length=False
```

Use this setting in `OpenAIEmbeddings` only if it is compatible with your installed package version and server behavior.

### 3. HTTP 422 during PDF upload

Check that the request uses `multipart/form-data` and that the upload field matches the FastAPI parameter:

```python
files: List[UploadFile] = File(...)
```

For this signature, the field name is `files`.

### 4. FAISS index not found

If no saved index exists, ingest a PDF before submitting a question. Confirm