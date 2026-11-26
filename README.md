# 📄 LangGraph PDF RAG Assistant

A local AI-powered PDF question-answering application built using **Python, LangGraph, LangChain, FastAPI, Streamlit, FAISS, and LM Studio**.

The application allows users to upload PDF documents, extract and index their content, and ask questions based on the uploaded documents.

It uses Retrieval-Augmented Generation (RAG) to retrieve relevant document content and generate context-aware answers using a locally hosted Large Language Model through LM Studio.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Technology Stack](#-technology-stack)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [LM Studio Configuration](#-lm-studio-configuration)
- [Running the Application](#-running-the-application)
- [API Documentation](#-api-documentation)
- [How the RAG Workflow Works](#-how-the-rag-workflow-works)
- [Vector Database](#-vector-database)
- [Troubleshooting](#-troubleshooting)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 🎯 Project Overview

The LangGraph PDF RAG Assistant is designed to make information retrieval from PDF documents easier through natural-language questions.

Instead of manually searching through lengthy documents, users can upload PDFs and ask questions about their contents.

The application follows a Retrieval-Augmented Generation architecture:

1. Extract text from uploaded PDF documents.
2. Split extracted text into smaller chunks.
3. Generate embeddings using an LM Studio embedding model.
4. Store document vectors in a FAISS index.
5. Retrieve relevant chunks based on user questions.
6. Process retrieval and answer generation through LangGraph.
7. Generate answers using a locally hosted chat model.
8. Return answers and available source references.

The application supports local model inference through LM Studio.

## ✨ Key Features

- **PDF Upload:** Upload PDF documents through a Streamlit interface.
- **Document Processing:** Extract text and split documents into smaller chunks.
- **Vector Embeddings:** Generate document embeddings using LM Studio.
- **Semantic Search:** Retrieve relevant document chunks using FAISS.
- **Local LLM Integration:** Generate answers using locally hosted language models.
- **LangGraph Workflow:** Organize retrieval and answer generation into graph nodes.
- **REST API:** Provide backend functionality through FastAPI.
- **Source References:** Return source information for retrieved document chunks.
- **Persistent Vector Index:** Save the FAISS index locally for reuse.
- **Interactive Interface:** Ask questions about uploaded documents through Streamlit.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application development |
| Streamlit | Frontend user interface |
| FastAPI | Backend REST API |
| LangChain | LLM integration and document processing |
| LangGraph | Workflow orchestration |
| FAISS | Vector storage and similarity search |
| LM Studio | Local LLM and embedding inference |
| PyPDF | PDF text extraction |
| Uvicorn | ASGI application server |
| python-dotenv | Environment configuration |
| HTTPX | HTTP communication |

## 🏗️ System Architecture

The application follows a modular architecture with separate frontend, backend, retrieval, and generation responsibilities.

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
           Embedding Generation
                    |
                    v
              FAISS Index
                    |
                    |
              User Question
                    |
                    v
            LangGraph Workflow
                    |
              +-----+-----+
              |           |
              v           v
          Retrieval    Context
             Node      Preparation
              |           |
              +-----+-----+
                    |
                    v
             Local Chat Model
                LM Studio
                    |
                    v
             Generated Answer
                    |
                    v
            Source References
                    |
                    v
           Streamlit Frontend
```

### Component Responsibilities

**1. Streamlit Frontend**

- Provides the user interface.
- Allows PDF uploads.
- Accepts natural-language questions.
- Displays generated answers and source information.

**2. FastAPI Backend**

- Handles frontend API requests.
- Receives uploaded PDF documents.
- Coordinates document ingestion and retrieval.
- Connects application components.

**3. LangChain**

- Handles document processing.
- Integrates language models and embedding models.
- Supports text splitting and retrieval operations.

**4. LangGraph**

- Organizes the RAG workflow.
- Coordinates retrieval and answer-generation nodes.
- Manages the flow of information between workflow components.

**5. FAISS**

- Stores document embeddings.
- Performs similarity-based retrieval.
- Provides relevant document chunks for answering questions.

**6. LM Studio**

- Hosts the local chat model.
- Provides embedding generation.
- Exposes an OpenAI-compatible local API.

## 📁 Project Structure

```text
langgraph_pdf_rag/
│
├── main.py                  # Streamlit frontend
├── api.py                   # FastAPI backend and RAG workflow
│
├── .env                     # Local configuration (not committed)
├── .env.example             # Configuration template
├── .gitignore
├── requirements.txt
├── README.md
│
└── faiss_index/             # Persisted vector index (generated locally)
```

*Note: This is the project structure described in the original documentation. Adjust filenames and directories to match your actual repository.*

## ⚙️ Prerequisites

Before running the application, ensure you have:

- Python 3.12
- LM Studio installed
- A compatible chat model
- A compatible embedding model
- Git (for cloning the repository)

Both the chat and embedding models should be available in LM Studio.

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Navigate into the project directory:

```bash
cd langgraph_pdf_rag
```

### 2. Create a Virtual Environment

Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

If the requirements file is not available, install the main dependencies:

```bash
pip install fastapi "uvicorn[standard]" streamlit \
langchain langchain-community langchain-openai \
langchain-text-splitters langgraph faiss-cpu \
pypdf python-dotenv python-multipart httpx
```

## 🖥️ LM Studio Configuration

LM Studio provides local model inference through an OpenAI-compatible API.

### Configuration Steps

1. Open LM Studio.
2. Download or select a compatible chat model.
3. Download or select an embedding model.
4. Load the required models.
5. Start the local API server.
6. Note the server address, port and model identifiers.

### Environment Variables

Create a `.env` file in the project root.

Use the following template:

```dotenv
# FastAPI backend
RAG_API_URL=http://127.0.0.1:8000

# LM Studio API
LM_STUDIO_BASE_URL=http://127.0.0.1:1234/v1
LM_STUDIO_API_KEY=lm-studio

# Model configuration
LM_STUDIO_CHAT_MODEL=your-chat-model
LM_STUDIO_EMBEDDING_MODEL=your-embedding-model

# FAISS index
FAISS_INDEX_DIR=faiss_index
```

Replace the model identifiers with the actual model names configured in LM Studio.

If LM Studio runs on another machine, replace `127.0.0.1` with its reachable address and configure network access appropriately.

**Security:** Never commit your actual `.env` file, credentials or private configuration to GitHub. Publish `.env.example` with placeholder values instead.

## ▶️ Running the Application

The backend and frontend should run in separate terminal windows.

### 1. Start FastAPI Backend

Open the first terminal:

```bash
source .venv/bin/activate
```

Run:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

The API should be accessible at:

- API: http://127.0.0.1:8000
- Swagger Documentation: http://127.0.0.1:8000/docs

### 2. Start Streamlit Frontend

Open a second terminal:

```bash
source .venv/bin/activate
```

Run:

```bash
streamlit run main.py
```

Open:

http://localhost:8501

For Windows, activate the virtual environment using:

```bash
.venv\Scripts\activate
```

## 🔌 API Documentation

The FastAPI backend provides endpoints for document ingestion and question answering.

### 1. Document Ingestion

**Endpoint:** `POST /ingest`

Purpose:

- Upload PDF documents.
- Extract PDF text.
- Split documents into chunks.
- Generate embeddings.
- Store vectors in FAISS.

Example request:

```bash
curl -X POST "http://127.0.0.1:8000/ingest" \
-F "files=@sample.pdf"
```

The multipart field name must match the upload parameter defined in your FastAPI endpoint.

### 2. Question Answering

**Endpoint:** `POST /retrieve`

Purpose:

- Accept a user question.
- Retrieve relevant document chunks.
- Execute the LangGraph workflow.
- Generate an answer using the local LLM.
- Return available source references.

Example request:

```bash
curl -X POST "http://127.0.0.1:8000/retrieve" \
-H "Content-Type: application/json" \
-d '{"question":"Summarize the uploaded document."}'
```

*The endpoint paths, HTTP methods, request field names and response schemas above should be checked against the actual FastAPI implementation.*

## 🔄 How the RAG Workflow Works

The application follows two major stages.

### Stage 1: Document Ingestion

**Step 1 – PDF Upload**

The user uploads a PDF through the Streamlit application.

**Step 2 – Text Extraction**

The backend extracts readable text from the PDF.

**Step 3 – Text Chunking**

The extracted text is divided into smaller overlapping chunks.

**Step 4 – Embedding Generation**

The embedding model converts text chunks into numerical vector representations.

**Step 5 – Vector Storage**

FAISS stores the generated vectors for similarity-based retrieval.

### Stage 2: Question Answering

**Step 1 – User Question**

The user enters a natural-language question.

**Step 2 – Retrieval**

FAISS searches for document chunks relevant to the question.

**Step 3 – Context Preparation**

Retrieved chunks are prepared as context for the language model.

**Step 4 – Answer Generation**

LangGraph coordinates the workflow, and the local chat model generates a response based on the retrieved context.

**Step 5 – Response**

The application returns the generated answer and available source references.

## 🗄️ Vector Database

The project uses FAISS for vector storage and similarity search.

FAISS helps the application retrieve document chunks based on semantic similarity rather than relying only on exact keyword matching.

The index is stored locally in:

```text
faiss_index/
```

The directory can be reused when a compatible persisted index is available.

If no saved index exists, upload and ingest a PDF before asking questions.

## 🔍 Troubleshooting

### 1. LM Studio Connection Refused

Possible causes:

- LM Studio server is not running.
- Incorrect IP address or port.
- Incorrect API base URL.
- Network or firewall restrictions.

Check the server configuration and confirm that the API endpoint is reachable.

### 2. HTTP 400 During Embedding Generation

Check:

- Whether the embedding model is loaded.
- Whether the configured model identifier is correct.
- Whether the embeddings endpoint accepts the request format.
- Whether the installed LangChain integration is compatible with the local server.

If your integration sends token arrays instead of text strings, check the compatibility of its embedding context-length handling options with your installed package version.

### 3. HTTP 422 During PDF Upload

Check that:

- The request uses `multipart/form-data`.
- The upload field matches the FastAPI parameter.
- The uploaded file is a supported PDF.

For example:

```python
files: List[UploadFile] = File(...)
```

The corresponding multipart field should be named `files`.

### 4. FAISS Index Not Found

If the index does not exist:

1. Start the FastAPI backend.
2. Upload a PDF.
3. Run the document ingestion process.
4. Confirm that the embedding generation succeeds.
5. Verify that the FAISS index is saved successfully.
6. Submit a question after ingestion.

### 5. FastAPI Application Import Error

Check:

- The correct project directory.
- The existence of `api.py`.
- Whether required dependencies are installed.
- Whether the module contains import-time errors.

Run:

```bash
python -c "import api; print('API module loaded successfully')"
```

## 🧪 Testing

Automated tests should be added for the following areas:

- PDF text extraction.
- Document chunking.
- Embedding generation.
- FAISS indexing and retrieval.
- API ingestion endpoint.
- API retrieval endpoint.
- LangGraph workflow.
- Application error handling.

If Pytest is configured in the project, run:

```bash
python -m pytest -v
```

*This command is provided as a testing instruction; actual test coverage and results should be documented after running the tests.*

## 🔮 Future Improvements

Potential improvements include:

- Support multiple PDF documents.
- Improve document metadata management.
- Add document deletion functionality.
- Add conversation history.
- Improve source citation display.
- Add configurable chunk size and overlap.
- Add support for multiple embedding models.
- Improve retrieval accuracy using reranking.
- Add authentication and API security.
- Add automated integration testing.
- Add Docker support.
- Add logging and monitoring.
- Improve error handling.
- Deploy the application to a suitable hosting environment.

## ⚠️ Limitations

- Answer quality depends on the quality of uploaded documents and the selected language model.
- Scanned PDFs may require OCR support.
- Local inference performance depends on available hardware and model size.
- Retrieval quality depends on document chunking and embedding configuration.
- The application requires a functioning local LM Studio server.

## 👨‍💻 Author

**Om Suryakant Ugale**

Python Developer | Backend Development | Automation

- GitHub: [Omugale09](https://github.com/Omugale09)
- LinkedIn: [Om Ugale](https://www.linkedin.com/in/om-ugale-235243267/)

---

⭐ If you find this project interesting, feel free to explore the repository and learn more about local RAG application development.
