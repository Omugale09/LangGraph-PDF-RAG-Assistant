
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=True)

# LM Studio OpenAI-compatible API.
# This is the model server URL, NOT the FastAPI URL.
LM_STUDIO_BASE_URL = os.getenv(
    "LM_STUDIO_BASE_URL",
    "http://192.168.10.3:1234/v1",
).strip().rstrip("/")

LM_STUDIO_API_KEY = os.getenv(
    "LM_STUDIO_API_KEY",
    "lm-studio",
).strip()

CHAT_MODEL = os.getenv(
    "LM_STUDIO_CHAT_MODEL",
    "qwen2.5-coder-14b",
).strip()

EMBEDDING_MODEL = os.getenv(
    "LM_STUDIO_EMBEDDING_MODEL",
    "text-embedding-nomic-embed-text-v1.5",
).strip()

INDEX_DIR = Path(
    os.getenv("FAISS_INDEX_DIR", "faiss_index")
)

if not INDEX_DIR.is_absolute():
    INDEX_DIR = BASE_DIR / INDEX_DIR

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
RETRIEVAL_K = 4

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LangGraph PDF RAG API - LM Studio",
    description=(
        "Upload PDFs, create a FAISS index, "
        "and ask questions using LangGraph and LM Studio."
    ),
    version="2.1.0",
)

# The in-memory vector store used by the API.
vectorstore = None


# ============================================================
# 2. LM STUDIO CLIENTS
# ============================================================


def get_embeddings() -> OpenAIEmbeddings:
    """Create an LM Studio-compatible embedding client."""

    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=LM_STUDIO_BASE_URL,
        api_key=LM_STUDIO_API_KEY,
        check_embedding_ctx_length=False,
        max_retries=1,
        request_timeout=120,
    )



def get_llm() -> ChatOpenAI:
    """Create an LM Studio-compatible chat client."""

    return ChatOpenAI(
        model=CHAT_MODEL,
        base_url=LM_STUDIO_BASE_URL,
        api_key=LM_STUDIO_API_KEY,
        temperature=0,
        timeout=120,
        max_retries=1,
    )


# ============================================================
# 3. LOAD SAVED FAISS INDEX
# ============================================================

def load_saved_index() -> None:
    """Load an existing FAISS index, if available."""

    global vectorstore

    index_file = INDEX_DIR / "index.faiss"
    metadata_file = INDEX_DIR / "index.pkl"

    if not index_file.exists() or not metadata_file.exists():
        logger.info("No saved FAISS index found at %s", INDEX_DIR)
        return

    try:
        vectorstore = FAISS.load_local(
            str(INDEX_DIR),
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )

        logger.info("Saved FAISS index loaded successfully.")

    except Exception:
        vectorstore = None
        logger.exception(
            "Could not load FAISS index. "
            "Check the embedding model and index compatibility."
        )


@app.on_event("startup")
def startup() -> None:
    load_saved_index()


# ============================================================
# 4. REQUEST MODELS
# ============================================================

class ChatTurn(BaseModel):
    role: str
    content: str


class RetrieveRequest(BaseModel):
    question: str = Field(min_length=1)
    chat_history: List[ChatTurn] = Field(default_factory=list)


class RAGState(TypedDict, total=False):
    question: str
    chat_history: List[Dict[str, str]]
    documents: List[Document]
    context: str
    answer: str
    sources: List[Dict[str, Any]]


# ============================================================
# 5. HEALTH AND STATUS ENDPOINTS
# ============================================================

@app.get("/")
def home() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "LangGraph PDF RAG API",
        "index_loaded": vectorstore is not None,
        "lm_studio_url": LM_STUDIO_BASE_URL,
        "chat_model": CHAT_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "index_directory": str(INDEX_DIR),
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "index_loaded": vectorstore is not None,
        "lm_studio_configured": bool(LM_STUDIO_BASE_URL),
        "chat_model": CHAT_MODEL,
        "embedding_model": EMBEDDING_MODEL,
    }


# ============================================================
# 6. INGEST PDF FILES
# ============================================================

@app.post("/ingest")
async def ingest(
    files: List[UploadFile] = File(...)
) -> Dict[str, Any]:
    """
    Accept one or more PDFs through multipart/form-data.
    Extract text, split it into chunks, generate embeddings,
    and save the resulting FAISS index.
    """

    global vectorstore

    if not files:
        raise HTTPException(
            status_code=400,
            detail="Upload at least one PDF using the 'files' field.",
        )

    all_docs: List[Document] = []
    filenames: List[str] = []

    # --------------------------------------------------------
    # Load and extract text from uploaded PDFs.
    # --------------------------------------------------------

    for upload in files:
        filename = Path(upload.filename or "unnamed.pdf").name
        filenames.append(filename)

        if not filename.lower().endswith(".pdf"):
            await upload.close()
            raise HTTPException(
                status_code=400,
                detail=f"{filename} is not a PDF file.",
            )

        tmp_path = None

        try:
            content = await upload.read()

            if not content:
                raise HTTPException(
                    status_code=400,
                    detail=f"{filename} is empty.",
                )

            with tempfile.NamedTemporaryFile(
                mode="wb",
                delete=False,
                suffix=".pdf",
            ) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            pages = PyPDFLoader(tmp_path).load()

            readable_pages = [
                page
                for page in pages
                if page.page_content
                and page.page_content.strip()
            ]

            for page in readable_pages:
                page.metadata["source"] = filename

            all_docs.extend(readable_pages)

            logger.info(
                "Loaded %s: %d readable pages",
                filename,
                len(readable_pages),
            )

        except HTTPException:
            raise

        except Exception as exc:
            logger.exception("PDF loading failed for %s", filename)

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Could not read {filename}: "
                    f"{type(exc).__name__}: {exc}"
                ),
            ) from exc

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

            await upload.close()

    if not all_docs:
        raise HTTPException(
            status_code=400,
            detail=(
                "No readable text was found in the uploaded PDFs. "
                "Scanned PDFs may require OCR."
            ),
        )

    # --------------------------------------------------------
    # Split documents into text chunks.
    # --------------------------------------------------------

    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        chunks = [
            chunk
            for chunk in splitter.split_documents(all_docs)
            if chunk.page_content.strip()
        ]

        if not chunks:
            raise ValueError("No text chunks were generated.")

        logger.info("Created %d text chunks.", len(chunks))

    except Exception as exc:
        logger.exception("Text splitting failed")

        raise HTTPException(
            status_code=500,
            detail=f"Text splitting failed: {type(exc).__name__}: {exc}",
        ) from exc

    # --------------------------------------------------------
    # Generate embeddings and build the FAISS index.
    # --------------------------------------------------------

    try:
        embeddings = get_embeddings()

        logger.info(
            "Creating embeddings using LM Studio model: %s",
            EMBEDDING_MODEL,
        )

        new_store = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings,
        )

        # Add existing indexed documents to the new store.
        # FAISS.merge_from merges the other store into this store.
        if vectorstore is not None:
            new_store.merge_from(vectorstore)

        INDEX_DIR.mkdir(parents=True, exist_ok=True)

        new_store.save_local(str(INDEX_DIR))
        vectorstore = new_store

        logger.info("FAISS index saved to %s", INDEX_DIR)

    except Exception as exc:
        logger.exception("Embedding or FAISS indexing failed")

        raise HTTPException(
            status_code=500,
            detail=(
                f"Embedding/indexing failed: "
                f"{type(exc).__name__}: {exc}. "
                "Verify LM Studio is reachable, the embedding model "
                "is loaded, and its model ID is correct."
            ),
        ) from exc

    return {
        "message": "PDFs indexed successfully.",
        "files": filenames,
        "pages_read": len(all_docs),
        "chunks_indexed": len(chunks),
        "embedding_model": EMBEDDING_MODEL,
        "index_directory": str(INDEX_DIR),
    }


# ============================================================
# 7. LANGGRAPH RETRIEVAL NODE
# ============================================================

def retrieve_node(state: RAGState) -> RAGState:
    """Retrieve relevant document chunks from FAISS."""

    if vectorstore is None:
        raise ValueError(
            "No PDF index found. Upload a PDF using POST /ingest first."
        )

    docs = vectorstore.similarity_search(
        state["question"],
        k=RETRIEVAL_K,
    )

    context_parts: List[str] = []
    sources: List[Dict[str, Any]] = []
    seen = set()

    for doc in docs:
        source_name = doc.metadata.get("source", "PDF")
        page_number = int(doc.metadata.get("page", 0)) + 1

        context_parts.append(
            f"[Source: {source_name}, page {page_number}]\n"
            f"{doc.page_content}"
        )

        source_key = (source_name, page_number)

        if source_key not in seen:
            seen.add(source_key)

            sources.append({
                "source": source_name,
                "page": page_number,
            })

    return {
        "documents": docs,
        "context": "\n\n".join(context_parts),
        "sources": sources,
    }


# ============================================================
# 8. LANGGRAPH ANSWER GENERATION NODE
# ============================================================

def generate_node(state: RAGState) -> RAGState:
    """Generate an answer using the local LM Studio model."""

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a PDF question-answering assistant. "
            "Answer using only the supplied document context. "
            "If the answer is not present, say you could not "
            "find it in the uploaded PDFs. Do not invent facts. "
            "Mention source filenames and page numbers when useful.\n\n"
            "CONTEXT:\n{context}",
        ),
        (
            "human",
            "Question: {question}",
        ),
    ])

    chain = prompt | get_llm()

    result = chain.invoke({
        "context": state.get("context", ""),
        "question": state["question"],
    })

    answer = result.content

    if not isinstance(answer, str):
        answer = str(answer)

    return {
        "answer": answer,
    }


# ============================================================
# 9. BUILD LANGGRAPH WORKFLOW
# ============================================================

graph_builder = StateGraph(RAGState)

graph_builder.add_node("retrieve", retrieve_node)
graph_builder.add_node("generate", generate_node)

graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph_builder.add_edge("generate", END)

rag_graph = graph_builder.compile()


# ============================================================
# 10. RETRIEVE ANSWERS FROM PDFS
# ============================================================

@app.post("/retrieve")
def retrieve(
    request: RetrieveRequest,
) -> Dict[str, Any]:
    """Retrieve relevant PDF content and generate an answer."""

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:
        result = rag_graph.invoke({
            "question": question,
            "chat_history": [
                turn.model_dump()
                for turn in request.chat_history
            ],
        })

        return {
            "question": question,
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "chat_model": CHAT_MODEL,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("RAG workflow failed")

        raise HTTPException(
            status_code=500,
            detail=(
                f"RAG workflow failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        ) from exc
