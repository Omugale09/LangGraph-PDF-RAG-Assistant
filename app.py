import os
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()
API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="PDF RAG Chat", page_icon="📚", layout="wide")
st.title("📚 PDF RAG Chatbot")
st.caption("Upload PDFs, index their content, and ask questions with LangGraph + Google Generative AI.")

with st.sidebar:
    st.header("1. Add documents")
    uploads = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)
    if st.button("Index PDFs", type="primary", disabled=not uploads):
        with st.spinner("Reading and indexing PDFs..."):
            try:
                files_payload = [
                    ("files", (f.name, f.getvalue(), "application/pdf"))
                    for f in uploads
                ]
                response = requests.post(f"{API_URL}/ingest", files=files_payload, timeout=300)
                response.raise_for_status()
                data = response.json()
                st.success(data.get("message", "Documents indexed."))
                st.json(data)
            except requests.RequestException as exc:
                st.error(f"Could not index PDFs: {exc}")

    st.divider()
    st.header("2. Chat controls")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.write(f"**{source.get('source', 'PDF')}** — page {source.get('page', '?')}")

question = st.chat_input("Ask a question about your indexed PDFs...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating an answer..."):
            try:
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                    if m["role"] in ("user", "assistant")
                ]
                response = requests.post(
                    f"{API_URL}/retrieve",
                    json={"question": question, "chat_history": history},
                    timeout=180,
                )
                response.raise_for_status()
                data = response.json()
                answer = data.get("answer", "No answer returned.")
                st.markdown(answer)
                sources = data.get("sources", [])
                if sources:
                    with st.expander("Sources"):
                        for source in sources:
                            st.write(f"**{source.get('source', 'PDF')}** — page {source.get('page', '?')}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except requests.RequestException as exc:
                st.error(f"API request failed: {exc}")
                st.info("Make sure FastAPI is running at " + API_URL)
