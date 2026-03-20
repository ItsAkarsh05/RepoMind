# ==========================================================================
# RepoMind — Chat with any GitHub codebase powered by RAG + LLama3
# ==========================================================================
# Adapted from https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps

import os
import json
import gc
import uuid
import logging

import streamlit as st
import nest_asyncio

from llama_index.core import Settings, PromptTemplate, VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.langchain import LangchainEmbedding

from rag_101.retriever import load_embedding_model, generate_repo_ast
from repo_ingestion import ingest_github_repo, validate_github_url

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Weight directories — adjust for your environment
WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", os.path.join(os.getcwd(), "weights"))
os.environ.setdefault("HF_HOME", WEIGHTS_DIR)
os.environ.setdefault("TORCH_HOME", WEIGHTS_DIR)

# Directories for cloned repos and FAISS indices
CLONE_DIR = os.path.join(os.getcwd(), "cloned_repos")
FAISS_DIR = os.path.join(os.getcwd(), "faiss_indices")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM & Embedding setup
# ---------------------------------------------------------------------------

llm = Ollama(model="llama3", request_timeout=60.0)

lc_embedding_model = load_embedding_model()
embed_model = LangchainEmbedding(lc_embedding_model)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "query_engine" not in st.session_state:
    st.session_state.query_engine = None
if "repo_ast" not in st.session_state:
    st.session_state.repo_ast = {}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def reset_chat():
    """Clear conversation history and query engine."""
    st.session_state.messages = []
    st.session_state.query_engine = None
    st.session_state.repo_ast = {}
    gc.collect()


# ---------------------------------------------------------------------------
# Sidebar — GitHub repo input
# ---------------------------------------------------------------------------

with st.sidebar:
    github_url = st.text_input("GitHub Repository URL")
    process_button = st.button("Load Repository")
    message_container = st.empty()

    if process_button and github_url:
        # ---- Validate the URL first ----
        try:
            owner, repo = validate_github_url(github_url)
        except ValueError as e:
            st.error(str(e))
            st.stop()

        with st.spinner(f"Ingesting {owner}/{repo} — cloning, chunking, embedding …"):
            try:
                # ========================================================
                # NEW: Use the modular repo_ingestion pipeline
                # ========================================================
                vectorstore, repo_path = ingest_github_repo(
                    github_url=github_url,
                    clone_dir=CLONE_DIR,
                    faiss_dir=FAISS_DIR,
                    embedding_model=lc_embedding_model,
                )

                # Convert the FAISS vectorstore into a llama_index query
                # engine by loading documents back via SimpleDirectoryReader
                # (the FAISS store is langchain-based, so we bridge through
                # the vectorstore's retriever).
                from llama_index.core import SimpleDirectoryReader

                loader = SimpleDirectoryReader(
                    input_dir=repo_path,
                    required_exts=[
                        ".py", ".ipynb", ".js", ".jsx",
                        ".ts", ".tsx", ".md", ".java",
                        ".dart", ".cpp", ".c", ".go",
                        ".rs", ".kt", ".swift",
                    ],
                    recursive=True,
                )
                docs = loader.load_data()

                # Create llama_index vector index for the query engine
                Settings.embed_model = embed_model
                index = VectorStoreIndex.from_documents(docs)

                # Setup query engine with streaming
                Settings.llm = llm
                query_engine = index.as_query_engine(
                    streaming=True,
                    similarity_top_k=4,
                )

                # Custom prompt template
                qa_prompt_tmpl_str = (
                    "Context information is below.\n"
                    "---------------------\n"
                    "{context_str}\n"
                    "---------------------\n"
                    "You are a helpful AI coding assistant. Given the "
                    "context information above, think step by step to "
                    "answer the query in a crisp manner. If you don't "
                    "know the answer say 'I don't know!'.\n"
                    "Query: {query_str}\n"
                    "Answer: "
                )
                qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
                query_engine.update_prompts(
                    {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
                )

                if docs:
                    message_container.success("Repository loaded successfully!")
                else:
                    message_container.warning(
                        "No supported files found — is the repo empty?"
                    )

                st.session_state.query_engine = query_engine
                st.session_state.repo_ast = generate_repo_ast(repo_path)

            except Exception as e:
                st.error(f"An error occurred: {e}")
                logger.exception("Ingestion failed")
                st.stop()

        st.success("Ready to Chat!")


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

col1, col2 = st.columns([6, 1])

with col1:
    st.header("Chat With Your Code! Powered by LLama3 🦙🚀")

with col2:
    st.button("Clear ↺", on_click=reset_chat)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What's up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        query_engine = st.session_state.query_engine

        if query_engine is None:
            full_response = "⚠️ Please load a GitHub repository first."
        else:
            # Include repo AST context in the query for structural awareness
            streaming_response = query_engine.query(
                f"Given the repository AST:\n"
                f"{json.dumps(st.session_state.repo_ast, indent=2)}\n\n"
                f"And the following question: {prompt}"
            )

            for chunk in streaming_response.response_gen:
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )
