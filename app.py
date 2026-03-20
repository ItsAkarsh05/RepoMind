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
from memory import ChatMemory

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

# Maximum number of recent chat messages to include in LLM context.
# Keeps token usage bounded while providing conversational continuity.
MAX_HISTORY_MESSAGES = 10

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM & Embedding setup
# ---------------------------------------------------------------------------

llm = Ollama(model="qwen2.5-coder:3b", request_timeout=60.0)

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
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = ChatMemory(max_messages=MAX_HISTORY_MESSAGES)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def reset_chat():
    """Clear conversation history and query engine."""
    st.session_state.messages = []
    st.session_state.query_engine = None
    st.session_state.repo_ast = {}
    st.session_state.chat_memory.clear_history()
    gc.collect()


def _build_history_context() -> str:
    """
    Build a formatted conversation-history block for the LLM prompt.

    **Follow-up handling**:  Users often send short, context-dependent
    messages such as "Explain it better", "What about this function?",
    or "Can you optimize it?".  These rely on the LLM understanding
    *what* "it" or "this" refers to from earlier turns.

    By injecting the most recent exchanges (capped by
    ``MAX_HISTORY_MESSAGES``) into every prompt, the model can:
      - Resolve pronouns and references ("it", "that", "this function")
      - Maintain topical continuity across turns
      - Build on its own prior answers when asked to refine/optimize

    Returns an empty string when there is no history yet, so the very
    first query is unaffected.
    """
    history = st.session_state.chat_memory.get_history()
    if not history:
        return ""

    lines = []
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")

    return (
        "Previous conversation (use this to resolve references like "
        "'it', 'this', 'that function', etc.):\n"
        + "\n".join(lines)
        + "\n---------------------\n"
    )


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

                # ---------------------------------------------------------
                # Prompt template — follow-up & grounding aware
                # ---------------------------------------------------------
                # The prompt is structured in clear sections:
                #   1. System role + follow-up handling instruction
                #   2. Conversation history (recent turns)
                #   3. Retrieved code context (RAG chunks)
                #   4. Grounding rule (avoid hallucination)
                #   5. Current query
                #
                # This layout lets the LLM resolve vague follow-ups
                # ("explain it better", "optimize it") by reading the
                # conversation history *before* the retrieved code, so
                # it knows what the user is referring to.
                # ---------------------------------------------------------
                qa_prompt_tmpl_str = (
                    "You are a helpful AI coding assistant specialised in "
                    "understanding codebases.\n\n"
                    #
                    # --- Follow-up handling instruction ---
                    "IMPORTANT: Use previous conversation context to "
                    "understand the user's intent. If the user says "
                    "'explain it', 'what about this function', "
                    "'can you optimize it', or similar short follow-ups, "
                    "refer to the conversation history below to determine "
                    "what 'it' or 'this' refers to.\n\n"
                    #
                    # --- Conversation history (filled at query time) ---
                    "{chat_history}"
                    #
                    # --- Retrieved code context (RAG) ---
                    "Retrieved code context is below.\n"
                    "---------------------\n"
                    "{context_str}\n"
                    "---------------------\n\n"
                    #
                    # --- Grounding rule ---
                    "GROUNDING RULE: Base your answer strictly on the "
                    "retrieved code context above. Do NOT invent code "
                    "that does not appear in the context. If the context "
                    "does not contain enough information, say "
                    "'I don't know!' rather than guessing.\n\n"
                    #
                    # --- Current query ---
                    "Given the conversation history and the retrieved "
                    "code context, think step by step to answer the "
                    "following query in a crisp manner.\n"
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
    # Record the user message in both Streamlit state and ChatMemory
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.chat_memory.add_user_message(prompt)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        query_engine = st.session_state.query_engine

        if query_engine is None:
            full_response = "⚠️ Please load a GitHub repository first."
        else:
            # ----------------------------------------------------------
            # Enriched query construction (follow-up support)
            # ----------------------------------------------------------
            # We inject three pieces of context into the query string:
            #
            # 1. **Conversation history** — so the LLM can resolve
            #    references like "it", "this function", "optimize it".
            #    This is the key mechanism for follow-up support.
            #
            # 2. **Repository AST** — structural overview of the repo
            #    for file/class/function awareness.
            #
            # 3. **The user's current question** — the actual prompt.
            #
            # Together with the RAG-retrieved code chunks (injected by
            # llama_index via {context_str}), this gives the LLM the
            # full picture: what was discussed, what the code looks
            # like, and what the user is currently asking.
            # ----------------------------------------------------------
            history_ctx = _build_history_context()
            enriched_query = (
                f"{history_ctx}"
                f"Given the repository AST:\n"
                f"{json.dumps(st.session_state.repo_ast, indent=2)}\n\n"
                f"And the following question: {prompt}"
            )

            streaming_response = query_engine.query(enriched_query)

            for chunk in streaming_response.response_gen:
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    # Record the assistant response in both Streamlit state and ChatMemory
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )
    st.session_state.chat_memory.add_assistant_message(full_response)
