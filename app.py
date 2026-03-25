# ==========================================================================
# RepoMind — Chat with any GitHub codebase powered by RAG + LLama3
# ==========================================================================
# Adapted from https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps

"""
Main Streamlit application for RepoMind.

This module provides the User Interface for the RepoMind application, allowing users
to input a GitHub repository URL, ingest the codebase into a vector database,
and interactively query the codebase using a Retrieval-Augmented Generation (RAG) pipeline.
It also provides a visualization tab for repository structure, call graphs,
and dependency graphs.
"""
import os
import json
import gc
import uuid
import logging
import html

import streamlit as st
import nest_asyncio

st.set_page_config(
    page_title="RepoMind — Chat with Code",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

from llama_index.core import Settings, PromptTemplate, VectorStoreIndex, Document
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.langchain import LangchainEmbedding

from rag_101.retriever import load_embedding_model, generate_repo_ast
from repo_ingestion import (
    validate_github_url, clone_repository,
    traverse_repository, chunk_file,
)
from memory import ChatMemory
from visualization import (
    get_repo_structure, build_call_graph, build_dependency_graph,
    render_repo_tree, render_call_graph, render_dependency_graph,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Weight directories — adjust for your environment
WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", os.path.join(os.getcwd(), "weights"))
os.environ.setdefault("HF_HOME", WEIGHTS_DIR)
os.environ.setdefault("TORCH_HOME", WEIGHTS_DIR)

# Directories for cloned repos
CLONE_DIR = os.path.join(os.getcwd(), "cloned_repos")

# Maximum number of recent chat messages to include in LLM context.
# Reduced from 10 to 4 to speed up LLM generation speeds.
MAX_HISTORY_MESSAGES = 4

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
if "repo_path" not in st.session_state:
    st.session_state.repo_path = None
if "viz_cache" not in st.session_state:
    st.session_state.viz_cache = {}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def reset_chat():
    """Clear conversation history and query engine."""
    st.session_state.messages = []
    st.session_state.query_engine = None
    st.session_state.repo_ast = {}
    st.session_state.repo_path = None
    st.session_state.viz_cache = {}
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
    # ── Sidebar branding ──
    st.markdown(
        '<div class="sidebar-brand">'
        '<span style="font-size:2rem;">🧠</span>'
        '<h1 style="margin:0;font-size:1.4rem;font-weight:700;'
        'background:linear-gradient(135deg,#667eea,#764ba2);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;">'
        'RepoMind</h1>'
        '<p style="margin:0;font-size:0.78rem;opacity:0.6;">'
        'Chat with any codebase</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    github_url = st.text_input(
        "🔗 GitHub Repository URL",
        placeholder="https://github.com/owner/repo",
    )
    process_button = st.button(
        "🚀 Load Repository", type="primary", use_container_width=True,
    )
    message_container = st.empty()

    st.markdown("---")
    st.markdown(
        '<p style="text-align:center;font-size:0.72rem;opacity:0.4;">'
        'Powered by Qwen Coder · FAISS · LlamaIndex</p>',
        unsafe_allow_html=True,
    )

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
                # Single-pass pipeline: clone → traverse → chunk → embed
                # (No double-indexing: FAISS build removed, documents
                #  are embedded only once via VectorStoreIndex)
                # ========================================================
                repo_path = clone_repository(github_url, clone_dir=CLONE_DIR)

                # Traverse & chunk source files using AST-aware chunking
                import concurrent.futures
                file_paths = [p for p, _ in traverse_repository(repo_path)]
                all_chunks = []
                if file_paths:
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        for chunks in executor.map(chunk_file, file_paths):
                            if chunks:
                                all_chunks.extend(chunks)

                # Convert CodeChunks → llama_index Documents (single pass)
                docs = [
                    Document(
                        text=chunk.content,
                        metadata=chunk.to_metadata(),
                    )
                    for chunk in all_chunks
                ]

                # Build vector index — embeddings computed ONCE here
                Settings.embed_model = embed_model
                index = VectorStoreIndex.from_documents(
                    docs, show_progress=True,
                )

                # Setup query engine with streaming
                Settings.llm = llm
                query_engine = index.as_query_engine(
                    streaming=True,
                    similarity_top_k=3,
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
                st.session_state.repo_path = repo_path
                st.session_state.viz_cache = {}  # reset viz cache for new repo

            except Exception as e:
                st.error(f"An error occurred: {e}")
                logger.exception("Ingestion failed")
                st.stop()

        st.success("Ready to Chat!")


# ---------------------------------------------------------------------------
# Custom CSS for premium look
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    /* ── Import premium font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* ── Global typography ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Sidebar branding ── */
    .sidebar-brand {
        text-align: center;
        padding: 12px 0 4px;
    }

    /* ── Tab styling ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        padding: 0 4px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        border-radius: 10px 10px 0 0;
        font-weight: 600;
        font-size: 1.05rem;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }

    /* ── Chat message bubbles ── */
    [data-testid="stChatMessage"] {
        border-radius: 16px;
        padding: 14px 18px;
        margin-bottom: 12px;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        animation: fadeSlideIn 0.3s ease-out;
    }

    /* User bubble */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: rgba(102, 126, 234, 0.10);
    }
    /* Assistant bubble */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: rgba(118, 75, 162, 0.08);
    }

    /* ── Avatar styling ── */
    [data-testid="stChatMessage"] [data-testid^="chatAvatarIcon"] {
        font-size: 1.5rem;
    }

    /* ── Chat input ── */
    [data-testid="stChatInput"] textarea {
        border-radius: 14px !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        transition: border-color 0.2s ease;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.15) !important;
    }

    /* ── Buttons ── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 14px rgba(102, 126, 234, 0.35);
    }

    /* ── Code blocks ── */
    code {
        border-radius: 6px;
        font-size: 0.88em;
    }

    /* ── Empty state card ── */
    .empty-state {
        text-align: center;
        padding: 60px 24px;
        border-radius: 20px;
        border: 1px dashed rgba(102, 126, 234, 0.25);
        background: rgba(102, 126, 234, 0.03);
        margin: 20px 0;
    }
    .empty-state .icon { font-size: 3rem; margin-bottom: 8px; }
    .empty-state h3 { margin: 8px 0 4px; opacity: 0.7; }
    .empty-state p { opacity: 0.5; font-size: 0.95rem; }

    /* ── History bubbles ── */
    .history-msg {
        padding: 12px 16px;
        border-radius: 14px;
        margin-bottom: 10px;
        font-size: 0.92rem;
        line-height: 1.55;
        animation: fadeSlideIn 0.25s ease-out;
    }
    .history-msg.user {
        background: rgba(102, 126, 234, 0.10);
        border-left: 3px solid #667eea;
    }
    .history-msg.assistant {
        background: rgba(118, 75, 162, 0.08);
        border-left: 3px solid #764ba2;
    }
    .history-msg .role {
        font-weight: 600;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        opacity: 0.6;
        margin-bottom: 4px;
    }

    /* ── Chat container (ChatGPT-style) ── */
    [data-testid="stVerticalBlockBorderWrapper"]:has(
        > div > [data-testid="stVerticalBlock"] [data-testid="stChatMessage"]
    ) {
        border-radius: 16px !important;
        border: 1px solid rgba(102, 126, 234, 0.12) !important;
    }
    /* Auto-scroll chat to bottom */
    [data-testid="stVerticalBlockBorderWrapper"]:has(
        > div > [data-testid="stVerticalBlock"] [data-testid="stChatMessage"]
    ) > div {
        display: flex;
        flex-direction: column-reverse;
    }

    /* ── Chat input pinned ── */
    [data-testid="stChatInput"] {
        position: sticky;
        bottom: 0;
        z-index: 100;
        padding-top: 8px;
    }

    /* ── Graphviz container ── */
    .stGraphVizChart {
        border-radius: 12px;
        padding: 8px;
    }

    /* ── Smooth scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-thumb {
        background: rgba(102, 126, 234, 0.3);
        border-radius: 4px;
    }

    /* ── Fade-in animation ── */
    @keyframes fadeSlideIn {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main content — three top-level tabs
# ---------------------------------------------------------------------------

tab_chat, tab_viz, tab_history = st.tabs(["💬 Chat", "📊 Visualize", "📜 History"])

# ═══════════════════════════════════════════════════════════════════════
# TAB 1: Chat
# ═══════════════════════════════════════════════════════════════════════

with tab_chat:
    col1, col2 = st.columns([6, 1])

    with col1:
        st.markdown(
            '<h2 style="margin:0;background:linear-gradient(135deg,#667eea,#764ba2);'
            '-webkit-background-clip:text;-webkit-text-fill-color:transparent;'
            'font-weight:700;">Chat With Your Code</h2>'
            '<p style="margin:0 0 8px;opacity:0.5;font-size:0.88rem;">'
            'Ask questions about the loaded repository</p>',
            unsafe_allow_html=True,
        )

    with col2:
        st.button("🗑 Clear", on_click=reset_chat)

    # ── Scrollable chat container (ChatGPT-style) ──
    chat_container = st.container(height=500)

    with chat_container:
        # Empty state when no messages
        if not st.session_state.messages:
            st.markdown(
                '<div class="empty-state">'
                '<div class="icon">💬</div>'
                '<h3>Start a conversation</h3>'
                '<p>Load a GitHub repository from the sidebar, '
                'then ask anything about the code.</p></div>',
                unsafe_allow_html=True,
            )

        # Display chat history
        for message in st.session_state.messages:
            avatar = "🧑\u200d💻" if message["role"] == "user" else "🧠"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

    # ── Chat input (pinned below the container) ──
    if prompt := st.chat_input("Ask anything about the codebase…"):
        # Record the user message in both Streamlit state and ChatMemory
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.chat_memory.add_user_message(prompt)

        with chat_container:
            with st.chat_message("user", avatar="🧑\u200d💻"):
                st.markdown(prompt)

            with st.chat_message("assistant", avatar="🧠"):
                message_placeholder = st.empty()
                full_response = ""

                query_engine = st.session_state.query_engine

                if query_engine is None:
                    full_response = "⚠️ Please load a GitHub repository first."
                else:
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


# ═══════════════════════════════════════════════════════════════════════
# TAB 2: Visualize
# ═══════════════════════════════════════════════════════════════════════

with tab_viz:
    if not st.session_state.repo_path:
        st.markdown(
            '<div class="empty-state">'
            '<div class="icon">📊</div>'
            '<h3>No Repository Loaded</h3>'
            '<p>Enter a GitHub URL in the sidebar and click '
            '<b>Load Repository</b> to unlock visualizations.</p></div>',
            unsafe_allow_html=True,
        )
    else:
        repo_name = os.path.basename(st.session_state.repo_path)
        st.markdown(
            f'<h2 style="margin-bottom:4px;">'
            f'🔍 Exploring <code>{repo_name}</code></h2>',
            unsafe_allow_html=True,
        )

        viz_struct, viz_calls, viz_deps = st.tabs([
            "📂 File Structure",
            "🔗 Call Graph",
            "🌐 Dependencies",
        ])

        repo_path_viz = st.session_state.repo_path

        with viz_struct:
            if "repo_tree" not in st.session_state.viz_cache:
                with st.spinner("Building file tree…"):
                    st.session_state.viz_cache["repo_tree"] = (
                        get_repo_structure(repo_path_viz)
                    )
            render_repo_tree(st.session_state.viz_cache["repo_tree"])

        with viz_calls:
            if "call_graph" not in st.session_state.viz_cache:
                with st.spinner("Analysing function calls…"):
                    st.session_state.viz_cache["call_graph"] = (
                        build_call_graph(repo_path_viz)
                    )
            render_call_graph(st.session_state.viz_cache["call_graph"])

        with viz_deps:
            if "dep_graph" not in st.session_state.viz_cache:
                with st.spinner("Mapping dependencies…"):
                    st.session_state.viz_cache["dep_graph"] = (
                        build_dependency_graph(repo_path_viz)
                    )
            render_dependency_graph(st.session_state.viz_cache["dep_graph"])

# ═══════════════════════════════════════════════════════════════════════
# TAB 3: History
# ═══════════════════════════════════════════════════════════════════════

with tab_history:
    col_hist1, col_hist2 = st.columns([6, 1])
    with col_hist1:
        st.markdown(
            '<h2 style="margin:0;background:linear-gradient(135deg,#667eea,#764ba2);'
            '-webkit-background-clip:text;-webkit-text-fill-color:transparent;'
            'font-weight:700;">Conversation History</h2>',
            unsafe_allow_html=True,
        )
    with col_hist2:
        st.button(
            "🗑 Reset", type="primary", on_click=reset_chat,
            key="reset_chat_history_tab",
        )

    history = st.session_state.chat_memory.get_history()
    if not history:
        st.markdown(
            '<div class="empty-state">'
            '<div class="icon">📜</div>'
            '<h3>No History Yet</h3>'
            '<p>Your conversation will appear here as you chat.</p></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<p style="opacity:0.5;font-size:0.85rem;margin-bottom:12px;">'
            f'💬 {len(history)} message{"s" if len(history) != 1 else ""} '
            f'in this session</p>',
            unsafe_allow_html=True,
        )
        for msg in history:
            role = msg["role"]
            label = "🧑\u200d💻 You" if role == "user" else "🧠 RepoMind"
            css_class = "user" if role == "user" else "assistant"
            # Escape HTML in content to prevent injection
            safe_content = html.escape(msg["content"]).replace("\n", "<br>")
            st.markdown(
                f'<div class="history-msg {css_class}">'
                f'<div class="role">{label}</div>'
                f'{safe_content}</div>',
                unsafe_allow_html=True,
            )
