"""
Embedding Store
===============
Converts code chunks into embeddings via the existing HuggingFace BGE model
and persists them in a FAISS vector store for fast retrieval.
"""

import os
import logging
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from .code_chunker import CodeChunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Build & Persist
# ---------------------------------------------------------------------------

def build_faiss_index(
    chunks: List[CodeChunk],
    embedding_model,
    faiss_dir: str = "faiss_indices",
    index_name: str = "repo_index",
) -> FAISS:
    """
    Create a FAISS vector store from a list of code chunks and save it to disk.

    Each chunk is stored as a langchain ``Document`` whose metadata contains
    file_path, file_name, start_line, end_line, and chunk_type — ensuring
    that retrieval results carry full provenance information.

    Args:
        chunks:          List of CodeChunk objects to embed.
        embedding_model: A langchain-compatible embeddings model instance.
        faiss_dir:       Directory to persist the FAISS index.
        index_name:      Name for the saved index (used as subfolder).

    Returns:
        The populated FAISS vectorstore instance.

    Raises:
        ValueError: If ``chunks`` is empty.
    """
    if not chunks:
        raise ValueError("No code chunks provided — cannot build an index.")

    # Convert CodeChunks → langchain Documents with metadata
    documents: List[Document] = []
    for chunk in chunks:
        doc = Document(
            page_content=chunk.content,
            metadata=chunk.to_metadata(),
        )
        documents.append(doc)

    logger.info(
        "Building FAISS index from %d documents using '%s' …",
        len(documents),
        type(embedding_model).__name__,
    )

    # Create the FAISS vectorstore (embeddings are computed here)
    vectorstore = FAISS.from_documents(documents, embedding_model)

    # Persist to disk so subsequent runs can reload without re-embedding
    save_path = os.path.join(faiss_dir, index_name)
    os.makedirs(save_path, exist_ok=True)
    vectorstore.save_local(save_path)
    logger.info("FAISS index saved to '%s'.", save_path)

    return vectorstore


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_faiss_index(
    faiss_dir: str,
    embedding_model,
    index_name: str = "repo_index",
) -> Optional[FAISS]:
    """
    Load a previously persisted FAISS index from disk.

    Args:
        faiss_dir:       Parent directory containing saved indices.
        embedding_model: The same embeddings model used during indexing.
        index_name:      The subfolder name of the saved index.

    Returns:
        The loaded FAISS vectorstore, or ``None`` if the index does not exist.
    """
    save_path = os.path.join(faiss_dir, index_name)
    if not os.path.isdir(save_path):
        logger.warning("No FAISS index found at '%s'.", save_path)
        return None

    logger.info("Loading FAISS index from '%s' …", save_path)
    vectorstore = FAISS.load_local(
        save_path,
        embedding_model,
        allow_dangerous_deserialization=True,
    )
    return vectorstore


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

def query_index(
    vectorstore: FAISS,
    query: str,
    k: int = 5,
) -> List[Document]:
    """
    Retrieve the top-*k* most relevant chunks for a natural-language query.

    Args:
        vectorstore: A FAISS vectorstore instance.
        query:       The user's question / search string.
        k:           Number of results to return.

    Returns:
        A list of langchain Document objects (with metadata).
    """
    results = vectorstore.similarity_search(query, k=k)
    logger.info("Retrieved %d chunks for query: '%s'", len(results), query[:80])
    return results
