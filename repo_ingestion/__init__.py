"""
repo_ingestion
==============
End-to-end pipeline for ingesting a GitHub repository into a FAISS vector
store.  This is the public API consumed by ``app.py`` and other entry-points.

Usage::

    from repo_ingestion import ingest_github_repo
    vectorstore, repo_path = ingest_github_repo(
        github_url="https://github.com/pallets/flask",
    )
"""

import logging
from typing import List, Tuple

from langchain_community.vectorstores import FAISS

from .github_handler import validate_github_url, clone_repository
from .file_traversal import traverse_repository
from .code_chunker import chunk_file, CodeChunk
from .embedding_store import build_faiss_index, load_faiss_index, query_index

logger = logging.getLogger(__name__)

__all__ = [
    # High-level orchestrator
    "ingest_github_repo",
    # Re-exports for convenience
    "validate_github_url",
    "clone_repository",
    "traverse_repository",
    "chunk_file",
    "CodeChunk",
    "build_faiss_index",
    "load_faiss_index",
    "query_index",
]


def ingest_github_repo(
    github_url: str,
    clone_dir: str = "cloned_repos",
    faiss_dir: str = "faiss_indices",
    embedding_model=None,
) -> Tuple[FAISS, str]:
    """
    Full pipeline: validate URL → clone → traverse → chunk → embed → store.

    Args:
        github_url:      Public GitHub repository HTTPS URL.
        clone_dir:       Directory to clone repositories into.
        faiss_dir:       Directory to persist the FAISS index.
        embedding_model: A langchain-compatible embeddings model.  Must be
                         provided (typically ``HuggingFaceBgeEmbeddings``).

    Returns:
        A tuple of (FAISS vectorstore, absolute path to cloned repo).

    Raises:
        ValueError:  If the URL is invalid or no code files are found.
        RuntimeError: If cloning fails.
    """
    if embedding_model is None:
        raise ValueError("An embedding_model must be provided.")

    # Step 1 — Validate & clone
    owner, repo = validate_github_url(github_url)
    logger.info("Starting ingestion of '%s/%s' …", owner, repo)
    repo_path = clone_repository(github_url, clone_dir=clone_dir)

    # Step 2 — Traverse source files
    all_chunks: List[CodeChunk] = []
    file_count = 0

    for abs_path, rel_path in traverse_repository(repo_path):
        file_count += 1
        chunks = chunk_file(abs_path)
        all_chunks.extend(chunks)

    logger.info(
        "Traversed %d files → produced %d chunks.", file_count, len(all_chunks)
    )

    if not all_chunks:
        raise ValueError(
            f"No code chunks produced from repository '{github_url}'. "
            "The repo may be empty or contain only unsupported file types."
        )

    # Step 3 — Build embeddings & persist FAISS index
    index_name = f"{owner}__{repo}"
    vectorstore = build_faiss_index(
        chunks=all_chunks,
        embedding_model=embedding_model,
        faiss_dir=faiss_dir,
        index_name=index_name,
    )

    logger.info("Ingestion complete for '%s/%s'.", owner, repo)
    return vectorstore, repo_path
