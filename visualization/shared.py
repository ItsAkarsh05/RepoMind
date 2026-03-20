"""
Shared Constants
================
Duplicates of the IGNORED_DIRS and SUPPORTED_EXTENSIONS sets from
``repo_ingestion.file_traversal`` so the visualization package can be
imported and tested independently without pulling in heavy dependencies
like ``langchain_community``.

These should be kept in sync with ``repo_ingestion/file_traversal.py``.
"""

import os
from typing import Generator, Set, Tuple

# Directories to skip entirely (same as repo_ingestion.file_traversal)
IGNORED_DIRS: Set[str] = {
    ".git",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    "build",
    "dist",
    ".next",
    ".nuxt",
    ".vscode",
    ".idea",
    "venv",
    "env",
    ".env",
    ".venv",
    "vendor",
    "target",
    ".cache",
    ".tox",
    "eggs",
    "*.egg-info",
    ".mypy_cache",
    ".gradle",
    ".dart_tool",
    ".pub-cache",
    "coverage",
    ".coverage",
    "htmlcov",
    ".terraform",
    ".repomind_cache",
}

# Source-code extensions we care about
SUPPORTED_EXTENSIONS: Set[str] = {
    ".py",
    ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    ".java", ".kt", ".kts",
    ".dart",
    ".c", ".h", ".cpp", ".hpp", ".cc", ".cxx",
    ".cs",
    ".go",
    ".rb",
    ".rs",
    ".swift",
    ".php",
    ".scala",
    ".sh", ".bash",
    ".md",
    ".yaml", ".yml", ".toml", ".json",
}


def traverse_source_files(
    repo_path: str,
    ignored_dirs: Set[str] | None = None,
    supported_exts: Set[str] | None = None,
) -> Generator[Tuple[str, str], None, None]:
    """
    Walk a repository tree and yield relevant source-code file paths.

    This is a lightweight version of ``repo_ingestion.file_traversal.traverse_repository``
    that does not import any heavy dependencies.

    Args:
        repo_path:      Absolute path to the root of the repository.
        ignored_dirs:   Directories to skip. Defaults to ``IGNORED_DIRS``.
        supported_exts: File extensions to include. Defaults to ``SUPPORTED_EXTENSIONS``.

    Yields:
        Tuples of (absolute_file_path, relative_path_from_repo_root).
    """
    repo_path = os.path.abspath(repo_path)
    skip = ignored_dirs if ignored_dirs is not None else IGNORED_DIRS
    exts = supported_exts if supported_exts is not None else SUPPORTED_EXTENSIONS

    for root, dirs, files in os.walk(repo_path, topdown=True):
        # Prune ignored directories in-place
        dirs[:] = [
            d for d in dirs
            if d not in skip and not d.endswith(".egg-info")
        ]

        for filename in files:
            _, ext = os.path.splitext(filename)
            if ext.lower() in exts:
                abs_path = os.path.join(root, filename)
                rel_path = os.path.relpath(abs_path, repo_path)
                yield abs_path, rel_path
