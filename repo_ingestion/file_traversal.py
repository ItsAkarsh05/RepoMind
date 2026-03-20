"""
File Traversal
==============
Recursively walks a cloned repository and yields paths to relevant source-code
files while skipping directories and extensions that are not useful for
code-understanding (e.g. .git, node_modules, binaries).
"""

import os
import logging
from typing import Generator, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration — easy to extend
# ---------------------------------------------------------------------------

# Source-code extensions we care about
SUPPORTED_EXTENSIONS: set[str] = {
    # Python
    ".py",
    # JavaScript / TypeScript
    ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    # Java / Kotlin
    ".java", ".kt", ".kts",
    # Dart
    ".dart",
    # C / C++
    ".c", ".h", ".cpp", ".hpp", ".cc", ".cxx",
    # C#
    ".cs",
    # Go
    ".go",
    # Ruby
    ".rb",
    # Rust
    ".rs",
    # Swift
    ".swift",
    # PHP
    ".php",
    # Scala
    ".scala",
    # Shell
    ".sh", ".bash",
    # Markdown (often contains useful docs)
    ".md",
    # Config-as-code (useful for context)
    ".yaml", ".yml", ".toml", ".json",
}

# Directories to skip entirely
IGNORED_DIRS: set[str] = {
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
}

# Binary / non-code extensions to skip even if they slip through
IGNORED_EXTENSIONS: set[str] = {
    ".pyc", ".pyo", ".pyd",
    ".exe", ".dll", ".so", ".dylib", ".o", ".a", ".lib",
    ".class", ".jar", ".war",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".ico", ".webp",
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    ".mp3", ".mp4", ".avi", ".mov", ".wav",
    ".zip", ".tar", ".gz", ".bz2", ".rar", ".7z",
    ".lock", ".map",
    ".min.js", ".min.css",
    ".db", ".sqlite", ".sqlite3",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def traverse_repository(
    repo_path: str,
) -> Generator[Tuple[str, str], None, None]:
    """
    Walk a repository tree and yield relevant source-code file paths.

    Args:
        repo_path: Absolute path to the root of the cloned repository.

    Yields:
        Tuples of (absolute_file_path, relative_path_from_repo_root).
    """
    repo_path = os.path.abspath(repo_path)

    if not os.path.isdir(repo_path):
        raise FileNotFoundError(f"Repository path does not exist: {repo_path}")

    file_count = 0

    for root, dirs, files in os.walk(repo_path, topdown=True):
        # ---- Prune ignored directories in-place (prevents os.walk from
        #      descending into them at all) ----------------------------------
        dirs[:] = [
            d for d in dirs
            if d not in IGNORED_DIRS and not d.endswith(".egg-info")
        ]

        for filename in files:
            _, ext = os.path.splitext(filename)
            ext = ext.lower()

            # Skip explicitly ignored extensions
            if ext in IGNORED_EXTENSIONS:
                continue

            # Only yield files with a supported extension
            if ext not in SUPPORTED_EXTENSIONS:
                continue

            abs_path = os.path.join(root, filename)
            rel_path = os.path.relpath(abs_path, repo_path)

            file_count += 1
            yield abs_path, rel_path

    logger.info(
        "Traversal complete — found %d relevant source files in '%s'.",
        file_count,
        repo_path,
    )
