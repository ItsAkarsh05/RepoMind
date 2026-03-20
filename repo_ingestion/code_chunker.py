"""
Code Chunker
=============
Splits source-code files into meaningful chunks for embedding.

Strategy:
  * **Python files** → AST-aware chunking (one chunk per function / class).
    Falls back to fixed-size if the file has syntax errors.
  * **All other files** → Fixed-size line-window chunking with overlap.

Every chunk carries metadata: file_path, file_name, start_line, end_line,
and chunk_type (function / class / block).
"""

import ast
import os
import logging
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Model
# ---------------------------------------------------------------------------

@dataclass
class CodeChunk:
    """A single chunk of source code with rich metadata."""
    content: str                          # The actual source lines
    file_path: str                        # Absolute path to the file
    file_name: str                        # Basename of the file
    start_line: int                       # 1-indexed inclusive
    end_line: int                         # 1-indexed inclusive
    chunk_type: str = "block"             # "function", "class", or "block"

    def to_metadata(self) -> dict:
        """Return a flat dict suitable for vector-store metadata."""
        return {
            "file_path": self.file_path,
            "file_name": self.file_name,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "chunk_type": self.chunk_type,
        }


# ---------------------------------------------------------------------------
# Fixed-size Chunker (language-agnostic)
# ---------------------------------------------------------------------------

def chunk_file_fixed_size(
    source: str,
    file_path: str,
    chunk_size: int = 60,
    overlap: int = 10,
) -> List[CodeChunk]:
    """
    Split source code into overlapping line-windows.

    Args:
        source:     The full file content as a string.
        file_path:  Absolute path (used for metadata).
        chunk_size: Number of lines per chunk.
        overlap:    Number of lines overlapping between consecutive chunks.

    Returns:
        A list of CodeChunk objects.
    """
    lines = source.splitlines(keepends=True)
    file_name = os.path.basename(file_path)
    chunks: List[CodeChunk] = []

    if not lines:
        return chunks

    step = max(chunk_size - overlap, 1)

    for start in range(0, len(lines), step):
        end = min(start + chunk_size, len(lines))
        chunk_lines = lines[start:end]
        content = "".join(chunk_lines)

        if content.strip():  # skip empty chunks
            chunks.append(
                CodeChunk(
                    content=content,
                    file_path=file_path,
                    file_name=file_name,
                    start_line=start + 1,       # 1-indexed
                    end_line=end,                # 1-indexed inclusive
                    chunk_type="block",
                )
            )

        # If we've reached the end of the file, stop
        if end >= len(lines):
            break

    return chunks


# ---------------------------------------------------------------------------
# AST-aware Chunker (Python only)
# ---------------------------------------------------------------------------

def _extract_ast_nodes(source: str, file_path: str) -> List[CodeChunk]:
    """
    Use the Python ``ast`` module to extract top-level and nested function /
    class definitions as individual chunks.

    Each node gets its own chunk.  Code that falls *between* top-level
    definitions (e.g. module-level statements) is captured as a "block" chunk
    so nothing is lost.
    """
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    file_name = os.path.basename(file_path)
    chunks: List[CodeChunk] = []

    # Collect line-ranges covered by top-level nodes
    node_ranges: List[tuple] = []

    for node in ast.iter_child_nodes(tree):
        if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
            continue

        start = node.lineno            # 1-indexed
        end = node.end_lineno          # 1-indexed inclusive

        # Determine chunk_type
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            chunk_type = "function"
        elif isinstance(node, ast.ClassDef):
            chunk_type = "class"
        else:
            chunk_type = "block"

        content = "".join(lines[start - 1 : end])
        if content.strip():
            chunks.append(
                CodeChunk(
                    content=content,
                    file_path=file_path,
                    file_name=file_name,
                    start_line=start,
                    end_line=end,
                    chunk_type=chunk_type,
                )
            )
            node_ranges.append((start, end))

    # Capture any lines not covered by the top-level nodes (e.g. imports,
    # module docstrings, standalone statements)
    covered = set()
    for s, e in node_ranges:
        covered.update(range(s, e + 1))

    uncovered_start = None
    for i in range(1, len(lines) + 1):
        if i not in covered:
            if uncovered_start is None:
                uncovered_start = i
        else:
            if uncovered_start is not None:
                content = "".join(lines[uncovered_start - 1 : i - 1])
                if content.strip():
                    chunks.append(
                        CodeChunk(
                            content=content,
                            file_path=file_path,
                            file_name=file_name,
                            start_line=uncovered_start,
                            end_line=i - 1,
                            chunk_type="block",
                        )
                    )
                uncovered_start = None

    # Handle trailing uncovered lines
    if uncovered_start is not None:
        content = "".join(lines[uncovered_start - 1 :])
        if content.strip():
            chunks.append(
                CodeChunk(
                    content=content,
                    file_path=file_path,
                    file_name=file_name,
                    start_line=uncovered_start,
                    end_line=len(lines),
                    chunk_type="block",
                )
            )

    # Sort by start_line so chunks appear in reading order
    chunks.sort(key=lambda c: c.start_line)
    return chunks


def chunk_python_file(source: str, file_path: str) -> List[CodeChunk]:
    """
    Chunk a Python file using AST-aware splitting.
    Falls back to fixed-size chunking on syntax errors.

    Args:
        source:    Full file content.
        file_path: Absolute path (for metadata).

    Returns:
        List of CodeChunk objects.
    """
    try:
        return _extract_ast_nodes(source, file_path)
    except SyntaxError:
        logger.warning(
            "SyntaxError parsing '%s' — falling back to fixed-size chunking.",
            file_path,
        )
        return chunk_file_fixed_size(source, file_path)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def chunk_file(file_path: str) -> List[CodeChunk]:
    """
    Read a file and split it into chunks.

    * ``.py`` files use AST-aware chunking.
    * All other files use fixed-size line-window chunking.

    Args:
        file_path: Absolute path to the source file.

    Returns:
        A list of CodeChunk objects (may be empty for unreadable files).
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
            source = fh.read()
    except (OSError, IOError) as exc:
        logger.error("Could not read file '%s': %s", file_path, exc)
        return []

    if not source.strip():
        return []

    _, ext = os.path.splitext(file_path)

    if ext.lower() == ".py":
        return chunk_python_file(source, file_path)

    return chunk_file_fixed_size(source, file_path)
