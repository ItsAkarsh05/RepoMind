"""
Repository Structure Generator
===============================
Builds a hierarchical JSON tree of the repository's file system, suitable
for frontend tree-view rendering.

Reuses the IGNORED_DIRS set from ``repo_ingestion.file_traversal`` so the
exclusion list is consistent across the whole application.
"""

import os
import logging
from typing import Dict, Any, Optional, Set

# Use the shared constants (avoids pulling in heavy langchain deps)
from .shared import IGNORED_DIRS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_repo_structure(
    repo_path: str,
    ignored_dirs: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Generate a hierarchical tree of the repository's directory structure.

    Args:
        repo_path:    Absolute path to the root of the repository.
        ignored_dirs: Optional custom set of directory names to skip.
                      Defaults to the shared ``IGNORED_DIRS`` list.

    Returns:
        A nested dict with the shape::

            {
              "name": "repo-root",
              "path": ".",
              "type": "directory",
              "children": [ ... ]
            }

        File nodes additionally carry ``extension`` and ``size_bytes``.
    """
    repo_path = os.path.abspath(repo_path)
    if not os.path.isdir(repo_path):
        raise FileNotFoundError(f"Repository path does not exist: {repo_path}")

    skip = ignored_dirs if ignored_dirs is not None else IGNORED_DIRS
    root_name = os.path.basename(repo_path) or repo_path

    tree = _build_tree(repo_path, repo_path, skip)
    tree["name"] = root_name
    tree["path"] = "."

    logger.info("Built repository structure tree for '%s'.", repo_path)
    return tree


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_tree(
    current_path: str,
    repo_root: str,
    ignored: Set[str],
) -> Dict[str, Any]:
    """Recursively build the tree dict for *current_path*."""
    name = os.path.basename(current_path)
    rel_path = os.path.relpath(current_path, repo_root).replace("\\", "/")

    if os.path.isfile(current_path):
        _, ext = os.path.splitext(name)
        try:
            size = os.path.getsize(current_path)
        except OSError:
            size = 0

        return {
            "name": name,
            "path": rel_path,
            "type": "file",
            "extension": ext.lower(),
            "size_bytes": size,
        }

    # Directory node
    children = []
    try:
        entries = sorted(os.listdir(current_path))
    except PermissionError:
        entries = []

    for entry in entries:
        # Skip ignored directories
        if os.path.isdir(os.path.join(current_path, entry)):
            if entry in ignored or entry.endswith(".egg-info"):
                continue

        child_path = os.path.join(current_path, entry)
        children.append(_build_tree(child_path, repo_root, ignored))

    return {
        "name": name,
        "path": rel_path,
        "type": "directory",
        "children": children,
    }
