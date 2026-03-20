"""
visualization
=============
Code-visualization module for RepoMind.

Provides three main capabilities:
  1. Repository file-structure tree generation
  2. Function / class call-graph construction
  3. File-level dependency-graph mapping

All outputs are JSON-serialisable dicts ready for frontend graph libraries
(D3.js, Cytoscape, React Flow).
"""

from .repo_structure import get_repo_structure
from .call_graph import build_call_graph
from .dependency_graph import build_dependency_graph

__all__ = [
    "get_repo_structure",
    "build_call_graph",
    "build_dependency_graph",
]
