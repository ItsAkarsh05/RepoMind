"""
Call-Graph Builder
==================
Traverses a repository, analyses source files with the appropriate
language-specific analyzer, and produces a directed graph of function
calls and class-method containment relationships.

Output format (Cytoscape / React Flow compatible)::

    {
      "nodes": [ { "id", "label", "type", "file", "line" }, … ],
      "edges": [ { "source", "target", "type" }, … ],
    }
"""

import os
import logging
from typing import Any, Dict, List, Set, Tuple

from .shared import traverse_source_files
from .ast_analyzers import get_analyzer
from .cache import AnalysisCache

logger = logging.getLogger(__name__)


def build_call_graph(
    repo_path: str,
    use_cache: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build a function-level call graph for an entire repository.

    Args:
        repo_path:  Absolute path to the repo root.
        use_cache:  Whether to use the file-hash cache.

    Returns:
        Dict with ``"nodes"`` and ``"edges"`` lists.
    """
    repo_path = os.path.abspath(repo_path)
    cache = AnalysisCache(repo_path) if use_cache else None

    # Collect raw analysis data per file
    all_definitions: List[Dict] = []
    all_calls: List[Dict] = []

    for abs_path, rel_path in traverse_source_files(repo_path):
        _, ext = os.path.splitext(abs_path)
        analyzer = get_analyzer(ext)
        if analyzer is None:
            continue

        # Try cache first
        cached = cache.get(abs_path) if cache else None
        if cached is not None:
            all_definitions.extend(cached["definitions"])
            all_calls.extend(cached["calls"])
            continue

        # Analyse the file
        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
                source = fh.read()
        except OSError:
            continue

        defs = analyzer.extract_definitions(source, rel_path)
        calls = analyzer.extract_calls(source, rel_path)

        all_definitions.extend(defs)
        all_calls.extend(calls)

        # Persist to cache
        if cache:
            cache.set(abs_path, {"definitions": defs, "calls": calls})

    # Build the graph
    return _assemble_graph(all_definitions, all_calls)


def _assemble_graph(
    definitions: List[Dict],
    calls: List[Dict],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Transform raw definitions and calls into nodes + edges.
    """
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    # --- Nodes from definitions ---
    # node ID = "file::name" (unique per file-scope)
    node_ids: Set[str] = set()
    # Mapping from short name → list of node IDs (for call resolution)
    name_to_ids: Dict[str, List[str]] = {}

    for d in definitions:
        node_id = f"{d['file']}::{d['name']}"
        if node_id in node_ids:
            continue
        node_ids.add(node_id)

        nodes.append({
            "id": node_id,
            "label": d["name"],
            "type": d["type"],
            "file": d["file"],
            "line": d["line"],
        })

        # Index by short name for call resolution
        name_to_ids.setdefault(d["name"], []).append(node_id)

        # "contains" edges: class → method
        if d["type"] == "method" and d.get("parent"):
            parent_id = f"{d['file']}::{d['parent']}"
            edges.append({
                "source": parent_id,
                "target": node_id,
                "type": "contains",
            })

    # --- Edges from calls ---
    for c in calls:
        caller_id = f"{c['file']}::{c['caller']}" if c.get("caller") else None
        if caller_id and caller_id not in node_ids:
            caller_id = None  # caller not in our definitions

        callee_name = c["name"].split(".")[-1]  # use the final segment

        # Resolve callee to a known node
        target_ids = name_to_ids.get(callee_name, [])

        if not target_ids:
            continue  # external / unresolved call — skip

        for tid in target_ids:
            if caller_id and caller_id != tid:
                edges.append({
                    "source": caller_id,
                    "target": tid,
                    "type": "calls",
                })

    # Deduplicate edges
    seen_edges: Set[Tuple[str, str, str]] = set()
    unique_edges: List[Dict[str, Any]] = []
    for e in edges:
        key = (e["source"], e["target"], e["type"])
        if key not in seen_edges:
            seen_edges.add(key)
            unique_edges.append(e)

    logger.info(
        "Call graph built — %d nodes, %d edges.", len(nodes), len(unique_edges)
    )
    return {"nodes": nodes, "edges": unique_edges}
