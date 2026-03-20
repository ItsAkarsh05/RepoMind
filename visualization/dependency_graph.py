"""
Dependency-Graph Builder
========================
Analyses imports across all source files in a repository and builds a
file-level dependency graph.

Output format::

    {
      "nodes": [ { "id", "label", "language" }, … ],
      "edges": [ { "source", "target", "module" }, … ],
    }
"""

import os
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from .shared import traverse_source_files, SUPPORTED_EXTENSIONS
from .ast_analyzers import get_analyzer
from .cache import AnalysisCache

logger = logging.getLogger(__name__)

# Extension → language label mapping
_LANG_MAP: Dict[str, str] = {
    ".py": "python",
    ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".mjs": "javascript", ".cjs": "javascript",
    ".dart": "dart",
}


def build_dependency_graph(
    repo_path: str,
    use_cache: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build a file-level dependency graph for the repository.

    Args:
        repo_path:  Absolute path to the repo root.
        use_cache:  Whether to use the file-hash cache.

    Returns:
        Dict with ``"nodes"`` and ``"edges"`` lists.
    """
    repo_path = os.path.abspath(repo_path)
    cache = AnalysisCache(repo_path) if use_cache else None

    # Collect all relative file paths in the repo (for import resolution)
    file_index: Dict[str, str] = {}   # basename → rel_path
    full_index: Set[str] = set()       # all rel_paths

    all_files: List[Tuple[str, str]] = []   # (abs_path, rel_path)

    for abs_path, rel_path in traverse_source_files(repo_path):
        rel_path_norm = rel_path.replace("\\", "/")
        all_files.append((abs_path, rel_path_norm))
        full_index.add(rel_path_norm)
        base = os.path.basename(rel_path_norm)
        file_index[base] = rel_path_norm
        # Also index without extension
        stem, _ = os.path.splitext(base)
        file_index.setdefault(stem, rel_path_norm)

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    node_ids: Set[str] = set()
    seen_edges: Set[Tuple[str, str]] = set()

    for abs_path, rel_path in all_files:
        _, ext = os.path.splitext(abs_path)
        analyzer = get_analyzer(ext)

        # Add file as a node regardless
        if rel_path not in node_ids:
            node_ids.add(rel_path)
            nodes.append({
                "id": rel_path,
                "label": os.path.basename(rel_path),
                "language": _LANG_MAP.get(ext.lower(), "other"),
            })

        if analyzer is None:
            continue

        # Try cache
        cached_key = f"deps_{abs_path}"
        cached = cache.get(abs_path) if cache else None
        imports_data = None
        if cached and "imports" in cached:
            imports_data = cached["imports"]
        else:
            try:
                with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
                    source = fh.read()
            except OSError:
                continue

            imports_data = analyzer.extract_imports(source, rel_path)

            # Merge with existing cache entry if present
            if cache:
                existing = cache.get(abs_path)
                if existing and isinstance(existing, dict):
                    existing["imports"] = imports_data
                    cache.set(abs_path, existing)
                else:
                    cache.set(abs_path, {"imports": imports_data})

        # Resolve imports to files in the repo
        for imp in imports_data:
            target = _resolve_import(
                imp["module"], rel_path, file_index, full_index
            )
            if target and target != rel_path:
                edge_key = (rel_path, target)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    edges.append({
                        "source": rel_path,
                        "target": target,
                        "module": imp["module"],
                    })
                    # Ensure target node exists
                    if target not in node_ids:
                        node_ids.add(target)
                        _, t_ext = os.path.splitext(target)
                        nodes.append({
                            "id": target,
                            "label": os.path.basename(target),
                            "language": _LANG_MAP.get(t_ext.lower(), "other"),
                        })

    logger.info(
        "Dependency graph built — %d nodes, %d edges.", len(nodes), len(edges)
    )
    return {"nodes": nodes, "edges": edges}


# ---------------------------------------------------------------------------
# Import resolution helpers
# ---------------------------------------------------------------------------

def _resolve_import(
    module: str,
    source_file: str,
    file_index: Dict[str, str],
    full_index: Set[str],
) -> Optional[str]:
    """
    Best-effort resolution of an import string to a file path in the repo.

    Handles:
      - Relative Python imports ( ``.foo`` → sibling ``foo.py`` )
      - Relative JS imports ( ``./utils`` → ``./utils.js`` )
      - Dart package imports ( ``package:xxx/yyy.dart`` )
      - Python 'from X import Y' (module is ``X.Y``, file is ``X.py``)
      - Simple module name lookup by basename
    """
    # Strip leading dots (Python relative imports)
    stripped = module.lstrip(".")
    dots = len(module) - len(stripped)

    # Convert module path to file path candidates
    parts = stripped.replace(".", "/")

    source_dir = os.path.dirname(source_file)

    candidates: List[str] = []

    if dots > 0:
        # Relative import — resolve from source file's directory
        # Go up (dots - 1) levels from source_dir
        base = source_dir
        for _ in range(dots - 1):
            base = os.path.dirname(base)
        candidates.append(f"{base}/{parts}.py" if base else f"{parts}.py")
        candidates.append(f"{base}/{parts}/__init__.py" if base else f"{parts}/__init__.py")

    if module.startswith("./") or module.startswith("../"):
        # JS-style relative import
        rel = os.path.normpath(os.path.join(source_dir, module)).replace("\\", "/")
        candidates.extend([
            rel,
            f"{rel}.js", f"{rel}.ts", f"{rel}.jsx", f"{rel}.tsx",
            f"{rel}/index.js", f"{rel}/index.ts",
        ])

    # Direct path match (full module path)
    candidates.extend([
        f"{parts}.py",
        f"{parts}.js", f"{parts}.ts",
        f"{parts}.dart",
        f"{parts}/__init__.py",
        f"{parts}/index.js", f"{parts}/index.ts",
    ])

    # For 'from X import Y' patterns the Python analyzer stores 'X.Y' as
    # the module string.  The actual *file* is X.py (or X/__init__.py),
    # not X/Y.py.  Generate parent-module candidates by progressively
    # stripping the last path component.
    segments = parts.split("/")
    for i in range(len(segments) - 1, 0, -1):
        parent = "/".join(segments[:i])
        candidates.extend([
            f"{parent}.py",
            f"{parent}/__init__.py",
            f"{parent}.js", f"{parent}.ts",
            f"{parent}.dart",
        ])

    # Check candidates against repo index
    for c in candidates:
        c_norm = c.replace("\\", "/")
        if c_norm in full_index:
            return c_norm

    # Fallback: try basename match (first segment, then last segment)
    for segment in [segments[0], segments[-1]]:
        if segment in file_index:
            return file_index[segment]

    return None

