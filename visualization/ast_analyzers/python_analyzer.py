"""
Python Analyzer
================
Uses the built-in ``ast`` module to extract function/class definitions,
imports, and function call sites from Python source files.
"""

import ast
import logging
from typing import Dict, List, Optional

from .base_analyzer import BaseAnalyzer

logger = logging.getLogger(__name__)


class PythonAnalyzer(BaseAnalyzer):
    """AST-based analyzer for ``.py`` files."""

    # ------------------------------------------------------------------
    # Definitions
    # ------------------------------------------------------------------

    def extract_definitions(
        self, source: str, file_path: str
    ) -> List[Dict]:
        """
        Extract all function, async-function, and class definitions,
        including methods nested inside classes.
        """
        try:
            tree = ast.parse(source)
        except SyntaxError:
            logger.warning("SyntaxError in '%s' — skipping definitions.", file_path)
            return []

        definitions: List[Dict] = []
        self._walk_definitions(tree, file_path, parent=None, results=definitions)
        return definitions

    def _walk_definitions(
        self,
        node: ast.AST,
        file_path: str,
        parent: Optional[str],
        results: List[Dict],
    ) -> None:
        """Recursively walk the AST to collect definitions."""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = "method" if parent else "function"
                results.append({
                    "name": child.name,
                    "type": kind,
                    "line": child.lineno,
                    "end_line": child.end_lineno,
                    "file": file_path,
                    "parent": parent,
                })
                # Check for nested functions inside this function
                self._walk_definitions(child, file_path, parent=child.name, results=results)

            elif isinstance(child, ast.ClassDef):
                results.append({
                    "name": child.name,
                    "type": "class",
                    "line": child.lineno,
                    "end_line": child.end_lineno,
                    "file": file_path,
                    "parent": parent,
                })
                # Walk class body for methods
                self._walk_definitions(child, file_path, parent=child.name, results=results)

    # ------------------------------------------------------------------
    # Imports
    # ------------------------------------------------------------------

    def extract_imports(
        self, source: str, file_path: str
    ) -> List[Dict]:
        """Extract ``import`` and ``from … import`` statements."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            logger.warning("SyntaxError in '%s' — skipping imports.", file_path)
            return []

        imports: List[Dict] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        "module": alias.name,
                        "alias": alias.asname,
                        "line": node.lineno,
                        "file": file_path,
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                # Prefix with dots for relative imports
                prefix = "." * (node.level or 0)
                for alias in node.names:
                    imports.append({
                        "module": f"{prefix}{module}.{alias.name}" if module else f"{prefix}{alias.name}",
                        "alias": alias.asname,
                        "line": node.lineno,
                        "file": file_path,
                    })

        return imports

    # ------------------------------------------------------------------
    # Calls
    # ------------------------------------------------------------------

    def extract_calls(
        self, source: str, file_path: str
    ) -> List[Dict]:
        """
        Extract function/method call-sites.  Each call is annotated with
        the name of the enclosing function (``caller``) when available.
        """
        try:
            tree = ast.parse(source)
        except SyntaxError:
            logger.warning("SyntaxError in '%s' — skipping calls.", file_path)
            return []

        calls: List[Dict] = []
        self._walk_calls(tree, file_path, caller=None, results=calls)
        return calls

    def _walk_calls(
        self,
        node: ast.AST,
        file_path: str,
        caller: Optional[str],
        results: List[Dict],
    ) -> None:
        """Recursively walk the AST to find ``Call`` nodes."""
        for child in ast.iter_child_nodes(node):
            # Track which function/method we're currently inside
            current_caller = caller
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                current_caller = child.name
            elif isinstance(child, ast.ClassDef):
                # Continue walking inside classes but keep the class name
                # as context (methods inside will override current_caller)
                self._walk_calls(child, file_path, caller=child.name, results=results)
                continue

            if isinstance(child, ast.Call):
                callee = self._resolve_call_name(child.func)
                if callee:
                    results.append({
                        "name": callee,
                        "caller": current_caller,
                        "line": child.lineno,
                        "file": file_path,
                    })

            self._walk_calls(child, file_path, caller=current_caller, results=results)

    @staticmethod
    def _resolve_call_name(func_node: ast.AST) -> Optional[str]:
        """
        Best-effort resolution of a Call node's function name.

        Handles:
          - ``foo()``              → "foo"
          - ``obj.method()``       → "obj.method"
          - ``a.b.c()``            → "a.b.c"
        """
        if isinstance(func_node, ast.Name):
            return func_node.id
        if isinstance(func_node, ast.Attribute):
            parts = []
            node = func_node
            while isinstance(node, ast.Attribute):
                parts.append(node.attr)
                node = node.value
            if isinstance(node, ast.Name):
                parts.append(node.id)
            return ".".join(reversed(parts)) if parts else None
        return None
