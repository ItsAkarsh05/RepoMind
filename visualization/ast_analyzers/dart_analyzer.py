"""
Dart Analyzer
=============
Regex-based analyzer for ``.dart`` files.

Extracts function/method definitions, class declarations, import
statements, and function call sites.
"""

import re
import logging
from typing import Dict, List

from .base_analyzer import BaseAnalyzer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

# Top-level or static function:  ReturnType functionName(  or  void main(
_RE_FUNC = re.compile(
    r"^\s*(?:static\s+)?(?:Future<[^>]+>|[\w<>,\s\?]+)\s+(\w+)\s*\(",
    re.MULTILINE,
)

# Class declarations:  class Foo { | class Foo extends Bar with Mixin {
_RE_CLASS = re.compile(
    r"^\s*(?:abstract\s+)?class\s+(\w+)",
    re.MULTILINE,
)

# Imports: import 'package:xxx'; | import 'relative.dart';
_RE_IMPORT = re.compile(
    r"""^\s*import\s+['"]([^'"]+)['"]\s*;""",
    re.MULTILINE,
)

# Part directives (treated as a form of dependency)
_RE_PART = re.compile(
    r"""^\s*part\s+['"]([^'"]+)['"]\s*;""",
    re.MULTILINE,
)

# Function calls: identifier( — but not control-flow keywords
_RE_CALL = re.compile(
    r"\b([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*)\s*\(",
)

_DART_KEYWORDS = {
    "if", "else", "for", "while", "do", "switch", "case", "catch",
    "try", "finally", "throw", "return", "new", "class", "void",
    "import", "export", "part", "library", "show", "hide", "as",
    "abstract", "extends", "implements", "with", "mixin", "enum",
    "typedef", "is", "in", "assert", "super", "this", "static",
    "final", "const", "var", "late", "required", "async", "await",
    "yield", "sync", "on", "get", "set", "factory", "operator",
}


class DartAnalyzer(BaseAnalyzer):
    """Regex-based analyzer for Dart source files."""

    # ------------------------------------------------------------------
    # Definitions
    # ------------------------------------------------------------------

    def extract_definitions(
        self, source: str, file_path: str
    ) -> List[Dict]:
        definitions: List[Dict] = []
        lines = source.split("\n")

        current_class = None
        brace_depth = 0
        class_start_depth = 0

        for lineno, line in enumerate(lines, start=1):
            brace_depth += line.count("{") - line.count("}")

            # Detect class end
            if current_class and brace_depth <= class_start_depth:
                current_class = None

            # Class
            m = _RE_CLASS.match(line)
            if m:
                current_class = m.group(1)
                class_start_depth = brace_depth - line.count("{")
                definitions.append({
                    "name": m.group(1),
                    "type": "class",
                    "line": lineno,
                    "file": file_path,
                    "parent": None,
                })
                continue

            # Function / method
            m = _RE_FUNC.match(line)
            if m:
                name = m.group(1)
                if name in _DART_KEYWORDS:
                    continue
                definitions.append({
                    "name": name,
                    "type": "method" if current_class else "function",
                    "line": lineno,
                    "file": file_path,
                    "parent": current_class,
                })

        return definitions

    # ------------------------------------------------------------------
    # Imports
    # ------------------------------------------------------------------

    def extract_imports(
        self, source: str, file_path: str
    ) -> List[Dict]:
        imports: List[Dict] = []
        lines = source.split("\n")

        for lineno, line in enumerate(lines, start=1):
            m = _RE_IMPORT.match(line)
            if m:
                imports.append({
                    "module": m.group(1),
                    "alias": None,
                    "line": lineno,
                    "file": file_path,
                })
                continue

            m = _RE_PART.match(line)
            if m:
                imports.append({
                    "module": m.group(1),
                    "alias": None,
                    "line": lineno,
                    "file": file_path,
                })

        return imports

    # ------------------------------------------------------------------
    # Calls
    # ------------------------------------------------------------------

    def extract_calls(
        self, source: str, file_path: str
    ) -> List[Dict]:
        calls: List[Dict] = []
        lines = source.split("\n")

        current_func = None

        for lineno, line in enumerate(lines, start=1):
            # Update enclosing function context
            m = _RE_FUNC.match(line)
            if m and m.group(1) not in _DART_KEYWORDS:
                current_func = m.group(1)

            for m in _RE_CALL.finditer(line):
                name = m.group(1)
                base = name.split(".")[0]
                if base not in _DART_KEYWORDS:
                    calls.append({
                        "name": name,
                        "caller": current_func,
                        "line": lineno,
                        "file": file_path,
                    })

        return calls
