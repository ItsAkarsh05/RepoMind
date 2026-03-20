"""
JavaScript / TypeScript Analyzer
=================================
Regex-based analyzer for ``.js``, ``.jsx``, ``.ts``, ``.tsx``, ``.mjs``,
and ``.cjs`` files.

We intentionally avoid heavyweight JS AST libraries (like ``esprima``)
to keep the dependency footprint minimal.  The regex patterns cover the
most common declaration and import styles.
"""

import re
import logging
from typing import Dict, List

from .base_analyzer import BaseAnalyzer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

# Function declarations: function foo(...)
_RE_FUNC_DECL = re.compile(
    r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(",
    re.MULTILINE,
)

# Arrow / expression functions: const foo = (...) => | const foo = function(
_RE_ARROW_FUNC = re.compile(
    r"^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)|[\w]+)\s*=>",
    re.MULTILINE,
)
_RE_EXPR_FUNC = re.compile(
    r"^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?function\s*\(",
    re.MULTILINE,
)

# Class declarations: class Foo { | export class Foo extends Bar {
_RE_CLASS = re.compile(
    r"^\s*(?:export\s+)?class\s+(\w+)",
    re.MULTILINE,
)

# Class methods: methodName(...) { or async methodName(...)
_RE_METHOD = re.compile(
    r"^\s+(?:async\s+)?(\w+)\s*\([^)]*\)\s*\{",
    re.MULTILINE,
)

# Import statements: import ... from '...' | require('...')
_RE_IMPORT_FROM = re.compile(
    r"""^\s*import\s+.*?\s+from\s+['"]([^'"]+)['"]""",
    re.MULTILINE,
)
_RE_IMPORT_SIDE = re.compile(
    r"""^\s*import\s+['"]([^'"]+)['"]""",
    re.MULTILINE,
)
_RE_REQUIRE = re.compile(
    r"""require\s*\(\s*['"]([^'"]+)['"]\s*\)""",
)

# Function calls: identifier( but NOT keywords
_RE_CALL = re.compile(
    r"\b([a-zA-Z_$][\w$]*(?:\.[a-zA-Z_$][\w$]*)*)\s*\(",
)
_JS_KEYWORDS = {
    "if", "else", "for", "while", "do", "switch", "case", "catch",
    "try", "finally", "throw", "return", "new", "typeof", "instanceof",
    "void", "delete", "in", "of", "class", "function", "import", "export",
    "default", "from", "async", "await", "yield", "const", "let", "var",
    "super", "this", "extends", "implements", "static", "get", "set",
}


class JavaScriptAnalyzer(BaseAnalyzer):
    """Regex-based analyzer for JavaScript and TypeScript files."""

    # ------------------------------------------------------------------
    # Definitions
    # ------------------------------------------------------------------

    def extract_definitions(
        self, source: str, file_path: str
    ) -> List[Dict]:
        definitions: List[Dict] = []
        lines = source.split("\n")

        # Track which class body we're inside (simplistic brace counting)
        current_class = None
        brace_depth = 0
        class_start_depth = 0

        for lineno, line in enumerate(lines, start=1):
            # Count braces for class scope tracking
            brace_depth += line.count("{") - line.count("}")

            # Detect class end
            if current_class and brace_depth <= class_start_depth:
                current_class = None

            # Class declaration
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

            # Function declarations
            m = _RE_FUNC_DECL.match(line)
            if m:
                definitions.append({
                    "name": m.group(1),
                    "type": "method" if current_class else "function",
                    "line": lineno,
                    "file": file_path,
                    "parent": current_class,
                })
                continue

            # Arrow / expression functions
            m = _RE_ARROW_FUNC.match(line) or _RE_EXPR_FUNC.match(line)
            if m:
                definitions.append({
                    "name": m.group(1),
                    "type": "function",
                    "line": lineno,
                    "file": file_path,
                    "parent": current_class,
                })
                continue

            # Methods inside a class body
            if current_class:
                m = _RE_METHOD.match(line)
                if m and m.group(1) not in _JS_KEYWORDS:
                    definitions.append({
                        "name": m.group(1),
                        "type": "method",
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
            for pattern in (_RE_IMPORT_FROM, _RE_IMPORT_SIDE):
                m = pattern.match(line)
                if m:
                    imports.append({
                        "module": m.group(1),
                        "alias": None,
                        "line": lineno,
                        "file": file_path,
                    })
                    break
            else:
                # Check for require()
                for m in _RE_REQUIRE.finditer(line):
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

        # Simple heuristic: track the nearest enclosing function
        current_func = None

        for lineno, line in enumerate(lines, start=1):
            # Update enclosing function context
            m = _RE_FUNC_DECL.match(line)
            if m:
                current_func = m.group(1)
            m = _RE_ARROW_FUNC.match(line) or _RE_EXPR_FUNC.match(line)
            if m:
                current_func = m.group(1)

            # Find all calls on this line
            for m in _RE_CALL.finditer(line):
                name = m.group(1)
                # Filter out keywords and common false positives
                base_name = name.split(".")[0]
                if base_name not in _JS_KEYWORDS:
                    calls.append({
                        "name": name,
                        "caller": current_func,
                        "line": lineno,
                        "file": file_path,
                    })

        return calls
