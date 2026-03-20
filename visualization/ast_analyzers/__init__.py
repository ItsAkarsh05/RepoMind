"""
AST Analyzers
=============
Language-specific source-code analyzers.  Each analyzer extracts
definitions (functions, classes, methods), imports, and function calls
from a single source file.

Use :func:`get_analyzer` to obtain the correct analyzer for a given
file extension.
"""

from .base_analyzer import BaseAnalyzer
from .python_analyzer import PythonAnalyzer
from .javascript_analyzer import JavaScriptAnalyzer
from .dart_analyzer import DartAnalyzer

# Maps file extensions → analyzer classes
_ANALYZER_MAP = {
    ".py": PythonAnalyzer,
    ".js": JavaScriptAnalyzer,
    ".jsx": JavaScriptAnalyzer,
    ".ts": JavaScriptAnalyzer,
    ".tsx": JavaScriptAnalyzer,
    ".mjs": JavaScriptAnalyzer,
    ".cjs": JavaScriptAnalyzer,
    ".dart": DartAnalyzer,
}


def get_analyzer(file_extension: str) -> BaseAnalyzer | None:
    """
    Return an analyzer instance for the given file extension, or
    ``None`` if the language is not yet supported.
    """
    cls = _ANALYZER_MAP.get(file_extension.lower())
    return cls() if cls else None


__all__ = [
    "get_analyzer",
    "BaseAnalyzer",
    "PythonAnalyzer",
    "JavaScriptAnalyzer",
    "DartAnalyzer",
]
