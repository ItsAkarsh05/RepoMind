"""
Base Analyzer
=============
Abstract base class that every language-specific analyzer must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List


class BaseAnalyzer(ABC):
    """
    Contract for source-code analyzers.

    Each method receives the raw source text and the file path (for
    metadata), and returns a list of dicts describing the extracted items.
    """

    # ------------------------------------------------------------------
    # Definitions: functions, classes, methods
    # ------------------------------------------------------------------

    @abstractmethod
    def extract_definitions(
        self, source: str, file_path: str
    ) -> List[Dict]:
        """
        Extract function, class, and method definitions.

        Each dict should contain at least::

            {
                "name": str,
                "type": "function" | "class" | "method",
                "line": int,
                "file": str,           # relative or absolute path
                "parent": str | None,  # enclosing class name (for methods)
            }
        """

    # ------------------------------------------------------------------
    # Imports
    # ------------------------------------------------------------------

    @abstractmethod
    def extract_imports(
        self, source: str, file_path: str
    ) -> List[Dict]:
        """
        Extract import / include statements.

        Each dict should contain at least::

            {
                "module": str,     # what is being imported
                "alias": str | None,
                "line": int,
                "file": str,
            }
        """

    # ------------------------------------------------------------------
    # Function / method calls
    # ------------------------------------------------------------------

    @abstractmethod
    def extract_calls(
        self, source: str, file_path: str
    ) -> List[Dict]:
        """
        Extract function and method call-sites.

        Each dict should contain at least::

            {
                "name": str,       # callee name
                "caller": str | None,  # enclosing function/method
                "line": int,
                "file": str,
            }
        """
