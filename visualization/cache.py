"""
Analysis Cache
==============
Simple file-hash caching so unchanged files are not re-analysed on
repeated requests.

Cache data is stored as a JSON file inside a ``.repomind_cache/``
directory at the repository root.
"""

import hashlib
import json
import os
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

CACHE_DIR_NAME = ".repomind_cache"
CACHE_FILE_NAME = "analysis_cache.json"


class AnalysisCache:
    """
    Persists per-file analysis results keyed by SHA-256 content hash.

    Usage::

        cache = AnalysisCache(repo_path)
        cached = cache.get(file_path)
        if cached is None:
            result = expensive_analysis(file_path)
            cache.set(file_path, result)
    """

    def __init__(self, repo_path: str) -> None:
        self._repo_path = os.path.abspath(repo_path)
        self._cache_dir = os.path.join(self._repo_path, CACHE_DIR_NAME)
        self._cache_file = os.path.join(self._cache_dir, CACHE_FILE_NAME)
        self._data: Dict[str, Dict[str, Any]] = self._load()

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def get(self, file_path: str) -> Optional[Any]:
        """
        Return the cached analysis result for *file_path*, or ``None``
        if the file has changed or was never cached.
        """
        file_path = os.path.abspath(file_path)
        current_hash = self._hash_file(file_path)
        if current_hash is None:
            return None

        entry = self._data.get(file_path)
        if entry and entry.get("hash") == current_hash:
            logger.debug("Cache HIT for '%s'.", file_path)
            return entry.get("result")

        logger.debug("Cache MISS for '%s'.", file_path)
        return None

    def set(self, file_path: str, result: Any) -> None:
        """Store *result* for *file_path* alongside its current hash."""
        file_path = os.path.abspath(file_path)
        current_hash = self._hash_file(file_path)
        if current_hash is None:
            return

        self._data[file_path] = {
            "hash": current_hash,
            "result": result,
        }
        self._save()

    def invalidate(self, file_path: str) -> None:
        """Remove a single file's entry from the cache."""
        file_path = os.path.abspath(file_path)
        self._data.pop(file_path, None)
        self._save()

    def clear(self) -> None:
        """Wipe the entire cache."""
        self._data.clear()
        self._save()
        logger.info("Cache cleared for '%s'.", self._repo_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_file(path: str) -> Optional[str]:
        """Return the SHA-256 hex-digest of a file's contents."""
        try:
            h = hashlib.sha256()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(8192), b""):
                    h.update(chunk)
            return h.hexdigest()
        except OSError:
            return None

    def _load(self) -> Dict[str, Dict[str, Any]]:
        """Load cache from disk; return empty dict on failure."""
        if not os.path.isfile(self._cache_file):
            return {}
        try:
            with open(self._cache_file, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupt cache file — starting fresh.")
            return {}

    def _save(self) -> None:
        """Persist current cache data to disk."""
        os.makedirs(self._cache_dir, exist_ok=True)
        try:
            with open(self._cache_file, "w", encoding="utf-8") as fh:
                json.dump(self._data, fh, indent=2)
        except OSError as exc:
            logger.error("Failed to write cache: %s", exc)
