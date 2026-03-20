"""
Chat Memory
============
In-memory conversation history for the RepoMind chatbot.

Stores messages as a list of dicts with ``role`` and ``content`` keys,
following the standard chat-message format used by most LLM APIs.

Designed for easy extension — swap the in-memory list with Redis, SQLite,
or any persistent backend by subclassing :class:`ChatMemory` and
overriding the storage methods.

Usage::

    from memory import ChatMemory

    memory = ChatMemory()
    memory.add_user_message("What does this function do?")
    memory.add_assistant_message("It calculates the factorial of n.")
    print(memory.get_history())
"""

from typing import Dict, List, Literal


class ChatMemory:
    """
    Simple in-memory conversation history.

    Each message is stored as::

        {"role": "user" | "assistant", "content": "..."}

    To persist history to a database later, subclass this and override
    :meth:`_append`, :meth:`get_history`, and :meth:`clear_history`.
    """

    def __init__(self, max_messages: int | None = None) -> None:
        """
        Args:
            max_messages: Optional cap on stored messages.  When set, the
                          oldest messages are dropped once the limit is
                          reached.  ``None`` means unlimited.
        """
        self._history: List[Dict[str, str]] = []
        self._max_messages = max_messages

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_user_message(self, message: str) -> None:
        """Record a user message."""
        self._append({"role": "user", "content": message})

    def add_assistant_message(self, message: str) -> None:
        """Record an assistant (AI) response."""
        self._append({"role": "assistant", "content": message})

    def get_history(self) -> List[Dict[str, str]]:
        """Return the full conversation history (oldest → newest)."""
        return list(self._history)          # return a copy for safety

    def clear_history(self) -> None:
        """Erase all stored messages."""
        self._history.clear()

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def length(self) -> int:
        """Number of messages currently stored."""
        return len(self._history)

    def get_last_n(self, n: int) -> List[Dict[str, str]]:
        """Return the *n* most recent messages."""
        return list(self._history[-n:])

    def to_prompt_string(self, separator: str = "\n") -> str:
        """
        Format history as a single string suitable for injecting into
        an LLM prompt.

        Example output::

            User: Hello
            Assistant: Hi! How can I help?
        """
        lines = []
        for msg in self._history:
            role = msg["role"].capitalize()
            lines.append(f"{role}: {msg['content']}")
        return separator.join(lines)

    # ------------------------------------------------------------------
    # Internal (override these for persistent storage)
    # ------------------------------------------------------------------

    def _append(self, message: Dict[str, str]) -> None:
        """
        Append a message to the history.

        Override this method in a subclass to persist messages to a
        database (e.g. Redis, SQLite, PostgreSQL).
        """
        self._history.append(message)

        # Enforce max_messages cap if set
        if self._max_messages and len(self._history) > self._max_messages:
            self._history = self._history[-self._max_messages:]
