"""
RepoMind API
============
Flask Blueprints exposing:

**Visualization**

- ``GET  /repo/structure``     → hierarchical file tree
- ``GET  /repo/call-graph``    → function call graph
- ``GET  /repo/dependencies``  → file dependency graph

**Chat Memory**

- ``POST /chat/reset``         → clears chat history
- ``GET  /chat/history``       → returns chat history

All endpoints return JSON.

Run standalone::

    python -m visualization.api          # starts on http://localhost:5000
"""

import os
import logging

from flask import Blueprint, Flask, jsonify, request

from .repo_structure import get_repo_structure
from .call_graph import build_call_graph
from .dependency_graph import build_dependency_graph

# ChatMemory lives outside the visualization package, so we use a
# sys.path-friendly import (the project root is on sys.path at runtime).
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from memory import ChatMemory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global session memory  (simple single-session version)
# ---------------------------------------------------------------------------
# For now a single global instance is used.  To support multiple isolated
# sessions later, swap this for a dict keyed by session / user ID.

_chat_memory = ChatMemory()


def get_chat_memory() -> ChatMemory:
    """Return the global chat memory instance."""
    return _chat_memory


# ---------------------------------------------------------------------------
# Visualization Blueprint
# ---------------------------------------------------------------------------

viz_bp = Blueprint("visualization", __name__)


def _get_repo_path() -> str:
    """Extract and validate the ``repo_path`` query parameter."""
    repo_path = request.args.get("repo_path", "").strip()
    if not repo_path:
        raise ValueError("Missing required query parameter: repo_path")
    repo_path = os.path.abspath(repo_path)
    if not os.path.isdir(repo_path):
        raise FileNotFoundError(f"Repository path does not exist: {repo_path}")
    return repo_path


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@viz_bp.route("/repo/structure", methods=["GET"])
def repo_structure():
    """Return the hierarchical file / folder tree of the repository."""
    try:
        repo_path = _get_repo_path()
        tree = get_repo_structure(repo_path)
        return jsonify(tree), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        logger.exception("Unexpected error in /repo/structure")
        return jsonify({"error": f"Internal server error: {exc}"}), 500


@viz_bp.route("/repo/call-graph", methods=["GET"])
def repo_call_graph():
    """Return the function-level call graph of the repository."""
    try:
        repo_path = _get_repo_path()
        graph = build_call_graph(repo_path)
        return jsonify(graph), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        logger.exception("Unexpected error in /repo/call-graph")
        return jsonify({"error": f"Internal server error: {exc}"}), 500


@viz_bp.route("/repo/dependencies", methods=["GET"])
def repo_dependencies():
    """Return the file-level dependency graph of the repository."""
    try:
        repo_path = _get_repo_path()
        graph = build_dependency_graph(repo_path)
        return jsonify(graph), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        logger.exception("Unexpected error in /repo/dependencies")
        return jsonify({"error": f"Internal server error: {exc}"}), 500


# ---------------------------------------------------------------------------
# Chat Memory Blueprint
# ---------------------------------------------------------------------------

chat_bp = Blueprint("chat", __name__)


@chat_bp.route("/chat/history", methods=["GET"])
def chat_history():
    """
    Return the full chat history for the current session.

    Response::

        {
          "history": [ {"role": "user", "content": "..."}, ... ],
          "message_count": 5
        }
    """
    memory = get_chat_memory()
    history = memory.get_history()
    count = memory.length

    logger.info("GET /chat/history — returning %d messages.", count)

    return jsonify({
        "history": history,
        "message_count": count,
    }), 200


@chat_bp.route("/chat/reset", methods=["POST"])
def chat_reset():
    """
    Clear the chat history for the current session.

    Response::

        { "status": "cleared", "message_count": 0 }
    """
    memory = get_chat_memory()
    prev_count = memory.length
    memory.clear_history()

    logger.info(
        "POST /chat/reset — cleared %d messages, history is now empty.",
        prev_count,
    )

    return jsonify({
        "status": "cleared",
        "message_count": 0,
    }), 200


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def create_app() -> Flask:
    """Create a minimal Flask application with all blueprints."""
    app = Flask(__name__)
    app.register_blueprint(viz_bp)
    app.register_blueprint(chat_bp)

    @app.route("/", methods=["GET"])
    def index():
        return jsonify({
            "message": "Welcome to the RepoMind API!",
            "endpoints": [
                "GET  /repo/structure?repo_path=<path>",
                "GET  /repo/call-graph?repo_path=<path>",
                "GET  /repo/dependencies?repo_path=<path>",
                "GET  /chat/history",
                "POST /chat/reset",
            ]
        }), 200

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = create_app()
    print("Starting RepoMind API on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
