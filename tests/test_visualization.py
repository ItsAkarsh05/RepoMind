"""
Tests for the visualization package
=====================================
Run with:  python -m pytest tests/test_visualization.py -v

Creates temporary mock repository structures on disk and validates
every module's output.
"""

import os
import sys
import json

import pytest

# Ensure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from visualization.repo_structure import get_repo_structure
from visualization.cache import AnalysisCache
from visualization.ast_analyzers.python_analyzer import PythonAnalyzer
from visualization.ast_analyzers.javascript_analyzer import JavaScriptAnalyzer
from visualization.ast_analyzers.dart_analyzer import DartAnalyzer
from visualization.ast_analyzers import get_analyzer
from visualization.call_graph import build_call_graph
from visualization.dependency_graph import build_dependency_graph
from visualization.api import create_app


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def mock_repo(tmp_path):
    """
    Create a small mock repository with Python, JS, and Dart files,
    plus directories that should be ignored.
    """
    # Python files
    (tmp_path / "main.py").write_text(
        'from utils import helper\n\n'
        'def main():\n'
        '    result = helper()\n'
        '    print(result)\n\n'
        'class App:\n'
        '    def run(self):\n'
        '        main()\n',
        encoding="utf-8",
    )
    (tmp_path / "utils.py").write_text(
        'import os\n\n'
        'def helper():\n'
        '    return os.getcwd()\n\n'
        'def unused():\n'
        '    pass\n',
        encoding="utf-8",
    )

    # JS files
    src = tmp_path / "src"
    src.mkdir()
    (src / "index.js").write_text(
        "import { greet } from './greet';\n\n"
        "function main() {\n"
        "  greet('world');\n"
        "}\n",
        encoding="utf-8",
    )
    (src / "greet.js").write_text(
        "export function greet(name) {\n"
        "  console.log(`Hello, ${name}!`);\n"
        "}\n",
        encoding="utf-8",
    )

    # Dart files
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "app.dart").write_text(
        "import 'package:flutter/material.dart';\n"
        "import 'widget.dart';\n\n"
        "class MyApp extends StatelessWidget {\n"
        "  Widget build(BuildContext context) {\n"
        "    return createWidget();\n"
        "  }\n"
        "}\n",
        encoding="utf-8",
    )
    (lib / "widget.dart").write_text(
        "import 'package:flutter/material.dart';\n\n"
        "Widget createWidget() {\n"
        "  return Container();\n"
        "}\n",
        encoding="utf-8",
    )

    # Ignored directories
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("gitconfig")

    nm = tmp_path / "node_modules"
    nm.mkdir()
    (nm / "lib.js").write_text("module.exports = {};")

    return tmp_path


# =========================================================================
# 1. Repository Structure Tests
# =========================================================================

class TestRepoStructure:

    def test_basic_tree(self, mock_repo):
        tree = get_repo_structure(str(mock_repo))
        assert tree["type"] == "directory"
        assert tree["path"] == "."
        assert isinstance(tree["children"], list)

    def test_excludes_git(self, mock_repo):
        tree = get_repo_structure(str(mock_repo))
        child_names = [c["name"] for c in tree["children"]]
        assert ".git" not in child_names

    def test_excludes_node_modules(self, mock_repo):
        tree = get_repo_structure(str(mock_repo))
        child_names = [c["name"] for c in tree["children"]]
        assert "node_modules" not in child_names

    def test_includes_source_files(self, mock_repo):
        tree = get_repo_structure(str(mock_repo))
        file_names = [c["name"] for c in tree["children"] if c["type"] == "file"]
        assert "main.py" in file_names
        assert "utils.py" in file_names

    def test_nested_directories(self, mock_repo):
        tree = get_repo_structure(str(mock_repo))
        dir_names = [c["name"] for c in tree["children"] if c["type"] == "directory"]
        assert "src" in dir_names
        assert "lib" in dir_names

    def test_file_has_extension_and_size(self, mock_repo):
        tree = get_repo_structure(str(mock_repo))
        files = [c for c in tree["children"] if c["type"] == "file"]
        for f in files:
            assert "extension" in f
            assert "size_bytes" in f
            assert isinstance(f["size_bytes"], int)

    def test_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            get_repo_structure("/nonexistent/path")


# =========================================================================
# 2. Python Analyzer Tests
# =========================================================================

class TestPythonAnalyzer:

    def setup_method(self):
        self.analyzer = PythonAnalyzer()

    def test_extract_functions(self):
        source = (
            "def foo():\n    pass\n\n"
            "async def bar():\n    pass\n"
        )
        defs = self.analyzer.extract_definitions(source, "test.py")
        names = [d["name"] for d in defs]
        assert "foo" in names
        assert "bar" in names
        assert all(d["type"] == "function" for d in defs if d["name"] in ("foo", "bar"))

    def test_extract_class_and_methods(self):
        source = (
            "class MyClass:\n"
            "    def method_a(self):\n"
            "        pass\n"
            "    def method_b(self):\n"
            "        pass\n"
        )
        defs = self.analyzer.extract_definitions(source, "test.py")
        class_defs = [d for d in defs if d["type"] == "class"]
        method_defs = [d for d in defs if d["type"] == "method"]
        assert len(class_defs) == 1
        assert class_defs[0]["name"] == "MyClass"
        assert len(method_defs) == 2
        assert all(m["parent"] == "MyClass" for m in method_defs)

    def test_extract_imports(self):
        source = "import os\nfrom sys import path\nfrom . import utils\n"
        imports = self.analyzer.extract_imports(source, "test.py")
        modules = [i["module"] for i in imports]
        assert "os" in modules
        assert "sys.path" in modules
        assert ".utils" in modules

    def test_extract_calls(self):
        source = (
            "def main():\n"
            "    result = helper()\n"
            "    os.path.join('a', 'b')\n"
        )
        calls = self.analyzer.extract_calls(source, "test.py")
        names = [c["name"] for c in calls]
        assert "helper" in names
        assert "os.path.join" in names
        assert all(c["caller"] == "main" for c in calls)

    def test_syntax_error_returns_empty(self):
        source = "def broken(:\n  pass"
        defs = self.analyzer.extract_definitions(source, "bad.py")
        assert defs == []


# =========================================================================
# 3. JavaScript Analyzer Tests
# =========================================================================

class TestJavaScriptAnalyzer:

    def setup_method(self):
        self.analyzer = JavaScriptAnalyzer()

    def test_extract_function_declaration(self):
        source = "function hello() {\n  return 'hi';\n}\n"
        defs = self.analyzer.extract_definitions(source, "test.js")
        names = [d["name"] for d in defs]
        assert "hello" in names

    def test_extract_arrow_function(self):
        source = "const greet = (name) => {\n  console.log(name);\n};\n"
        defs = self.analyzer.extract_definitions(source, "test.js")
        names = [d["name"] for d in defs]
        assert "greet" in names

    def test_extract_class(self):
        source = "class MyComponent {\n  render() {\n    return null;\n  }\n}\n"
        defs = self.analyzer.extract_definitions(source, "test.js")
        class_defs = [d for d in defs if d["type"] == "class"]
        method_defs = [d for d in defs if d["type"] == "method"]
        assert len(class_defs) == 1
        assert class_defs[0]["name"] == "MyComponent"
        assert any(m["name"] == "render" for m in method_defs)

    def test_extract_imports(self):
        source = (
            "import React from 'react';\n"
            "import { useState } from 'react';\n"
            "const fs = require('fs');\n"
        )
        imports = self.analyzer.extract_imports(source, "test.js")
        modules = [i["module"] for i in imports]
        assert "react" in modules
        assert "fs" in modules

    def test_extract_calls(self):
        source = (
            "function main() {\n"
            "  greet('world');\n"
            "  console.log('done');\n"
            "}\n"
        )
        calls = self.analyzer.extract_calls(source, "test.js")
        names = [c["name"] for c in calls]
        assert "greet" in names
        assert "console.log" in names


# =========================================================================
# 4. Dart Analyzer Tests
# =========================================================================

class TestDartAnalyzer:

    def setup_method(self):
        self.analyzer = DartAnalyzer()

    def test_extract_function(self):
        source = "void main() {\n  print('hello');\n}\n"
        defs = self.analyzer.extract_definitions(source, "test.dart")
        names = [d["name"] for d in defs]
        assert "main" in names

    def test_extract_class(self):
        source = (
            "class MyWidget extends StatelessWidget {\n"
            "  Widget build(BuildContext context) {\n"
            "    return Container();\n"
            "  }\n"
            "}\n"
        )
        defs = self.analyzer.extract_definitions(source, "test.dart")
        class_defs = [d for d in defs if d["type"] == "class"]
        method_defs = [d for d in defs if d["type"] == "method"]
        assert len(class_defs) >= 1
        assert class_defs[0]["name"] == "MyWidget"
        assert any(m["name"] == "build" for m in method_defs)

    def test_extract_imports(self):
        source = (
            "import 'package:flutter/material.dart';\n"
            "import 'utils.dart';\n"
            "part 'generated.g.dart';\n"
        )
        imports = self.analyzer.extract_imports(source, "test.dart")
        modules = [i["module"] for i in imports]
        assert "package:flutter/material.dart" in modules
        assert "utils.dart" in modules
        assert "generated.g.dart" in modules


# =========================================================================
# 5. Analyzer Dispatcher Tests
# =========================================================================

class TestAnalyzerDispatcher:

    def test_python_ext(self):
        a = get_analyzer(".py")
        assert isinstance(a, PythonAnalyzer)

    def test_js_ext(self):
        a = get_analyzer(".js")
        assert isinstance(a, JavaScriptAnalyzer)

    def test_ts_ext(self):
        a = get_analyzer(".tsx")
        assert isinstance(a, JavaScriptAnalyzer)

    def test_dart_ext(self):
        a = get_analyzer(".dart")
        assert isinstance(a, DartAnalyzer)

    def test_unsupported_ext(self):
        assert get_analyzer(".rb") is None
        assert get_analyzer(".go") is None


# =========================================================================
# 6. Call Graph Tests
# =========================================================================

class TestCallGraph:

    def test_builds_graph(self, mock_repo):
        graph = build_call_graph(str(mock_repo), use_cache=False)
        assert "nodes" in graph
        assert "edges" in graph
        assert isinstance(graph["nodes"], list)
        assert isinstance(graph["edges"], list)

    def test_contains_expected_nodes(self, mock_repo):
        graph = build_call_graph(str(mock_repo), use_cache=False)
        labels = {n["label"] for n in graph["nodes"]}
        # Should find Python functions from main.py and utils.py
        assert "main" in labels
        assert "helper" in labels

    def test_contains_call_edges(self, mock_repo):
        graph = build_call_graph(str(mock_repo), use_cache=False)
        call_edges = [e for e in graph["edges"] if e["type"] == "calls"]
        # main() calls helper()
        assert len(call_edges) > 0

    def test_contains_containment_edges(self, mock_repo):
        graph = build_call_graph(str(mock_repo), use_cache=False)
        contains_edges = [e for e in graph["edges"] if e["type"] == "contains"]
        # App class contains run method
        assert len(contains_edges) > 0


# =========================================================================
# 7. Dependency Graph Tests
# =========================================================================

class TestDependencyGraph:

    def test_builds_graph(self, mock_repo):
        graph = build_dependency_graph(str(mock_repo), use_cache=False)
        assert "nodes" in graph
        assert "edges" in graph

    def test_nodes_have_language(self, mock_repo):
        graph = build_dependency_graph(str(mock_repo), use_cache=False)
        for node in graph["nodes"]:
            assert "language" in node
            assert node["language"] in ("python", "javascript", "dart", "other")

    def test_python_import_edge(self, mock_repo):
        graph = build_dependency_graph(str(mock_repo), use_cache=False)
        # main.py has `from utils import helper` — the resolver should
        # find utils.py and create an edge.  Verify at least one edge
        # has a Python file as the target.
        py_edges = [
            e for e in graph["edges"]
            if e["target"].endswith(".py")
        ]
        assert len(py_edges) > 0


# =========================================================================
# 8. Cache Tests
# =========================================================================

class TestCache:

    def test_set_and_get(self, tmp_path):
        # Create a test file
        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1")

        cache = AnalysisCache(str(tmp_path))
        cache.set(str(test_file), {"foo": "bar"})

        result = cache.get(str(test_file))
        assert result == {"foo": "bar"}

    def test_miss_on_changed_file(self, tmp_path):
        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1")

        cache = AnalysisCache(str(tmp_path))
        cache.set(str(test_file), {"foo": "bar"})

        # Modify the file
        test_file.write_text("x = 2")

        result = cache.get(str(test_file))
        assert result is None

    def test_miss_on_nonexistent(self, tmp_path):
        cache = AnalysisCache(str(tmp_path))
        result = cache.get(str(tmp_path / "nonexistent.py"))
        assert result is None

    def test_invalidate(self, tmp_path):
        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1")

        cache = AnalysisCache(str(tmp_path))
        cache.set(str(test_file), {"foo": "bar"})
        cache.invalidate(str(test_file))

        assert cache.get(str(test_file)) is None

    def test_clear(self, tmp_path):
        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1")

        cache = AnalysisCache(str(tmp_path))
        cache.set(str(test_file), {"foo": "bar"})
        cache.clear()

        assert cache.get(str(test_file)) is None


# =========================================================================
# 9. API Endpoint Tests
# =========================================================================

class TestAPI:

    @pytest.fixture
    def client(self):
        app = create_app()
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client

    def test_structure_missing_param(self, client):
        resp = client.get("/repo/structure")
        assert resp.status_code == 400

    def test_structure_nonexistent_path(self, client):
        resp = client.get("/repo/structure?repo_path=/nonexistent/abc")
        assert resp.status_code == 404

    def test_structure_success(self, client, mock_repo):
        resp = client.get(f"/repo/structure?repo_path={mock_repo}")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["type"] == "directory"
        assert "children" in data

    def test_call_graph_success(self, client, mock_repo):
        resp = client.get(f"/repo/call-graph?repo_path={mock_repo}")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "nodes" in data
        assert "edges" in data

    def test_dependencies_success(self, client, mock_repo):
        resp = client.get(f"/repo/dependencies?repo_path={mock_repo}")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "nodes" in data
        assert "edges" in data


# =========================================================================
# Run
# =========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
