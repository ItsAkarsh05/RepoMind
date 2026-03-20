"""
Tests for repo_ingestion package
=================================
Run with:  python -m pytest tests/test_repo_ingestion.py -v
"""

import os
import sys
import tempfile
import shutil

import pytest

# Ensure project root is on sys.path so imports work without installing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from repo_ingestion.github_handler import validate_github_url, clone_repository
from repo_ingestion.file_traversal import traverse_repository, SUPPORTED_EXTENSIONS, IGNORED_DIRS
from repo_ingestion.code_chunker import chunk_file, chunk_file_fixed_size, chunk_python_file, CodeChunk


# =========================================================================
# 1. URL Validation Tests
# =========================================================================

class TestValidateGithubUrl:
    """Tests for ``validate_github_url``."""

    def test_valid_url_basic(self):
        owner, repo = validate_github_url("https://github.com/pallets/flask")
        assert owner == "pallets"
        assert repo == "flask"

    def test_valid_url_with_git_suffix(self):
        owner, repo = validate_github_url("https://github.com/pallets/flask.git")
        assert owner == "pallets"
        assert repo == "flask"

    def test_valid_url_trailing_slash(self):
        owner, repo = validate_github_url("https://github.com/pallets/flask/")
        assert owner == "pallets"
        assert repo == "flask"

    def test_valid_url_with_dashes_dots(self):
        owner, repo = validate_github_url("https://github.com/my-org/my.repo-name")
        assert owner == "my-org"
        assert repo == "my.repo-name"

    def test_invalid_url_http(self):
        with pytest.raises(ValueError, match="Invalid GitHub URL"):
            validate_github_url("http://github.com/pallets/flask")

    def test_invalid_url_gitlab(self):
        with pytest.raises(ValueError, match="Invalid GitHub URL"):
            validate_github_url("https://gitlab.com/pallets/flask")

    def test_invalid_url_empty(self):
        with pytest.raises(ValueError, match="non-empty string"):
            validate_github_url("")

    def test_invalid_url_none(self):
        with pytest.raises(ValueError, match="non-empty string"):
            validate_github_url(None)

    def test_invalid_url_no_repo(self):
        with pytest.raises(ValueError, match="Invalid GitHub URL"):
            validate_github_url("https://github.com/pallets")

    def test_invalid_url_random_string(self):
        with pytest.raises(ValueError, match="Invalid GitHub URL"):
            validate_github_url("not a url at all")


# =========================================================================
# 2. File Traversal Tests
# =========================================================================

class TestTraverseRepository:
    """Tests for ``traverse_repository`` using a mock directory structure."""

    @pytest.fixture
    def mock_repo(self, tmp_path):
        """Create a temporary directory tree that looks like a small repo."""
        # Supported files
        (tmp_path / "main.py").write_text("print('hello')")
        (tmp_path / "utils.js").write_text("console.log('hi')")
        (tmp_path / "README.md").write_text("# Hello")

        # Nested directory
        sub = tmp_path / "src"
        sub.mkdir()
        (sub / "app.ts").write_text("const x = 1;")
        (sub / "helper.dart").write_text("void main() {}")

        # Ignored directories (should be skipped)
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text("gitconfig")

        nm_dir = tmp_path / "node_modules"
        nm_dir.mkdir()
        (nm_dir / "lodash.js").write_text("module.exports = {}")

        pycache = tmp_path / "__pycache__"
        pycache.mkdir()
        (pycache / "main.cpython-311.pyc").write_bytes(b"\x00\x00")

        # Binary / ignored extensions (should be skipped)
        (tmp_path / "logo.png").write_bytes(b"\x89PNG")
        (tmp_path / "data.db").write_bytes(b"\x00")
        (tmp_path / "compiled.pyc").write_bytes(b"\x00")

        return tmp_path

    def test_finds_supported_files(self, mock_repo):
        found = list(traverse_repository(str(mock_repo)))
        rel_paths = {rel for _, rel in found}

        assert "main.py" in rel_paths
        assert "utils.js" in rel_paths
        assert "README.md" in rel_paths
        assert os.path.join("src", "app.ts") in rel_paths
        assert os.path.join("src", "helper.dart") in rel_paths

    def test_ignores_git_dir(self, mock_repo):
        rel_paths = {rel for _, rel in traverse_repository(str(mock_repo))}
        for p in rel_paths:
            assert ".git" not in p.split(os.sep)

    def test_ignores_node_modules(self, mock_repo):
        rel_paths = {rel for _, rel in traverse_repository(str(mock_repo))}
        for p in rel_paths:
            assert "node_modules" not in p.split(os.sep)

    def test_ignores_pycache(self, mock_repo):
        rel_paths = {rel for _, rel in traverse_repository(str(mock_repo))}
        for p in rel_paths:
            assert "__pycache__" not in p.split(os.sep)

    def test_ignores_binary_files(self, mock_repo):
        rel_paths = {rel for _, rel in traverse_repository(str(mock_repo))}
        assert "logo.png" not in rel_paths
        assert "data.db" not in rel_paths
        assert "compiled.pyc" not in rel_paths

    def test_nonexistent_path_raises(self):
        with pytest.raises(FileNotFoundError):
            list(traverse_repository("/nonexistent/path/xyz"))

    def test_correct_count(self, mock_repo):
        found = list(traverse_repository(str(mock_repo)))
        assert len(found) == 5  # main.py, utils.js, README.md, app.ts, helper.dart


# =========================================================================
# 3. Code Chunker Tests
# =========================================================================

class TestCodeChunker:
    """Tests for chunking logic."""

    def test_fixed_size_basic(self):
        source = "\n".join(f"line {i}" for i in range(100))
        chunks = chunk_file_fixed_size(source, "/fake/test.js", chunk_size=20, overlap=5)

        assert len(chunks) > 1
        # First chunk starts at line 1
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == 20
        assert chunks[0].file_name == "test.js"
        assert chunks[0].chunk_type == "block"

    def test_fixed_size_overlap(self):
        source = "\n".join(f"line {i}" for i in range(50))
        chunks = chunk_file_fixed_size(source, "/fake/test.js", chunk_size=20, overlap=5)

        # Second chunk should start at line 16 (20 - 5 + 1)
        if len(chunks) > 1:
            assert chunks[1].start_line == 16

    def test_fixed_size_empty_source(self):
        chunks = chunk_file_fixed_size("", "/fake/empty.js")
        assert chunks == []

    def test_python_ast_chunking_functions(self):
        source = '''
def hello():
    print("hello")

def world():
    print("world")
'''
        chunks = chunk_python_file(source, "/fake/test.py")

        func_chunks = [c for c in chunks if c.chunk_type == "function"]
        assert len(func_chunks) == 2

        names = [c.content for c in func_chunks]
        assert any("hello" in n for n in names)
        assert any("world" in n for n in names)

    def test_python_ast_chunking_class(self):
        source = '''
class MyClass:
    def method_a(self):
        pass

    def method_b(self):
        pass
'''
        chunks = chunk_python_file(source, "/fake/test.py")

        class_chunks = [c for c in chunks if c.chunk_type == "class"]
        assert len(class_chunks) == 1
        assert "MyClass" in class_chunks[0].content

    def test_python_ast_fallback_on_syntax_error(self):
        source = "def broken(:\n  pass"
        chunks = chunk_python_file(source, "/fake/bad.py")

        # Should fall back to fixed-size — still produce chunks
        assert len(chunks) >= 1
        assert chunks[0].chunk_type == "block"

    def test_chunk_file_dispatcher_py(self, tmp_path):
        py_file = tmp_path / "example.py"
        py_file.write_text('def foo():\n    return 42\n')

        chunks = chunk_file(str(py_file))
        assert len(chunks) >= 1
        # Should use AST-aware chunking for .py
        func_chunks = [c for c in chunks if c.chunk_type == "function"]
        assert len(func_chunks) == 1

    def test_chunk_file_dispatcher_js(self, tmp_path):
        js_file = tmp_path / "example.js"
        js_file.write_text("function foo() { return 42; }\n" * 10)

        chunks = chunk_file(str(js_file))
        assert len(chunks) >= 1
        assert all(c.chunk_type == "block" for c in chunks)

    def test_metadata_fields(self):
        source = "x = 1\ny = 2\n"
        chunks = chunk_file_fixed_size(source, "/path/to/script.py")
        assert len(chunks) == 1

        meta = chunks[0].to_metadata()
        assert meta["file_path"] == "/path/to/script.py"
        assert meta["file_name"] == "script.py"
        assert meta["start_line"] == 1
        assert meta["end_line"] == 2
        assert meta["chunk_type"] == "block"


# =========================================================================
# Run
# =========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
