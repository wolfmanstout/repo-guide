from pathlib import Path

import git
import pytest
from click.testing import CliRunner

from repo_guide.cli import DocGenerator, cli


def test_version():
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert result.output.startswith("cli, version ")


@pytest.fixture
def test_repo(tmp_path):
    """Create a test repository with some sample files."""
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()

    # Create some files and directories
    (repo_path / "src").mkdir()
    (repo_path / "src/main.py").write_text("def main():\n    pass")
    (repo_path / "README.md").write_text("# Test Repo")

    # Initialize git repo with a remote and commit files
    repo = git.Repo.init(repo_path)
    repo.create_remote("origin", "https://github.com/test/test_repo.git")
    repo.index.add(["src/main.py", "README.md"])
    repo.index.commit("Initial commit")

    return repo_path


def test_prompt_construction(test_repo, tmp_path):
    generator = DocGenerator(
        repo_path=test_repo,
        output_path=tmp_path / "output",
        model_name="",
        count_tokens=False,
        ignore_patterns=[],
    )

    # Add some dummy generated readmes
    generated_readmes = {
        test_repo / "src": "## Source Code\nContains main application code."
    }

    # Test prompt construction
    dirs = [test_repo / "src"]
    files = [test_repo / "README.md"]
    prompt = generator._build_prompt(test_repo, dirs, files, generated_readmes)

    expected_prompt = f"""Current directory (path relative to repo root): .

=====

Subdirectories (paths relative to repo root):
src

=====

Files (paths relative to repo root):
README.md
---
# Test Repo

---

=====

Previously generated documentation (paths relative to current directory):

{Path("src/README.md")}:
## Source Code
Contains main application code.
---
"""

    assert prompt == expected_prompt


def test_system_prompt_construction(test_repo, tmp_path):
    generator = DocGenerator(
        repo_path=test_repo,
        output_path=tmp_path / "output",
        model_name="",
        count_tokens=False,
        ignore_patterns=[],
    )

    system_prompt = generator._build_system_prompt(is_repo_root=True)

    expected_system = (
        "Provide an overview of what this directory does in Markdown, "
        "including a summary of each subdirectory and file, starting with "
        "the subdirectories. "
        "Omit heading level 1 (#) as it will be added automatically. "
        "If adding links to previously generated documentation, use the "
        "relative path to the file from the current directory. "
        "Link any files mentioned to an absolute URL starting with "
        "https://github.com/test/test_repo/blob/main/ followed by the repo-relative file path. "
        "Begin with an overall description of the repository. List the "
        "dependencies and how they are used."
    )

    assert system_prompt == expected_system


def test_file_decoding_failures(test_repo, tmp_path, capfd):
    """Test handling of files that can't be decoded."""
    # Create test files with different encodings
    (test_repo / "utf8.txt").write_text("Hello", encoding="utf-8")
    (test_repo / "latin1.txt").write_text("Café", encoding="latin-1")

    # Create a file with partial UTF-16 bytes that will fail UTF-8/Latin1/CP1252 decoding
    bad_file = test_repo / "broken.txt"
    bad_file.write_bytes(b"Hello\x00 \x00W\x00o\x00r\x00l\x00d")  # UTF-16LE without BOM

    generator = DocGenerator(
        repo_path=test_repo,
        output_path=tmp_path / "output",
        model_name="",
        count_tokens=False,
        ignore_patterns=[],
    )

    files = [
        test_repo / "utf8.txt",
        test_repo / "latin1.txt",
        test_repo / "broken.txt",
    ]

    prompt = generator._build_prompt(test_repo, [], files, {})

    # Verify only readable files are in prompt
    assert "utf8.txt" in prompt
    assert "latin1.txt" in prompt
    assert "binary.dat" not in prompt


@pytest.fixture
def mock_model(monkeypatch):
    """Mock LLM model that returns truncated input as response."""

    class MockResponse:
        def __init__(self, text):
            self._text = text

        def text(self):
            return self._text

        def usage(self):
            return type("Usage", (), {"input": 0, "output": 0})()

    class MockModel:
        model_id = "mock"

        def prompt(self, prompt, system=None):
            return MockResponse(prompt[:5] + "...")

    monkeypatch.setattr("llm.get_model", lambda *args: MockModel())


def test_doc_generation_end_to_end(mock_model, test_repo):
    """Test full documentation generation process using CLI."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        # Run the CLI command with default output dir
        result = runner.invoke(cli, [str(test_repo)])
        assert result.exit_code == 0

        # Verify generated documentation
        main_readme = Path("generated_docs/docs/README.md")
        assert main_readme.exists()
        assert main_readme.read_text().startswith("# test_repo\n\nCurre...")

        src_readme = Path("generated_docs/docs/src/README.md")
        assert src_readme.exists()
        assert src_readme.read_text().startswith("# src\n\nCurre...")
