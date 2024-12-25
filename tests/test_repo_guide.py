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

    # Create nested directory structure
    (repo_path / "src").mkdir()
    (repo_path / "src/main.py").write_text("def main():\n    pass")
    (repo_path / "src/utils").mkdir()
    (repo_path / "src/utils/helpers.py").write_text("def helper():\n    pass")
    (repo_path / "README.md").write_text("# Test Repo")

    # Initialize git repo with a remote and commit files
    repo = git.Repo.init(repo_path, initial_branch="main")
    repo.create_remote("origin", "https://github.com/test/test_repo.git")
    repo.index.add(["src/main.py", "src/utils/helpers.py", "README.md"])
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
        test_repo / "src": "## Source Code\nContains main application code.",
        test_repo / "src/utils": "## Utilities\nHelper functions and utilities.",
    }

    # Test prompt construction
    dirs = [test_repo / "src", test_repo / "src/utils"]
    files = [test_repo / "README.md"]
    prompt = generator._build_prompt(test_repo, dirs, files, generated_readmes)

    expected_prompt = """\
<current_directory>.</current_directory>

<subdirectories>
<subdirectory>src</subdirectory>
<subdirectory>src/utils</subdirectory>
</subdirectories>

<files>
<file>
<url>https://github.com/test/test_repo/blob/main/README.md</url>
<content>
# Test Repo
</content>
</file>
</files>

<existing_docs>
<readme>
<url>src/README.md</url>
<content>## Source Code
Contains main application code.</content>
</readme>
<readme>
<url>src/utils/README.md</url>
<content>## Utilities
Helper functions and utilities.</content>
</readme>
</existing_docs>"""

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
        "You are repo-guide. Your responses will be used to build a field guide to a code repository. "
        "Analyze the provided XML and explain what the current directory does in Markdown. "
        "The <current_directory> tag contains the current directory relative to repo root. "
        "The <subdirectories> tag lists every <subdirectory> relative to repo root. "
        "The <files> tag contains files in the current directory, each in its own <file> tag with <url> and <content>. "
        "The <existing_docs> tag contains docs you previously generated in <readme> tags with <url> and <content>. "
        "Focus on the subdirectories and files that are most important or interesting. Describe how they work together. "
        "If a large group of files or subdirectories do something similar, provide a summary for the group instead of summarizing each one. "
        "Omit heading level 1 (#) as it will be added automatically. "
        "Link any <file> or <readme> references to its provided <url> without modification. "
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
        result = runner.invoke(cli, [str(test_repo), "--no-serve"])
        assert result.exit_code == 0

        # Verify generated documentation
        main_readme = Path("generated_docs/docs/README.md")
        assert main_readme.exists()
        assert main_readme.read_text() == "# test_repo\n\n<curr..."

        src_readme = Path("generated_docs/docs/src/README.md")
        assert src_readme.exists()
        assert src_readme.read_text() == "# src\n\n<curr..."

        utils_readme = Path("generated_docs/docs/src/utils/README.md")
        assert utils_readme.exists()
        assert utils_readme.read_text() == "# src/utils\n\n<curr..."
