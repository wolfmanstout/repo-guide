from pathlib import Path

import git
from click.testing import CliRunner

from gitscout.cli import DocGenerator, cli


def test_version():
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert result.output.startswith("cli, version ")


def test_prompt_construction(tmp_path):
    # Create a simple repository structure
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()

    # Create some files and directories
    (repo_path / "src").mkdir()
    (repo_path / "src/main.py").write_text("def main():\n    pass")
    (repo_path / "README.md").write_text("# Test Repo")

    # Initialize git repo with a remote
    repo = git.Repo.init(repo_path)
    repo.create_remote("origin", "https://github.com/test/test_repo.git")

    # Create DocGenerator instance
    generator = DocGenerator(repo_path, tmp_path / "output")

    # Add some dummy generated readmes
    generated_readmes = {
        repo_path / "src": "## Source Code\nContains main application code."
    }

    # Test prompt construction directly
    dirs = [repo_path / "src"]
    files = [repo_path / "README.md"]
    prompt = generator._build_prompt(repo_path, dirs, files, generated_readmes)
    system_prompt = generator._build_system_prompt(is_repo_root=True)

    expected_prompt = f"""Current directory: {repo_path}

=====

Subdirectories:
{repo_path / "src"}

=====

Files:
{repo_path / "README.md"}
---
# Test Repo

---

=====

Previously generated documentation:

{Path("src/README.md")}:
## Source Code
Contains main application code.
---
"""

    expected_system = (
        "Provide an overview of what this directory does in Markdown, "
        "including a summary of each subdirectory and file, starting with "
        "the subdirectories. "
        "Omit heading level 1 (#) as it will be added automatically. "
        "If adding links to previously generated documentation, use the "
        "relative path to the file from the *current* directory, not the "
        "repo root. "
        "Link any files mentioned to an absolute URL starting with "
        "https://github.com/test/test_repo/blob/main/ followed by the relative file path. "
        "Begin with an overall description of the repository. List the "
        "dependencies and how they are used."
    )

    assert prompt == expected_prompt
    assert system_prompt == expected_system
