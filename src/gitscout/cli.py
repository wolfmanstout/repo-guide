import os
import shutil
import textwrap
from datetime import datetime
from pathlib import Path

import click
import git
import llm
from mkdocs.commands.serve import serve as mkdocs_serve


class DocGenerator:
    def __init__(
        self, repo_path: str, model_name: str | None = None, count_tokens: bool = False
    ):
        self.repo_path = Path(repo_path)
        self.repo = git.Repo(repo_path)
        self.model = llm.get_model(model_name) if model_name else llm.get_model()
        self.count_tokens = count_tokens
        self.total_tokens = 0

    def get_recent_changes(self, num_commits=5):
        commits = list(self.repo.iter_commits("main", max_count=num_commits))
        changes = []

        for commit in commits:
            # Get the diff for this commit
            diff = (
                commit.parents[0].diff(commit)
                if commit.parents
                else commit.diff(git.NULL_TREE)
            )

            # Extract modified files and their diffs (truncated)
            files_changed = []
            for d in diff:
                if d.a_path:
                    try:
                        # Get truncated diff content
                        diff_content = d.diff.decode("utf-8")[
                            :500
                        ]  # Truncate long diffs
                        files_changed.append({"path": d.a_path, "diff": diff_content})
                    except Exception as e:
                        print(f"Error processing diff for {d.a_path}: {e}")
                        continue

            changes.append(
                {
                    "hash": commit.hexsha[:8],
                    "message": commit.message,
                    "author": commit.author.name,
                    "date": datetime.fromtimestamp(commit.committed_date),
                    "files": files_changed,
                }
            )

        return changes

    def generate_changelog(self, changes):
        # Start a conversation for maintaining context
        response = self.model.prompt(
            """Generate a detailed changelog entry for the following git commits.
            Focus on user-facing changes and group similar changes together.
            Format the output in markdown with appropriate headers and bullet points.

            Commit details:
            """
            + str(changes),
            system="You are a technical writer creating clear, organized changelog entries.",
        )

        if self.count_tokens:
            self.total_tokens += response.usage().input or 0
            self.total_tokens += response.usage().output or 0

        return response.text()

    def generate_docs(self):
        all_files = set(
            str(Path(f).resolve()) for f in self.repo.git.ls_files().splitlines()
        )
        resolved_repo_path = self.repo_path.resolve()
        all_directories = set(
            str(d)
            for f in all_files
            for d in Path(f).parents
            if d.is_relative_to(resolved_repo_path)
        )
        generated_readmes = {}
        for root, dirs, files in os.walk(self.repo_path, topdown=False):
            root = Path(root)
            if str(root.resolve()) not in all_directories:
                continue

            dirs = [root / Path(d) for d in dirs]
            dirs = [d for d in dirs if str(d.resolve()) in all_directories]
            files = [root / Path(f) for f in files]
            files = [f for f in files if str(f.resolve()) in all_files]

            # Get the directory name for the prompt
            is_repo_root = root.resolve() == resolved_repo_path
            dir_name = resolved_repo_path.name if is_repo_root else str(root)

            # Get READMEs from all subdirectories
            readme_context = ""
            for subdir, content in generated_readmes.items():
                if subdir != root and subdir.is_relative_to(root):
                    rel_path = subdir.relative_to(self.repo_path) / "README.md"
                    readme_context += f"\n{rel_path}:\n"
                    readme_context += content
                    readme_context += "\n---\n"

            prompt_parts = [f"Directory: {dir_name}"]

            if dirs:
                dir_list = "\n".join(str(d) for d in dirs)
                prompt_parts.append(f"Subdirectories:\n{dir_list}")

            if files:
                file_template = textwrap.dedent(
                    """\
                    {path}
                    ---
                    {content}

                    ---
                    """
                )
                file_contents = "".join(
                    file_template.format(path=f, content=f.read_text()) for f in files
                )
                prompt_parts.append(f"Files:\n{file_contents}")

            if readme_context:
                prompt_parts.append(
                    f"Previously generated documentation:\n{readme_context}"
                )

            prompt = "=====\n\n".join(prompt_parts)
            response = self.model.prompt(
                prompt,
                system=(
                    "Provide an overview of what this directory does in Markdown, "
                    "including a summary of each subdirectory and file, starting with "
                    "the subdirectories. "
                    "The title should be the directory name."
                    "If adding links to previously generated documentation, use the "
                    "relative path to the file from the current directory."
                )
                + (
                    (
                        "Begin with an overall description of the repository. List the "
                        "dependencies and how they are used."
                    )
                    if is_repo_root
                    else ""
                ),
            )

            if self.count_tokens:
                self.total_tokens += response.usage().input or 0
                self.total_tokens += response.usage().output or 0

            output_path = Path("generated_docs/docs") / root / "README.md"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(response.text())

            # Store the generated README
            generated_readmes[root] = response.text()


@click.command()
@click.version_option()
@click.argument("repo_path")
@click.option("--model", help="LLM model to use (defaults to system default)")
@click.option(
    "--serve/--no-serve", default=False, help="Start local documentation server"
)
@click.option("--port", default=5000, help="Port for local server")
@click.option("--gen/--no-gen", default=True, help="Generate documentation")
@click.option(
    "--count-tokens/--no-count-tokens", default=False, help="Count tokens used"
)
def cli(
    repo_path: str, model: str, serve: bool, port: int, gen: bool, count_tokens: bool
):
    "Uses AI to help understand repositories and their changes."
    generated_docs_path = Path("generated_docs")

    if gen:
        # Remove existing generated docs
        if generated_docs_path.exists():
            shutil.rmtree(generated_docs_path)
        docs_path = generated_docs_path / "docs"
        docs_path.mkdir(parents=True)

        generator = DocGenerator(repo_path, model, count_tokens)

        # Generate documentation
        generator.generate_docs()

        # Generate changelog from recent commits
        changes = generator.get_recent_changes()
        changelog = generator.generate_changelog(changes)

        Path("generated_docs/docs/CHANGELOG.md").write_text(changelog)

        if count_tokens:
            click.echo(f"Total tokens used: {generator.total_tokens:,}")
    else:
        # Ensure the docs directory exists when serving
        if serve and not generated_docs_path.exists():
            click.echo(
                "Error: No generated documentation found. Use --gen to generate docs first."
            )
            return

    mkdocs_content = textwrap.dedent("""\
        site_name: Repository Documentation
        theme: material
        exclude_docs: |
            !.*
            !/templates/
        """)
    Path("generated_docs/mkdocs.yml").write_text(mkdocs_content)

    if serve:
        # app.run(port=port)  # Commented out Flask server
        click.echo("Serving docs at http://127.0.0.1:8000/")
        mkdocs_serve(
            f"{generated_docs_path}/mkdocs.yml",
            dev_addr="127.0.0.1:8000",
            livereload=True,
        )
