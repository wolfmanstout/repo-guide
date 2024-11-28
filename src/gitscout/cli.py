import os
import re
import shutil
import textwrap
from datetime import datetime
from pathlib import Path

import click
import git
import llm
from flask import Flask, render_template_string
from mistletoe import markdown

app = Flask(__name__)


def escape_markdown(text):
    # List of characters to escape in Markdown
    markdown_chars = r"[\`\*\_\{\}\[\]\(\)\#\+\-\.\!]"
    # Escape each character by prefixing it with a backslash
    escaped_text = re.sub(markdown_chars, r"\\\g<0>", text)
    return escaped_text


class DocGenerator:
    def __init__(self, repo_path: str, model_name: str | None = None):
        self.repo_path = Path(repo_path)
        self.repo = git.Repo(repo_path)
        self.model = llm.get_model(model_name) if model_name else llm.get_model()

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
        conversation = self.model.conversation()
        for root, dirs, files in os.walk(self.repo_path, topdown=False):
            root = Path(root)
            if str(root.resolve()) not in all_directories:
                continue
            dirs = [root / Path(d) for d in dirs]
            dirs = [d for d in dirs if str(d.resolve()) in all_directories]
            files = [root / Path(f) for f in files]
            files = [f for f in files if str(f.resolve()) in all_files]
            dir_list = "\n".join(str(d) for d in dirs)
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
            prompt_template = textwrap.dedent(
                """\
                Directory: {root}

                Subdirectories:
                {dir_list}

                Files:
                {file_contents}
                """
            )
            response = conversation.prompt(
                prompt_template.format(
                    root=root, dir_list=dir_list, file_contents=file_contents
                ),
                system="Provide an overview of what this directory does in Markdown, including a summary of each subdirectory and file.",
            )

            output_path = Path("generated_docs") / root / "README.md"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(response.text())


@app.route("/")
def serve_docs():
    # Combine and convert all generated markdown files to HTML
    docs_path = Path("generated_docs")
    all_content = []

    for md_file in docs_path.glob("**/*.md"):
        content = md_file.read_text()
        file_template = textwrap.dedent(
            """\
            {path}
            ---
            {content}

            ---
            """
        )
        all_content.append(
            file_template.format(
                path=escape_markdown(str(md_file.relative_to(docs_path))),
                content=content,
            )
        )

    html = markdown("\n\n".join(all_content))

    return render_template_string(
        """
        <!DOCTYPE html>
        <html>
            <head>
                <title>Repository Documentation</title>
                <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.2.0/github-markdown.min.css">
                <style>
                    .markdown-body {
                        box-sizing: border-box;
                        min-width: 200px;
                        max-width: 980px;
                        margin: 0 auto;
                        padding: 45px;
                    }
                </style>
            </head>
            <body class="markdown-body">
                {{ content|safe }}
            </body>
        </html>
        """,
        content=html,
    )


@click.command()
@click.version_option()
@click.argument("repo_path")
@click.option("--model", help="LLM model to use (defaults to system default)")
@click.option(
    "--serve/--no-serve", default=False, help="Start local documentation server"
)
@click.option("--port", default=5000, help="Port for local server")
@click.option("--gen/--no-gen", default=True, help="Generate documentation")
def cli(repo_path: str, model: str, serve: bool, port: int, gen: bool):
    "Uses AI to help understand repositories and their changes."
    generated_docs_path = Path("generated_docs")

    if gen:
        # Remove existing generated docs
        if generated_docs_path.exists():
            shutil.rmtree(generated_docs_path)
        generated_docs_path.mkdir()

        generator = DocGenerator(repo_path, model)

        # Generate documentation
        generator.generate_docs()

        # Generate changelog from recent commits
        changes = generator.get_recent_changes()
        changelog = generator.generate_changelog(changes)

        Path("generated_docs/CHANGELOG.md").write_text(changelog)
    else:
        # Ensure the docs directory exists when serving
        if serve and not generated_docs_path.exists():
            click.echo(
                "Error: No generated documentation found. Use --gen to generate docs first."
            )
            return

    if serve:
        app.run(port=port)
