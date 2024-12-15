import os
import shutil
import textwrap
import webbrowser
from collections.abc import Sequence
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path

import click
import git
import llm
from mkdocs.commands.serve import serve as mkdocs_serve
from tqdm import tqdm


class DocGenerator:
    def __init__(
        self,
        repo_path: Path,
        output_path: Path,
        model_name: str,
        count_tokens: bool,
        ignore_patterns: Sequence[str],
        verbose: bool = False,
    ):
        self.repo_path = repo_path
        self.output_path = output_path
        self.docs_path = output_path / "docs"
        self.repo = git.Repo(repo_path)
        self.model = llm.get_model(model_name) if model_name else llm.get_model()
        self.verbose = verbose
        click.echo(f"Using model: {self.model.model_id}")
        self.count_tokens = count_tokens
        self.total_tokens = 0
        self.ignore_patterns = ignore_patterns

        # Parse repo URL and default branch to construct file URLs
        self.repo_url = None
        self.repo_url_file_prefix = None
        origins = [remote for remote in self.repo.remotes if remote.name == "origin"]
        if origins:
            origin = origins[0]
            self.repo_url = (
                origin.url.replace(".git", "")
                .replace("git@", "https://")
                .replace(".com:", ".com/")
            )
            if "github.com" in self.repo_url:
                default_branch = origin.refs[0].remote_head if origin.refs else "main"
                self.repo_url_file_prefix = f"{self.repo_url}/blob/{default_branch}/"

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

    def _safe_read_file(self, file_path: Path) -> str | None:
        """Safely read file content, handling encoding errors."""
        encodings = ["utf-8", "cp1252"]
        for encoding in encodings:
            try:
                return file_path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
            except Exception as e:
                click.echo(f"Error reading {file_path}: {e}", err=True)
                return None

        click.echo(
            f"Failed to decode {file_path} with any supported encoding", err=True
        )
        return None

    def _build_prompt(self, root, dirs, files, generated_readmes):
        prompt_parts = [
            f"Current directory (path relative to repo root): {root.relative_to(self.repo_path)}\n"
        ]

        if dirs:
            dir_list = "\n".join(str(d.relative_to(self.repo_path)) for d in dirs)
            prompt_parts.append(
                f"Subdirectories (paths relative to repo root):\n{dir_list}\n"
            )

        if files:
            file_contents = ""
            for f in files:
                content = self._safe_read_file(f)
                if content is not None:
                    file_template = textwrap.dedent("""\
                        {path}
                        ---
                        {content}

                        ---
                        """)
                    file_contents += file_template.format(
                        path=f.relative_to(self.repo_path), content=content
                    )
            if file_contents:
                prompt_parts.append(
                    f"Files (paths relative to repo root):\n{file_contents}"
                )

        if generated_readmes:
            readme_context = ""
            for subdir, content in generated_readmes.items():
                if subdir.is_relative_to(root):
                    rel_path = subdir.relative_to(root) / "README.md"
                    readme_context += f"\n{rel_path}:\n"
                    readme_context += content
                    readme_context += "\n---\n"
            if readme_context:
                prompt_parts.append(
                    f"Previously generated documentation (paths relative to current directory):\n{readme_context}"
                )

        return "\n=====\n\n".join(prompt_parts)

    def _build_system_prompt(self, is_repo_root):
        parts = [
            "Provide an overview of what this directory does in Markdown, "
            "including a summary of each subdirectory and file, starting with "
            "the subdirectories. "
            "Omit heading level 1 (#) as it will be added automatically. "
            "If adding links to previously generated documentation, use the "
            "relative path to the file from the current directory."
        ]
        if self.repo_url_file_prefix:
            parts.append(
                "Link any files mentioned to an absolute URL starting with "
                f"{self.repo_url_file_prefix} followed by the repo-relative file path."
            )
        if is_repo_root:
            parts.append(
                "Begin with an overall description of the repository. List the "
                "dependencies and how they are used."
            )
        return " ".join(parts)

    def load_existing_docs(self) -> dict[Path, str]:
        """Load existing documentation from the filesystem."""
        generated_readmes = {}
        if not self.docs_path.exists():
            return generated_readmes

        for readme in self.docs_path.rglob("README.md"):
            rel_path = readme.parent.relative_to(self.docs_path)
            source_path = self.repo_path / rel_path
            if source_path.exists():
                generated_readmes[source_path] = readme.read_text()
        return generated_readmes

    def generate_docs(self, resume=False):
        all_files = set(
            str((self.repo_path / f).resolve())
            for f in self.repo.git.ls_files().splitlines()
        )
        resolved_repo_path = self.repo_path.resolve()
        all_directories = set(
            str(d)
            for f in all_files
            for d in Path(f).parents
            if d.is_relative_to(resolved_repo_path)
        )
        generated_readmes = self.load_existing_docs() if resume else {}
        walk_triples = [
            (r, d, f)
            for r, d, f in os.walk(self.repo_path, topdown=False)
            if str(Path(r).resolve()) in all_directories
        ]
        total_dirs = len(walk_triples)
        if self.verbose:
            iter = walk_triples
        else:
            iter = tqdm(walk_triples, desc="Generating documentation")

        current_dir = 0
        for root, dirs, files in iter:
            current_dir += 1
            root = Path(root)
            resolved_root = root.resolve()
            # Skip directories that already have documentation when resuming
            if resume and root in generated_readmes:
                if self.verbose:
                    click.echo(
                        f"[{current_dir}/{total_dirs}] Skipping {root} (already documented)"
                    )
                continue

            rel_root = root.relative_to(self.repo_path)
            if self.verbose:
                if self.count_tokens:
                    click.echo(
                        f"[{current_dir}/{total_dirs}] Processing {rel_root} (tokens used so far: {self.total_tokens:,})"
                    )
                else:
                    click.echo(f"[{current_dir}/{total_dirs}] Processing {rel_root}")

            is_repo_root = resolved_root == resolved_repo_path

            dirs = [
                root / d for d in dirs if str((root / d).resolve()) in all_directories
            ]
            files = [
                root / f
                for f in files
                if str((root / f).resolve()) in all_files
                and not any(
                    fnmatch(str(f), pattern) for pattern in self.ignore_patterns
                )
            ]

            prompt = self._build_prompt(root, dirs, files, generated_readmes)
            system_prompt = self._build_system_prompt(is_repo_root)

            response = self.model.prompt(
                prompt,
                system=system_prompt,
            )

            if self.count_tokens:
                self.total_tokens += response.usage().input or 0
                self.total_tokens += response.usage().output or 0

            readme_path = self.docs_path / rel_root / "README.md"
            readme_path.parent.mkdir(parents=True, exist_ok=True)
            dir_name = (
                resolved_repo_path.name
                if is_repo_root
                else str(root.relative_to(self.repo_path))
            )
            readme_path.write_text(f"# {dir_name}\n\n{response.text()}")

            # Store the generated README
            generated_readmes[root] = response.text()

    def write_mkdocs_configuration(self):
        config_template = textwrap.dedent("""\
            site_name: {repo_name} docs by repo-guide
            theme: material
            exclude_docs: |
                !.*
                !/templates/
            hooks:
                - my_hooks.py
            """)
        if self.repo_url:
            config_template += textwrap.dedent("""\
                repo_url: {repo_url}
                edit_uri:
                """)
        config_content = textwrap.dedent(
            config_template.format(
                repo_name=self.repo_path.resolve().name, repo_url=self.repo_url
            )
        )
        hooks_content = textwrap.dedent("""\
            import bleach
            from bleach_allowlist import markdown_tags, markdown_attrs

            def on_page_content(html, **kwargs):
                return bleach.clean(html, markdown_tags, markdown_attrs)
            """)
        Path(self.output_path / "mkdocs.yml").write_text(config_content)
        Path(self.output_path / "my_hooks.py").write_text(hooks_content)


@click.command()
@click.version_option()
@click.argument(
    "repo_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output",
)
@click.option(
    "--model",
    default="gemini-2.0-flash-exp",
    show_default=True,
    help="LLM model to use",
)
@click.option(
    "--serve",
    is_flag=True,
    help="Start local documentation server",
)
@click.option(
    "--open",
    is_flag=True,
    help="Open documentation in browser (implies --serve)",
)
@click.option("--port", default=8000, show_default=True, help="Port for local server")
@click.option(
    "--gen/--no-gen", default=True, show_default=True, help="Generate documentation"
)
@click.option(
    "--count-tokens/--no-count-tokens",
    default=True,
    show_default=True,
    help="Count tokens used",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, writable=True, path_type=Path),
    default="generated_docs",
    show_default=True,
    help="Output directory for generated documentation",
)
@click.option(
    "--include-changelog/--no-include-changelog",
    default=False,
    show_default=True,
    help="Generate changelog from recent commits",
)
@click.option(
    "--ignore",
    multiple=True,
    help="Pattern to ignore (may be specified multiple times)",
)
@click.option(
    "--resume",
    is_flag=True,
    help="Resume documentation generation from last stopping point",
)
@click.option(
    "--public/--local",
    default=False,
    show_default=True,
    help="Serve documentation on all network interfaces (0.0.0.0). Warning: This makes docs accessible to other devices on your network.",
)
def cli(
    repo_dir: Path,
    model: str,
    serve: bool,
    open: bool,
    port: int,
    gen: bool,
    count_tokens: bool,
    output_dir: Path,
    include_changelog: bool,
    ignore: tuple[str],
    resume: bool,
    public: bool,
    verbose: bool,
):
    "Uses AI to help understand repositories and their changes."
    generator = DocGenerator(
        repo_dir,
        output_dir,
        model,
        count_tokens,
        ignore_patterns=ignore,
        verbose=verbose,
    )
    if gen:
        # Only remove existing docs if not resuming
        if not resume and output_dir.exists():
            shutil.rmtree(output_dir)

        docs_path = output_dir / "docs"
        docs_path.mkdir(parents=True, exist_ok=True)

        # Generate documentation
        generator.generate_docs(resume=resume)

        # Generate changelog only if requested
        if include_changelog:
            changes = generator.get_recent_changes()
            changelog = generator.generate_changelog(changes)
            Path(output_dir / "docs/CHANGELOG.md").write_text(changelog)

        if count_tokens:
            if generator.total_tokens:
                click.echo(f"Total tokens used: {generator.total_tokens:,}")
            else:
                click.echo("Unable to count tokens. Add --no-count-tokens to disable.")
    else:
        # Ensure the docs directory exists when serving
        if serve and not output_dir.exists():
            click.echo(
                "Error: No generated documentation found. Use --gen to generate docs first."
            )
            return

    generator.write_mkdocs_configuration()

    if open or serve:
        url = f"http://127.0.0.1:{port}/"
        click.echo(f"Serving docs at {url}" + (" (public)" if public else ""))
        if open:
            webbrowser.open(url)
        host = "0.0.0.0" if public else "127.0.0.1"
        mkdocs_serve(
            f"{output_dir}/mkdocs.yml",
            dev_addr=f"{host}:{port}",
            livereload=True,
        )
