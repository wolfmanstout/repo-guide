import os
import shutil
import textwrap
import threading
import time
import webbrowser
from collections.abc import Sequence
from fnmatch import fnmatch
from pathlib import Path

import click
import git
import llm
import tiktoken
from mkdocs.commands.serve import serve as mkdocs_serve
from tqdm import tqdm

# Try to import magika, set to None if not available
try:
    from magika import Magika  # type: ignore
except ImportError:
    Magika = None


class DocGenerator:
    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        model_name: str,
        count_tokens: bool,
        ignore_patterns: Sequence[str],
        verbose: bool = False,
        token_budget: int = 0,
        use_magika: bool = False,
        files_token_limit: int = 0,
        write_gitignore: bool = True,
        custom_instructions: str = "",
        rename_dot_github: bool = False,
    ) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.docs_dir = output_dir / "docs"
        self.rename_dot_github = rename_dot_github
        self.repo = git.Repo(input_dir, search_parent_directories=True)
        self.resolved_repo_dir = Path(self.repo.working_dir).resolve()
        self.repo_relative_input_dir = self.input_dir.resolve().relative_to(
            self.resolved_repo_dir
        )
        self.model = llm.get_model(model_name) if model_name else llm.get_model()
        self.verbose = verbose
        click.echo(f"Using model: {self.model.model_id}")
        self.count_tokens = count_tokens
        self.total_tokens = 0
        self.ignore_patterns = ignore_patterns
        self.token_budget = token_budget
        self.use_magika = use_magika
        self.files_token_limit = files_token_limit
        self.write_gitignore = write_gitignore
        self.custom_instructions = custom_instructions
        # Encoding used by gpt-4o.
        self.token_encoding = (
            tiktoken.get_encoding("o200k_base") if files_token_limit else None
        )

        # Parse repo URL and default branch to construct file URLs
        self.repo_url = None
        self.repo_url_file_prefix = None
        origin = next(
            (remote for remote in self.repo.remotes if remote.name == "origin"), None
        )
        if origin:
            self.repo_url = (
                origin.url.replace(".git", "")
                .replace("git@", "https://")
                .replace(".com:", ".com/")
            )
            if "github.com" in self.repo_url:
                current_branch = None
                try:
                    current_branch = self.repo.active_branch.name
                except TypeError:
                    click.echo("Unable to determine current branch")
                if current_branch:
                    repo_subdir_str = (
                        self._forward_slash_path(self.repo_relative_input_dir) + "/"
                        if self.repo_relative_input_dir != Path(".")
                        else ""
                    )
                    self.repo_url_file_prefix = (
                        f"{self.repo_url}/blob/{current_branch}/{repo_subdir_str}"
                    )
        if not self.repo_url:
            click.echo("Unable to determine repository URL")
        if not self.repo_url_file_prefix:
            click.echo("Unable to determine repository file URL prefix")

    def _prompt_with_backoff(self, prompt: str, system: str = "") -> llm.Response:
        """Make model.prompt call with exponential backoff."""
        max_attempts = 5
        base_wait = 2  # seconds

        for attempt in range(max_attempts):
            try:
                response = self.model.prompt(prompt, system=system)
                response.text()  # Trigger any exceptions
                return response
            except llm.ModelError as e:
                if attempt == max_attempts - 1:
                    raise
                wait_time = base_wait * (2**attempt)
                if self.verbose:
                    click.echo(f"Model error: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
        raise AssertionError("Unreachable code")

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

    def _forward_slash_path(self, path: Path) -> str:
        return str(path).replace("\\", "/")

    def _build_prompt(
        self,
        root: Path,
        files: list[Path],
        generated_readmes: dict[Path, str],
    ) -> str:
        prompt_parts: list[str] = []
        prompt_parts.append("<current_directory>")
        # Include input directory name as it is often helpful context.
        current_directory = self._forward_slash_path(
            root.resolve().relative_to(self.input_dir.resolve().parent)
        )
        prompt_parts.append(f"<path>{current_directory}</path>\n")

        subdirs = [p for p in generated_readmes if p.is_relative_to(root) and p != root]
        # List subdirectories in order of depth.
        subdirs.sort(key=lambda p: len(p.parts))
        if subdirs:
            prompt_parts.append("<subdirectories>")
            for d in subdirs:
                root_rel_path = self._forward_slash_path(d.relative_to(root))
                prompt_parts.append("<subdirectory>")
                prompt_parts.append(f"<path>{root_rel_path}</path>")
                if self.rename_dot_github:
                    parts = [
                        "_github" if p == ".github" else p
                        for p in d.relative_to(root).parts
                    ]
                    renamed_rel_path = self._forward_slash_path(Path(*parts))
                else:
                    renamed_rel_path = root_rel_path
                prompt_parts.append(
                    f"<link_url>{renamed_rel_path}/README.md</link_url>"
                )
                prompt_parts.append(f"<readme>\n{generated_readmes[d]}\n</readme>")
                prompt_parts.append("</subdirectory>")
            prompt_parts.append("</subdirectories>\n")

        if files:
            prompt_parts.append("<files>")
            total_file_tokens = 0
            for f in files:
                content = self._safe_read_file(f)
                if content is not None:
                    if self.files_token_limit:
                        # Tokens average 4 characters. If the number of characters
                        # exceeds the limit by 10x, we skip the expensive tokenization.
                        if (
                            total_file_tokens + len(content) / 10
                            > self.files_token_limit
                        ):
                            click.echo(
                                f"\nTruncating <files> in {root} due to --files-token-limit."
                            )
                            break
                        try:
                            file_tokens = len(self.token_encoding.encode(content))  # type: ignore
                        except BaseException as e:
                            # See https://github.com/openai/tiktoken/issues/15
                            if "StackOverflow" in str(e):
                                click.echo(
                                    f"\nTokenization error in {f}: {e}. Skipping file."
                                )
                                continue
                            raise
                        if total_file_tokens + file_tokens > self.files_token_limit:
                            click.echo(
                                f"\nTruncating <files> in {root} due to --files-token-limit."
                            )
                            break
                        total_file_tokens += file_tokens

                    input_rel_path = self._forward_slash_path(
                        f.relative_to(self.input_dir)
                    )
                    root_rel_path = self._forward_slash_path(f.relative_to(root))
                    prompt_parts.append("<file>")
                    prompt_parts.append(f"<path>{root_rel_path}</path>")
                    if self.repo_url_file_prefix:
                        prompt_parts.append(
                            f"<link_url>{self.repo_url_file_prefix}{input_rel_path}</link_url>"
                        )
                    prompt_parts.append(f"<content>\n{content}\n</content>")
                    prompt_parts.append("</file>")
            prompt_parts.append("</files>")

        prompt_parts.append("</current_directory>")
        return "\n".join(prompt_parts)

    def _build_system_prompt(self, is_repo_root: bool) -> str:
        parts = [
            "You are a principal software engineer. Your responses will be used to build a guide to a code repository. "
            "Analyze the provided XML and explain (in Markdown) what <current_directory> does and how it works. "
            "\n\n"
            "The <current_directory> <path> is relative to the path to the repo. "
            "The <subdirectories> tag contains subdirectories of <current_directory>, each in its own <subdirectory> tag with <path>, <link_url>, and <readme>. "
            "Each <readme> is a doc that you previously authored that will be included in the guide you are adding to. "
            "The <files> tag contains files in the current directory, each in its own <file> tag with <path>, <link_url>, and <content>. "
            "Each <subdirectory> <path> and <file> <path> is relative to <current_directory> <path>. "
            "Focus on the subdirectories and files that are most important or interesting. Describe how they work together. "
            "If a large group of files or subdirectories do something similar, provide a summary for the group instead of summarizing each one. "
            "Refer to any <file> or <subdirectory> by its <path> and hyperlink it to its <link_url> without modification. "
            "Only link to absolute URLs within <readme> content. Do not link to relative paths except when provided by <link_url>. "
            "\n\n"
            "Output Markdown compatible with John Gruber's reference implementation, for use in MkDocs. "
            "For example, always add a blank line before a list. "
            'Do not refer to the XML tags used to describe the input (e.g. "<current_directory>") in your response. '
            "Omit heading level 1 (#) as it will be added automatically. "
        ]
        if is_repo_root:
            parts.append(
                "\n\nBegin with an overall description of the repository. List the "
                "dependencies and how they are used."
            )
        if self.custom_instructions:
            parts.append("\n\n" + self.custom_instructions)
        return " ".join(parts)

    def load_existing_docs(self) -> dict[Path, str]:
        """Load existing documentation from the filesystem."""
        generated_readmes = {}
        if not self.docs_dir.exists():
            return generated_readmes

        for readme in self.docs_dir.rglob("README.md"):
            rel_path = readme.parent.relative_to(self.docs_dir)
            if self.rename_dot_github:
                # Reverse the mapping.
                parts = [".github" if p == "_github" else p for p in rel_path.parts]
                rel_path = Path(*parts)
            source_path = self.input_dir / rel_path
            if source_path.exists():
                generated_readmes[source_path] = readme.read_text()
        return generated_readmes

    def generate_docs(self, resume: bool = False) -> None:
        all_files: list[Path] = [
            self.resolved_repo_dir / f
            for f in self.repo.git.ls_files(self.repo_relative_input_dir).splitlines()
        ]
        # Filter out binary files if magika is enabled/available
        if self.use_magika:
            if not Magika:
                raise ImportError(
                    "Magika is not available but was explicitly requested"
                )
            magika = Magika()
            magika_results = magika.identify_paths(all_files)
            filtered_files = set(
                f
                for f, r in zip(all_files, magika_results, strict=True)
                if r.output.is_text or f.stat().st_size == 0
            )
        else:
            # Use GitHub's eol attribute to filter out binary files
            is_binary_files = []
            for line in self.repo.git.ls_files(
                self.repo_relative_input_dir, eol=True
            ).splitlines():
                # Avoid parsing the file path
                parts = line.split(None, 3)
                working_tree_info = next(
                    (p for p in parts[:-1] if p.startswith("w/")), None
                )
                if working_tree_info:
                    is_binary_files.append(working_tree_info[2:] == "-text")
                else:
                    click.echo(f"Error parsing ls-files output: {line}")
                    is_binary_files.append(False)
            filtered_files = set(
                f
                for f, is_binary in zip(all_files, is_binary_files, strict=True)
                if not is_binary
            )
        filtered_files = set(
            f
            for f in filtered_files
            if not any(fnmatch(f.name, pattern) for pattern in self.ignore_patterns)
        )

        resolved_input_dir = self.input_dir.resolve()
        filtered_directories = set(
            d
            for f in filtered_files
            for d in f.parents
            if d.is_relative_to(resolved_input_dir)
        )
        generated_readmes = self.load_existing_docs() if resume else {}
        if generated_readmes:
            click.echo("Resuming documentation. To start fresh, use --no-resume.")
        walk_triples = [
            (r, d, f)
            for r, d, f in os.walk(self.input_dir, topdown=False)
            if Path(r).resolve() in filtered_directories
        ]
        total_dirs = len(walk_triples)
        if self.verbose:
            iter = walk_triples
        else:
            iter = tqdm(
                walk_triples,
                desc="Generating documentation",
                postfix={"tokens": 0} if self.count_tokens else None,
            )

        current_dir = 0
        for root, _, files in iter:
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

            rel_root = root.relative_to(self.input_dir)
            if self.verbose:
                if self.count_tokens:
                    click.echo(
                        f"[{current_dir}/{total_dirs}] Processing {rel_root} (tokens used so far: {self.total_tokens:,})"
                    )
                else:
                    click.echo(f"[{current_dir}/{total_dirs}] Processing {rel_root}")

            files = [root / f for f in files if (root / f).resolve() in filtered_files]

            prompt = self._build_prompt(root, files, generated_readmes)
            system_prompt = self._build_system_prompt(
                resolved_root == self.resolved_repo_dir
            )

            response = self._prompt_with_backoff(
                prompt,
                system=system_prompt,
            )

            if self.count_tokens:
                self.total_tokens += response.usage().input or 0
                self.total_tokens += response.usage().output or 0
                if not self.verbose:
                    iter.set_postfix({"tokens": f"{self.total_tokens:,}"})  # type: ignore
                if self.token_budget and self.total_tokens >= self.token_budget:
                    click.echo(
                        f"\nToken budget of {self.token_budget:,} exceeded. To continue, set a higher --token-budget."
                    )
                    return

            if self.rename_dot_github:
                parts = ["_github" if p == ".github" else p for p in rel_root.parts]
                renamed_rel_root = Path(*parts)
            else:
                renamed_rel_root = rel_root
            readme_path = self.docs_dir / renamed_rel_root / "README.md"
            readme_path.parent.mkdir(parents=True, exist_ok=True)
            readme_path.write_text(
                f"# {resolved_root.name}\n\n{response.text()}", encoding="utf-8"
            )

            # Store the generated README
            generated_readmes[root] = response.text()
        click.echo("Documentation generation complete.")

    def write_mkdocs_configuration(self) -> None:
        if self.write_gitignore:
            Path(self.output_dir / ".gitignore").write_text("*\n", encoding="utf-8")

        config_template = textwrap.dedent("""\
            site_name: {input_dir_name} docs by repo-guide
            theme: material
            exclude_docs: |
                !.*
                !/templates/
            hooks:
                - my_hooks.py
            copyright: AI-generated by <a href="https://github.com/wolfmanstout/repo-guide/" target="_blank">repo-guide</a>
            """)
        if self.repo_url:
            config_template += textwrap.dedent("""\
                repo_url: {repo_url}
                edit_uri:
                """)
        config_content = textwrap.dedent(
            config_template.format(
                input_dir_name=self.input_dir.resolve().name, repo_url=self.repo_url
            )
        )
        hooks_content = textwrap.dedent("""\
            import bleach
            from bleach_allowlist import markdown_tags, markdown_attrs

            def on_page_content(html, **kwargs):
                return bleach.clean(html, markdown_tags, markdown_attrs)
            """)
        Path(self.output_dir / "mkdocs.yml").write_text(
            config_content, encoding="utf-8"
        )
        Path(self.output_dir / "my_hooks.py").write_text(
            hooks_content, encoding="utf-8"
        )


@click.command()
@click.version_option()
@click.argument(
    "input_dir",
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
    default="gemini-1.5-flash-latest",
    show_default=True,
    help="LLM model to use",
)
@click.option(
    "--serve/--no-serve",
    default=True,
    show_default=True,
    help="Start local documentation server",
)
@click.option(
    "--open/--no-open",
    default=False,
    show_default=True,
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
    "--ignore",
    multiple=True,
    help="File name pattern to ignore (may be specified multiple times)",
)
@click.option(
    "--resume/--no-resume",
    default=True,
    show_default=True,
    help="Resume documentation generation from last stopping point",
)
@click.option(
    "--public/--local",
    default=False,
    show_default=True,
    help=(
        "Serve documentation on all network interfaces (0.0.0.0). "
        "Warning: This makes docs accessible to other devices on your network."
    ),
)
@click.option(
    "--token-budget",
    type=int,
    default=0,
    show_default=True,
    help="Approximate maximum number of tokens to use (0 for unlimited).",
)
@click.option(
    "--magika/--no-magika",
    default=False,
    show_default=True,
    help="Use magika to filter binary files instead of git metadata. Requires 'magika' extra.",
)
@click.option(
    "--files-token-limit",
    type=int,
    default=0,
    show_default=True,
    help=(
        "Maximum number of tokens for total file contents in a directory "
        "(0 for unlimited). Truncate file list if exceeded."
    ),
)
@click.option(
    "--write-gitignore/--no-write-gitignore",
    default=True,
    show_default=True,
    help="Write .gitignore file to output directory to ignore generated files.",
)
@click.option(
    "--custom-instructions",
    default="",
    help="Custom instructions to append to the system prompt",
)
@click.option(
    "--custom-instructions-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="File containing custom instructions to append to the system prompt",
)
@click.option(
    "--rename-dot-github",
    is_flag=True,
    help="Rename .github directory to _github in generated docs to avoid GitHub Pages issues",
)
def cli(
    input_dir: Path,
    model: str,
    serve: bool,
    open: bool,
    port: int,
    gen: bool,
    count_tokens: bool,
    output_dir: Path,
    ignore: tuple[str, ...],
    resume: bool,
    public: bool,
    verbose: bool,
    token_budget: int,
    magika: bool,
    files_token_limit: int,
    write_gitignore: bool,
    custom_instructions: str,
    custom_instructions_file: Path | None,
    rename_dot_github: bool,
) -> None:
    "Use AI to generate guides to code repositories."
    if custom_instructions_file:
        if custom_instructions:
            raise click.UsageError(
                "Only one of --custom-instructions and --custom-instructions-file may be specified"
            )
        custom_instructions = custom_instructions_file.read_text(encoding="utf-8")

    generator = DocGenerator(
        input_dir,
        output_dir,
        model,
        count_tokens,
        ignore_patterns=ignore,
        verbose=verbose,
        token_budget=token_budget,
        use_magika=magika,
        files_token_limit=files_token_limit,
        write_gitignore=write_gitignore,
        custom_instructions=custom_instructions,
        rename_dot_github=rename_dot_github,
    )

    # Create docs directory and write mkdocs config
    docs_dir = output_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    generator.write_mkdocs_configuration()

    # Start server if requested
    if open:
        serve = True
    server_thread = None
    if serve:
        url = f"http://127.0.0.1:{port}/"
        click.echo(f"Starting docs server at {url}" + (" (public)" if public else ""))
        host = "0.0.0.0" if public else "127.0.0.1"
        server_thread = threading.Thread(
            target=mkdocs_serve,
            args=(f"{output_dir}/mkdocs.yml",),
            kwargs={"dev_addr": f"{host}:{port}", "livereload": True},
            daemon=True,
        )
        server_thread.start()
        if open:
            webbrowser.open(url)

    if gen:
        # Only remove existing docs if not resuming
        if not resume and docs_dir.exists():
            for item in docs_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()

        # Generate documentation
        generator.generate_docs(resume=resume)

        if count_tokens:
            if generator.total_tokens:
                click.echo(f"Total tokens used: {generator.total_tokens:,}")
            else:
                click.echo("Unable to count tokens. Add --no-count-tokens to disable.")

    # If we're serving but not generating, verify docs exist
    elif serve and not any(docs_dir.iterdir()):
        click.echo("Warning: No documentation found in output directory.")
        click.echo(
            "The server is running but you may want to use --gen to generate docs."
        )

    # If serving, keep the main thread alive
    if server_thread:
        click.echo(f"Server running at http://127.0.0.1:{port}/. Press Ctrl+C to stop.")
        try:
            while True:
                server_thread.join(1.0)
        except KeyboardInterrupt:
            click.echo("\nStopping server...")
