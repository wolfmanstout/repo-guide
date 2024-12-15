# repo-guide

[![PyPI](https://img.shields.io/pypi/v/repo-guide.svg)](https://pypi.org/project/repo-guide/)
[![Changelog](https://img.shields.io/github/v/release/wolfmanstout/repo-guide?include_prereleases&label=changelog)](https://github.com/wolfmanstout/repo-guide/releases)
[![Tests](https://github.com/wolfmanstout/repo-guide/actions/workflows/test.yml/badge.svg)](https://github.com/wolfmanstout/repo-guide/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/wolfmanstout/repo-guide/blob/master/LICENSE)

Uses AI to help understand repositories and their changes.

NOTE: this has not yet been released and is still in active development.

## Installation

Install this tool using `pip` or `pipx`:

```bash
pip install repo-guide
```

## Usage

For help, run:

```bash
repo-guide --help
```

You can also use:

```bash
python -m repo_guide --help
```

## Development

To contribute to this tool, use uv. The following command will establish the
venv and run tests:

```bash
uv run pytest
```

To run repo-guide locally, use:

```bash
uv run repo-guide
```
