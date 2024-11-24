# gitscout

[![PyPI](https://img.shields.io/pypi/v/gitscout.svg)](https://pypi.org/project/gitscout/)
[![Changelog](https://img.shields.io/github/v/release/wolfmanstout/gitscout?include_prereleases&label=changelog)](https://github.com/wolfmanstout/gitscout/releases)
[![Tests](https://github.com/wolfmanstout/gitscout/actions/workflows/test.yml/badge.svg)](https://github.com/wolfmanstout/gitscout/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/wolfmanstout/gitscout/blob/master/LICENSE)

Uses AI to help understand repositories and their changes.

NOTE: this has not yet been released and is still in active development.

## Installation

Install this tool using `pip` or `pipx`:

```bash
pip install gitscout
```

## Usage

For help, run:

```bash
gitscout --help
```

You can also use:

```bash
python -m gitscout --help
```

## Development

To contribute to this tool, use uv. The following command will establish the
venv and run tests:

```bash
uv run pytest
```

To run gitscout locally, use:

```bash
uv run gitscout
```
