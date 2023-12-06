# Representation Engineering Benchmark

Codebase for benchmarking Representation Engineering

## Quickstart

### Dependencies

We use [PDM](https://github.com/pdm-project/pdm) to manage dependencies. We recommend `pipx`` installation. The documentation describes alternative installation methods.
```bash
pipx install pdm
```

### Installation

Setup is as simple as:
```bash
pdm install
pdm run pre-commit install
```

This creates a virtual environment in `.venv` with the appropriate libraries, installed from the PDM lock file. We support Python >=3.10

The python interpreter can be accessed with `pdm run`. Scripts can be run with `pdm run <my_script.py>`.

### Troubleshooting
If you cannot install the dependencies, try re-compiling with `pdm lock`.


## Development

Activate the virtual environment with:
``` bash
source .venv/bin/activate
```
This will enable the pre-commit hooks to run on commit.

Branching follows [Github workflow](https://githubflow.github.io/)

### Code Style

Run `make check` to check code format.

Run `make fmt` to run Black, Ruff linters.

Run `make typecheck` to run Pyright type checking

Run `make test` to run Pytest
