# Representation Engineering Benchmark

Codebase for benchmarking Representation Engineering

## Installation

We use [PDM](https://github.com/pdm-project/pdm) to manage dependencies. We recommend `pipx`` installation. The documentation describes alternative installation methods.
```bash
pipx install pdm
```

Setup is as simple as:
```bash
pdm install
```

This creates a virtual environment in `.venv` with the appropriate libraries, installed from the PDM lock file.

The python interpreter can be accessed with `pdm run`. Scripts can be run with `pdm run <my_script.py>`.

### Troubleshooting
If you cannot install the dependencies, try re-compiling with `pdm lock`.


## Development

branch structure:
`main` -> `dev` -> `[insert-name]-dev`

To update dependencies, modify `requirements/main.in`.

Then compile and freeze dependencies:
```bash
make update-deps
```

This creates frozen requirements in `requirements/main.txt` and `requirements.dev/txt` for reproducible installs.

### Code Style

Run `make check` to check code format.

Run `make fmt` to run Black, Ruff linters.
