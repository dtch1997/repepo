# Representation Engineering Benchmark

Codebase for benchmarking Representation Engineering

## Installation

First make a virtual environment and activate it. We support Python 3.9, 3.10

```bash
# Example with virtualenv; you can also use Conda
python3.10 -m venv .venv
source .venv/bin/activate
```

Then, install dependencies
```bash 
make init
```

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
