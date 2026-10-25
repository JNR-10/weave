# One entry point for people, coding agents and CI. `make check` is the definition of green.
# The venv interpreter is called by path, so shell aliases and an unactivated venv do not matter.

PY311 ?= python3.11
PYTHON ?= .venv/bin/python
CODE := src scripts tests

.PHONY: setup setup-m4 check lint test fmt hooks

setup: ## Portable environment: Intel or Apple Silicon, no MLX
	$(PY311) -m venv .venv
	$(PYTHON) -m pip install -r experiments/phase1/requirements-portable-lock.txt
	$(PYTHON) -m pip install --no-build-isolation --no-deps -e .

setup-m4: ## Apple Silicon environment with MLX (native arm64 Python, never Rosetta)
	$(PY311) -m venv .venv
	$(PYTHON) -m pip install -r experiments/phase1/requirements-m4-lock.txt
	$(PYTHON) -m pip install --no-build-isolation --no-deps -e .

check: lint test ## Everything CI runs

lint:
	$(PYTHON) -m ruff check $(CODE)
	$(PYTHON) -m ruff format --check $(CODE)

test:
	$(PYTHON) -m pytest -q

fmt: ## Apply lint fixes and formatting
	$(PYTHON) -m ruff check --fix $(CODE)
	$(PYTHON) -m ruff format $(CODE)

hooks: ## Run the fast checks on every commit (optional, local only)
	$(PYTHON) -m pip install pre-commit==4.6.2
	$(PYTHON) -m pre_commit install
