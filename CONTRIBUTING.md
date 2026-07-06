# Contributing to SysPlug

Thanks for your interest in improving SysPlug! This document describes how to
set up a development environment and the conventions we follow.

## Development setup

```bash
git clone https://github.com/arpitsinghgautam/sysplug.git
cd sysplug
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Optional framework extras: `pip install -e ".[dev,hf,deepspeed]"`.

## Running the checks

All of these must pass before a PR is merged (CI enforces them):

```bash
pytest                      # unit + integration tests; coverage must stay >= 85%
ruff check sysplug tests    # lint
ruff format --check .       # formatting (or: black --check .)
mypy sysplug                # static types
```

A one-shot pre-commit hook is provided:

```bash
pre-commit install          # runs ruff/black/mypy on staged files
```

### GPU validation (optional)

Most of the suite runs on CPU with mocked hardware. If you have an NVIDIA GPU,
you can validate the analytic models against real measurements:

```bash
python -m paper.experiments.measure_gpu --configs gpt2-small gpt2-medium
```

## Conventions

- **Style:** ruff + black, line length 100. Prefer clear names over comments.
- **Types:** the package is fully typed and ships a `py.typed` marker; keep
  `mypy --strict` clean.
- **Tests:** every bug fix adds a regression test; every feature adds coverage.
  Numeric model changes should pin exact values (see
  `tests/unit/test_*_numeric.py`).
- **Commits:** use [Conventional Commits](https://www.conventionalcommits.org/)
  (`fix:`, `feat:`, `docs:`, `chore:`, `test:`…).
- **Changelog:** add a bullet under `## [Unreleased]` in `CHANGELOG.md`.

## Pull requests

1. Fork and create a feature branch: `git checkout -b feat/my-change`.
2. Make the change with tests and docs.
3. Ensure `pytest`, `ruff`, and `mypy` are green locally.
4. Open a PR describing **what** changed and **why**, and link any issue.

## Scientific honesty

SysPlug makes quantitative predictions. Any performance or accuracy number in
the README, docs, or paper must be **measured and reproducible**, or explicitly
labelled as an illustrative/analytic estimate. Please do not add benchmark
numbers you have not run.

## Reporting bugs

Open an issue using the bug-report template. Include your OS, Python version,
GPU (if any), and a minimal reproduction.
