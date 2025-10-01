import os
from pathlib import Path

import nox

# Default Nox behavior
nox.options.default_venv_backend = "uv|virtualenv"
nox.options.reuse_existing_virtualenvs = True
nox.options.sessions = ["pre-commit", "unit"]

# Environment variables for consistent runs
PROJECT_ENV = {
    "PYTHONIOENCODING": "utf-8",
    "MPLBACKEND": "Agg",  # avoids GUI issues with matplotlib
}
VENV_DIR = Path("./venv").resolve()


def set_env(session, env_dict):
    """Helper: apply environment variables to session."""
    for key, value in env_dict.items():
        session.env[key] = value


@nox.session(name="unit")
def run_unit(session):
    """Run unit tests."""
    set_env(session, PROJECT_ENV)
    session.install("-e", ".[all,dev]", silent=False)  # editable + dev deps
    session.run("pytest", "-m", "unit")


@nox.session(name="integration")
def run_integration(session):
    """Run integration tests."""
    set_env(session, PROJECT_ENV)
    session.install("-e", ".[all,dev]", silent=False)  # editable + dev deps
    session.run("pytest", "-m", "integration")


@nox.session(name="coverage")
def run_coverage(session):
    """Run tests with coverage tracking."""
    set_env(session, PROJECT_ENV)
    session.install("coverage", "pytest-cov", silent=False)
    session.install("-e", ".[all,dev]", silent=False)
    session.run("pytest", "-m", "unit or integration", "--cov=modularml", "--cov-report=xml", "tests/")


@nox.session(name="examples")
def run_examples(session):
    """Run Jupyter notebook examples with nbmake."""
    set_env(session, PROJECT_ENV)
    session.install("-e", ".[all,dev]", "nbmake", silent=False)
    notebooks = session.posargs if session.posargs else ["examples/"]
    session.run("pytest", "--nbmake", *notebooks, external=True)


@nox.session(name="pre-commit")
def lint(session):
    """Run pre-commit hooks (lint, format, etc.)."""
    session.install("pre-commit", silent=False)
    session.run("pre-commit", "run", "--all-files")


@nox.session(name="dev")
def dev_env(session):
    """Create a reusable developer venv with all dependencies."""
    set_env(session, PROJECT_ENV)
    session.install("virtualenv")
    session.run("virtualenv", os.fsdecode(VENV_DIR), silent=True)
    python = os.fsdecode(VENV_DIR.joinpath("bin/python"))
    session.run(python, "-m", "pip", "install", "-e", ".[all,dev]", external=True)


@nox.session(name="all_tests", reuse_venv=True)
def run_all_tests(session):
    """Run unit tests and integration tests."""
    run_unit(session)
    run_integration(session)
