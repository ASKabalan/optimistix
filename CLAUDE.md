# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Optimistix is a JAX library for nonlinear solvers: root finding, minimisation, fixed points, and least squares. Built on JAX, Equinox, and Lineax.

## Commands

### Testing
```bash
pytest                          # Run all tests
pytest tests/test_minimise.py   # Run single test file
pytest -k "test_name"           # Run tests matching pattern
```

Tests require JAX x64 precision and strict dtype/rank promotion (configured in `tests/conftest.py`).

### Linting and Formatting
```bash
pre-commit run --all-files      # Run all pre-commit hooks
ruff format optimistix tests    # Format code
ruff check --fix optimistix tests  # Lint with auto-fix
pyright                         # Type checking
```

### Installation
```bash
pip install -e '.[dev,tests]'   # Development install
pre-commit install              # Set up git hooks
```

## Architecture

### Core Abstractions (`optimistix/_iterate.py`)
All solvers inherit from `AbstractIterativeSolver`, which defines the iterative solve loop. Problem-specific base classes:
- `AbstractRootFinder`
- `AbstractMinimiser`
- `AbstractLeastSquaresSolver`
- `AbstractFixedPointSolver`

### Entry Points
Four main functions with unified signatures:
- `minimise(fn, solver, y0, ...)` - minimize scalar functions
- `root_find(fn, solver, y0, ...)` - find roots
- `least_squares(fn, solver, y0, ...)` - solve least-squares
- `fixed_point(fn, solver, y0, ...)` - find fixed points

All return a `Solution` object with `value`, `result` (status enum), `aux`, `stats`, and `state`.

### Solver Implementations (`optimistix/_solver/`)
26 solvers including Newton, BFGS, L-BFGS, Gauss-Newton, Levenberg-Marquardt, Nelder-Mead, NonlinearCG, and Optax integration.

### Interoperability
Problem types can be converted: root-finding via least-squares/minimisation, least-squares via minimisation, fixed-point via root-finding.

### Adjoint Methods (`optimistix/_adjoint.py`)
- `ImplicitAdjoint` (default) - implicit function theorem
- `RecursiveCheckpointAdjoint` - checkpoint iterates

## Code Patterns

### Type Annotations
Uses `jaxtyping` for array shape/dtype annotations. Common type variables: `Y` (solution), `Out` (output), `Aux` (auxiliary).

### Equinox Integration
- Solvers are `eqx.Module` subclasses
- Use `@eqx.filter_jit` for JIT compilation
- `AbstractVar` for solver hyperparameters (rtol, atol, norm)
- PyTree-based state for JAX compatibility

### Import Aliases
Configured in pyproject.toml:
- `collections` as `co`
- `functools` as `ft`
- `itertools` as `it`

## Pyright Configuration
- `reportIncompatibleVariableOverride = false` - required for `eqx.AbstractVar`
- `reportInvalidTypeForm = false` - required for `FunctionInfo.Foo`
