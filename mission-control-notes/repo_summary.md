# ModularML repository summary

ModularML is a pure-Python, backend-agnostic machine learning framework for
composing, training, and serializing research ML pipelines. This document maps the
repository for developers (including agents) working on the codebase: layout, main
components, tech stack, entry points, and the conventions the repo enforces.

## Contents

- [What this repository is](#what-this-repository-is)
- [Tech stack](#tech-stack)
- [Repository layout](#repository-layout)
- [Core architecture](#core-architecture)
- [Entry points](#entry-points)
- [Serialization and the mml format](#serialization-and-the-mml-format)
- [Tests, linting, and commands](#tests-linting-and-commands)
- [Code conventions](#code-conventions)
- [Documentation](#documentation)
- [Git remotes and history](#git-remotes-and-history)
- [Mission Control notes in this repo](#mission-control-notes-in-this-repo)

---

## What this repository is

The distributable is a single Python package, `modularml`, published to PyPI as
`modularml` and documented at `modular-ml.readthedocs.io`. It is a library, not an
application: there is no server, no frontend, no database, and no CLI entry point.

There is no frontend, no FastAPI service, and no Alembic migration directory in this
repository, so the frontend, API, and database guides listed in
`mission-control-notes/guides/index.md` do not apply to any task here. The guides
that do apply are the Python ones (docstrings, error handling, testing, TDD,
documentation writing, library docs, context efficiency, security, logging) plus the
git and PR guide.

Scale as of this summary: 204 Python modules and roughly 54,800 lines under
`modularml/`, plus 16 test modules under `tests/`.

Upstream project: `github.com/REIL-UConn/modular-ml`. Authors are researchers at the
University of Connecticut; the development status is Alpha and the active roadmap
milestone is v1.0.0 in `ROADMAP.md`.

## Tech stack

| Area | Choice |
|---|---|
| Language | Python, `requires-python >= 3.10`, CI matrix 3.10 / 3.11 / 3.12 |
| Build | setuptools with `setuptools_scm` for dynamic versioning |
| Data storage | Apache Arrow via `pyarrow`, with `numpy`, `pandas`, `scipy` |
| ML backends | Optional `torch`, optional `tensorflow`, `scikit-learn` (core dep) |
| Optional extras | `optuna` and `scikit-optimize` for optimization, `jupyter` for examples |
| Terminal output | `rich` (progress bars, formatters) |
| Config and serialization | `pyyaml`, `joblib`, zipped artifact bundles |
| Plotting | `matplotlib`, plus Mermaid diagrams for structural views |
| Lint and format | `ruff` (lint + format) through `pre-commit` |
| Tests | `pytest` with `pytest-cov`, `pytest-xdist`, `nbmake`, `pytest-doctestplus` |
| Task runner | `nox` (`noxfile.py`) |
| Docs | Sphinx with `myst_nb`, `pydata-sphinx-theme`, `sphinxcontrib.mermaid` |
| CI | GitHub Actions (`.github/workflows/test_on_push.yml`), Codecov, Read the Docs |

Core dependencies stay lightweight on purpose. `torch` and `tensorflow` are optional
extras and must never be imported at module import time (see
[Code conventions](#code-conventions)).

Install extras: `torch`, `tensorflow`, `optimize`, `docs`, `examples`, `dev`, and the
aggregates `all`, `all-torch`, `all-tf`.

## Repository layout

```text
modularml/            The library (see table below)
tests/                pytest suite: tests/unit/, tests/integration/
docs/                 Sphinx site: explanation/, how_to/, tutorials/, reference/
.github/              CI workflows, issue templates, PR template
mission-control-notes/ Mission Control guides, specs, decisions, this summary
noxfile.py            All test, lint, docs, and doctest sessions
pyproject.toml        Package metadata, dependencies, pytest and coverage config
ruff.toml             Lint and format rules (line length 88, broad rule set)
.pre-commit-config.yaml  ruff, hygiene hooks, nbstripout
ROADMAP.md            Milestone checklists; badges auto-updated by CI
CONTRIBUTING.md       Contributor workflow and local check commands
```

### Package map

| Path | Responsibility |
|---|---|
| `modularml/api.py` | Curated re-export surface; the single list of public names |
| `modularml/__init__.py` | Re-exports `api`, calls `register_all()`, activates a default `ExperimentContext` |
| `modularml/registry.py` | `register_all()`: wires every subpackage registry at import |
| `modularml/predicates.py` | Serializable `Predicate` conditions for `ConditionSplitter` |
| `modularml/core/data/` | `FeatureSet`, `FeatureSetView`, `SampleCollection`, `Batch`, schema constants |
| `modularml/core/topology/` | `GraphNode`, `ComputeNode`, `ModelNode`, `MergeNode`, `ConcatNode`, `ModelGraph` |
| `modularml/core/models/` | `BaseModel` plus Torch, TensorFlow, and scikit-learn wrappers |
| `modularml/core/experiment/` | `Experiment`, `ExperimentContext`, phases, results containers, checkpointing |
| `modularml/core/execution/` | Execution strategies; `CrossValidation`, `CVBinding`, `CVResults` |
| `modularml/core/training/` | `Loss`, `AppliedLoss`, `Optimizer`, loss records |
| `modularml/core/sampling/` | `BaseSampler`, batcher, `SimilarityCondition` |
| `modularml/core/splitting/` | `BaseSplitter`, split mixin, `SplitterRecord` |
| `modularml/core/transforms/` | `Scaler` wrapper and `ScalerRecord` |
| `modularml/core/references/` | Lazy symbolic references resolved at execution time |
| `modularml/core/io/` | Serialization: handlers, YAML translators, symbol registry, migrations |
| `modularml/core/settings.py` | `settings` singleton and `configure()` |
| `modularml/samplers/` | Built-ins: simple, N-sampler, paired, triplet |
| `modularml/splitters/` | Built-ins: random, condition-based |
| `modularml/scalers/` | Built-ins: absolute, negate, per-sample min-max, per-sample zero, segmented |
| `modularml/models/torch/` | Built-in Torch templates: `SequentialMLP`, `SequentialCNN`, `TemporalCNN` |
| `modularml/callbacks/` | `EarlyStopping`, evaluation, metrics, artifact results |
| `modularml/visualization/` | Mermaid `Visualizer` for FeatureSets, graphs, and phases |
| `modularml/utils/` | Data conversion, backend detection, logging, progress bars, errors, nn helpers |

The largest subpackages by line count are `core/experiment` (about 11,600 lines),
`core/data` (7,200), `core/topology` (7,100), `utils` (6,500), and `core/io` (5,800).

## Core architecture

The framework is designed as four layers, each unaware of the layer above it.
`docs/explanation/architecture_overview.md` is the authoritative long-form version
and should be read before any structural change.

1. **Data storage.** `FeatureSet` wraps a `SampleCollection`, an immutable Arrow
   table whose columns are named `<domain>.<key>.<representation>` (for example
   `features.velocity.raw`, `targets.label.transformed`). Domains are features,
   targets, tags, and sample_uuids. Scalers add `transformed` columns instead of
   overwriting `raw`, so transforms are reversible. Every sample carries a UUID that
   survives splits, views, batches, and folds.
2. **Data processing.** Splitters partition a FeatureSet into named
   `FeatureSetView` objects (row indices plus column names, zero-copy). Scalers are
   applied through `fit_transform()` and recorded as `ScalerRecord` entries so
   `undo_last_transform()` and `undo_all_transforms()` can roll back. Samplers yield
   `Batch` objects organized by **roles** (concurrent inputs such as anchor,
   positive, negative) and **streams** (named output branches).
3. **Graph topology.** `ModelGraph` is a DAG of `GraphNode` subclasses. `ModelNode`
   holds a `BaseModel` (Torch, TensorFlow, or scikit-learn); `MergeNode` and
   `ConcatNode` combine upstream outputs. Nodes are wired with references
   (`FeatureSetReference`, `ModelIOReference`) resolved at execution time, which is
   what makes serialization, templating, and cross-validation possible.
   `build()` propagates shapes so mismatches surface before training.
4. **Orchestration.** `ExperimentContext` is a `ContextVar`-backed registry mapping
   node IDs and labels, and references resolve against it. `Experiment` owns a
   `PhaseGroup` of phases and runs them in order: `TrainPhase` (iterative,
   backpropagating), `FitPhase` (single-pass `.fit()` for closed-form models), and
   `EvalPhase` (forward only). `AppliedLoss` binds a `Loss` to a node output and a
   target domain. Per-batch transient state lives in an `ExecutionContext` that
   callbacks and losses can inspect.

Cross-cutting design points worth knowing before editing:

- **Configurable / Stateful split.** `get_config()` captures structure;
  `get_state()` and `set_state()` capture runtime values (weights, fitted scaler
  parameters, optimizer momentum). Keep that separation when adding components.
- **Registries.** Samplers, splitters, scalers, losses, optimizers, models, and
  node types resolve from named registries so they can be referenced by string.
  `CaseInsensitiveRegistry` in `modularml/utils/registries.py` backs lookups. New
  built-ins must be added to the relevant `registry.py` and picked up by
  `register_all()`.
- **Backend neutrality.** `Backend` (`modularml/utils/nn/backend.py`) tags data and
  models as `torch`, `tensorflow`, `numpy`, `scikit`, or `none`. The framework
  validates backend compatibility across a graph to decide whether graph-wide
  training is possible or whether stages must train independently.
- **Zero-copy by default.** Splits, column selections, and batch construction
  produce views or slices. Materialization happens as late as possible, at the
  boundary where tensors are handed to a backend.

## Entry points

There is no executable entry point. The surfaces that matter are:

- `import modularml as mml` runs `modularml/__init__.py`, which re-exports the
  public API, calls `register_all()`, and installs a default active
  `ExperimentContext`. Anything intended to be public must be added to
  `modularml/api.py` and its `__all__`, and then to `docs/reference/api.rst`.
- `Experiment.run()` (`modularml/core/experiment/experiment.py`) is the main
  execution call. `Experiment.save()` / `Experiment.load()` and
  `Experiment.save_checkpoint()` are the persistence surface.
- `FeatureSet.from_dict()`, `.from_pandas()`, and `.from_pyarrow_table()` are the
  data ingestion constructors; `FeatureSet.save()` / `.load()` persist one set.
- Subpackage imports are the documented path for built-ins:
  `from modularml.samplers import ...`, `modularml.splitters`,
  `modularml.scalers`, `modularml.models`, `modularml.callbacks`.
- `nox` sessions in `noxfile.py` are the developer entry points for every check.

## Serialization and the mml format

Reproducibility is the project's core goal, so serialization is first-class and
lives in `modularml/core/io/`.

- Artifacts are zipped bundles with the extension `.<kind>.mml`. Kind codes are
  registered in each subpackage's `registry.py` through `SerializationKind`:
  `fs` (FeatureSet), `mg` (ModelGraph), `exp` (Experiment), `md` (BaseModel),
  `sm` (sampler), `sp` (splitter), `sc` (Scaler), `ls` (Loss), `op` (Optimizer),
  `smc` (SimilarityCondition), `ckpt` (Checkpoint), `cvr` (CVResults).
- Every artifact carries an `ArtifactHeader` with `mml_version`, `schema_version`,
  `object_version`, `kind`, and a `symbol_spec` payload, so loads can be migrated.
  `modularml/core/io/migrations/` holds the migration registry.
- `KindRegistry` resolves kinds by MRO, so subclasses inherit the first matching
  base class kind.
- `SymbolRegistry` and `SymbolSpec` resolve user-defined classes and callables on
  load; packaged code loaders handle code shipped inside an artifact.
- YAML translators under `modularml/core/io/yaml/` convert configs for experiments,
  model graphs, phases, samplers, splitters, callbacks, and references.
- Avoid raw lambdas in anything that must serialize. Use `Predicate`
  (`modularml/predicates.py`) for `ConditionSplitter` conditions and the `Loss` /
  `Scaler` wrappers for user callables.

## Tests, linting, and commands

Run everything through `nox`; the CI workflow runs the same sessions.

```bash
nox -s pre-commit    # ruff lint + format and hygiene hooks, all files
nox -s unit          # pytest -m unit
nox -s integration   # pytest -m integration (exit code 5 is accepted)
nox -s coverage      # unit + integration with coverage XML
nox -s examples      # nbstripout then pytest --nbmake over docs notebooks
nox -s doctests      # pytest --doctest-modules --doctest-plus modularml
nox -s docs          # sphinx-build -W (sphinx-autobuild when interactive)
nox -s dev           # create ./venv with .[all,dev]
```

Test layout and facts:

- `tests/unit/` currently covers only `core/data` (batch, batch view, execution
  context, featureset, featureset view, registry, sample collection, sample data,
  sampled view, sample schema, schema constants). The rest of the library has no
  unit tests yet, so new work in `core/experiment`, `core/topology`, or `core/io`
  is adding coverage rather than extending it.
- `tests/integration/` holds only `test_placeholder.py`.
- Markers are strict (`--strict-markers`) and only `unit` and `integration` exist.
  Every test needs an explicit `@pytest.mark.unit` or `@pytest.mark.integration`,
  because the nox sessions select by marker and an unmarked test runs in neither.
- `tests/conftest.py` provides an autouse `_fresh_experiment_context` fixture that
  activates a new `ExperimentContext` per test to avoid label collisions, plus
  `generate_dummy_data()` and `generate_dummy_featureset()` helpers. Use those
  helpers rather than hand-building FeatureSets.
- Docs notebooks are executed in CI, so a public API change that breaks a notebook
  breaks the build. Notebook outputs are stripped by `nbstripout`.
- Sphinx runs with `-W` in CI, so any new docs warning is a failure.

## Code conventions

- **Formatting.** ruff with line length 88, double quotes, 4-space indent, target
  `py310`. `fix = true` is set, so ruff auto-fixes on run. The lint rule set is
  broad (`E`, `W`, `I`, `N`, `D`, `UP`, `B`, `S`, `TRY`, `PL`, `RUF`, and more);
  see `ruff.toml` for the ignore list and per-file exemptions.
- **Naming.** Classes are CamelCase, functions and variables are snake_case.
- **Docstrings.** PEP257, Google-style sections with types in the signature line
  (`shape (tuple[int, ...]): ...`). Many classes use a `Description:` block and an
  `Attributes:` block. Follow `mission-control-notes/guides/python_docstring_style.md`.
- **Errors.** Raise from the `ModularMLError` hierarchy in
  `modularml/utils/errors/exceptions.py` rather than bare built-ins where a typed
  error exists (`BackendMismatchError`, `NodeNotFoundError`, `FeatureSetError`,
  `ShapeSpecError`, and others). `EM101` is ignored but the codebase still assigns
  messages to `msg` before raising, which matches `EM` style; keep that pattern.
- **Optional backends.** Never import `torch` or `tensorflow` at module scope in
  core code. Use `ensure_torch()`, `check_torch()`, `ensure_tensorflow()`, and
  `check_tensorflow()` from `modularml/utils/environment/optional_imports.py`.
- **Logging.** Use `get_logger(...)` from `modularml/utils/logging/logger.py`.
- **Public API.** Adding a public class means touching three places:
  `modularml/api.py` (import plus `__all__`), the subpackage `registry.py` if it is
  registrable or serializable, and `docs/reference/api.rst`.
- **Settings.** Package-wide options live in `modularml/core/settings.py` and are
  settable via `configure()` or an env var (for example
  `MODULARML_REQUIRE_UNIQUE_LABELS`).
- **Markdown and docs text.** ASCII only, no em dashes, no section symbols; see
  `mission-control-notes/guides/documentation_writing_guide.md`.

## Documentation

`docs/` is a Sphinx site in the Diataxis shape, published to Read the Docs via
`.readthedocs.yaml` (Python 3.10, `docs` extra).

- `docs/explanation/` holds the design rationale: `architecture_overview.md`,
  `model_graph_design.md`, `training_phases.md`, `experiment_design.md`. Read these
  before changing core abstractions.
- `docs/how_to/` holds nine notebooks covering FeatureSets, ModelNodes, ModelGraphs,
  MergeNodes, Experiments, scalers, hardware acceleration, callbacks, and cross
  validation, with shared helpers in `docs/how_to/utils/`.
- `docs/tutorials/` holds eight end-to-end notebooks, including battery-domain
  examples, with helpers in `docs/tutorials/utils/`.
- `docs/reference/api.rst` autodocs `modularml.api`, which is why the curated
  export surface matters.

## Git remotes and history

- `origin` is `github.com/ben-nowacki/modular-ml.git` (the fork used day to day);
  `upstream` is `github.com/REIL-UConn/modular-ml.git`. Default branch is `main`.
- Commit subjects in history are terse, capitalized, sentence-like, with no
  conventional-commit prefixes (for example
  `Added package-wide settings. Added support for serialized result containers`).
  Match that style; follow
  `mission-control-notes/guides/git_branch_and_pr_guide.md` for branch naming and
  PR contents.
- `.github/PULL_REQUEST_TEMPLATE.md` asks for a description, related issues, how
  the change was tested, and a checklist referencing the nox sessions. Treat it as
  an outline and fill it with facts about the change.
- `ROADMAP.md` progress badges are regenerated by
  `.github/workflows/update_roadmap.yml` on every push that touches the file, and
  that workflow commits back to the branch. Do not hand-edit the badge URLs.
- Never attribute repository work to Claude, Codex, an AI, or an agent anywhere in
  commits, PRs, comments, docs, or history.

## Mission Control notes in this repo

- `CLAUDE.md` and `AGENTS.md` are identical copies of the Mission Control
  conventions and point here plus at `mission-control-notes/guides/index.md`.
- `mission-control-notes/guides/index.md` is written for a different project (a
  FastAPI plus React application) and lists guides for Alembic, FastAPI routes,
  SQLAlchemy queries, React, Tailwind, and a Radix design system. None of those
  domains exist in this repository; skip those guides and follow the Python, docs,
  security, and git ones.
- `.claude/skills/frontend-design/` and `.agents/skills/frontend-design/` are
  duplicated skill assets from that same template and are unused here.
