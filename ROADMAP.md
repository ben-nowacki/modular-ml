# ModularML Roadmap

## v1.0.0 — Core ModularML Pipeline  
![Progress](https://img.shields.io/badge/progress-67%25-yellow)

**Target Release:** Dec 2025
*(Single breaking overhaul release; all features land together)*


---

### Data Structures & Serialization
- [x] Refactor `FeatureSet`, `FeatureSubset`, `Batch`, and related structures to use **PyArrow tables**
- [x] Implement **zero-copy subset & sampler views** over parent FeatureSet tables
- [x] Ensure data loads into memory **only when needed** for ModelGraph execution
- [ ] Make all components fully **serializable** (FeatureSets, ModelGraphs, Stages, Samplers, Losses, Phases)
- [ ] Support exporting Experiments as:
  - [ ] Full state (post-training, weights included)
  - [ ] Config-only (reproducible structure, no weights)

---

### Experiment Context & Tracking
- [x] Implement automatic **Experiment context binding** for all defined components
- [ ] Add conflict detection for mismatched component/Experiment associations
- [ ] Store all outputs (loss curves, metrics, results, figures) linked to their source phase

---

### FeatureSet / Splitting / Sampling

#### FeatureSet
- [x] Fully structured feature–target–tag schema
- [x] Per-column scaling/normalization with tracked transform pipelines

#### Splitting
- [x] Ratio-based random splits
- [x] Rule-based conditional splits (user-defined criteria)

#### Sampling
- [x] Sample-wise batching
- [x] N-Sampler-based paired sampling
- [ ] N-Sampler-based triplet sampling

---

### ModelGraph
- [x] Support sequential, branching, and merging DAGs
- [x] Validate graph connectivity before training
- [x] Add graph visualization utility (Graphviz/Dot)

---

### ModelStage
- [x] Unified wrappers for PyTorch, TensorFlow, and scikit-learn
- [x] Built-in PyTorch models (Sequential MLP, CNN encoder)
- [x] Merge stages supporting concatenation, stacking, and padded stacking
- [x] Backend-agnostic forward, training-step, and eval-step APIs

---

### Experiment / TrainingPhase / EvaluationPhase
- [x] Experiment holds static FeatureSets, splits, and ModelGraph
- [ ] Support multiple independent **Training** and **Evaluation** phases
- [ ] Each phase configurable with samplers, losses, optimizers, and trackers
- [ ] Store and version phase results in the Experiment instance

---
### Unit Testing
- [x] Add nox-based automated unit, integration, example, and doc test routines
- [ ] Increase code coverage to $\geq$ 90%

---

## v1.1.0 — Multi-Experiment Container & Comparison  
![Progress](https://img.shields.io/badge/progress-0%25-red)

**Target Release:** March 2026

- [ ] Add higher-level **ExperimentCollection** container
- [ ] Support grouping Experiments for shared evaluation pipelines
- [ ] Provide unified comparison utilities across Experiments (metrics, plots, tables)
- [ ] Enable rapid testing of alternative ModelGraphs, architectures, or FeatureSets within the same task