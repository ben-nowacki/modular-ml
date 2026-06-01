---
title: 'ModularML: A Backend-Agnostic, Declarative Framework for Reproducible and Modular Machine Learning Experiments'
tags:
  - python
  - machine-learning
  - experimentation
  - reproducibility
  - research-tools
authors:
  - name: Ben Nowacki
    orcid: 0009-0000-6723-3085
    affiliation: "1"
  - name: Tingkai Li            # TODO: add orcID?
    affiliation: "1"
  - name: Mohammad Mundiwala    # TODO: include? add orcID?
    affiliation: "1"
  - name: Hui Hua               # TODO: include? add orcID?
    affiliation: "1"
  - name: Fei Miao              # TODO: add orcID?
    affiliation: "1"
  - name: Chao Hu               # TODO: add orcID?
    affiliation: "1"
    corresponding: true
affiliations:
 - name: University of Connecticut, United States
   index: 1
date: 2 June 2026
bibliography: paper.bib
---

# Summary

Modern machine learning (ML), particularly deep learning, has become central to scientific research across domains such as energy systems, materials science, structural health monitoring, and biomedical engineering.
However, most ML frameworks remain optimized for users with formal computer science backgrounds.
As a result, domain researchers often rely on fragmented scripts, backend-specific conventions, and ad hoc experiment management workflows that limit reproducibility and collaboration.

`ModularML` is an open-source Python framework designed to address this gap.
It provides a backend-agnostic, declarative, and fully serializable architecture for ML experimentation, enabling users to define complete workflows---including data curation, feature processing, model architecture, training strategy, loss configuration, and evaluation---using modular, composable components that serialize into a single reproducible artifact.

At its core, `ModularML` introduces:
* a FeatureSet abstraction for structured, backend-agnostic data handling and splitting,
* a directed acyclic ModelGraph composed of independent ModelNodes,
* multi-phase training orchestration with configurable freeze/unfreeze policies and phase- and node-specific loss definitions,
* and full pipeline serialization and visualization for transparent sharing and inspection of experiments.

By separating experiment definition from backend implementation, `ModularML` allows researchers to collaborate and audit ML studies without requiring expertise in `PyTorch` [@pytorch], `Keras` [@keras], or other specific ML libraries.


# Statement of need

ML workflows in academia differ substantially from typical industry-focused ML development.
Scientific researchers emphasize experimentation with domain-specific feature selection, sampling, and model architecture.
Existing libraries such as `PyTorch Lightning` [@pytorch_lightning], `Catalyst` [@catalyst], `FastAI` [@fastai], and `PyTorch Ignite` [@pytorch_ignite] can streamline training loops, but typically remain tied to a single backend, prioritize development productivity over experimentation standardization, and depend on implicit code-based configurations with limited traceability of dataset splitting.

In contrast, `ModularML` is built around the principle of *configuration-as-contract*: the entire ML experiment is represented as a structured, inspectable object. This design offers three major benefits:

1. **Reproducibility through serialization.** Entire experiments, from initial data curation to multi-stage training and evaluation, can be saved and shared as a single file. A collaborator can load this artifact and inspect the full pipeline without having to trace the source code.
2. **Backend abstraction.** Model nodes are agnostic to the backend in which the underlying model is defined. `ModularML` does not replace existing ML libraries such as `PyTorch`, `Keras`, or `scikit-learn` [@scikit_learn]; rather, it improves their usage by providing a unified interface that researchers can intuitively understand. This backend-agnostic design also enables mixed-backend experiments, allowing direct reuse of a collaborator's model regardless of which library it uses.
3. **Transparency for non-ML experts.** Built-in `summary` and `visualize` methods for feature sets , model graphs, and experiment execution phases (`modularml.ExperimentPhase`) allow users and code reviewers to quickly inspect data splits, model architecture, loss routing, and training sequencing without needing to understand backend-specific implementation details.

These features directly address common reproducibility and collaboration challenges in research, while still supporting a comprehensive suite of modeling techniques and existing ML libraries.


# State of the field

General-purpose deep learning frameworks (e.g., `PyTorch`, `Keras`) streamline model training but typically assume homogeneous backends and single-phase workflows.
Data-centric libraries (e.g., `Apache Arrow` [@pyarrow], `Hugging Face` [@huggingface_datasets]) manage datasets but leave sampling and experiment scheduling to custom user code.
Experiment managers (e.g., `MLflow` [@mlflow], `Weights & Biases` [@wandb]) track runs but do not define how data flows through modular graphs.

`ModularML` sits between these layers.
It provides concrete abstractions for data (`modularml.FeatureSet`), sampling logic (`modularml.Sampler`), modeling (`modularml.ModelGraph` composed of `modularml.ModelNode`), losses (`modularml.AppliedLoss`), and orchestration (`modularml.ExperimentPhase`), while remaining backend-agnostic and lightweight enough for research scripts or notebooks.


# Software design

The `ModularML` architecture provides abstractions for data storage and processing, graph-based modeling, and execution of distinct training sequences (\autoref{fig:architecture}).

![ModularML architecture overview: FeatureSet curation into samples with feature, target, and tag domains; flexible ModelGraph construction for rapid experimentation of model topology; and multi-phase training workflows sequenced within a single Experiment container.\label{fig:architecture}](docs/_static/figures/modularml_overview_diagram.png)

## FeatureSets

A `modularml.FeatureSet` organizes data into three intent-driven domains---features, targets, and tags---reflecting the core design philosophy of `ModularML`.Features represent model inputs (what the model learns from), targets represent model outputs (what the model is trained to predict or reproduce), and tags store optional metadata associated with each sample.

Each sample, defined by its feature, target, and tag attributes, is assigned a globally unique identifier to ensure explicit traceability throughout the entire lifecycle of an experiment.
This identifier propagates through splitting, sampling, batching, model execution, and evaluation, enabling transparent lineage tracking and reproducible analysis.

Data within a `modularml.FeatureSet` is stored in backend-agnostic containers supporting `NumPy` [@numpy] arrays, `Pandas` [@pandas] dataframes, `PyTorch` tensors, and `TensorFlow` [@tensorflow] tensors.
All downstream operations, such as splitting, subsetting, and batching, operate using no-copy views of the underlying data. 
This design makes split definitions explicit and inspectable, reduces memory overhead, and minimizes the risk of data leakage or unintended experimentation bias.

## Sampling
All subclasses of `modularml.Sampler` consume a `modularml.FeatureSet`, or subset views of one, and emit aligned batches, supporting stratification, grouping, and multi-role sampling needed for contrastive or paired training schemes.  

## ModelGraph
The full experiment model is defined as a directed acyclic graph (DAG) of `modularml.ModelNode`s, where each node wraps a user-defined or built-in model and exposes optimizer hooks, build routines, and freeze/unfreeze controls.
Nodes are subset into several subclasses (e.g., `modularml.MergeNode`) depending on the connectivity constraints of each.
A `modularml.ModelNode` supports single-input, single-output operation, whereas the `modularml.MergeNode` supports multiple inputs.
`modularml.MergeNode` has expensive support for merge logic, including concatenation axes, aggregation operators, and padding, providing experimentation with branding and multi-task ML models.
All nodes belong to an overarching `modularml.ModelGraph` which handles DAG-ordered execution during both forward and backward propagation with either per-node or global optimizers.

## Losses and phases
`modularml.AppliedLoss` binds objectives to nodes, enabling composite-loss training and targeted loss aggregation over specific nodes and loss types.
These node-targeted losses are recorded for all execution phases, providing tracked losses with fast filtering and analysis upon execution completion.

TrainPhase orchestrates iterative training with callbacks and checkpointing, FitPhase performs single-pass fit-based workflows (e.g., for scikit-learn estimators), and EvalPhase runs inference-only sweeps.

## Experiment orchestration
The `modularml.Experiment` class serves as the top-level container for a machine learning workflow, binding together one or more `modularml.FeatureSet`s, a `modularml.ModelGraph`, and a sequence of execution phases.
Each `modularml.ExperimentPhase` defines a reproducible unit of execution, including data sampling, loss definitions, callbacks, and training or evaluation behavior.

By explicitly separating experiment definition from execution, `ModularML` supports multi-stage workflows such as pretraining, fine-tuning, and evaluation while retaining all information required to reproduce each phase.
The `modularml.Experiment` also provides serialization, checkpointing, and execution tracking, allowing complete experiments to be exported, shared, and reloaded as self-contained artifacts.

# Research impact statement

`ModularML` lowers the barrier for scientists prototyping hybrid ML systems that combine learned encoders, classical regressors/classifiers, and domain-specific samplers.
By guaranteeing that every component can be serialized, checkpointed, and replayed, the framework supports reproducible experiments and facilitates sharing of trained graphs or datasets between collaborators.
Its backend-agnostic graph execution simplifies comparisons across `PyTorch`, `TensorFlow`, and `scikit-learn` implementations, encouraging rigorous benchmarking and cross-validation in applied research domains.

The unique contributions can be summarized as follows:

1. **Full pipeline serialization.** Rather than serializing only model weights, `ModularML` serializes data definitions with assigned unique identifiers, split logic, sampling configuration, graph topology, per-phase loss routing, and training and evaluation configurations. This enables complete auditability of published ML studies, as shown in \autoref{fig:collaboration}.
2. **Backend-agnostic DAG modeling.** By supporting mixed-backend modeling, `ModularML` extends adoption of existing ML libraries rather than replacing them, reducing friction between works that use different packages.
3. **Declarative syntax.** All aspects of an ML experiment are constructed via declarative, configuration-driven objects. This reduces boilerplate, simplifies code review, and significantly accelerates experimentation across different training techniques and model architectures.
4. **Built-in visualization for validation.** Visual summaries help detect data leakage, misconfigured sampling, incorrect loss routing, unexpected model topologies, and training sequencing issues (\autoref{fig:collaboration}). This enables validation at definition time, ensures execution matches intent, and lowers the barrier for domain scientists to review ML-based results.

![Backend-agnostic and fully serializable experiment workflow in ModularML with built-in visualization utilities.\label{fig:collaboration}](docs/_static/figures/modularml_collaboration.png)

# Mathematics

`ModularML` does not introduce new mathematical formulations; instead, it codifies well-established training loops, loss aggregation, and optimizer steps across common ML backends.
Any model- or loss-specific mathematics is delegated to user-defined modules or external libraries, ensuring that `ModularML` remains an orchestration layer that provides continued support for existing ML packages.

# AI usage disclosure

Generative AI tools were used in a limited capacity during the development of this software, confined to helping structure docstrings and scaffold unit tests.
All AI-assisted contributions were reviewed and verified by the authors.
AI was not used in the writing of this manuscript.

# Acknowledgements

We thank the `ModularML` contributor community for feature ideas, bug reports, and documentation improvements, and acknowledge Professor Ryan Cooper for guidance in navigating the open-source environment.

# References
