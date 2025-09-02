
<div align="center">

[![ModularML Banner](assets/modularml_logo_banner.png)](https://github.com/REIL-UConn/modular-ml)

**Modular, fast, and reproducible ML experimentation built for R\&D.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/modularml.svg)](https://pypi.org/project/modularml/)
[![Docs](https://app.readthedocs.org/projects/modular-ml/badge/?version=latest&style=flat)](https://modular-ml.readthedocs.io/en/latest/)
[![License](https://img.shields.io/badge/License-Apache%202.0-orange
)](LICENSE)

</div>


ModularML is a flexible, backend-agnostic machine learning framework for designing, training, and evaluating modular ML pipelines, tailored specifically for research and scientific workflows. 
It enables rapid experimentation with complex model architectures, supports domain-specific feature engineering, and provides full reproducibility through configuration-driven declaration.

> ModularML provides a plug-and-play ecosystem of interoperable components for data preprocessing, sampling, modeling, training, and evaluation — all wrapped in a unified experiment container.


<p align="center">
  <img src="assets/modularml_overview_diagram.png" alt="ModularML Overview Diagram" width="600"/>
</p>
<p align="center"><em>Figure 1. Overview of the ModularML framework, highlighting the three core abstractions: feature set preprocessing and splitting, modular model graph construction, and staged training orchestration.</em></p>




## Features

ModularML includes a comprehensive set of components for scientific ML workflows:

### Data Handling
- **`FeatureSet` abstraction** for organizing structured datasets with features, targets, tags, and metadata.
- **`Data` class** with unified support for multiple backends (`torch.Tensor`, `tf.Tensor`, `np.ndarray`).
- **Built-in splitters**: Supports sample-based and rule-based splitting with condition-based filtering by feature, target, or tags values.
- **Sample grouping** and multi-part splits for paired, triplet, or grouped training tasks.

### Advanced Sampling
- **Flexible `FeatureSampler` interface** with support for advanced sampling during different stages of model training, including:
  - Triplet sampling (e.g., anchor/positive/negative)
  - Paired samples
  - Class-balanced, cluster-based, or time-windowed sampling strategies.
- **Condition-aware sampling** using any tags or metadata fields.

### Model Architecture
- **`ModelGraph`**: A Directed Acyclic Graph (DAG)-based model builder where:
  - Each node is a `ModelStage` (e.g., encoder, head, discriminator).
  - Each stage can use a different backend (PyTorch, TensorFlow, scikit-learn, LightGBM, etc).
  - Mixed-backend models are supported with seamless input/output routing.
- **Stage-wise training**: Custom `TrainingPhase` configuration enables fine-tuning, freezing, and transfer learning across sub-models.

### Training & Experiments
- **`Experiment` class** encapsulates all training logic (via multiple `TrainingPhase` objects), ModelGraph and FeatureSet definition, and a `TrackingManager` that logs all configuration files and training, validation, and evaluation metrics for rapid and reproducible ML experimentation.
- Each `TrainingPhase` defines training loop logic with early stopping, validation hooks, loss weighting, and optimizer configs.
- **Multi-objective loss support** with configurable stage-level targets, sample-based loss functions, and weighted combinations.
- **Config-driven experiments**: Every experiment is fully seriallizable and reproducible from a single configuration file.
- **Built-in experiment tracking** via a `TrackingManager`, with optional integration into external managers like MLflow or other logging backends.



## Getting Started

Requires Python >= 3.9

### Installation
Install from PyPI:
```bash
pip install modularml
```

To install the latest development version:
```bash
pip install git+https://github.com/REIL-UConn/modular-ml.git
```


## Explore More
- **[Examples](examples/)** – Explore complete examples of how to set up FeatureSets, apply feature preprocessing, construct model graphs, and run training configurations.
- **[Documentation](https://modular-ml.readthedocs.io/en/latest/)** – API reference, component explanations, configuration guides, and tutorials.
- **[Discussions](https://github.com/REIL-UConn/modular-ml/discussions)** – Join the community, ask questions, suggest features, or share use cases.

---


<!-- ## Cite ModularML

If you use ModularML in your research, please cite the following:

```bibtex
@misc{nowacki2025modularml,
  author       = {Ben Nowacki and contributors},
  title        = {ModularML: Modular, fast, and reproducible ML experimentation built for R&D.
  },
  year         = {2025},
  note         = {https://github.com/REIL-UConn/modular-ml},
} -->
<!-- 
## The Team
ModularML was initiated in 2025 by Ben Nowacki as part of graduate research at the University of Connecticut. 
It is actively developed in collaboration with researchers and contributors across academia and industry, including partners from the Honda Research Institute, MathWorks, and the University of South Carolina.

The project is community-driven and welcomes contributors interested in building modular, reproducible ML workflows for science and engineering. -->

## License
**[Apache 2.0](https://github.com/REIL-UConn/modular-ml/license)**


