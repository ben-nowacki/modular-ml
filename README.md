# ModularML

Modular, fast, and reproducible ML experimentation built for R\&D.


[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/modularml.svg)](https://pypi.org/project/modularml/)
[![Docs](https://img.shields.io/badge/docs-coming--soon-lightgrey)](https://github.com/REIL-UConn/modular-ml/)
[![License](https://img.shields.io/badge/License-Apache%202.0-orange
)](LICENSE)

---

# Why ModularML?

**ModularML** is a flexible, backend-agnostic machine learning framework for designing, training, and evaluating modular ML pipelines, tailored specifically for research and scientific workflows. 
It enables rapid experimentation with complex model architectures, supports domain-specific feature engineering, and provides full reproducibility through configuration-driven declaration.

> ModularML provides a plug-and-play ecosystem of interoperable components for data preprocessing, sampling, modeling, training, and evaluation — all wrapped in a unified experiment container.


![ModularML Overview Diagram](docs/images/modularml_overview.png)




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
- **Stage-wise training**: Custom `TrainingStage` configuration enables fine-tuning, freezing, and transfer learning across sub-models.

### Training & Experiments
- **`Experiment` class** encapsulates all training logic (via multiple `TrainingStage` objects), ModelGraph and FeatureSet definition, and a `TrackingManager` that logs all configuration files and training, validation, and evaluation metrics for rapid and reproducible ML experimentation.
- Each `TrainingStage` defines training loop logic with early stopping, validation hooks, loss weighting, and optimizer configs.
- **Multi-objective loss support** with configurable stage-level targets, sample-based loss functions, and weighted combinations.
- **Config-driven experiments**: Every experiment is fully seriallizable and reproducible from a single configuration file.
- **Built-in experiment tracking** via a `TrackingManager`, with optional integration into external managers like MLflow or other logging backends.



## Getting Started

### Requirements
- Python >= 3.8  
- PyTorch >= 1.10  
- TensorFlow >= 2.8  
- NumPy >= 1.22  

### Installation
Install from PyPI:
```bash
pip install modularml
```

To install the latest development version:
```bash
pip install git+https://github.com/REIL-UConn/modular-ml.git
```

Below outlines a basic toy example of using ModularML:

```python
# EXAMPLE COMING SOON ...
```



## Explore More
- **[Examples (coming soon)](examples/)** – Explore complete examples of how to set up FeatureSets, samplers, model graphs, and training configurations.
- **[Documentation (coming soon)](https://github.com/REIL-UConn/modular-ml/)** – API reference, component explanations, YAML configuration guides, and tutorials.
- **[Discussions (coming soon)](https://github.com/REIL-UConn/modular-ml/discussions)** – Join the community, ask questions, suggest features, or share use cases.

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


