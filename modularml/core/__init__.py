import importlib

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modularml.core.api import (
        Activation,
        AppliedLoss,
        Batch,
        ConcatStage,
        ConditionSplitter,
        CrossValidationSplitter,
        Data,
        EvaluationPhase,
        Experiment,
        FeatureSet,
        FeatureSubset,
        FeatureTransform,
        GraphNode,
        Loss,
        LossCollection,
        LossRecord,
        ModelGraph,
        ModelStage,
        Optimizer,
        PairedSampler,
        RandomSplitter,
        Sample,
        SampleCollection,
        ShapeSpec,
        SimilarityCondition,
        SimpleSampler,
        TrainingPhase,
    )

__all__ = [
    "Activation",
    "AppliedLoss",
    "Batch",
    "ConcatStage",
    "ConditionSplitter",
    "CrossValidationSplitter",
    "Data",
    "EvaluationPhase",
    "Experiment",
    "FeatureSet",
    "FeatureSubset",
    "FeatureTransform",
    "GraphNode",
    "Loss",
    "LossCollection",
    "LossRecord",
    "ModelGraph",
    "ModelStage",
    "Optimizer",
    "PairedSampler",
    "RandomSplitter",
    "Sample",
    "SampleCollection",
    "ShapeSpec",
    "SimilarityCondition",
    "SimpleSampler",
    "TrainingPhase",
]


def __getattr__(name):
    if name in __all__:
        module = importlib.import_module("modularml.core.api")
        return getattr(module, name)
    msg = f"module 'modularml.core' has no attribute '{name}'"
    raise AttributeError(msg)
