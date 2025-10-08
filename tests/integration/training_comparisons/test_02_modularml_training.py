# tests/integration/test_modularml_training.py
import json
from pathlib import Path

import numpy as np
import pytest

import modularml as mml
from modularml.core import (
    AppliedLoss,
    Experiment,
    Loss,
    ModelGraph,
    ModelStage,
    Optimizer,
    SimpleSampler,
    TrainingPhase,
)
from modularml.models.torch import SequentialMLP
from modularml.utils.data_conversion import align_ranks


@pytest.mark.integration
@pytest.mark.order(2)
@pytest.mark.dependency(name="modularml_training", scope="session")
def test_modularml_training(model_config, battery_soh_featureset, metrics):
    """Train and evaluate the ModularML model using the same config."""
    fs = battery_soh_featureset

    ms_regressor = ModelStage(
        model=SequentialMLP(
            output_shape=model_config["output_shape"],
            n_layers=model_config["n_layers"],
            hidden_dim=model_config["hidden_dim"],
        ),
        label="Regressor",
        upstream_node=fs,
        optimizer=Optimizer("adam", lr=model_config["lr"]),
    )

    mg = ModelGraph(nodes=[fs, ms_regressor])
    mg.build_all()

    mse_loss = AppliedLoss(
        label="MSELoss",
        loss=Loss(name="mse", backend=mml.Backend.TORCH),
        all_inputs={"0": f"{fs.label}.targets", "1": "Regressor.output"},
    )

    sampler = SimpleSampler(
        shuffle=False,
        seed=model_config["seed"],
    )

    phase = TrainingPhase(
        label="train_phase",
        losses=[mse_loss],
        train_samplers={f"{fs.label}.train": sampler},
        val_samplers={f"{fs.label}.val": sampler},
        batch_size=model_config["batch_size"],
        n_epochs=model_config["n_epochs"],
    )

    exp = Experiment(graph=mg, phases=[phase])
    result = exp.run_training_phase(phase)

    # Evaluate
    from modularml.core import EvaluationPhase

    results = {}
    for subset in ["train", "val", "test"]:
        phase_eval = EvaluationPhase(
            label=f"eval_{subset}",
            samplers={f"{fs.label}.{subset}": sampler},
            batch_size=model_config["batch_size"],
            losses=[mse_loss],
        )
        val_res = exp.run_evaluation_phase(phase_eval)
        df_res = val_res.outputs.to_dataframe()
        df_res = df_res.loc[df_res["node"] == "Regressor"]
        y_pred = np.vstack(df_res["output"].to_numpy()).reshape(-1)
        y_true = np.vstack(df_res["target"].to_numpy()).reshape(-1)

        y_true, y_pred = align_ranks(y_true, y_pred)
        results[subset] = metrics(y_true, y_pred)

    # Save results for cross-framework tests
    DIR_RES = Path("tests/integration/training_comparisons/results")
    DIR_RES.mkdir(exist_ok=True, parents=True)
    with Path.open(DIR_RES / "results_modularml.json", "w", encoding="utf-8") as f:
        json.dump(results, f)

    for split, vals in results.items():
        print(f"{split.upper()} → MSE: {vals['MSE']:.5f} | R2: {vals['R2']:.4f}")
