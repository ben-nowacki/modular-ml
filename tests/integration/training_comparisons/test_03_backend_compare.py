# tests/integration/test_crossframework_compare.py
import json
from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.order(3)
@pytest.mark.dependency(depends=["pytorch_training", "modularml_training"], scope="session")
def test_crossframework_results_close():
    DIR_RES = Path("tests/integration/training_comparisons/results")
    with Path.open(DIR_RES / "results_pytorch.json", "r", encoding="utf-8") as f:
        pytorch_res = json.load(f)
    with Path.open(DIR_RES / "results_modularml.json", "r", encoding="utf-8") as f:
        modularml_res = json.load(f)

    tol_maps = {
        "MSE": 0.20,
        "R2": 0.02,
    }
    for subset in ["train", "val", "test"]:
        for metric in ["MSE", "R2"]:
            diff = abs(pytorch_res[subset][metric] - modularml_res[subset][metric])
            rel_err = diff / max(pytorch_res[subset][metric], 1e-8)
            print(f"{subset.upper()} {metric}: Δ={diff:.6e}, RelErr={rel_err:.3%}")
            assert rel_err < tol_maps[metric], (
                f"{subset.upper()} {metric} differs >{int(tol_maps[metric] * 100)}% between frameworks!"
            )
