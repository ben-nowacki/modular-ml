import json
from pathlib import Path

import numpy as np
import pytest
import torch

# from matplotlib import cm
# from matplotlib import pyplot as plt
# from matplotlib.colors import Normalize
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import modularml as mml
from modularml.core.graph.feature_subset import FeatureSubset
from modularml.utils.data_conversion import align_ranks


@pytest.mark.integration
@pytest.mark.order(1)
@pytest.mark.dependency(name="pytorch_training", scope="session")
def test_pytorch_training(model_config, battery_soh_featureset, torch_model, metrics):
    """Train and evaluate the baseline PyTorch model."""
    fss_train, fss_val, fss_test = (
        battery_soh_featureset.get_subset("train"),
        battery_soh_featureset.get_subset("val"),
        battery_soh_featureset.get_subset("test"),
    )

    # for subset in [fss_train, fss_val, fss_test]:
    #     Xs = subset.get_all_features(fmt=mml.DataFormat.NUMPY)
    #     ys = subset.get_all_targets(fmt=mml.DataFormat.NUMPY)

    #     norm = Normalize(vmin=ys.min(), vmax=ys.max())
    #     scm = cm.ScalarMappable(norm=norm, cmap=plt.cm.Blues)
    #     plt.figure()
    #     for i in range(Xs.shape[0]):
    #         plt.plot(Xs[i], color=scm.to_rgba(ys[i]))
    #     plt.show()

    def to_tensors(fss: FeatureSubset):
        X = torch.tensor(fss.get_all_features(fmt=mml.DataFormat.NUMPY), dtype=torch.float32)
        y = torch.tensor(fss.get_all_targets(fmt=mml.DataFormat.NUMPY), dtype=torch.float32)
        return X, y

    dataset = {
        "train": to_tensors(fss_train),
        "val": to_tensors(fss_val),
        "test": to_tensors(fss_test),
    }

    model = torch_model
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config["lr"])

    # DataLoaders
    def make_loader(split):
        X, y = dataset[split]
        return DataLoader(TensorDataset(X, y), batch_size=model_config["batch_size"], shuffle=False)

    loaders = {k: make_loader(k) for k in ["train", "val", "test"]}

    # Train
    for _epoch in range(model_config["n_epochs"]):
        model.train()
        for xb, yb in loaders["train"]:
            optimizer.zero_grad()
            preds = model(xb)
            preds, yb = align_ranks(preds, yb)  # noqa: PLW2901
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    results = {}
    with torch.no_grad():
        for subset, loader in loaders.items():
            y_true, y_pred = [], []
            for xb, yb in loader:
                preds = model(xb)
                y_true.append(yb.numpy())
                y_pred.append(preds.numpy())
            y_true = np.vstack(y_true).ravel()
            y_pred = np.vstack(y_pred).ravel()
            y_true, y_pred = align_ranks(y_true, y_pred)

            # plt.scatter(y_true, y_pred)
            # plt.xlabel("TRUE")
            # plt.ylabel("PRED")
            # plt.title(subset.upper())
            # plt.xlim(0, 1)
            # plt.ylim(0, 1)
            # plt.show()
            results[subset] = metrics(y_true, y_pred)

    # Save results for cross-framework tests
    DIR_RES = Path("tests/integration/training_comparisons/results")
    DIR_RES.mkdir(exist_ok=True, parents=True)
    with Path.open(DIR_RES / "results_pytorch.json", "w", encoding="utf-8") as f:
        json.dump(results, f)

    for split, vals in results.items():
        print(f"{split.upper()} → MSE: {vals['MSE']:.5f} | R2: {vals['R2']:.4f}")
