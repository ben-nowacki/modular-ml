
from pathlib import Path
from typing import Optional, Union, Dict, Any
import joblib
import numpy as np

from modularml.preprocessing import SCALER_REGISTRY


class FeatureTransform:
    def __init__(
        self,
        scaler: Union[str, Any] = "standard",
        scaler_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        FeatureTransform can be initiallized by the name of a supported scaler
        or the scaler instance/class. 
        Use `FeatureTransform.supported_scalers` to see a mapping of supported
        scaler names and corresponding classes.

        Args:
            scaler (Union[str, Any], optional): _description_. Defaults to "standard".
            scaler_kwargs (Optional[Dict[str, Any]], optional): _description_. Defaults to None.
        """
        self.scaler_name = scaler if isinstance(scaler, str) else scaler.__class__.__name__
        self.scaler_kwargs = scaler_kwargs or {}
        self._scaler = self._initialize_scaler(scaler)

    @classmethod
    def get_supported_scalers(cls) -> Dict[str, Any]:
        return SCALER_REGISTRY

    def _initialize_scaler(self, scaler):
        if isinstance(scaler, str):
            if scaler not in SCALER_REGISTRY:
                raise ValueError(f"Scaler '{scaler}' not recognized.")
            return SCALER_REGISTRY[scaler](**self.scaler_kwargs)
        elif hasattr(scaler, "fit") and hasattr(scaler, "transform"):
            return scaler
        else:
            raise TypeError("Scaler must be a string or an sklearn-like transformer object.")

    def fit(self, data: np.ndarray):
        """data.shape: (n_samples, n_features)"""
        self._scaler.fit(data)

    def transform(self, data: np.ndarray) -> np.ndarray:
        """data.shape: (n_samples, n_features)"""
        return self._scaler.transform(data)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """data.shape: (n_samples, n_features)"""
        if hasattr(self._scaler, "fit_transform"):
            return self._scaler.fit_transform(data)
        else:
            self._scaler.fit(data)
            return self._scaler.transform(data)


    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if hasattr(self._scaler, "inverse_transform"):
            return self._scaler.inverse_transform(data)
        else:
            raise NotImplementedError(f"{self.scaler_name} does not support inverse_transform.")


    def get_config(self) -> dict:
        """
        This returns the FeatureTransform configuration but DOES NOT
        return the fitted state of the ._scaler attribute. 
        
        Use .save and .load for full state seriallizability.
        """
        return {
            "scaler_name": self.scaler_name,
            "scaler_kwargs": self.scaler_kwargs,
        }

    @classmethod
    def from_config(cls, config: dict) -> "FeatureTransform":
        """
        This reloads the FeatureTransform with the same configuration but DOES NOT
        reload the fitted state of the previous ._scaler attribute. 
        
        Use .save and .load for full state reload.
        """
        return cls(
            scaler=config.get("scaler_name", None),
            scaler_kwargs=config.get("scaler_kwargs", {}),
        )
            
                
    def to_serializable(self) -> dict:
        return {
            "config": self.get_config(),
            "scaler": self._scaler
        }

    @classmethod
    def from_serializable(cls, obj: dict) -> "FeatureTransform":
        config = obj["config"]
        scaler = obj["scaler"]
        return cls(scaler=scaler)


    def save(self, path: Union[str, Path], overwrite_existing:bool=False):
        """
        Save the FeatureTransform (config + scaler states) into a single file using joblib.
        
        Args:
            path (Union[str, Path]): File path to save the FeatureTransform.
        """
        path = Path(path).with_suffix('.joblib')
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if path.exists() and not overwrite_existing:
            raise FileExistsError(
                f"File already exists at: {path}."
                f"Use `overwrite_existing=True` to overwrite."
            )
        joblib.dump(self.to_serializable(), path)


    @classmethod
    def load(cls, path: Union[str, Path]) -> "FeatureTransform":
        """
        Load the FeatureTransform from a single joblib file containing config + state.
        
        Args:
            path (Union[str, Path]): Path to the saved joblib file.
        
        Returns:
            FeatureTransform: Loaded instance.
        """
        path = Path(path).with_suffix('.joblib')
        if not path.exists():
            raise FileNotFoundError(f"No file found at: {path}")

        data = joblib.load(path)
        return cls.from_serializable(data)


    def __repr__(self):
        return f"FeatureTransform(scaler={self.scaler_name})"
    
    