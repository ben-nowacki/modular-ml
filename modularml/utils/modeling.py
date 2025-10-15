# from enum import Enum

# import numpy as np

# from modularml.core.data_structures.batch import Batch
# from modularml.core.data_structures.data import Data
# from modularml.core.data_structures.sample import Sample
# from modularml.core.data_structures.sample_collection import SampleCollection
# from modularml.core.graph.shape_spec import ShapeSpec
# from modularml.utils.backend import Backend


# class PadMode(str, Enum):
#     """
#     Enum representing universal padding modes.

#     Description:
#         This enum standardizes padding mode names across different backends
#         (e.g., PyTorch, TensorFlow, NumPy). Use this to ensure consistent
#         behavior regardless of the chosen backend.

#     Values:
#         CONSTANT: Pad with a constant value.
#         REFLECT: Reflect without including edge (mirror).
#         REPLICATE: Repeat edge values.
#         CIRCULAR: Wrap around (circular padding).
#     """

#     CONSTANT = "constant"
#     REFLECT = "reflect"
#     REPLICATE = "replicate"
#     CIRCULAR = "circular"


# def map_pad_mode_to_backend(mode: PadMode, backend: str | Backend) -> str:
#     """
#     Map a universal PadMode to backend-specific string.

#     Args:
#         mode (PadMode): Universal padding mode.
#         backend (str): One of ['torch', 'tensorflow', 'scikit'].

#     Returns:
#         str: Backend-specific padding mode string.

#     Raises:
#         ValueError: If mode or backend is unsupported.

#     """
#     if isinstance(backend, str):
#         backend = Backend(backend.lower())

#     match (mode, backend):
#         case (PadMode.CONSTANT, _):
#             return "constant"
#         case (PadMode.REFLECT, "torch") | (PadMode.REFLECT, "tensorflow") | (PadMode.REFLECT, "scikit"):
#             return "reflect"
#         case (PadMode.REPLICATE, "torch"):
#             return "replicate"
#         case (PadMode.REPLICATE, "tensorflow"):
#             return "SYMMETRIC"
#         case (PadMode.REPLICATE, "scikit"):
#             return "edge"
#         case (PadMode.CIRCULAR, "torch"):
#             return "circular"
#         case (PadMode.CIRCULAR, "scikit"):
#             return "wrap"
#         case _:
#             msg = f"Pad mode {mode} not supported for backend '{backend}'"
#             raise ValueError(msg)


# def make_dummy_data(shape: ShapeSpec | tuple[int, ...]) -> Data:
#     """
#     Creates a dummy `Data` object filled with ones for testing or placeholder use.

#     Args:
#         shape (ShapeSpec | tuple[int, ...]): The shape of the data tensor to create.

#     Returns:
#         Data: A `Data` object containing a tensor of ones with the specified shape.

#     """
#     if isinstance(shape, ShapeSpec):
#         shape = shape.merged_shape

#     return Data(np.ones(shape=shape))


# def make_dummy_batch(
#     feature_shape: ShapeSpec,
#     target_shape: ShapeSpec | None = None,
#     tag_shape: ShapeSpec | None = None,
#     batch_size: int = 8,
# ) -> Batch:
#     """
#     Creates a dummy `Batch` object with synthetic samples for testing model components.

#     Each sample contains multiple named feature and target entries, and dummy tags.

#     Args:
#         feature_shape (ShapeSpec): Feature keys and shape (n_features, feature_dim).
#         target_shape (ShapeSpec | None): Target keys and shape (n_targets, target_dim). \
#             Defaults to ShapeSpec({"Y0": (1, 1)}).
#         tag_shape (ShapeSpec | None): Tag keys and shape. Defaults to \
#             ShapeSpec({"T0": (1, 1), "T1": (1, 1)}).
#         batch_size (int, optional): Number of samples in the batch. Defaults to 8.

#     Returns:
#         Batch: A `Batch` object with randomly generated dummy features, targets, and tags.

#     """
#     if target_shape is None:
#         target_shape = ShapeSpec({"Y0": (1, 1)})
#     if tag_shape is None:
#         tag_shape = ShapeSpec({"T0": (1, 1), "T1": (1, 1)})

#     samples = [
#         Sample(
#             features={k: make_dummy_data(v) for k, v in feature_shape.items()},
#             targets={k: make_dummy_data(v) for k, v in target_shape.items()},
#             tags={k: make_dummy_data(v) for k, v in tag_shape.items()},
#         )
#         for _ in range(batch_size)
#     ]
#     sample_coll = SampleCollection(samples=samples)

#     return Batch(
#         role_samples={"default": sample_coll},
#         label="DummyBatch",
#     )
