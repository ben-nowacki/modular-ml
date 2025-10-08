from dataclasses import dataclass

import pandas as pd

from modularml.core.data_structures.batch import Batch, BatchOutput
from modularml.utils.data_conversion import to_python
from modularml.utils.data_format import DataFormat


@dataclass
class NodeOutputs:
    """
    Aggregated outputs from all nodes (FeatureSets and ModelStages).

    Attributes:
        node_batches (dict[str, list[Batch | BatchOutput]]):
            Mapping from node label -> list of Batch or BatchOutput objects.
        node_source_metadata (dict[str, str | tuple[str], None]): Mappings of Node.label to original source label \
            (eg. FeatureSet.label). FeatureSet nodes have no source and values will be None. Node \
            downstream from a merge will have multiple sources.

    Example:
        After a training or evaluation run, this object stores outputs like::
        ```
            {
                "Encoder": [BatchOutput(...), BatchOutput(...)],
                "Regressor": [BatchOutput(...)]
            }
        ```

        Each node may contain multiple batches.

    """

    node_batches: dict[str, list[Batch | BatchOutput]]
    node_source_metadata: dict[str, str]

    def __repr__(self):
        return f"NodeOutputs({self.available_nodes})"

    @property
    def available_nodes(self) -> list[str]:
        """
        List all node labels that produced outputs.

        Returns:
            list[str]: List of node labels.

        """
        return list(self.node_batches.keys())

    def available_roles(self, node: str) -> list[str]:
        """
        Return all roles (e.g., "default", "anchor", "positive") observed for a given node.

        Args:
            node (str): Node label to query.

        Returns:
            list[str]: Sorted list of roles available in that node's outputs.

        """
        roles = set()
        for b in self.node_batches[node]:
            roles.update(b.available_roles)
        return sorted(roles)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Flatten outputs into a DataFrame.

        Columns:
            - node: Node label (FeatureSet or ModelStage).
            - role: Role name (e.g., "default", "anchor").
            - sample_uuid: Unique identifier(s) of the sample(s).
            - batch: Batch number of sample(s).
            - output: Model output features (converted to Python/NumPy types).
            - target: Expected target values, if available.
            - tags / tag.<key>: Tag values associated with each sample.

        For FeatureSet nodes with structured tags (dict of lists), each tag key
        is expanded into its own column with the prefix "tag.".

        Returns:
            pd.DataFrame: Flattened outputs for all nodes and roles.

        Raises:
            TypeError: If a batch object is neither BatchOutput nor Batch.
            ValueError: If tag list lengths do not match sample lengths.

        """
        rows = []
        for node, batches in self.node_batches.items():
            for b_id, b in enumerate(batches):
                for role in b.available_roles:
                    outs, ids, tgts, tags = None, None, None, None
                    if isinstance(b, BatchOutput):
                        outs = to_python(b.features[role])
                        ids = b.sample_uuids[role]
                        tgts = (
                            to_python(b.targets[role])
                            if getattr(b, "targets", None) is not None
                            else [None] * len(outs)
                        )
                        tags = b.tags[role]

                    elif isinstance(b, Batch):
                        outs = b.role_samples[role].get_all_features(fmt=DataFormat.LIST)
                        ids = b.role_samples[role].sample_uuids
                        tgts = b.role_samples[role].get_all_targets(fmt=DataFormat.LIST)
                        tags = b.role_samples[role].get_all_tags(fmt=DataFormat.DICT_LIST)

                    else:
                        msg = f"Unkown batch type: {type(b)}"
                        raise TypeError(msg)

                    # For structured FeatureSet tags (ie, a dict of lists), expand columns
                    if isinstance(tags, dict):
                        # Ensure length consistency
                        for k, v in tags.items():
                            if len(v) != len(ids):
                                msg = f"Tag '{k}' length {len(v)} does not match sample length {len(ids)}"
                                raise ValueError(msg)

                        for i, (u, o, t) in enumerate(zip(ids, outs, tgts, strict=True)):
                            r = {
                                "node": node,
                                "role": role,
                                "sample_uuid": u,
                                "batch": b_id,
                                "output": o,
                                "target": t,
                            }
                            # Expand each tag into its own column
                            for k, v in tags.items():
                                r[f"tag.{k}"] = v[i]
                            rows.append(r)

                    # If tags aren't structured, just make a single column
                    else:
                        for u, o, t, g in zip(ids, outs, tgts, [tags] * len(ids), strict=True):
                            r = {
                                "node": node,
                                "role": role,
                                "sample_uuid": u,
                                "batch": b_id,
                                "output": o,
                                "target": t,
                            }
                            if g is not None:
                                r["tags"] = g
                            rows.append(r)
        return pd.DataFrame(rows)
