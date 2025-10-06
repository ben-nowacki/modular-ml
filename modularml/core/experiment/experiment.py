import copy
from collections import defaultdict
from typing import Any, Literal

import numpy as np
import pandas as pd
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from modularml.core.data_structures.data import Data
from modularml.core.data_structures.node_outputs import NodeOutputs
from modularml.core.data_structures.sample import Sample
from modularml.core.data_structures.sample_collection import SampleCollection
from modularml.core.experiment.eval_phase import EvaluationPhase
from modularml.core.experiment.phase_results import EvaluationResult, TrainingResult
from modularml.core.experiment.training_phase import TrainingPhase
from modularml.core.graph.feature_set import FeatureSet
from modularml.core.graph.model_graph import ModelGraph
from modularml.core.loss.loss_collection import LossCollection
from modularml.core.loss.loss_record import LossRecord
from modularml.utils.data_conversion import to_python
from modularml.utils.data_format import DataFormat
from modularml.utils.exceptions import NotInvertibleError
from modularml.utils.formatting import format_value_to_sig_digits


class Experiment:
    def __init__(
        self,
        graph: ModelGraph,
        phases: list[TrainingPhase | EvaluationPhase] | None = None,
    ):
        """
        Initialize an Experiment with a model graph and a list of training and/or evaluation phases.

        Args:
            graph (ModelGraph): The modular computation graph containing model stages and feature sets.
            phases (list[Union[TrainingPhase, EvaluationPhase]] | None): Ordered list of training and evaluation phases
                that define the experiment procedure. Defaults to None.

        """
        self.graph = graph
        self.phases = [phases] if isinstance(phases, (TrainingPhase, EvaluationPhase)) else phases

        # Build graph (required before .run())
        if not self.graph.is_built:
            self.graph.build_all()

    def run(self):
        for phase in self.phases:
            if isinstance(phase, TrainingPhase):
                self.run_training_phase(phase)
            elif isinstance(phase, EvaluationPhase):
                self.run_evaluation_phase(phase)
            else:
                msg = f"Unknown phase type: {type(phase)}"
                raise TypeError(msg)

    def _should_stop_early(
        self,
        phase: TrainingPhase,
        train_losses: list[LossCollection],
        val_losses: list[LossCollection],
        bad_epoch_count: int,
        best_value: float | None,
    ) -> tuple[bool, int, float]:
        """
        Check if training should stop early based on the phase's early stopping configuration.

        Args:
            phase (TrainingPhase): Current training phase (contains early stopping config).
            train_losses (list[LossCollection]): Accumulated training losses per epoch.
            val_losses (list[LossCollection]): Accumulated validation losses per epoch.
            bad_epoch_count (int): Current count of epochs without improvement.
            best_value (float | None): Best metric value seen so far.

        Returns:
            tuple:
                - stop (bool): Whether to stop training early.
                - bad_epoch_count (int): Updated bad epoch count.
                - best_value (float): Updated best metric value.

        """
        if phase.early_stop_patience is None:
            return False, bad_epoch_count, best_value

        # Choose history
        if phase.early_stop_metric == "val_loss":
            history = val_losses
        elif phase.early_stop_metric == "train_loss":
            history = train_losses
        else:
            msg = f"Unsupported early_stop_metric: {phase.early_stop_metric}"
            raise ValueError(msg)

        if not history:
            return False, bad_epoch_count, best_value

        current_value = history[-1].total  # you could swap to .trainable if desired

        # First epoch → initialize best_value
        if best_value is None:
            return False, 0, current_value

        # Check improvement
        improved = False
        if phase.early_stop_mode == "min":
            if best_value - current_value > phase.early_stop_min_delta:
                improved = True
        elif phase.early_stop_mode == "max":
            if current_value - best_value > phase.early_stop_min_delta:
                improved = True
        else:
            msg = f"Unknown early stop mode: {phase.early_stop_mode}"
            raise ValueError(msg)

        if improved:
            best_value = current_value
            bad_epoch_count = 0
        else:
            bad_epoch_count += 1

        stop = bad_epoch_count >= phase.early_stop_patience
        return stop, bad_epoch_count, best_value

    def run_training_phase(self, phase: TrainingPhase) -> TrainingResult:
        # Resolve losses
        if not phase.resolved:
            phase.resolve(graph=self.graph)

        # Determine frozen vs trainable nodes:
        trainable_nodes = set(self.graph._nodes_req_opt.keys()).difference(set(phase.frozen_nodes))

        # Record losses and model outputs over each epoch
        train_losses_per_epoch: list[LossCollection] = []
        val_losses_per_epoch: list[LossCollection] = []
        train_outputs_per_epoch: list[dict[str, Any]] = []

        # Get training batches
        train_batches = phase.train_io.get_batches(
            featuresets={k: self.graph._nodes[k] for k in self.graph.source_node_labels},
        )
        n_train_batches = min([len(b) for b in train_batches.values()])

        # Get validation batches
        val_batches = None
        if phase.val_io is not None:
            val_batches = phase.val_io.get_batches(
                featuresets={k: self.graph._nodes[k] for k in self.graph.source_node_labels},
            )
        n_val_batches = 0 if val_batches is None else min([len(b) for b in val_batches.values()])

        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            transient=False,
        ) as progress:
            task_id = progress.add_task(f"TrainingPhase: {phase.label}", total=phase.n_epochs)

            last_epoch, bad_epoch_count, best_value = 0, 0, None
            for epoch in range(phase.n_epochs):
                # 1. Perform Training
                n_train_samples = 0
                epoch_outputs = defaultdict(list)
                epoch_train_loss: LossCollection | None = None

                for i in range(n_train_batches):
                    ref_key = next(iter(train_batches.keys()))
                    n_train_samples += train_batches[ref_key][i].n_samples
                    batch = {k: v[i] for k, v in train_batches.items()}
                    step_res = self.graph.train_step(
                        batch_input=batch,
                        losses=phase.train_io.losses_mapped_by_node,
                        trainable_stages=trainable_nodes,
                    )
                    for k, v in batch.items():
                        epoch_outputs[k].append(v)
                    for k in step_res.node_outputs:
                        epoch_outputs[k].append(step_res.node_outputs[k])

                    if epoch_train_loss is None:
                        epoch_train_loss = step_res.losses.to_float()
                    else:
                        epoch_train_loss += step_res.losses.to_float()

                # Scale losses (average loss per sample)
                _conv_lrs = [
                    LossRecord(
                        value=rec.value / n_train_samples,
                        label=rec.label,
                        contributes_to_update=rec.contributes_to_update,
                    )
                    for rec in epoch_train_loss._records
                ]
                epoch_train_loss = LossCollection(records=_conv_lrs)
                train_losses_per_epoch.append(epoch_train_loss)
                train_outputs_per_epoch.append(epoch_outputs)

                # 2. Perform Evaluation (optional)
                n_val_samples = 0
                epoch_val_loss: LossCollection | None = None
                if phase.val_io is not None:
                    for i in range(n_val_batches):
                        ref_key = next(iter(val_batches.keys()))
                        n_val_samples += val_batches[ref_key][i].n_samples
                        batch = {k: v[i] for k, v in val_batches.items()}
                        step_res = self.graph.eval_step(
                            batch_input=batch,
                            losses=phase.val_io.losses_mapped_by_node,
                        )
                        for k in step_res.node_outputs:
                            epoch_outputs[k].append(step_res.node_outputs[k])

                        if epoch_val_loss is None:
                            epoch_val_loss = step_res.losses.to_float()
                        else:
                            epoch_val_loss += step_res.losses.to_float()

                    # Scale losses (average loss per sample)
                    _conv_lrs = [
                        LossRecord(
                            value=rec.value / n_val_samples,
                            label=rec.label,
                            contributes_to_update=rec.contributes_to_update,
                        )
                        for rec in epoch_val_loss._records
                    ]
                    epoch_val_loss = LossCollection(records=_conv_lrs)
                    val_losses_per_epoch.append(epoch_val_loss)

                # 3. Update progress bar
                dsc = (
                    f"{phase.label} "
                    f"[{epoch + 1}/{phase.n_epochs}] "
                    f"T={format_value_to_sig_digits(epoch_train_loss.total, sig_digits=4)}"
                    f"({format_value_to_sig_digits(epoch_train_loss.trainable, sig_digits=4)})"
                )
                if epoch_val_loss is not None:
                    dsc += f" V={format_value_to_sig_digits(epoch_val_loss.total, sig_digits=4)}"
                progress.update(
                    task_id,
                    advance=1,
                    description=dsc,
                )
                last_epoch = epoch

                # 4. Early stopping check
                stop, bad_epoch_count, best_value = self._should_stop_early(
                    phase=phase,
                    train_losses=train_losses_per_epoch,
                    val_losses=val_losses_per_epoch,
                    bad_epoch_count=bad_epoch_count,
                    best_value=best_value,
                )
                if stop:
                    break

            # 5. Update final progress bar
            dsc = (
                f"{phase.label} "
                f"[{epoch + 1}/{phase.n_epochs}] "
                f"T={format_value_to_sig_digits(epoch_train_loss.total, sig_digits=4)}"
                f"({format_value_to_sig_digits(epoch_train_loss.trainable, sig_digits=4)})"
            )
            if epoch_val_loss is not None:
                dsc += f" V={format_value_to_sig_digits(epoch_val_loss.total, sig_digits=4)}"
            progress.update(
                task_id,
                completed=last_epoch + 1,
                description=dsc,
            )
            progress.refresh()

        # 6. Convert raw outputs to user-friendly format
        train_outputs = NodeOutputs(
            node_batches=train_outputs_per_epoch[-1],
            node_source_metadata={k: self.graph.get_node_source(k) for k in train_outputs_per_epoch[-1]},
        )

        return TrainingResult(
            phase_label=phase.label,
            final_train_loss=train_losses_per_epoch[-1],
            train_losses=train_losses_per_epoch,
            final_val_loss=val_losses_per_epoch[-1] if len(val_losses_per_epoch) > 0 else val_losses_per_epoch,
            val_losses=val_losses_per_epoch,
            train_outputs=train_outputs,
            last_epoch=last_epoch,
            stopped_early=last_epoch < phase.n_epochs,
        )

    def run_evaluation_phase(self, phase: EvaluationPhase) -> EvaluationResult:
        # Resolve losses
        if not phase.resolved:
            phase.resolve(graph=self.graph)

        # Get batches
        eval_batches = phase.eval_io.get_batches(
            featuresets={k: self.graph._nodes[k] for k in self.graph.source_node_labels},
        )
        n_eval_batches = min([len(b) for b in eval_batches.values()])

        n_eval_samples = 0
        outputs = defaultdict(list)
        loss: LossCollection | None = None

        # 1. Forward pass + eval of all batches
        for i in range(n_eval_batches):
            ref_key = next(iter(eval_batches.keys()))
            n_eval_samples += eval_batches[ref_key][i].n_samples
            batch = {k: v[i] for k, v in eval_batches.items()}
            step_res = self.graph.eval_step(
                batch_input=batch,
                losses=phase.eval_io.losses_mapped_by_node,
            )
            for k, v in batch.items():
                outputs[k].append(v)
            for k in step_res.node_outputs:
                outputs[k].append(step_res.node_outputs[k])

            if loss is None:
                loss = step_res.losses.to_float()
            else:
                loss += step_res.losses.to_float()

        # 2. Scale losses (average loss per sample)
        _conv_lrs = [
            LossRecord(
                value=rec.value / n_eval_samples,
                label=rec.label,
                contributes_to_update=rec.contributes_to_update,
            )
            for rec in loss._records
        ]
        loss = LossCollection(records=_conv_lrs)

        # 3. Convert raw outputs to user-friendly format
        # For each node, get its source FeatureSets and annotate with subset
        node_source_metadata = {}
        # Build mapping: FeatureSet -> subset (from PhaseIO)
        fs_to_fss = {fs_lbl: fss_lbl for (fs_lbl, fss_lbl) in phase.eval_io.samplers}  # noqa: C416
        for node_lbl in outputs:
            sources = self.graph.get_node_source(node_lbl)
            if sources is None:
                node_source_metadata[node_lbl] = None
            elif isinstance(sources, str):
                node_source_metadata[node_lbl] = (sources, fs_to_fss.get(sources))
            else:  # tuple of sources
                node_source_metadata[node_lbl] = [(src, fs_to_fss.get(src)) for src in sources]

        node_outputs = NodeOutputs(
            node_batches=outputs,
            node_source_metadata=node_source_metadata,
        )

        return EvaluationResult(
            phase_label=phase.label,
            losses=loss,
            outputs=node_outputs,
        )

    def inverse_transform_node_outputs(
        self,
        all_outputs: NodeOutputs,
        node: str,
        *,
        component: Literal["features", "targets"] = "targets",
        which: Literal["all", "last"] = "all",
        strict: bool = True,
    ) -> pd.DataFrame:
        """
        Apply inverse transforms to the outputs of a single node.

        Results are mapping them back into the original feature/target space defined by the \
        originating FeatureSet.

        This method reconstructs the pre-transform values for model predictions or FeatureSet
        outputs by applying the inverse of the transform chain recorded in the relevant
        FeatureSet. Only nodes that trace unambiguously to a single FeatureSet (with an optional
        subset) can be inverted.

        Args:
            all_outputs (NodeOutputs):
                Aggregated outputs from a training or evaluation phase.
            node (str):
                The node label to invert (e.g., "Regressor").
            component ({"features", "targets"}, optional):
                Which transform family to apply. For predictions, this is typically "targets".
                Currently, only "targets" is supported. Default = "targets".
            which ({"all", "last"}, optional):
                Whether to invert all transforms in the recorded chain, or only the most
                recently applied one. Default = "all".
            strict (bool, optional):
                If True, raise an error when inversion is impossible or ambiguous.
                If False, return the original scaled values unchanged. Default = True.

        Returns:
            pd.DataFrame:
                A DataFrame containing the outputs for the specified node with the
                "output" and "target" columns inverse-transformed to the original space.
                Other metadata columns (node, role, sample_uuid, batch, tags, etc.) are preserved.

        Raises:
            KeyError:
                If `node` is not found in `all_outputs`.
            NotInvertibleError:
                If the node is downstream of a merge or has no valid FeatureSet source,
                and `strict=True`.
            NotImplementedError:
                If `component="features"` is requested (not yet supported).

        Notes:
            - This method does **not** modify `all_outputs`. It returns a new DataFrame.
            - Nodes with multiple upstream sources (e.g., MergeStages) cannot be inverted,
            because the transform chain is ambiguous.
            - For `component="targets"`, the method reconstructs a temporary SampleCollection
            and delegates inversion to the FeatureSet's `inverse_transform` method.
            - If transforms were applied to only a specific subset, inversion will respect
            that subset if available in metadata.

        """
        # region: Step 1 - Prelim Error Checking
        if node not in all_outputs.node_batches:
            msg = f"Node `{node}` not found in outputs. Available: {list(all_outputs.node_batches.keys())}"
            raise KeyError(msg)

        meta = all_outputs.node_source_metadata.get(node, None)

        # Determine (fs_label, fss_label) or refuse if merged
        fs_label: str | None = None
        fss_label: str | None = None

        def _fail_or_passthrough(msg: str):
            if strict:
                raise NotInvertibleError(msg)
            # non-strict: just return original
            return all_outputs

        # Case 1 - Node has no source (ie, is a FeatureSet)
        if meta is None:
            # If meta is None, only invertible when the node is itself a FeatureSet (source node).
            # We can still try (component may be "features" or "targets").
            # Figure out if node is a FeatureSet:
            src_is_featureset = node in self.graph.source_node_labels
            if not src_is_featureset:
                return _fail_or_passthrough(
                    f"Node `{node}` has no source metadata and is not a FeatureSet source; cannot invert.",
                )
            fs_label = node
            fss_label = None

        # Case 2 - Node has a single source (FeatureSet or FeatureSubset)
        elif isinstance(meta, tuple) and len(meta) == 2 and all(isinstance(x, (str, type(None))) for x in meta):
            fs_label, fss_label = meta
        elif isinstance(meta, str):
            fs_label, _subset_label = meta, None

        # Case 3 - Node has multiple sources (ie, node is downstream of a MergeStage)
        else:
            return _fail_or_passthrough(
                f"Node `{node}` appears to be downstream of a merge (sources={meta}); cannot invert.",
            )
        # endregion

        # region: Step 2 - Get FeatureSet node
        if fs_label not in self.graph._nodes:
            return _fail_or_passthrough(f"Origin FeatureSet `{fs_label}` not found in graph.")

        fs_node = self.graph._nodes[fs_label]
        if not isinstance(fs_node, FeatureSet):
            return _fail_or_passthrough(f"Origin node `{fs_label}` is not a FeatureSet. Received: {type(fs_node)}.")
        # endregion

        # region: Step 3 - Apply inverse_transform
        df = all_outputs.to_dataframe()
        if component == "targets":
            # Filter all_outputs to only specified node and convert to dict of lists
            df_node = df.loc[df["node"] == node]
            results = copy.deepcopy(df_node.to_dict(orient="list"))
            # Overwrite
            for k in ["output", "target"]:
                scaled = np.vstack(df_node[k].values)
                temp = SampleCollection(
                    [
                        Sample(
                            features={"dummy": Data(0.0)},
                            targets={k: Data(x[i]) for i, k in enumerate(fs_node.target_keys)},
                            tags={},
                        )
                        for x in scaled
                    ],
                )
                temp = fs_node.inverse_transform(
                    data=temp,
                    component=component,
                    subset=fss_label,
                    which=which,
                    inplace=False,
                )
                unscaled = temp.get_all_targets(fmt=DataFormat.NUMPY).reshape(scaled.shape)
                results[k] = to_python(unscaled)

        else:
            msg = "Inverse transform of node outputs using feature transforms is not supported yet."
            raise NotImplementedError(msg)
        # endregion

        return pd.DataFrame(results)
