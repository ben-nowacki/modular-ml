

from abc import ABC
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
import warnings

from torch import Generator

from modularml.core.data_structures.batch import Batch
from modularml.core.data_structures.feature_set import FeatureSet
from modularml.core.model_graph.loss import AppliedLoss
from modularml.core.model_graph.model_graph import ModelGraph
from modularml.core.model_graph.model_stage import ModelStage
from modularml.core.samplers.feature_sampler import FeatureSampler




class BasePhase(ABC):
    def __init__(
        self,
        label: str,
        losses : Optional[List[AppliedLoss]] = None,
        samplers: Optional[Dict[str, FeatureSampler]] = None,
        batch_size: int = 32,
        
        merge_policy: Optional[str] = None,
        merge_mapping: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            label (str): Name of this training phase (e.g., "pretrain_encoder").
            losses (List[AppliedLoss]): List of loss functions and their input mappings.
            samplers (Dict[str, FeatureSampler]): Mapping from source string to FeatureSampler. 
                Keys must be of the form "FeatureSet" or "FeatureSet.subset".
            batch_size (int): Batch size to enforce across all samplers (overrides existing sampler batch sizes).
            
            merge_policy (str, optional): Strategy to merge roles if needed. Not yet implemented. TODO
            merge_mapping (Dict[str, Any], optional): Custom mapping for role merges. Not yet implemented. TODO
        """

        self.label = label
        self.losses = losses or []
        self.samplers = samplers or {}
        self.batch_size = batch_size

        # Only needed if losses is non-null
        self._available_loss_inputs = None
        self._required_loss_inputs = None
        self._loss_mapping: Dict[str, List[AppliedLoss]] = defaultdict(list)
    
        # Parsed samplers if provided
        if self.samplers:
            self.samplers : Dict[Tuple[str, Optional[str]], FeatureSampler] = {}
            for k,v in samplers.items():
                v.batch_size = batch_size       # overwrite pre-existing batch_size
                self.samplers[self._parse_sample_spec(k)] = v
       
        self._is_resolved = False        
    
    def _parse_sample_spec(self, spec:str) -> Tuple[str, Optional[str]]:
        """
        Parses a string that references a FeatureSet or FeatureSubset.

        Accepted formats:
            - "FeatureSet" (use all samples in FeatureSet)
            - "FeatureSet.subset" (use only a named subset)

        Returns:
            Tuple[str, Optional[str]]: (FeatureSet label, Subset label or None)

        Raises:
            ValueError: If the string format is invalid.
        """

        featureset, subset = None, None
        
        parts = spec.split(".")
        if len(parts) == 1:
            featureset, subset = parts[0], None
        elif len(parts) == 2:
            featureset, subset = parts
        else:
            raise ValueError(f"Invalid `samplers` key: {spec}")
    
        return featureset, subset
       
    def resolve_loss_inputs_and_roles(self, graph: ModelGraph,) -> Dict[str, Any]:
        """
        Resolves all possible input keys for AppliedLosses after a full model pass.

        This performs a single forward pass on one batch to:
        - Determine available `FeatureSet.sample_attribute.role` mappings
        - Determine available `ModelStage.output` keys
        - Identify if roles need merging (e.g. 'anchor', 'default')
        """
        self._loss_mapping: Dict[str, List[AppliedLoss]] = defaultdict(list)
        
        # Step 1: Sample batches from all samplers
        batches = self.get_batches(graph._feature_sets)
        
        # Step 2: Use the first batch from each source
        batch_input = {fs: b[0] for fs, b in batches.items()}
        
        # TODO: merge policy needs to be check prior to forward pass. Need to:
        #  - For each stage in graph._sorted_stage_labels, get required inputs.
        #  - If multiple inputs, check if suitable merge policy
        
        # Step 3: Run forward pass
        outputs = graph.forward(batch_input)
        for fs_key in graph._feature_sets.keys():
            outputs.pop(fs_key) 

        # Step 4: Collect available input keys
        available_inputs : Set[str] = set()
        
        # Valid FeatureSet-based AppliedLoss input keys (eg, "FeatureSet.targets" or "FeatureSet.features.anchor")
        for fs_label, batch in batch_input.items():
            for role in batch.available_roles:
                available_inputs.add(
                    f"{fs_label}.features.{role}"
                )
                available_inputs.add(
                    f"{fs_label}.targets.{role}"
                )
                
        # Valid ModelStage-based AppliedLoss input keys (eg, "Encoder.output" or "Encoder.output.anchor")
        for stage_name, batch in outputs.items():
            for role in batch.available_roles:
                available_inputs.add(
                    f"{stage_name}.output.{role}"
                )
                
        # Step 5: Determine loss-required inputs
        required_inputs = set()
        for loss in self.losses:
            for source in loss.inputs.values():
                # source is of type: Tuple[str, str, Optional[str]] for node, attribute, role
                required_inputs.add(
                    f"{source[0]}.{source[1]}" +\
                    (f".{source[2]}" if source[2] is not None else "")
                )
                
                # Record mapping of each loss to any ModelStages
                if source[0] in graph.model_stage_labels:
                    self._loss_mapping[source[0]].append(loss)
                

        # Check is inputs required by AppliedLosses is missing from forward pass
        missing_inputs = required_inputs - available_inputs
        if missing_inputs:
            raise ValueError(f"Missing inputs required by `losses`: {missing_inputs}")

        # Save available & required loss terms
        self._available_loss_inputs = list(available_inputs)
        self._required_loss_inputs = list(required_inputs)
        
        self._is_resolved = True

    @property
    def available_loss_inputs(self) -> List[str]:
        if self._available_loss_inputs is None:
            raise RuntimeError(
                f"You must call `resolve_loss_inputs_and_roles()` to generate available_loss_inputs."
            )
        return self._available_loss_inputs
     
    @property
    def is_resolved(self) -> bool:
        return self._is_resolved
     
    def get_losses_for_stage(self, stage_name:str) -> Optional[List[AppliedLoss]]:
        if not self._is_resolved:
            raise RuntimeError(
                f"You must call Phase.resolve_loss_inputs_and_roles(...) before accessing losses."
                )
        return self._loss_mapping.get(stage_name, None)
    
    def get_batches(self, featuresets:Dict[str, FeatureSet]) -> Dict[str, List[Batch]]:
        """
        Binds each FeatureSampler to its corresponding FeatureSet or Subset and 
        builds batches for training.

        Args:
            featuresets (Dict[str, FeatureSet]): Dictionary of available FeatureSets keyed by \
                their labels. E.g., `{FeatureSet.label: FeatureSet, ...}`

        Returns:
            Dict[str, List[Batch]]: Dictionary mapping each FeatureSet label to a list of Batches.

        Raises:
            ValueError: If a required FeatureSet or Subset is missing.

        Warnings:
            If samplers yield differing numbers of batches, only the minimum number will be used.
        """

        # 1. Check that featuresets match sample keys
        for sampler_spec in self.samplers.keys():
            if not sampler_spec[0] in featuresets.keys():
                raise ValueError(f"Required FeatureSet (`{sampler_spec[0]}`) is missing: {featuresets.keys()}")
        
            if sampler_spec[1] is not None:
                if sampler_spec[1] not in featuresets[sampler_spec[0]].available_subsets:
                    raise ValueError(
                        f"Required FeatureSubset (`{sampler_spec}`) is missing: "
                        f"{featuresets[sampler_spec[0]].available_subsets}"
                    )
        
        # Batches are keyed by FeatureSet (drop subset label?)
        batches: Dict[str, List[Batch]] = {}
        
        # 2. Build batches (ensure samplers are bound to sources)
        for sampler_spec, sampler in self.samplers.items():
            if not sampler.is_bound():
                # Get FeatureSet
                source = featuresets[sampler_spec[0]]
                # Get Subset if specified
                if sampler_spec[1] is not None:
                    source = source.get_subset(sampler_spec[1])
                # Bind source to sampler
                sampler.bind_source(source=source)

            batches[sampler_spec[0]] = sampler.batches
    

        # 3. Check n_batches for mismatch
        batch_lens = set([len(b) for b in batches.values()])
        if len(batch_lens) > 1:
            warnings.warn(
                f"`samplers` resulted in a differing number of batches: {batch_lens}. "
                f"Only the lesser will be used: {min(batch_lens)} batches."
            )
        
        return batches
    
    
    
    
    
    
    
    
        