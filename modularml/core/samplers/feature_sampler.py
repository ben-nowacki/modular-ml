

from typing import TYPE_CHECKING, Dict, List, Optional, Union
import warnings
import numpy as np

from modularml.core.data_structures.sample_collection import SampleCollection


if TYPE_CHECKING:
    from modularml.core.data_structures.sample import Sample
    from modularml.core.data_structures.feature_set import FeatureSet
    from modularml.core.data_structures.feature_subset import FeatureSubset

from modularml.core.data_structures.batch import Batch


class FeatureSampler:
    def __init__(
        self,
        source: Optional[Union["FeatureSet", "FeatureSubset"]] = None,
        batch_size: int = 1,
        shuffle: bool = False,
        stratify_by: Optional[List[str]] = None,
        group_by: Optional[List[str]] = None,
        drop_last: bool = False,
        seed: Optional[int] = None
    ):
        """
        Initializes a sampler for a FeatureSet or FeatureSubset.

        Args:
            source (FeatureSet or FeatureSubset, optional): The source to sample from. If provided, 
                batches are built immediately. If not provided, you must call `bind_source()` before 
                using the sampler.
            batch_size (int, optional): Number of samples per batch. Defaults to 1.
            shuffle (bool, optional): Whether to shuffle samples before batching. Defaults to False.
            stratify_by (List[str], optional): One or more `Sample.tags` keys to stratify batches by. 
                Ensures each batch has a representative distribution of tag values. For example, 
                `stratify_by=["cell_id"]` ensures balanced sampling across different cell IDs.
            group_by (List[str], optional): One or more `Sample.tags` keys to group samples into 
                batches. Ensures each batch contains samples only from a single group. For example, 
                `group_by=["cell_id"]` yields batches grouped by cell.
            drop_last (bool, optional): Whether to drop the final batch if it's smaller than 
                `batch_size`. Defaults to False.
            seed (int, optional): Random seed for reproducible shuffling.
        """
        
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        
        self.source = source
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.stratify_by = stratify_by if isinstance(stratify_by, list) else [stratify_by, ] if isinstance(stratify_by, str) else None
        self.group_by = group_by if isinstance(group_by, list) else [group_by, ] if isinstance(group_by, str) else None
        if self.stratify_by is not None and self.group_by is not None:
            raise ValueError(f"Both `group_by` and `stratify_by` cannot be applied at the same.")
        self.drop_last = bool(drop_last)
        
        self.batches : Optional[List[Batch]] = \
            self._build_batches() if self.source is not None else None
        
        self._batch_by_id: Optional[Dict[str, Batch]] = None  # Lazy cache of Batch.batch_id : Batch
        self._batch_by_index: Optional[Dict[int, Batch]] = None  # Lazy cache of Batch.index : Batch
        
        
    @property
    def batch_ids(self) -> List[str]:
        """A list of `Batch.batch_id`."""
        if self.source is None:
            raise RuntimeError(f"`bind_source` must be called before batches are available.")
        
        if self._batch_by_id is None:
            self._batch_by_id = {b.batch_id: b for b in self.batches}
        return list(self._batch_by_id.keys())
    
    @property
    def batch_by_id(self) -> Dict[str, Batch]:
        """Contains Batches mapped by the unique `Batch.batch_id` attribute."""
        if self.source is None:
            raise RuntimeError(f"`bind_source` must be called before batches are available.")
        
        if self._batch_by_id is None:
            self._batch_by_id = {b.batch_id: b for b in self.batches}
        return self._batch_by_id
    
    @property
    def batch_by_index(self) -> Dict[int, Batch]:
        """Contains Batches mapped by the user-defined `Batch.index` attribute."""
        if self.source is None:
            raise RuntimeError(f"`bind_source` must be called before batches are available.")
        
        if self._batch_by_index is None:
            self._batch_by_index = {b.index: b for b in self.batches}
        return self._batch_by_index

    @property
    def available_roles(self) -> List[str]:
        if not self.batches: 
            return None
        return self.batches[0].available_roles


    def _build_batches(self) -> List[Batch]:
        """
        Builds and returns the batches according to shuffle, stratify, or group settings.
        """
        if self.source is None:
            raise RuntimeError(f"`bind_source` must be called before batches can be built.")
        
        
        samples : List["Sample"] = self.source.samples
        batches : List[Batch] = []
        batch_idx = 0
        
        # Group samples if group_by is specified
        if self.group_by is not None:
            # Assign sample to their respective groups
            grouped_samples = {}
            for sample in samples:
                key = tuple(sample.tags[k] for k in self.group_by)
                grouped_samples.setdefault(key, []).append(sample)

            for g_key, g_samples in grouped_samples.items():
                # Shuffle samples in this group, if specified
                if self.shuffle:
                    self.rng.shuffle(g_samples)
                    
                # Split group samples into batches
                for i in range(0, len(g_samples), self.batch_size):
                    batch_samples = g_samples[i : i+self.batch_size]
                    if self.drop_last and len(batch_samples) < self.batch_size:
                        if i == 0:
                            warnings.warn(
                                f"The current group (`{g_key}`) has fewer than `{self.batch_size}` samples "
                                f"and `drop_last=True`. No batches will be created with this group.",
                                category=UserWarning
                            )
                        continue
                    
                    batches.append(Batch(
                        role_samples={'default':SampleCollection(batch_samples)}, 
                        label=batch_idx
                    ))
                    batch_idx += 1
                        
        # stratify_by logic
        elif self.stratify_by is not None:
            # Assign sample to their respective strat groups
            strat_groups = {}
            for sample in samples:
                key = tuple(sample.tags[k] for k in self.stratify_by)
                strat_groups.setdefault(key, []).append(sample)
            if len(strat_groups) > self.batch_size:
                raise ValueError(
                    f"Batch size (n={self.batch_size}) must be greater than or equal "
                    f"to the number of unique strata (n={len(strat_groups)})"
                )
            
            # Shuffle samples within each stratum
            if self.shuffle:
                for key in strat_groups:
                    self.rng.shuffle(strat_groups[key])
            
            # Flatten in interleaved fashion to ensure class balance
            interleaved_samples = []
            group_iters = {k: iter(v) for k, v in strat_groups.items()}
            group_keys = list(group_iters.keys())
            exhausted = set()
            while len(exhausted) < len(group_keys):
                for k in group_keys:
                    if k in exhausted:
                        continue
                    try:
                        interleaved_samples.append(next(group_iters[k]))
                    except StopIteration:
                        exhausted.add(k)
            
            # Split interleaved_samples into batches
            for i in range(0, len(interleaved_samples), self.batch_size):
                batch_samples = interleaved_samples[i: i + self.batch_size]
                if self.drop_last and len(batch_samples) < self.batch_size:
                    if i == 0:
                        warnings.warn(
                            f"Stratified sampling results in a batch smaller than batch_size={self.batch_size} "
                            f"and `drop_last=True`. No batches will be created.",
                            category=UserWarning
                        )
                    continue
                batches.append(
                    Batch(
                        role_samples={'default':SampleCollection(batch_samples)}, 
                        label=batch_idx
                    )
                )
                batch_idx += 1
              
        # default logic
        else:
            # Shuffle all samples, if specified
            if self.shuffle:
                self.rng.shuffle(samples)
            
            # Create batches
            for batch_idx, i in enumerate(range(0, len(samples), self.batch_size)):
                batch_samples = samples[i : i+self.batch_size]
                
                if self.drop_last and len(batch_samples) < self.batch_size:
                    if i == 0:
                        warnings.warn(
                            f"The current FeatureSampler strategy results in only 1 batch with "
                            f"fewer than `{self.batch_size}` samples and `drop_last=True`. "
                            f"Thus, no batches will be created.",
                            category=UserWarning
                        )
                    continue
                
                batches.append(
                    Batch(
                        role_samples={'default':SampleCollection(batch_samples)}, 
                        label=batch_idx
                    )
                )
                
        # Shuffle batch order, if specified
        if self.shuffle:
            self.rng.shuffle(batches)
            
        return batches


    def is_bound(self) -> bool:
        """Returns True if the sampler has a bound source and batches are built."""
        return self.source is not None and self.batches is not None

    def bind_source(self, source:Union["FeatureSet", "FeatureSubset"]):
        """
        Binds a source FeatureSet or FeatureSubset to this sampler and builds batches.

        This method is required if `source` was not provided during initialization. 
        It resets and rebuilds internal batches based on the sampler configuration.

        Args:
            source (FeatureSet or FeatureSubset): The data source to bind for sampling.
        """
        self.source = source
        self.batches = self._build_batches()
        
    
    def __len__(self):
        """Returns the number of valid batches."""
        return len(self.batches)
        
    def __iter__(self):
        """
        Generator over valid batches in `self.source`

        Yields:
            Batch: A single Batch of samples.
        """
        for batch in self.batches:
            yield batch
            
    def __getitem__(self, key: Union[int, str]) -> "Batch":
        """
        Get a Batch by index (position in list) or `Batch.batch_id` (str). \n
        **Note that by index is not using the user-defined `Batch.index` attribute, \
        it uses the position of the sample in `Batch._batches`. To access Batches \
        by their `.index` attribute, use the `FeatureSampler.get_batch_with_index()`
        method.**
        """
        if isinstance(key, int):
            return self.batches[key]
        elif isinstance(key, str):
            return self.get_batch_with_id(key)
        else:
            raise TypeError(f"Invalid key type: {type(key)}. Expected int or str.")
    
    def get_batch_with_id(self, batch_id: str) -> "Batch":
        """
        Return the Batch with `Batch.batch_id` == `batch_id`, if it exists.
        """        
        if not batch_id in self.batches:
            raise ValueError(f"`batch_id={batch_id}` does not exist in `self.batch_by_id`")
        return self.batch_by_id[batch_id]
    
    def get_batch_with_index(self, batch_index: int) -> "Batch":
        """
        Return the Batch with `Batch.index` == `batch_index`, if it exists.
        """
        if not batch_index in self.batch_by_index:
            raise ValueError(f"`batch_index={batch_index}` does not exist in `self.batch_by_index`")
        return self.batch_by_index[batch_index]
      
      
    