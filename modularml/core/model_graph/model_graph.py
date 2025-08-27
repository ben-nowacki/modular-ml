

'''
DAG of FeatureSets and ModelStages

Responsibilities:
    Stores all nodes: FeatureSets and ModelStages.
    Stores all edges: how outputs of one stage feed another.
    Performs:
        Topological sorting
        Lazy build() propagation of shapes
        Forward pass through the graph
'''


from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict, deque
import warnings
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from modularml.core.data_structures.batch import Batch
from modularml.core.data_structures.data import Data
from modularml.core.data_structures.feature_set import FeatureSet
from modularml.core.data_structures.multi_batch import MultiBatch
from modularml.core.data_structures.sample import Sample
from modularml.core.data_structures.sample_collection import SampleCollection
from modularml.core.model_graph.model_stage import ModelStage, StageInput
from modularml.utils.backend import Backend
from modularml.utils.data_format import get_data_format_for_backend



def make_dummy_data(shape: Tuple[int, ...], batch_size:int=None) -> Data:
    """
    Creates a dummy Data object. A batch dimension is added if `batch_size` is not None.
    """
    
    # Add batch dimension if specified
    if batch_size is not None:
        shape = (batch_size, ) + shape
        
    # Create dummy data
    d = Data(np.ones(shape=shape))
    
    return d

def make_dummy_batch(feature_shape: Tuple[int, ...], batch_size:int=8) -> Batch:
    sample_coll = SampleCollection([
        Sample(
            features={'features_1': make_dummy_data(shape=feature_shape)},
            targets={'targets_1': make_dummy_data(shape=(1,1))},
            tags={'tags_1': make_dummy_data(shape=(1,)), 'tags_2': make_dummy_data(shape=(1,))},
        )
        for i in range(batch_size)
    ])
    return Batch(
        role_samples = {'default': sample_coll}, 
        label='dummy', 
    )    
    

class ModelGraph:
    def __init__(
        self,
        nodes: List[Union[FeatureSet, ModelStage]],
    ):
        self.all_nodes : Dict[str, Union[FeatureSet, ModelStage]] = {}
        self._model_stages : Dict[str, ModelStage] = {}
        self._feature_sets : Dict[str, FeatureSet] = {}
        for node in nodes:
            self.all_nodes[node.label] = node
            if isinstance(node, FeatureSet): 
                self._feature_sets[node.label] = node
            elif isinstance(node, ModelStage): 
                self._model_stages[node.label] = node
            else:
                raise TypeError(
                    f"Unknown node type. Must be of type ModelStage or FeatureSet. "
                    f"Received: {type(node)}"
                )
        
        self._validate_graph()
        self._sorted_stage_labels = self._topological_sort()
    
    @property
    def feature_set_labels(self) -> List[str]:
        return list(self._feature_sets.keys())
    
    @property
    def model_stage_labels(self) -> List[str]:
        return list(self._model_stages.keys())
    
    
    def __repr__(self):
        return self.summary()
    
    def summary(self) -> str:
        """Returns a compact summary of all stages in the graph."""
        
        msg = f"ModelGraph(n_nodes={len(self.all_nodes)})"
        for label, fs in self._feature_sets.items():
            msg += f"\n  + {repr(fs)}"
        for label, stage in self._model_stages.items():
            msg += f"\n  + {stage.description_short()}"
        
        return msg
    
 
 
    def _validate_graph(self):
        """Check for backend consistency and graph connectivity."""
        
        # Warn if using mixed backend: not thoroughly tested
        used_backends = set(stage.backend for stage in self._model_stages.values())
        if len(used_backends) > 1:
            warnings.warn(
                "Mixed backends detected in ModelGraph. Though allowed, minimal testing has been "
                "conducted. Gradient flow may break during training.",
                category=UserWarning,
                stacklevel=2
            )
        
        # Check for unreachable stages
        reachable = set()
        frontier = [
            inp.source
            for label, stage in self._model_stages.items()
            for inp in stage.inputs
            if inp.source not in self._model_stages
        ]
        # BFS traversal
        while frontier:
            current = frontier.pop()			# Current stage in search
            if current in reachable: continue	# Ignore if already seen
            reachable.add(current)	
            for label, stage in self._model_stages.items():
                inputs: List[StageInput] = stage.inputs if isinstance(stage.inputs, list) else [stage.inputs, ]
                if current in [inp.source for inp in inputs]:
                    frontier.append(label)		# Add next connected stage

        unreachable = set(self._model_stages) - reachable
        if unreachable:
            warnings.warn(
                f"Unreachable ModelStages detected in ModelGraph: {sorted(unreachable)}. ",
                category=UserWarning,
                stacklevel=2
            )
    
    def _topological_sort(self) -> List[str]:
        """Topological sort of model graph using Kahn's algorithm."""
        in_degree = defaultdict(int)        # Number of dependencies (value) for each stage (key)
        children = defaultdict(list)        # List of child stage names (value) for each stage (key)
        all_stage_names = set(self._model_stages.keys())

        # Initialize in-degree of all nodes to 0
        for stage_name in all_stage_names:
            in_degree[stage_name] = 0

        for stage_name, stage in self._model_stages.items():
            # Get all parent stages for this current stage
            parents = stage.inputs if isinstance(stage.inputs, list) else [stage.inputs]
            for p in parents:
                # If p is a base node (ie, it's name is FeatureSet.label), continue
                if p.source in self._feature_sets:
                    continue
                # Otherwise, increment the in_degree (how many parents this stage has)
                if p.source not in self.all_nodes:
                    raise ValueError(
                        f"Invalid input source '{p.source}' for stage '{stage_name}'.")
                in_degree[stage_name] += 1
                children[p.source].append(stage_name)

        sorted_stage_names = []
        queue = deque([k for k in all_stage_names if in_degree[k] == 0])

        while queue:
            stage_name = queue.popleft()
            sorted_stage_names.append(stage_name)
            for child in children[stage_name]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(sorted_stage_names) != len(all_stage_names):
            unresolved = all_stage_names - set(sorted_stage_names)
            raise ValueError(f"Cyclic dependency detected in ModelGraph: {unresolved}")

        return sorted_stage_names
    
    
    def build_all(self):
        """Build all ModelStage contain in this ModelGraph"""
        for stage_label in self._sorted_stage_labels:
            node : ModelStage = self._model_stages[stage_label]
        
            if not node.is_built:
                # Try building without input shapes
                try: node.build()
                except: pass
                
                # Try inferring input and output shapes
                try:
                    input_shape = self._infer_input_shape(node.inputs)
                    output_shape = self._infer_output_shape(node, input_shape)

                    node.build(
                        input_shape=input_shape,
                        output_shape=output_shape
                    )
                    
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to build node: {node.label}. {e}"
                    )
            
            print(f"Inferred shapes for `{node.label}`: ", node.input_shape, "->", node.output_shape)
    
    def _infer_input_shape(self, inputs: List[StageInput]) -> Tuple[int, ...]:
        """Attempts to infer the input shape given the StageInput specs"""
        
        input_shapes = []
        for inp in inputs:
            # Get previous node
            prev_node = self.all_nodes[inp.source]
            
            if isinstance(prev_node, FeatureSet):
                input_shapes.append(tuple(int(d) for d in prev_node.feature_shape))
                continue
            
            elif isinstance(prev_node, ModelStage):
                if prev_node.output_shape is None:
                    raise ValueError(
                        f"Previous ModelStage has no output shape. "
                        f"Run .build() to perform model input/output shape inference."
                    )
                input_shapes.append(tuple(int(d) for d in prev_node.output_shape))
                continue
            
            else:
                raise TypeError(f"Unknown node type: {prev_node}")
            
        if len(input_shapes) == 1:
            return input_shapes[0]
        else:
            # TODO: how to concantenate
            raise NotImplementedError(
                f"Output shape determination for multiple input sources is not "
                f"implemented yet. Received shapes: {input_shapes}"
            )
              
    def _infer_output_shape(self, node: ModelStage, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Attempts to infer the output shape of a ModelStage
        Runs a Dummy input with `input_shape` and check outputs size.
        """
        # Make dummy input for ModelStage with batch_size = 1
        X: Data = make_dummy_data(shape=input_shape, batch_size=1)
    
        # Collect model output
        y = node.forward(X)
        
        # Drop batch dimension
        return tuple(int(dim) for dim in y.shape[1:])



    def forward(self, batch: MultiBatch) -> Dict[str, Batch]:
        
        missing_featuresets = []
        for fs in self._feature_sets.keys():
            if fs not in batch.keys(): missing_featuresets.append(fs)
        if missing_featuresets:
            raise ValueError(
                f"The MultiBatch provided to ModelGraph is missing data from required "
                f"FeatureSets. Missing: {missing_featuresets}"
            )
        
        # Stores output from each node in ModelGraph
        cache : Dict[str, Batch] = {}
        
        # Add FeatureSet data
        for fs_label in self.feature_set_labels:
            cache[fs_label] = batch[fs_label]
            
        # Topological forward pass through ModelStages
        for label in self._sorted_stage_labels:
            stage = self._model_stages[label]
            inputs: List[Batch] = []
            for inp in stage.inputs:
                if inp.source not in cache:
                    raise ValueError(f"Missing input `{inp.source}` for stage `{label}`")
                inputs.append(cache[inp.source])
                
            # TODO: Combine multiple inputs into one input (e.g., tuple, dict, or concat)
            stage_input = inputs[0] if len(inputs) == 1 else tuple(inputs)
            
            # Model forward pass
            output: Batch = stage.forward(stage_input)
            cache[label] = output
            
        return cache
    
    def dummy_foward(self, batch_size:int = 8) -> Batch:
        """
        A foward pass through the entire ModelGraph with a dummy input to test connections.
        """
        if len(self._feature_sets.keys()) > 1:
            raise NotImplementedError(
                f"`dummy_forward` doesn't currently support ModelGraphs with multiple FeatureSets."
            )
        fs = self._feature_sets[self.feature_set_labels[0]]
        batch = make_dummy_batch(feature_shape=fs.feature_shape, batch_size=batch_size)
        multi_batch = MultiBatch({self.feature_set_labels[0]: batch})
        
        res = self.forward(multi_batch)
        output = res[self._sorted_stage_labels[-1]]
        return output.get_samples('default').get_all_features(format='np').tolist()
        
    def visualize(
        self, 
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Visualize the structure of the model graph.

        Args:
            save_path (str, optional): If provided, saves the figure to this path.

        Returns:
            Tuple[plt.Figure, plt.Axes]: matplotlib.pyplot Figure and Axes.
        """
        import networkx as nx
        
        graph = nx.DiGraph()

        for label, stage in self._model_stages.items():
            inputs : List[StageInput] = stage.inputs if isinstance(stage.inputs, list) else [stage.inputs]
            for inp in inputs:
                graph.add_edge(inp.source, label,)

        for layer, nodes in enumerate(nx.topological_generations(graph)):
            for node in nodes:
                graph.nodes[node]["layer"] = layer

        pos = nx.multipartite_layout(graph, subset_key="layer")

        # Assign numeric IDs and build label map
        node_to_id = {node: idx for idx, node in enumerate(graph.nodes())}
        id_to_node = {v: k for k, v in node_to_id.items()}
        labels = {node: str(idx) for node, idx in node_to_id.items()}

        # Identify feature nodes vs stage nodes
        feature_nodes = [n for n in graph.nodes if n not in self._model_stages]
        stage_nodes = [n for n in graph.nodes if n in self._model_stages]


        fig, ax = plt.subplots(figsize=(6, 3))

        # Draw features as squares
        nx.draw_networkx_nodes(
            graph, pos, ax=ax, nodelist=feature_nodes,
            node_shape='s', node_color='lightgray',
            edgecolors='black', linewidths=2, node_size=2000
        )

        # Draw stages as circles
        nx.draw_networkx_nodes(
            graph, pos, ax=ax, nodelist=stage_nodes,
            node_shape='o', node_color='skyblue',
            edgecolors='black', linewidths=2, node_size=2000
        )

        # Draw edges
        nx.draw_networkx_edges(
            graph, pos, ax=ax, 
            width=2, 
            style='solid',
            arrows=True,             # Enable arrows
            arrowstyle='-|>',         # Style of the arrow
            arrowsize=20,            # Size of the arrow head
            node_size=2200
            )

        # Draw labels (all)
        nx.draw_networkx_labels(graph, pos, labels=labels, font_size=12)

        # Add legend mapping node ID to stage name
        legend_elements = [
            Line2D(
                [0], [0], 
                marker='s' if name in feature_nodes else 'o', 
                color='w', 
                label=f"{idx}: {'FeatureSet' if name in feature_nodes else 'ModelStage'} `{name}`",
                markerfacecolor='lightgray' if name in feature_nodes else 'skyblue', 
                markersize=10, 
                markeredgecolor='black')
            for idx, name in id_to_node.items()
        ]
        ax.legend(handles=legend_elements, title="ModelGraph Legend", bbox_to_anchor=(1, 0.5), loc='center left')

        ax.margins(0.20)
        ax.axis('off')
        if save_path:
            fig.savefig(save_path)
        return fig, ax

    
    
    