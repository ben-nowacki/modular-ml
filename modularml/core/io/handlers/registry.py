from modularml.core.io.handlers.handler import HandlerRegistry

from modularml.core.transforms.scaler import Scaler
from modularml.core.io.handlers.scaler_handler import ScalerHandler

from modularml.core.splitting.base_splitter import BaseSplitter
from modularml.core.io.handlers.splitter_handler import SplitterHandler

from modularml.core.data.featureset import FeatureSet
from modularml.core.io.handlers.featureset_handler import FeatureSetHandler

from modularml.core.io.handlers.similarity_condition_handler import SimilarityConditionHandler
from modularml.core.sampling.similiarity_condition import SimilarityCondition

from modularml.core.sampling.base_sampler import BaseSampler
from modularml.core.io.handlers.sampler_handler import SamplerHandler

from modularml.core.models.base_model import BaseModel
from modularml.core.io.handlers.model_handler import ModelHandler

from modularml.core.training.optimizer import Optimizer
from modularml.core.io.handlers.optimizer_handler import OptimizerHandler

from modularml.core.training.loss import Loss
from modularml.core.io.handlers.loss_handler import LossHandler


handler_registry = HandlerRegistry()
handler_registry.register(cls=Scaler, handler=ScalerHandler())
handler_registry.register(cls=BaseSplitter, handler=SplitterHandler())
handler_registry.register(cls=FeatureSet, handler=FeatureSetHandler())
handler_registry.register(cls=SimilarityCondition, handler=SimilarityConditionHandler())
handler_registry.register(cls=BaseSampler, handler=SamplerHandler())
handler_registry.register(cls=BaseModel, handler=ModelHandler())
handler_registry.register(cls=Optimizer, handler=OptimizerHandler())
handler_registry.register(cls=Loss, handler=LossHandler())
