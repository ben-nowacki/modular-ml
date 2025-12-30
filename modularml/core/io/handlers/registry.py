from modularml.core.io.handlers.handler import HandlerRegistry

from modularml.core.splitting.base_splitter import BaseSplitter
from modularml.core.io.handlers.splitter_handler import SplitterHandler

from modularml.core.transforms.scaler import Scaler
from modularml.core.io.handlers.scaler_handler import ScalerHandler

from modularml.core.data.featureset import FeatureSet
from modularml.core.io.handlers.featureset_handler import FeatureSetHandler

handler_registry = HandlerRegistry()
handler_registry.register(cls=Scaler, handler=ScalerHandler())
handler_registry.register(cls=BaseSplitter, handler=SplitterHandler())
handler_registry.register(cls=FeatureSet, handler=FeatureSetHandler())
