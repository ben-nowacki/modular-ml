# from modularml.core.data.featureset import FeatureSet
# from modularml.core.io.handlers.featureset_handler import FeatureSetHandler
from modularml.core.io.handlers.handler import HandlerRegistry
from modularml.core.io.handlers.scaler_handler import ScalerHandler
from modularml.core.transforms.scaler import Scaler

handler_registry = HandlerRegistry()
handler_registry.register(cls=Scaler, handler=ScalerHandler())
# handler_registry.register(cls=FeatureSet, handler=FeatureSetHandler())
