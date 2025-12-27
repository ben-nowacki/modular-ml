from modularml.core.io.handlers.handler import HandlerRegistry
from modularml.core.io.handlers.scaler_handler import ScalerHandler
from modularml.core.transforms.scaler import Scaler

handler_registry = HandlerRegistry()
handler_registry.register(cls=Scaler, handler=ScalerHandler())
