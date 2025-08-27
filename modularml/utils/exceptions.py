


from modularml.utils.backend import Backend



class ModularMLError(Exception):
    """Base class for all ModularML-specific exceptions."""

class BackendNotSupportedError(ModularMLError):
    """Raised when an operation is called on a backend that is not supported."""

    def __init__(self, backend:Backend, method:str=None, message: str = None):
        if message is None:
            if method is None:
                message = f"Unsupported backend '{backend}'."
            else:
                message = f"Unsupported backend '{backend}' in method `{method}`"
        super().__init__(message)
        
class BackendMismatchError(ModularMLError):
    """Raised when a component receives input or configuration with the wrong backend."""

    def __init__(self, expected: str, received: str, message: str = None):
        if message is None:
            message = f"Expected backend '{expected}', but got '{received}'."
        super().__init__(message)
        self.expected = expected
        self.received = received
        



class ActivationError(ModularMLError):
    """Raised when error occurs with Activation class"""
    def __init__(self, message:str = None):
        if message is None:
            message = f"Error with Activation."
        super().__init__(message)
        
        
class OptimizerError(ModularMLError):
    """Raised when error occurs with Optimizer class"""
    def __init__(self, message:str = None):
        if message is None:
            message = f"Error with Optimizer."
        super().__init__(message)

class OptimizerNotSetError(OptimizerError):
    """Raised when training is called on a ModelStage with no Optimizer"""
    def __init__(self, method:str = None, message:str = None):
        if message is None:
            if method is None:
                message = f"Missing Optimizer."
            else:
                message = f"Missing Optimizer in method `{method}`."
        super().__init__(message)



class LossError(ModularMLError):
    """Raised when error occurs with Loss class"""
    def __init__(self, message:str = None):
        if message is None:
            message = f"Error with Loss."
        super().__init__(message)
        
class LossNotSetError(LossError):
    """Raised when loss is None but required."""
    def __init__(self, method:str = None, message:str = None):
        if message is None:
            if method is None:
                message = f"Missing AppliedLoss."
            else:
                message = f"Missing AppliedLoss in method `{method}`."
        super().__init__(message)

   
        
class ModelStageError(ModularMLError):
    """Raised when error occurs within ModelStage class"""
    def __init__(self, message:str = None):
        if message is None:
            message = f"Error with ModelStage."
        super().__init__(message)
        
class ModelStageInputError(ModelStageError):
    def __init__(self, message:str = None):
        if message is None:
            message = f"Error with ModelStage input."
        super().__init__(message)
        
    
class FeatureSetError(ModularMLError):
    def __init__(self, message:str = None):
        if message is None:
            message = f"Error with FeatureSet."
        super().__init__(message)

class SampleLoadError(FeatureSetError):
    def __init__(self, message:str = None):
        if message is None:
            message = f"Failed to load Samples."
        super().__init__(message)
        

class SubsetOverlapWarning(UserWarning):
    pass