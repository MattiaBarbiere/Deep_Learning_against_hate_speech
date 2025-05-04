from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    Base class for all models in the project.
    
    This class provides a common interface for all models, including methods for training,
    evaluation, and saving/loading model weights. It also includes a method to ensure that
    the model parameters are frozen.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    @property
    def model_type(self):
        """
        Returns the model type.
        """
        pass
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def predict(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def save(self, path):
        pass
    
    def assert_frozen_params(self):
        """
        Assert that the model weights are frozen.
        """
        # All parameters from clip.pretrained_model should be frozen
        for param in self.clip.pretrained_model.parameters():
            assert not param.requires_grad, "CLIP model parameters should be frozen."