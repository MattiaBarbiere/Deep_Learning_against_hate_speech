"""
base_model.py

Defines the abstract base class for all models.
"""

from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    Base class for all models.
    
    This class provides a common interface for all models, including methods for training,
    evaluation, and saving/loading model weights. It also includes a method to ensure that
    the model parameters are frozen.
    """
    def __init__(self):
        super().__init__()

    @property
    def model_type(self):
        """
        Returns the model type as a string.
        """
        return self.__class__.__name__
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Forward pass for the model.
        """
        pass
    
    @abstractmethod
    def predict(self, *args, **kwargs):
        """
        Predict method for the model.
        """
        pass
    
    @abstractmethod
    def save(self, path):
        """
        Save the model to the specified path.
        """
        pass
    
    def assert_frozen_params(self):
        """
        Assert that the model weights are frozen.
        Raises an AssertionError if any parameter in clip.pretrained_model is trainable.
        """
        # All parameters from clip.pretrained_model should be frozen
        for param in self.clip.pretrained_model.parameters():
            assert not param.requires_grad, "CLIP model parameters should be frozen."