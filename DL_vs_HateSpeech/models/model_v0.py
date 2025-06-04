"""
model_v0.py

Defines ModelV0, a baseline multimodal model using CLIP and a transformer classifier.
"""

import torch
import torch.nn as nn
import os
from DL_vs_HateSpeech.models.base_model import BaseModel
from DL_vs_HateSpeech.CLIP import FineTunedCLIP
from DL_vs_HateSpeech.transformer_models.transformer import TransformerClassifier

class ModelV0(nn.Module, BaseModel):
    """
    Baseline model using CLIP for feature extraction and a transformer classifier.
    """
    def __init__(self, clip_model_type="32", hidden_dim=256, dropout=0.1):
        """
        Args:
            clip_model_type (str): Type of CLIP model ("32" or "16").
            hidden_dim (int): Hidden dimension for the classifier.
            dropout (float): Dropout rate for the classifier.
        """
        nn.Module.__init__(self)
        BaseModel.__init__(self)
        # Save the args
        self.clip_model_type = clip_model_type
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Multimodal CLIP
        self.clip = FineTunedCLIP(model_type=clip_model_type)
        
        # Classifier
        embedding_dim = self.clip.config.text_config.hidden_size  # Same as aligned image dim
        self.classifier = TransformerClassifier(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        # Make sure the clip weights are frozen
        self.assert_frozen_params()

    def forward(self, text, images, attention_mask=None):
        """
        Forward pass for the model.

        Args:
            text: List[str] or tensor (batch_size,)
            images: List[PIL.Image] or tensor (batch_size, channels, H, W)
            attention_mask: Optional tensor (batch_size, seq_len) for text padding

        Returns:
            logits: Output of the classifier.
        """
        # Make sure we are in training mode
        self.train()

        # Get joint embeddings from FineTunedCLIP
        joint_embeddings = self.clip(text, images) # (batch_size, seq_len + num_patches + 1, embedding_dim)
        
        # Classification
        logits = self.classifier(joint_embeddings, attention_mask)
        return logits
    
    def predict(self, text, images):
        """
        Predict probabilities for the given inputs.

        Args:
            text: List[str] or tensor (batch_size,)
            images: List[PIL.Image] or tensor (batch_size, channels, H, W)

        Returns:
            torch.Tensor: Softmax probabilities.
        """
        # Make sure we are in evaluation mode
        self.eval()

        # Get joint embeddings from FineTunedCLIP
        joint_embeddings = self.clip(text, images)

        # Classification
        logits = self.classifier(joint_embeddings)
        
        # Return softmax probabilities
        return torch.softmax(logits, dim=-1)

    def save(self, path=None):
        """
        Save the model state dictionary to the specified path.
        
        Args:
            path (str): The path to save the model. Default is None, which saves to the current directory.
        """
        # Make sure the clip weights are frozen
        self.assert_frozen_params()

        # Create the directory if it doesn't exist
        if path is None:
            path = os.path.join(os.getcwd())
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save the model state dictionary
        torch.save({
        "model_type": self.model_type,
        "model_args": {
            "clip_model_type": self.clip_model_type,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout
        },
        "clip.linear1": self.clip.linear1.state_dict(),
        "classifier": self.classifier.state_dict()
        }, os.path.join(path, "model.pth"))
