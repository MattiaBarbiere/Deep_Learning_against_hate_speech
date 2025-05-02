import torch
import torch.nn as nn
import os
from DL_vs_HateSpeech.CLIP import FineTunedCLIP
from DL_vs_HateSpeech.transformer_models.transformer import TransformerClassifier

class ModelV0(nn.Module):
    def __init__(self, clip_model_type="32", hidden_dim=256, dropout=0.1):
        super().__init__()
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
        text: List[str] or tensor (batch_size,)
        images: List[PIL.Image] or tensor (batch_size, channels, H, W)
        attention_mask: Optional tensor (batch_size, seq_len) for text padding
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
        text: List[str] or tensor (batch_size,)
        images: List[PIL.Image] or tensor (batch_size, channels, H, W)
        """
        # Make sure we are in evaluation mode
        self.eval()

        # Get joint embeddings from FineTunedCLIP
        joint_embeddings = self.clip(text, images)

        # Classification
        logits = self.classifier(joint_embeddings)
        
        # Return softmax probabilities
        return torch.softmax(logits, dim=-1)


    def save(self, path):
        """
        Save the model state dictionary to the specified path.
        
        Args:
            path (str): The path to save the model.
        """
        # Make sure the clip weights are frozen
        self.assert_frozen_params()

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save the model state dictionary
        torch.save({
        "clip.linear1": self.clip.linear1.state_dict(),
        "classifier": self.classifier.state_dict()
        }, path + "fine_tuned_params.pth")

    def load(self, path):
        """
        Load the model state dictionary from the specified path.
        
        Args:
            path (str): The path to load the model from.
        """
        checkpoint = torch.load(path + "fine_tuned_params.pth")
        self.clip.linear1.load_state_dict(checkpoint["clip.linear1"])
        self.classifier.load_state_dict(checkpoint["classifier"])

        # Make sure the clip weights are frozen
        self.assert_frozen_params()

    def assert_frozen_params(self):
        """
        Assert that the model weights are frozen.
        """
        # All parameters from clip.pretrained_model should be frozen
        for param in self.clip.pretrained_model.parameters():
            assert not param.requires_grad, "CLIP model parameters should be frozen."
        