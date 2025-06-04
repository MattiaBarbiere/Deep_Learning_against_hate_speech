"""
model_v2.py

Defines ModelV2, a multimodal model using AttentionCLIP and an attention classifier.
"""

import torch
import torch.nn as nn
import os
from DL_vs_HateSpeech.models.base_model import BaseModel
from DL_vs_HateSpeech.CLIP import AttentionCLIP
from DL_vs_HateSpeech.transformer_models.attention_transformer import AttentionClassifier

class ModelV2(nn.Module, BaseModel):
    """
    ModelV2: Uses AttentionCLIP and an attention-based classifier.
    """
    def __init__(self, clip_model_type="32", hidden_dim=256, dropout=0.1, output_dim=3):
        """
        Args:
            clip_model_type (str): Type of CLIP model ("32" or "16").
            hidden_dim (int): Hidden dimension for the classifier.
            dropout (float): Dropout rate for the classifier.
            output_dim (int): Output dimension (number of classes).
        """
        nn.Module.__init__(self)
        BaseModel.__init__(self)
        # Save the args
        self.clip_model_type = clip_model_type
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.output_dim = output_dim

        # Multimodal CLIP
        self.clip = AttentionCLIP(model_type=clip_model_type)
        
        # Classifier
        embedding_dim = self.clip.config.text_config.hidden_size  # Same as aligned image dim
        self.classifier = AttentionClassifier(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            output_dim=output_dim
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
        # print("Joint embeddings shape:", joint_embeddings.shape)
        
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
            torch.Tensor: Softmax or sigmoid probabilities.
        """
        # Make sure we are in evaluation mode
        self.eval()

        # Get joint embeddings from FineTunedCLIP
        joint_embeddings = self.clip(text, images)

        # Classification
        logits = self.classifier(joint_embeddings)
        
        # Return softmax or sigmoid probabilities depending on output_dim
        if self.output_dim == 3:
            return torch.softmax(logits, dim=-1)
        elif self.output_dim == 1:
            return torch.sigmoid(logits)
        else:
            raise ValueError(f"Unknown output dimension: {self.output_dim}. Must be 1 or 3.")

    def save(self, path=None, file_name=None):
        """
        Save the model state dictionary to the specified path.
        
        Args:
            path (str): The path to save the model. Default is None, which saves to the current directory.
            file_name (str): The name of the file to save the model. Default is None, which saves as "model.pth".
        """
        # Make sure the clip weights are frozen
        self.assert_frozen_params()

        # Create the directory if it doesn't exist
        if path is None:
            path = os.path.join(os.getcwd())
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)

        # Create the file name if not provided
        if file_name is None:
            file_name = "model.pth"
        
        # Make sure the clip weights are frozen
        self.assert_frozen_params()

        # Get attention weights from both components
        clip_text_attn, clip_image_attn = self.clip.get_attention_weights()
        classifier_attn = self.classifier.get_attention_weights()

        # Save the model state dictionary with attention weights
        torch.save({
            "model_type": self.model_type,
            "model_args": {
                "clip_model_type": self.clip_model_type,
                "hidden_dim": self.hidden_dim,
                "dropout": self.dropout,
                "output_dim": self.output_dim
            },
            "clip.linear1": self.clip.linear1.state_dict(),
            "classifier": self.classifier.state_dict(),
            # "attention_weights": {
            #     "clip_text": clip_text_attn,
            #     "clip_image": clip_image_attn,
            #     "classifier": classifier_attn
            # }
        }, os.path.join(path, file_name))
        
    def get_model_attention(self):
        """
        Get the attention weights from the model.
        
        Returns:
            dict: A dictionary containing the attention weights from both components.
        """
        clip_text_attn, clip_image_attn = self.clip.get_attention_weights()
        classifier_attn = self.classifier.get_attention_weights()
        return {
            "clip_text": clip_text_attn,
            "clip_image": clip_image_attn,
            "classifier": classifier_attn
        }
    
    def get_tokenizer(self):
        """
        Get the tokenizer from the CLIP model.
        
        Returns:
            tokenizer: The tokenizer from the CLIP model.
        """
        return self.clip.processor.tokenizer.tokenize