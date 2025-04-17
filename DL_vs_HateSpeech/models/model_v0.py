import torch.nn as nn
from DL_vs_HateSpeech.CLIP import FineTunedCLIP
from DL_vs_HateSpeech.transformer_models.transformer import TransformerClassifier

class ModelV0(nn.Module):
    def __init__(self, clip_model_type="32", hidden_dim=256, dropout=0.1):
        super().__init__()
        # Multimodal CLIP
        self.clip = FineTunedCLIP(model_type=clip_model_type)
        
        # Binary classifier
        embedding_dim = self.clip.config.text_config.hidden_size  # Same as aligned image dim
        self.classifier = TransformerClassifier(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

    def forward(self, text, images, attention_mask=None):
        """
        text: List[str] or tensor (batch_size,)
        images: List[PIL.Image] or tensor (batch_size, channels, H, W)
        attention_mask: Optional tensor (batch_size, seq_len) for text padding
        """
        # Get joint embeddings from FineTunedCLIP
        joint_embeddings = self.clip(text, images) # (batch_size, seq_len + num_patches + 1, embedding_dim)
        
        # Classification
        logits = self.classifier(joint_embeddings, attention_mask) # (batch_size,)
        return logits