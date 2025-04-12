from DL_vs_HateSpeech.CLIP import CLIP
from DL_vs_HateSpeech.embeddings import ImageEncoder, TextEncoder
import torch.nn as nn

class ModelV0(nn.Module):
    def __init__(self, image_emb_dim=512, text_emb_dim=512):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim=image_emb_dim)
        self.text_encoder = TextEncoder(embed_dim=text_emb_dim)
        self.clip = CLIP(image_encoder=self.image_encoder, text_encoder=self.text_encoder)

        # Simple linear layer for classification
        self.classifier = nn.Sequential(
            nn.Linear(image_emb_dim + text_emb_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        # Define the forward pass of the model here
        pass