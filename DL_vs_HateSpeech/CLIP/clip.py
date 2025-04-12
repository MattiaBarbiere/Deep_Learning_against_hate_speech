import torch.nn as nn

class CLIP(nn.Module):
    def __init__(self, text_encoder, image_encoder):
        super().__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder

    def forward(self, text, images):
        text_features = self.text_encoder(text)
        image_features = self.image_encoder(images)
        pass