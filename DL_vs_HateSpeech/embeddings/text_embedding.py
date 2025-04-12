import torch.nn as nn
from torchvision.models import resnet50

class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.resnet = resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()
        self.linear = nn.Linear(2048, embed_dim)

    def forward(self, images):
        x = self.resnet(images)
        x = self.linear(x)
        return x