from DL_vs_HateSpeech.CLIP import FineTunedCLIP
import torch.nn as nn

class ModelV0(nn.Module):
    def __init__(self, clip_model_type = "32"):
        super().__init__()
        self.clip = FineTunedCLIP(model_type=clip_model_type)

        

    def forward(self, x):
        # Define the forward pass of the model here
        pass