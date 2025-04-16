import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
import torch


class FineTunedCLIP(nn.Module):
    def __init__(self, model_type = "32"):
        """
        
        Parameters
        model_type: str or int
            Type of CLIP model to use. Can be "32" for ViT-B/32 or "16" for ViT-B/16.
        """ 
        super().__init__()
        if isinstance(model_type, int):
            model_type = str(model_type)
        elif not isinstance(model_type, str):
            raise ValueError("model_type must be a string or an integer.")
        if model_type not in ["32", "16"]:
            raise ValueError("model_type must be '32' or '16'.")

        self.pretrained_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch" + model_type, output_hidden_states=True)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch" + model_type)
        self.config = self.pretrained_model.config

        # Make sure the hidden state is of the same dim
        self.linear1 = nn.Linear(self.config.vision_config.hidden_size, self.config.text_config.hidden_size)

    def forward(self, text, images):
        inputs = self.processor(text=text, images=images, return_tensors="pt", padding=True, truncation=True)
        outputs = self.pretrained_model(**inputs)
        text_hidden_states = outputs.text_model_output.last_hidden_state    # Tensor of shape (batch_size, sequence_length, hidden_size)
        image_hidden_states = outputs.vision_model_output.last_hidden_state # Tensor of shape (batch_size, num_patches + 1, hidden_size)
        
        image_hidden_states = self.linear1(image_hidden_states)
        return torch.cat([text_hidden_states, image_hidden_states], dim=1)