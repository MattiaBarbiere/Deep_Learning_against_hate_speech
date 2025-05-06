import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel

class AttentionCLIP(nn.Module):
    def __init__(self, model_type = "32"):
        """
        Parameters
        model_type: str or int
            Type of CLIP model to use. Can be "32" for ViT-B/32 or "16" for ViT-B/16.
        """ 
        super().__init__()

        # Check if model_type is a string or an integer
        if isinstance(model_type, int):
            model_type = str(model_type)
        elif not isinstance(model_type, str):
            raise ValueError("model_type must be a string or an integer.")
        if model_type not in ["32", "16"]:
            raise ValueError("model_type must be '32' or '16'.")

        # Load the CLIP model and processor
        self.pretrained_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch" + model_type, output_hidden_states=True)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch" + model_type, use_fast=False)
        self.config = self.pretrained_model.config

        # Freeze the CLIP model parameters
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        # Ensure the hidden state is of the same dimension between text and image
        self.linear1 = nn.Linear(self.config.vision_config.hidden_size, self.config.text_config.hidden_size)

        self.text_attentions = None
        self.image_attentions = None

    def forward(self, text, images):
        inputs = self.processor(text=text, images=images, return_tensors="pt", padding=True, truncation=True)

        # Move inputs to the same device as the model
        inputs = {k: v.to(next(self.parameters()).device) for k, v in inputs.items()}

        # Forward pass through the CLIP model
        outputs = self.pretrained_model(**inputs, output_attentions=True)
        text_hidden_states = outputs.text_model_output.last_hidden_state
        image_hidden_states = outputs.vision_model_output.last_hidden_state

        self.text_attentions = outputs.text_model_output.attentions
        self.image_attentions = outputs.vision_model_output.attentions

        # Project image hidden states to match text hidden states
        image_hidden_states = self.linear1(image_hidden_states)
        return torch.cat([text_hidden_states, image_hidden_states], dim=1)

    def get_attention_weights(self):
        return self.text_attentions, self.image_attentions

    def set_attention_weights(self, text_attentions, image_attentions):
        self.text_attentions = text_attentions
        self.image_attentions = image_attentions