import torch
import os
from DL_vs_HateSpeech.models import *

MODEL_NAMES = {"ModelV0": ModelV0, "ModelV1": ModelV1, "ModelV2": ModelV2}

def load_model_from_path(path, file_name, device):
    """
    Load the model from the specified path.

    Args:
        path (str): The path where the folder where model was saved.

    Returns:
        An instance of the model with loaded weights.
    """
    checkpoint = torch.load(os.path.join(path, file_name), map_location=device)

    # Extract the saved model arguments
    model_type = checkpoint["model_type"]

    # Model class
    model_class = MODEL_NAMES[model_type]

    # Extract the model arguments
    model_args = checkpoint["model_args"]

    # Initialize the model using the saved arguments
    model = model_class(**model_args)

    # Load state dicts into submodules
    model.clip.linear1.load_state_dict(checkpoint["clip.linear1"])
    model.classifier.load_state_dict(checkpoint["classifier"])

    # Load attention weights if available
    if "attention_weights" in checkpoint:
        attention_weights = checkpoint["attention_weights"]
        model.clip.set_attention_weights(attention_weights["clip_text"], attention_weights["clip_image"])
        model.classifier.set_attention_weights(attention_weights["classifier"])
    
    # Put the model in evaluation mode
    model.eval()

    return model