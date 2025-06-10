"""
utils.py

Utility functions for label conversion, data loading, parameter checking, and configuration management.
"""

from DL_vs_HateSpeech.env_constants import PATH_TO_JSON_US_POL, PATH_TO_JSON_COVID_19
import pandas as pd
import json
import os
import yaml
import torch

# Dict that assigns each label to a number
LABEL_TO_NUM = {
    "not harmful": 0,
    "somewhat harmful": 1,
    "very harmful": 1
}

NUM_TO_LABEL = {
    0.0: "not harmful",
    1.0: "harmful",
}

def get_label_num(label):
    """
    Convert a label string to its corresponding numerical value.

    Args:
        label (str): The label to convert.

    Returns:
        int: The numerical value of the label.
    """
    if label in LABEL_TO_NUM:
        return LABEL_TO_NUM[label]
    else:
        raise ValueError(f"Label {label} not found in LABEL_TO_NUM dictionary.")
    
def get_label_num_list(label_list):
    """
    Convert a list of label strings to their corresponding numerical values.

    Args:
        label_list (list): The list of labels to convert.

    Returns:
        list: A list of numerical values corresponding to the labels.
    """
    return [get_label_num(label) for label in label_list]
    
def get_label_str(label_num):
    """
    Convert a numerical label to its corresponding string value.

    Args:
        label_num (int or torch.Tensor): The numerical label to convert.

    Returns:
        str: The string value of the label.
    """
    if isinstance(label_num, torch.Tensor):
        label_num = label_num.item()
    return NUM_TO_LABEL[label_num]

def get_label_str_list(label_num_list):
    """
    Convert a list of numerical labels to their corresponding string values.

    Args:
        label_num_list (list): The list of numerical labels to convert.

    Returns:
        list: A list of string values corresponding to the numerical labels.
    """
    return [get_label_str(label_num) for label_num in label_num_list]

def find_text_and_label_jsonl(image_name, type="train", subset="us_pol"):
    """
    Given an image name, find the text and labels of that image in the JSONL dataset.

    Args:
        image_name (str): The name of the image.
        type (str): Dataset type, e.g. "train", "val", or "test".
        subset (str): Dataset subset, e.g. "us_pol", "covid_19", or "both".

    Returns:
        tuple: A tuple containing the file path, text, and labels.
    """
    paths = []

    # Select the correct dataset paths based on the subset
    if subset == "us_pol":
        paths = [PATH_TO_JSON_US_POL[type]]
    elif subset == "covid_19":
        paths = [PATH_TO_JSON_COVID_19[type]]
    elif subset == "both":
        paths = [PATH_TO_JSON_US_POL[type], PATH_TO_JSON_COVID_19[type]]
    else:
        raise ValueError(f"Unknown subset: {subset}")

    # Search for the image in the selected datasets
    for path in paths:
        df = load_json(path)
        match = df[df["image"] == image_name]
        if not match.empty:
            text = match.iloc[0]["text"]
            labels = match.iloc[0]["labels"]
            return path, text, labels

    raise ValueError(f"Image name {image_name} not found in any JSONL file.")

def load_json(path):
    """
    Load a JSONL file and return its content as a pandas DataFrame.

    Args:
        path (str): The path to the JSONL file.

    Returns:
        pd.DataFrame: The content of the JSONL file.
    """
    with open(path, 'r') as f:
        data = [json.loads(line) for line in f]
    return pd.DataFrame(data)

def check_frozen_params(model, print_layers=False):
    """
    Check which parameters of the model are frozen and which are trainable.
    Optionally print the names of the layers and their requires_grad status.

    Args:
        model (nn.Module): The model to check.
        print_layers (bool): If True, print the names of the layers and their requires_grad status.

    Returns:
        None
    """
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if print_layers:
        for name, param in model.named_parameters():
            print(f"{name}: requires_grad = {param.requires_grad}")
    print(f"Trainable params: {trainable}, Frozen params: {frozen}")

def read_yaml_file(path):
    """
    Open the Hydra-generated YAML config file and return the parameters as a dictionary.

    Args:
        path (str): Path to the experiment folder containing .hydra/config.yaml

    Returns:
        dict: Parameters loaded from the YAML file.
    """
    # Get the absolute path
    path = os.path.abspath(path)
    with open(os.path.join(path, "config.yaml"), 'r') as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    # Organise the dict of parameters
    return params