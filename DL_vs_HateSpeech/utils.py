from DL_vs_HateSpeech.env_constants import PATH_TO_JSON_US_POL, PATH_TO_JSON_COVID_19
import pandas as pd
import json

# Dict that assigns each label to a number
LABEL_TO_NUM = {
    "not harmful": 0,
    "somewhat harmful": 1,
    "very harmful": 2
}

def get_label_num(label):
    """
    Convert a label to its corresponding numerical value.

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
    Convert a list of labels to their corresponding numerical values.

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
        label_num (int): The numerical label to convert.

    Returns:
        str: The string value of the label.
    """
    for key, value in LABEL_TO_NUM.items():
        if value == label_num:
            return key
    raise ValueError(f"Label number {label_num} not found in LABEL_TO_NUM dictionary.")

def get_label_str_list(label_num_list):
    """
    Convert a list of numerical labels to their corresponding string values.

    Args:
        label_num_list (list): The list of numerical labels to convert.

    Returns:
        list: A list of string values corresponding to the numerical labels.
    """
    return [get_label_str(label_num) for label_num in label_num_list]

# # A function that given an image name, find the text and label of that image
# def find_text_and_label(image_name):
#     """
#     Given an image name, find the text and label of that image in the dataframe.

#     Args:
#         image_name (str): The name of the image.

#     Returns:
#         tuple: A tuple containing the file, text and label of the image.
#     """
#     # Iterate over the files
#     for file in PATH_TO_JSON_FILES.values():
#         df = pd.read_csv(file)
#         # Check if the first column matches the image name
#         match = df[df.iloc[:, 0] == image_name]
#         if not match.empty:
#             # Get the values from second and third columns
#             text = match.iloc[0, 1]
#             label = match.iloc[0, 2]
#             return file, text, label
    
#     # If no match is found, return error
#     raise ValueError(f"Image name {image_name} not found in any CSV file.")

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

    if subset == "us_pol":
        paths = [PATH_TO_JSON_US_POL[type]]
    elif subset == "covid_19":
        paths = [PATH_TO_JSON_COVID_19[type]]
    elif subset == "both":
        paths = [PATH_TO_JSON_US_POL[type], PATH_TO_JSON_COVID_19[type]]
    else:
        raise ValueError(f"Unknown subset: {subset}")

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
    Load a JSON file and return its content.

    Args:
        path (str): The path to the JSON file.

    Returns:
        dict: The content of the JSON file.
    """
    with open(path, 'r') as f:
        data = [json.loads(line) for line in f]
    return pd.DataFrame(data)

def check_frozen_params(model, print_layers=False):
    """
    Check which parameters of the model are frozen and which are trainable. If prints
    the number of frozen and trainable parameters.

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