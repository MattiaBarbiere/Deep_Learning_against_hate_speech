from DL_vs_HateSpeech.env_constants import csv_files
import pandas as pd

# A function that given an image name, find the text and label of that image
def find_text_and_label(image_name, df):
    """
    Given an image name, find the text and label of that image in the dataframe.

    Args:
        image_name (str): The name of the image.
        df (pd.DataFrame): The dataframe containing the text and label.

    Returns:
        tuple: A tuple containing the file, text and label of the image.
    """
    # Iterate over the files
    for file in csv_files:
        df = pd.read_csv(file)
        # Check if the first column matches the image name
        match = df[df.iloc[:, 0] == image_name]
        if not match.empty:
            # Get the values from second and third columns
            text = match.iloc[0, 1]
            label = match.iloc[0, 2]
            return file, text, label
    
    # If no match is found, return error
    raise ValueError(f"Image name {image_name} not found in any CSV file.")