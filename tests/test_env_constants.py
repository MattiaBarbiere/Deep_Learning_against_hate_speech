from DL_vs_HateSpeech.env_constants import *
from DL_vs_HateSpeech.utils import find_text_and_label
import matplotlib.pyplot as plt
import pandas as pd

# Check if the path exists
def test_path_to_images(image_name):
    path = PATH_TO_IMAGES + image_name
    assert os.path.exists(path), f"Path {path} does not exist."

    # Plot the image
    image = plt.imread(path)
    plt.imshow(image)
    plt.axis('off')  # Hide the axes
    plt.show()

    # Find the text and label of the image
    file, text, label = find_text_and_label(image_name, pd.read_csv(csv_files[0]))
    print(f"Text: {text}")
    print(f"Label: {label}")



if __name__ == "__main__":
    # Check that all the paths to the CSV files exist
    for file in csv_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"CSV file {file} does not exist. Please check the path.")
    
    # Test the path to images
    test_path_to_images("0bOKK62.png")
    print("Test passed!")

