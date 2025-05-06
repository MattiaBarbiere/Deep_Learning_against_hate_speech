import os
import matplotlib.pyplot as plt
from DL_vs_HateSpeech.env_constants import PATH_TO_IMAGES, PATH_TO_JSON_US_POL, PATH_TO_JSON_COVID_19
from DL_vs_HateSpeech.utils import find_text_and_label_jsonl

# Check if the path exists
def test_path_to_images(image_name, type="train", subset="us_pol"):
    """
    Test if the image path exists, display the image, and print its associated text and label.

    Args:
        image_name (str): Name of the image file.
        type (str): Dataset split, e.g. "train", "val", or "test".
        subset (str): Dataset subset, e.g. "us_pol", "covid_19", or "both".
    """
    if subset == "us_pol":
        image_path = os.path.join(PATH_TO_IMAGES, "harmeme_images_us_pol", image_name)
    elif subset == "covid_19":
        image_path = os.path.join(PATH_TO_IMAGES, "harmeme_images_covid_19", image_name)
    else:
        raise ValueError(f"Subset '{subset}' is not supported in this test function.")

    assert os.path.exists(image_path), f"Path {image_path} does not exist."

    # Plot the image
    image = plt.imread(image_path)
    plt.imshow(image)
    plt.axis('off')
    plt.title(image_name)
    plt.show()

    # Retrieve text and label
    file, text, label = find_text_and_label_jsonl(image_name, type=type, subset=subset)
    print(f"JSONL Source: {file}")
    print(f"Text: {text}")
    print(f"Label: {label}")

if __name__ == "__main__":
    # Manually check that paths to individual JSON files exist
    for subset_name, json_map in [("us_pol", PATH_TO_JSON_US_POL), ("covid_19", PATH_TO_JSON_COVID_19)]:
        for split, path in json_map.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"{subset_name} JSON file for '{split}' does not exist at: {path}")

    # Test image loading and metadata lookup
    test_path_to_images("covid_memes_2.png", type="train", subset="covid_19")
    print("Test passed!")
