import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from DL_vs_HateSpeech.env_constants import PATH_TO_IMAGES, PATH_TO_JSON_US_POL, PATH_TO_JSON_COVID_19
from DL_vs_HateSpeech.utils import load_json, get_label_num

    
class DataLoader(Dataset):
    def __init__(self, type="train", subset="us_pol"):
        """
        Parameters
        type: str
            Type of dataset to load. Can be "train", "val", or "test".
        subset: str
            Subset of the dataset to load. Can be "us_pol", "covid_19" or "both".
        """
        self.subset = subset

        if subset == "us_pol":
            self.path_to_images = os.path.join(PATH_TO_IMAGES, "harmeme_images_us_pol/")
            self.json = load_json(PATH_TO_JSON_US_POL[type])
        elif subset == "covid_19":
            self.path_to_images = os.path.join(PATH_TO_IMAGES, "harmeme_images_covid_19/")
            self.json = load_json(PATH_TO_JSON_COVID_19[type])
        elif subset == "both":
            self.path_to_images = [
                os.path.join(PATH_TO_IMAGES, "harmeme_images_us_pol/"),
                os.path.join(PATH_TO_IMAGES, "harmeme_images_covid_19/")
            ]
            json_us_pol = load_json(PATH_TO_JSON_US_POL[type])
            json_covid = load_json(PATH_TO_JSON_COVID_19[type])
            # Add a source column to differentiate
            json_us_pol["source"] = "us_pol"
            json_covid["source"] = "covid_19"
            # Concatenate the two datasets
            self.json = pd.concat([json_us_pol, json_covid], ignore_index=True)

    def _ensure_rgb(self, img):
        """
        Convert grayscale/RGBA to RGB
        """
        if img.shape[0] == 1:  # Grayscale
            return img.repeat(3, 1, 1)  # Repeat the channel 3 times
        elif img.shape[0] == 4:  # RGBA
            return img[:3]  # Keep first 3 channels
        return img  # Already RGB

    def __len__(self):
        return len(self.json)

    def __getitem__(self, idx):
        """
        Returns the image and its corresponding label.
        """
        row = self.json.iloc[idx]
        image_name = row[1]
        text = row[3]
        label_raw = row[2][0]

        if len(label_raw) == 0:
            raise ValueError(f"Label is empty for image {image_name}")
        label = get_label_num(label_raw)
        label = torch.tensor(label).long()

        if self.subset == "both":
            source = row["source"]
            if source == "us_pol":
                image_path = os.path.join(self.path_to_images[0], image_name)
            else:  # "covid_19"
                image_path = os.path.join(self.path_to_images[1], image_name)
        else:
            image_path = os.path.join(self.path_to_images, image_name)

        image = Image.open(image_path).convert("RGB")

        return image, text, label
