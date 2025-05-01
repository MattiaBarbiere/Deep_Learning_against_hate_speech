import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from DL_vs_HateSpeech.env_constants import PATH_TO_IMAGES, PATH_TO_JSON_FILES
from DL_vs_HateSpeech.utils import load_json, get_label_num

    
class DataLoader(Dataset):
    def __init__(self, type = "train"):
        """
        Parameters
        type: str
            Type of dataset to load. Can be "train", "val", or "test".

        """
        self.path_to_image = PATH_TO_IMAGES
        self.path_to_json = PATH_TO_JSON_FILES[type]
        self.json = load_json(self.path_to_json)


    def _ensure_rgb(self, img):
        """
        Convert grayscale/RGBA to RGB
        """
        if img.shape[0] == 1: # Grayscale
            return img.repeat(3, 1, 1) # Repeat the channel 3 times
        elif img.shape[0] == 4: # RGBA
            return img[:3] # Keep first 3 channels
        return img # Already RGB

    def __len__(self):
        return len(self.json)

    def __getitem__(self, idx):
        """
        Returns the image and its corresponding label.
        """
        image_name = self.json.iloc[idx, 1]
        text = self.json.iloc[idx, 3]
        label_raw = self.json.iloc[idx, 2][0]

        # Convert labels to numerical values
        if len(label_raw) == 0:
            raise ValueError(f"Label is empty for image {image_name}")
        label = get_label_num(label_raw)
        label = torch.tensor(label, dtype=torch.float32)

        # Load the image and convert to RGB
        image_path = os.path.join(self.path_to_image, image_name)
        image = Image.open(image_path).convert("RGB")

        return image, text, label
