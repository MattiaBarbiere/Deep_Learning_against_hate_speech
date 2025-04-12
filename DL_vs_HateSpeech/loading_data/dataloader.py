import os
from torch.utils.data import Dataset
import pandas as pd
from torchvision.io import read_image
from DL_vs_HateSpeech.loading_data.preprocessing import *
from DL_vs_HateSpeech.env_constants import PATH_TO_IMAGES, PATH_TO_CSV_FILES

class DataLoader(Dataset):
    def __init__(self, type = "train"):
        """
        Parameters
        type: str
            Type of dataset to load. Can be "train", "val", or "test".

        """
        self.path_to_image = PATH_TO_IMAGES
        self.path_to_csv = PATH_TO_CSV_FILES[type]
        self.csv = pd.read_csv(self.path_to_csv)

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        """
        Returns the image and its corresponding label.
        """
        image_name = self.csv.iloc[idx, 0]
        text = self.csv.iloc[idx, 1]
        label = self.csv.iloc[idx, 2]

        # Load the image
        image_path = os.path.join(self.path_to_image, image_name)
        image = read_image(image_path)

        # Preprocess the image
        # image = preprocess_image(image)

        return image, text, label