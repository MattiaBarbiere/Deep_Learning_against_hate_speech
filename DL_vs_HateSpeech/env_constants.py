from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Constants for the environment variables
PATH_TO_IMAGES = os.getenv("PATH_TO_IMAGES")
PATH_TO_TEXT = os.getenv("PATH_TO_TEXT")

# Constants for the CSV files
PATH_TO_CSV_FILES = {
    "test": os.path.join(PATH_TO_TEXT, "Testing_meme_dataset.csv"),
    "train": os.path.join(PATH_TO_TEXT, "Training_meme_dataset.csv"),
    "val": os.path.join(PATH_TO_TEXT, "Validation_meme_dataset.csv"),
}