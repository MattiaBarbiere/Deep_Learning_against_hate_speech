# print current working directory
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Constants for the environment variables
PATH_TO_IMAGES = os.getenv("PATH_TO_IMAGES")
PATH_TO_TEXT = os.getenv("PATH_TO_TEXT")

# Constants for the CSV files
csv_files = [
    os.path.join(PATH_TO_TEXT, "Testing_meme_dataset.csv"),
    os.path.join(PATH_TO_TEXT, "Training_meme_dataset.csv"),
    os.path.join(PATH_TO_TEXT, "Validation_meme_dataset.csv"),
]