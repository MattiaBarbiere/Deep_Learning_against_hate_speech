from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Constants for the environment variables
PATH_TO_IMAGES = os.getenv("PATH_TO_IMAGES")
PATH_TO_METADATA = os.getenv("PATH_TO_METADATA")

# Constants for the JSON files
PATH_TO_JSON_FILES = {
    "test": os.path.join(PATH_TO_METADATA, "test.jsonl"),
    "train": os.path.join(PATH_TO_METADATA, "train.jsonl"),
    "val": os.path.join(PATH_TO_METADATA, "val.jsonl"),
}