from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Constants for the environment variables
PATH_TO_IMAGES = os.getenv("PATH_TO_IMAGES")
PATH_TO_METADATA = os.getenv("PATH_TO_METADATA")

# Constants for the JSON files
PATH_TO_JSON_US_POL = {
    "test": os.path.join(PATH_TO_METADATA, "us_pol/", "test.jsonl"),
    "train": os.path.join(PATH_TO_METADATA, "us_pol/", "train.jsonl"),
    "val": os.path.join(PATH_TO_METADATA, "us_pol/", "val.jsonl"),
}
PATH_TO_JSON_COVID_19 = {
    "test": os.path.join(PATH_TO_METADATA, "covid_19/", "test.jsonl"),
    "train": os.path.join(PATH_TO_METADATA, "covid_19/", "train.jsonl"),
    "val": os.path.join(PATH_TO_METADATA, "covid_19/", "val.jsonl"),
}