## Module to call the Kaggle API for downloading the images data.

import os
import gdown
from dotenv import load_dotenv

load_dotenv()

url = os.getenv('KAGGLE_JSON_URL')

if not url:
    raise ValueError("KAGGLE_JSON_URL is not set in the .env file")

kaggle_dir = os.path.expanduser('~/.kaggle')
kaggle_json_path = os.path.join(kaggle_dir, 'kaggle.json')

os.makedirs(kaggle_dir, exist_ok=True)

gdown.download(url, kaggle_json_path, quiet=False)
os.chmod(kaggle_json_path, 0o600)
os.remove(kaggle_json_path)
