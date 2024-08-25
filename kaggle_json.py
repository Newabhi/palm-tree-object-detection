import os
import gdown

# URL of your kaggle.json file on Google Drive
url = 'https://drive.google.com/file/d/1T-km5fAlGKbAc0GpxK_Spc06vzhUaR2_/view?usp=drive_link'

# Define the directory and file path
kaggle_dir = os.path.expanduser('~/.kaggle')
kaggle_json_path = os.path.join(kaggle_dir, 'kaggle.json')

# Create the .kaggle directory if it doesn't exist
os.makedirs(kaggle_dir, exist_ok=True)

# Download kaggle.json from Google Drive
gdown.download(url, kaggle_json_path, quiet=False)

# Ensure correct permissions
os.chmod(kaggle_json_path, 0o600)

# Run your data ingestion or other tasks here

# Delete kaggle.json after use
os.remove(kaggle_json_path)
