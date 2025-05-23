import os
import urllib.request
import zipfile

DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip"
DATA_DIR = "./data"
DEST_DIR = f"{DATA_DIR}/mini_speech_commands"
ZIP_PATH = f"{DATA_DIR}/mini_speech_commands.zip"

def downloadMiniSpeech():
    # make the data directory if it doesn't exist
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created directory: {DATA_DIR}")

    if not os.path.exists(ZIP_PATH):
        print("Downloading Mini Speech Commands dataset...")
        urllib.request.urlretrieve(DATASET_URL, ZIP_PATH)
        print("Download complete.")
    else:
        print("Zip file already exists, skipping download.")

    if not os.path.exists(DEST_DIR):
        print("Extracting dataset...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(DEST_DIR)
        print("Extraction complete.")
    else:
        print("Dataset already extracted.")

if __name__ == "__main__":
    downloadMiniSpeech()