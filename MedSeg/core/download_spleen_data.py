from monai.apps import download_and_extract
import os
import config


def download_spleen_data():
    # Setup
    _root_dir = config.DATA_DIR
    _compressed_file = os.path.join(_root_dir, "Task09_Spleen.tar")
    _data_dir = os.path.join(_root_dir, "Task09_Spleen")

    # Download
    if not os.path.exists(_data_dir):
        download_and_extract(
            config.SPLEEN_DATA_URL, _compressed_file, _root_dir, config.SPLEEN_DATA_MD5
        )
        print("Downloaded Data")
    else:
        print("Spleen Data Loaded...")
