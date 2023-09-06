import os
import sys
from fetch_data import download_zip, unzip_and_rename
from process_data import clean_data, resample_data


def create_raw_dataset():
    zip_url       = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip?1f54ac4889a7e6d01b17d729b1c02549"
    zip_path      = "eurofxref-hist.zip"
    unzip_dir     = "../dataset"
    original_name = "eurofxref-hist.csv"
    new_name      = "raw_dataset.csv"
    
    if not os.path.exists(unzip_dir):
        os.makedirs(unzip_dir)

    if download_zip(zip_url, zip_path):
        unzip_and_rename(zip_path, unzip_dir, original_name, new_name)
        print("Downloaded, unzipped, renamed, and deleted the original forex-zip file successfully.")
    else:
        print("Failed to download the file.")
        sys.exit(1)


def process_dataset():
    unzip_dir             = "../dataset"
    old_name              = "raw_dataset.csv"
    new_name              = "processed_dataset.csv"
    cleaned_csv_file_path = os.path.join(unzip_dir, new_name)

    clean_data(os.path.join(unzip_dir, old_name),
                            cleaned_csv_file_path)
        

def resample_dataset():
    processed_csv_path = "../dataset/processed_dataset.csv"
    ouput_path         = "../dataset"

    resample_data(processed_csv_path, ouput_path)


if __name__ == "__main__":
    create_raw_dataset()
    process_dataset()
    resample_dataset()
