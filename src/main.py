import os
import sys
import fetch_data
import process_data

def create_raw_dataset():
    zip_url = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip?1f54ac4889a7e6d01b17d729b1c02549"
    zip_path = "eurofxref-hist.zip"
    unzip_dir = "../dataset"
    original_name = "eurofxref-hist.csv"
    new_name = "raw_dataset.csv"
    
    # Create the directory if it does not exist
    if not os.path.exists(unzip_dir):
        os.makedirs(unzip_dir)

    if fetch_data.download_zip(zip_url, zip_path):
        fetch_data.unzip_and_rename(zip_path, unzip_dir, original_name, new_name)
        print("Downloaded, unzipped, renamed, and deleted the original forex-zip file successfully.")
    else:
        print("Failed to download the file.")
        sys.exit(1)

def process_dataset():
    unzip_dir = "../dataset"
    old_name = "raw_dataset.csv"
    new_name = "processed_dataset.csv"
    cleaned_csv_file_path = os.path.join(unzip_dir, new_name)

    # Clean the data
    process_data.clean_data(os.path.join(unzip_dir, old_name),
                            cleaned_csv_file_path)
        

if __name__ == "__main__":
    create_raw_dataset()
    process_dataset()
