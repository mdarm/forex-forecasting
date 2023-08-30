import os
import requests
import zipfile

def download_zip(url, save_path):
    """
    Downloads a zip file from a given URL and saves it locally.

    Parameters:
    - url (str): The URL of the zip file to download.
    - save_path (str): The local path where the zip file will be saved.

    Returns:
    - bool: True if successful, False otherwise.
    """
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return True
    else:
        return False

def unzip_and_rename(zip_path, unzip_dir, original_name, new_name):
    """
    Unzips a zip file, renames the unzipped CSV, and deletes the zip.

    Parameters:
    - zip_path (str): The local path of the zip file.
    - unzip_dir (str): The directory where the file will be unzipped.
    - original_name (str): The original name of the unzipped CSV file.
    - new_name (str): The new name for the unzipped CSV file.
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(unzip_dir)
    os.remove(zip_path)
    os.rename(os.path.join(unzip_dir, original_name),
              os.path.join(unzip_dir, new_name))
