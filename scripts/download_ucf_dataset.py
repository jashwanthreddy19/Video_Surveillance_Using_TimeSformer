# Please Download the UCF Crime Dataset from the following link: https://www.crcv.ucf.edu/data1/chenchen/UCF_Crimes.zip
# If the above link is not working, you can download the dataset from the following link: https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AABvnJSwZI7zXb8_myBA0CLHa?dl=0
# After downloading the dataset, extract the zip file and place the extracted folder in the data directory of the project.
# Else You can run the below script to download and extract the dataset.

import requests
import zipfile
import io
import os
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

def extract_file(zip_file, file_info, data_dir):
    """Extracts a single file from the zip archive."""
    try:
        zip_file.extract(file_info, data_dir)
    except Exception as e:
        print(f"Error extracting {file_info.filename}: {e}")

def download_and_extract_ucf_dataset(url, data_dir="data/"):
    """
    Downloads and extracts the UCF Anomaly Detection dataset from a given URL using multiprocessing.

    Args:
        url (str): The URL of the dataset zip file.
        data_dir (str): The directory where the extracted data will be stored.
    """

    try:
        os.makedirs(data_dir, exist_ok=True)

        print(f"Downloading dataset from: {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        print("Download complete. Extracting dataset...")
        zip_file = zipfile.ZipFile(io.BytesIO(response.content))
        file_list = zip_file.infolist()

        # use a ThreadPoolExecutor, as zip file extraction is mostly I/O bound.
        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            futures = [executor.submit(extract_file, zip_file, file_info, data_dir) for file_info in file_list]

            # Wait for all extraction tasks to complete
            for future in futures:
                future.result() # get result to raise any exceptions that occured during extraction.

        print(f"Dataset extracted to: {data_dir}")

    except requests.exceptions.RequestException as e:
        print(f"Error during download: {e}")
    except zipfile.BadZipFile as e:
        print(f"Error extracting zip file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    dataset_url = "https://www.crcv.ucf.edu/data1/chenchen/UCF_Crimes.zip"
    download_and_extract_ucf_dataset(dataset_url)