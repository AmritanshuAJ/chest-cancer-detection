import os
import zipfile
import gdown
from cancerDetection import logger
from cancerDetection.utils.common import get_size
from cancerDetection.entity.config_entity import DataIngestionConfig
from cancerDetection.components.duplicate_image_cleaner import DuplicateImageCleaner


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

     
    def download_file(self)-> str:
        '''
        Fetch data from the url
        '''

        try: 
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            file_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix+file_id,zip_download_dir)

            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")

        except Exception as e:
            raise e
        
    
    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        try:
            unzip_path = self.config.unzip_dir
            os.makedirs(unzip_path, exist_ok=True)
            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)
            logger.info(f"Extracted zip file into {unzip_path}")

            from glob import glob
            logger.info(len(glob(os.path.join(self.config.unzip_dir, "**", "*.*"), recursive=True)))


            # Call duplicate cleaner here after extraction
            cleaner = DuplicateImageCleaner(directory=unzip_path)
            cleaner.find_and_remove_duplicates()


        except Exception as e:
            logger.error("Failed to extract and clean duplicates")
            raise e