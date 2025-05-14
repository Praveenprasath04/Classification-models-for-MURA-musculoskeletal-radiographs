import os
import urllib.request as request
import zipfile
from Mura_model import logger
from Mura_model.utils.common import get_size
import shutil
from Mura_model.entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    def get_data_file(self):
         shutil.copyfile(self.config.source_dir,self.config.local_data_file)
        
         

    def extract_zip_file(self):
            """
            zip_file_path: str
            Extracts the zip file into the data directory
            Function returns None
            """
            unzip_path = self.config.unzip_dir
            os.makedirs(unzip_path, exist_ok=True)
            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(unzip_path) 