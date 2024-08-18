from palmTreesCounter import logger
from palmTreesCounter.definitions.config_entity import DataIngestionConfig
from palmTreesCounter.utils.common import create_directories
import kaggle


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
     
    def download_dataset(self)-> str:
        '''
        Fetch data from the Kaggle dataset and save it in the local directory
        '''

        try: 
            dataset_identifier = self.config.dataset_identifier
            download_dir = self.config.unzip_dir
            create_directories(["artifacts/data_ingestion"])
            logger.info(f"Downloading data from {dataset_identifier} into file {download_dir}")
            
            # code to download from kaggle
            kaggle.api.dataset_download_files(dataset_identifier, path=download_dir, unzip=True, quiet=False)

            logger.info(f"Downloaded data from {dataset_identifier} into file {download_dir}")

        except Exception as e:
            raise e

