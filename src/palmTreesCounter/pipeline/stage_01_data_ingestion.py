from palmTreesCounter.configurations.configuration import ConfigurationManager
from palmTreesCounter.core.data_ingestion import DataIngestion
from palmTreesCounter import logger



STAGE_NAME = "Data Ingestion Stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_dataset()



if __name__ == '__main__':
    try:
        logger.info(f">>>>>> Stage - {STAGE_NAME} Started <<<<<<")
        objective = DataIngestionTrainingPipeline()
        objective.main()
        logger.info(f">>>>>> Stage - {STAGE_NAME} Completed <<<<<<\n\n")
    except Exception as e:
        logger.exception(e)
        raise e
