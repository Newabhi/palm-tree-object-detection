from palmTreesCounter.configurations.configuration import ConfigurationManager
from palmTreesCounter.core.get_base_model import BaseModel
from palmTreesCounter import logger



STAGE_NAME = "Base Model"


class BaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        configure_base_model_config = config.get_base_model_config()
        configure_base_model = BaseModel(config=configure_base_model_config)
        configure_base_model.get_base_model()
        

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> Stage - {STAGE_NAME} Started <<<<<<")
        objective = BaseModelTrainingPipeline()
        objective.main()
        logger.info(f">>>>>> Stage - {STAGE_NAME} Completed <<<<<<\n\n")
    except Exception as e:
        logger.exception(e)
        raise e