from palmTreesCounter.configurations.configuration import ConfigurationManager
from palmTreesCounter.core.model_trainer import Training
from palmTreesCounter import logger


STAGE_NAME = "Training"


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        training.get_data_loaders()
        training.train()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> Stage - {STAGE_NAME} Started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> Stage - {STAGE_NAME} Completed <<<<<<\n\n")
    except Exception as e:
        logger.exception(e)
        raise e
        