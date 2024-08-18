import os
from palmTreesCounter.constants import *
from palmTreesCounter.utils.common import read_yaml, create_directories
from palmTreesCounter.definitions.config_entity import (BaseModelConfig, DataIngestionConfig, EvaluationConfig, TrainingConfig)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):
        """
        Initializes the ConfigurationManager with configuration and parameters file paths.

        Args:
            config_filepath (str, optional): Path to the configuration YAML file. 
                                              Defaults to CONFIG_FILE_PATH.
            params_filepath (str, optional): Path to the parameters YAML file. 
                                             Defaults to PARAMS_FILE_PATH.
        """
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Returns a DataIngestionConfig object containing the configuration 
        parameters for data ingestion.

        Returns:
            DataIngestionConfig: Object with data ingestion configuration parameters.
        """

        config = self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            dataset_identifier=config.dataset_identifier,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    

    def get_base_model_config(self) -> BaseModelConfig:
        config = self.config.base_model
        
        create_directories([config.root_dir])

        base_model_config = BaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.NUM_CLASSES,
        )

        return base_model_config
    

    
    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        base_model = self.config.base_model
        params = self.params
        training_data = os.path.join(self.config.data_ingestion.unzip_dir, "Palm-Counting-349images")
        create_directories([
            Path(training.root_dir)
        ])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            base_model_path=Path(base_model.base_model_path),
            training_data=Path(training_data),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_image_size=params.IMAGE_SIZE,
            params_classes=params.NUM_CLASSES,
            mlflow_uri= training.mlflow_uri,
            all_params=self.params,
            learning_rate=params.LEARNING_RATE,
        )

        return training_config
    

    def get_evaluation_config(self) -> EvaluationConfig:
        training = self.config.training
        training_data = os.path.join(self.config.data_ingestion.unzip_dir, "Palm-Counting-349images")

        eval_config = EvaluationConfig(
            path_to_model=training.trained_model_path,
            training_data=training_data,
            mlflow_uri=training.mlflow_uri,
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE,
            params_classes=self.params.NUM_CLASSES
        )
        
        return eval_config