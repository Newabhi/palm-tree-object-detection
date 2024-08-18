from palmTreesCounter.configurations.configuration import ConfigurationManager
from palmTreesCounter.core.model_evaluation import Evaluation
from palmTreesCounter import logger



STAGE_NAME = "Evaluation"


class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(eval_config)
        evaluation.get_test_dataloader()
        evaluation.evaluation()
        evaluation.send_eval_metric_to_mlflow()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> Stage - {STAGE_NAME} Started <<<<<<")
        objective = EvaluationPipeline()
        objective.main()
        logger.info(f">>>>>> Stage - {STAGE_NAME} Completed <<<<<<\n\n")
    except Exception as e:
        logger.exception(e)
        raise e
            