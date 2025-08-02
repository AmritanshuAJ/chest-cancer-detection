import dagshub
from cancerDetection.config.configuration import ConfigurationManager
from cancerDetection.components.model_evaluation import Evaluation
from cancerDetection import logger

STAGE_NAME = "Evaluation stage"

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(eval_config)
        evaluation.evaluation()
        evaluation.save_score()
        evaluation.log_into_mlflow()


if __name__ == '__main__':
    # Initialize DagsHub tracking once at the start
    dagshub.init(repo_owner='amritanshu.10819011622', repo_name='chest-cancer-detection', mlflow=True)

    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n{'='*20}")
    except Exception as e:
        logger.exception(e)
        raise e