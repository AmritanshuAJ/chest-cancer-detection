from cancerDetection import logger
from cancerDetection.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cancerDetection.pipeline.stage_02_data_split import DataSplitPipeline
from cancerDetection.pipeline.stage_03_prepare_base_model import PrepareBaseModelTrainingPipeline
from cancerDetection.pipeline.stage_04_model_trainer import ModelTrainingPipeline
from cancerDetection.pipeline.stage_05_model_evaluation import ModelEvaluationPipeline


STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n{'='*20}")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Split Stage"
try:
    logger.info(f"*******************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    split_obj = DataSplitPipeline()
    split_obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n{'='*20}")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Prepare base model"
try: 
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   prepare_base_model = PrepareBaseModelTrainingPipeline()
   prepare_base_model.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n{'='*20}")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Training"
try: 
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   model_trainer = ModelTrainingPipeline()
   model_trainer.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Evaluation stage"
try:
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   model_evalution = ModelEvaluationPipeline()
   model_evalution.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
        logger.exception(e)
        raise e