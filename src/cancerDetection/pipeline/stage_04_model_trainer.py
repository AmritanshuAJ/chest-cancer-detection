from cancerDetection.config.configuration import ConfigurationManager
from cancerDetection.components.model_trainer import Training
from cancerDetection import logger



STAGE_NAME = "Training"


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            # Existing training code
            config = ConfigurationManager()
            training_config = config.get_training_config()
            training = Training(config=training_config)
            
            # Step 1: Regular training (what you just completed)
            training.get_base_model()
            training.train_valid_generator()
            training.train()

            # Step 2: NEW - Fine-tuning stage
            print("\n Starting fine-tuning to improve performance...")
            fine_tune_history = training.fine_tune_model(
                unfreeze_layers=training_config.params_unfreeze_layers,
                fine_tune_epochs=training_config.params_fine_tune_epochs,
                fine_tune_lr=training_config.params_fine_tune_lr
            )

            logger.info(">>>>>> Fine-tuning completed <<<<<<")

        
        except Exception as e:
            logger.exception(e)
            raise e

if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n{'='*20}")
    except Exception as e:
        logger.exception(e)
        raise e