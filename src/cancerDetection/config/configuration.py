import os
from cancerDetection.constants import *
from cancerDetection.utils.common import read_yaml, create_directories
from cancerDetection.entity.config_entity import (DataIngestionConfig,
                                                DataSplitConfig,
                                                PrepareBaseModelConfig,
                                                TrainingConfig,
                                                EvaluationConfig)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config


    def get_data_split_config(self) -> DataSplitConfig:
        config = self.config.data_split
        create_directories([config.split_data_dir])
        
        data_split_config = DataSplitConfig(
            raw_data_dir=Path(config.raw_data_dir),
            split_data_dir=Path(config.split_data_dir),
            train_split=config.train_split,
            val_split=config.val_split,
            test_split=config.test_split
        )
        
        return data_split_config
    

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES
        )

        return prepare_base_model_config
    


    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        training_data = self.config.data_split.split_data_dir
        
        create_directories([
            Path(training.root_dir)
        ])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE,
            params_fine_tune_epochs=params.FINE_TUNE_EPOCHS,
            params_fine_tune_lr=params.FINE_TUNE_LR,
            params_unfreeze_layers=params.UNFREEZE_LAYERS
        )

        return training_config

    def get_evaluation_config(self) -> EvaluationConfig: 
        evaluation = self.config.evaluation

        eval_config = EvaluationConfig(
            path_of_model=Path(evaluation.model_path),
            training_data=Path(evaluation.data_dir),
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE
        )

        return eval_config