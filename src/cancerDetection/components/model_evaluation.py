import os
import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from cancerDetection.entity.config_entity import EvaluationConfig
from cancerDetection.utils.common import read_yaml,create_directories,save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    
    def _test_generator(self):
        """
        Data generator for the test set.
        - No data augmentation is applied.
        - Shuffling is turned off for consistent metric calculation.
        """

        datagenerator_kwargs = dict(
            rescale=1./255
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
            class_mode='categorical'  # Important for model.evaluate
        )

        # Use consistent path handling
        test_dir = os.path.join(self.config.training_data, "test")

        # Add directory existence check
        if not os.path.exists(test_dir):
            raise FileNotFoundError(f"[ERROR] Test directory not found: {test_dir}")
        
        print(f"[INFO] Found test directory at: {test_dir}")

        # Create the test ImageDataGenerator (no augmentation)
        test_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        # Create the test generator
        self.test_generator = test_datagenerator.flow_from_directory(
            directory=test_dir,
            shuffle=False,
            **dataflow_kwargs
        )

        print(f"[INFO] Found {self.test_generator.samples} test samples")
        print(f"[INFO] Class indices: {self.test_generator.class_indices}")


    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._test_generator()
        self.score = self.model.evaluate(self.test_generator)
        self.save_score()


    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)


    def log_into_mlflow(self):

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )
            
            if tracking_url_type_store != "file":
                mlflow.keras.log_model(self.model, "model", registered_model_name="DenseNet121Model")
            else:
                mlflow.keras.log_model(self.model, "model")