import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from cancerDetection.entity.config_entity import PrepareBaseModelConfig
                                                

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    
    def get_base_model(self):
        self.model = tf.keras.applications.DenseNet121(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        self.save_model(path=self.config.base_model_path, model=self.model)


    
    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        # Global Average Pooling
        global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()(model.output)
        
        # First dropout layer
        dropout1 = tf.keras.layers.Dropout(0.2)(global_avg_pool)
        
        # Dense layer - Updated to 1024 units based on ResNet50 findings
        dense_layer = tf.keras.layers.Dense(1024, activation=None)(dropout1)
        
        # BatchNormalization - Added based on ResNet50 findings
        batch_norm = tf.keras.layers.BatchNormalization()(dense_layer)
        
        # ReLU activation - Separated based on ResNet50 findings
        activation = tf.keras.layers.Activation('relu')(batch_norm)
        
        # Second dropout layer for regularization - Updated to 0.3 based on ResNet50
        dropout2 = tf.keras.layers.Dropout(0.3)(activation)


        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(dropout2)

        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model
    

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)
    

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)