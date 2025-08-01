import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
import numpy as np
from collections import Counter
from cancerDetection.entity.config_entity import TrainingConfig
from pathlib import Path



class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def train_valid_generator(self):
        # Different augmentation for train vs validation
        train_datagenerator_kwargs = dict(
            rescale=1./255,
            rotation_range=15,           # Medical image appropriate rotation
            width_shift_range=0.1,       # Reduced shifts for medical images
            height_shift_range=0.1,
            zoom_range=0.1,              # Gentle zoom
            horizontal_flip=True,
            fill_mode='nearest',
            brightness_range=[0.9, 1.1]  # Slight brightness variation
        )

        # Validation data only rescaled (no augmentation)
        valid_datagenerator_kwargs = dict(
            rescale=1./255
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
            class_mode='categorical'  # Explicitly set class_mode
        )

        train_dir = os.path.join(self.config.training_data, "train")
        valid_dir = os.path.join(self.config.training_data, "val")

        # Add directory existence checks
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"Training directory not found: {train_dir}")
        if not os.path.exists(valid_dir):
            raise FileNotFoundError(f"Validation directory not found: {valid_dir}")

        train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **train_datagenerator_kwargs
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **valid_datagenerator_kwargs
        )

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=train_dir,
            shuffle=True,
            **dataflow_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=valid_dir,
            shuffle=False,
            **dataflow_kwargs
        )

        # Add detailed logging and class weight calculation
        print("Class indices:", self.train_generator.class_indices)
        print(f"Found {self.train_generator.samples} training samples")
        print(f"Found {self.valid_generator.samples} validation samples")
        
        # Calculate class weights for imbalanced datasets
        class_counts = list(Counter(self.train_generator.classes).values())
        total_samples = sum(class_counts)
        self.class_weights = {i: total_samples / (len(class_counts) * count) 
                             for i, count in enumerate(class_counts)}
        print(f"Class distribution: {dict(zip(self.train_generator.class_indices.keys(), class_counts))}")
        print(f"Class weights: {self.class_weights}")

    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def train(self):
        # More robust steps calculation
        self.steps_per_epoch = max(1, self.train_generator.samples // self.train_generator.batch_size)
        self.validation_steps = max(1, self.valid_generator.samples // self.valid_generator.batch_size)
        
        print(f"Steps per epoch: {self.steps_per_epoch}")
        print(f"Validation steps: {self.validation_steps}")

        # Add comprehensive callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=8,                    # Increased patience
                restore_best_weights=True,
                verbose=1,
                min_delta=0.001
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,                   # Reduce LR by half
                patience=4,                   # Wait 4 epochs before reducing
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(self.config.trained_model_path).replace('.h5', '_best.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
        ]

        # Add class_weight parameter and return history
        history = self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            callbacks=callbacks,              # Add callbacks
            class_weight=self.class_weights,  # Handle class imbalance
            verbose=1
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
        
        # Print training summary
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        print(f"\nFinal training accuracy: {final_train_acc:.4f}")
        print(f"Final validation accuracy: {final_val_acc:.4f}")
        
        return history  # Return history for analysis

    # Add fine-tuning method
    def fine_tune_model(self, unfreeze_layers=None, fine_tune_epochs=None, fine_tune_lr=None):
        """
        Fine-tune the pre-trained model by unfreezing top layers
        
        Args:
            unfreeze_layers: Number of top layers to unfreeze (default: from config)
            fine_tune_epochs: Number of epochs for fine-tuning (default: from config)
            fine_tune_lr: Learning rate for fine-tuning (default: from config)
        """

        if unfreeze_layers is None:
            unfreeze_layers = self.config.params_unfreeze_layers
    
        if fine_tune_epochs is None:
            fine_tune_epochs = self.config.params_fine_tune_epochs
            
        if fine_tune_lr is None:
            fine_tune_lr = self.config.params_fine_tune_lr

        print("\n" + "="*50)
        print(" STARTING FINE-TUNING STAGE")
        print("="*50)
        
        # Load the best model from previous training
        best_model_path = str(self.config.trained_model_path).replace('.h5', '_best.h5')
        if os.path.exists(best_model_path):
            self.model = tf.keras.models.load_model(best_model_path)
            print(f" Loaded best model from: {best_model_path}")
        else:
            print("  Best model not found, using current model for fine-tuning")
        
        print(f" Current model performance check...")
        print(f"   Model input shape: {self.model.input_shape}")
        print(f"   Model output shape: {self.model.output_shape}")
        
        # Find the base DenseNet121 model (should be the first layer)
        base_model = None
        for layer in self.model.layers:
            if hasattr(layer, 'layers') and len(layer.layers) > 100:  # DenseNet121 has many layers
                base_model = layer
                break
        
        if base_model is None:
            # If not found, assume the whole model needs unfreezing
            base_model = self.model
            print("  Base model structure different than expected, unfreezing entire model")
        
        print(f" Base model found with {len(base_model.layers)} layers")
        
        # First, freeze all layers
        for layer in base_model.layers:
            layer.trainable = False
        
        # Then unfreeze top layers
        layers_to_unfreeze = min(unfreeze_layers, len(base_model.layers))
        for layer in base_model.layers[-layers_to_unfreeze:]:
            layer.trainable = True
        
        # Count trainable parameters
        trainable_params = np.sum([np.prod(v.get_shape().as_list()) for v in self.model.trainable_weights])
        total_params = self.model.count_params()
        
        print(f"   Unfrozen top {layers_to_unfreeze} layers")
        print(f"   Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.1f}%)")
        
        # Recompile with lower learning rate for fine-tuning
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )
        
        print(f"   Model recompiled with learning rate: {fine_tune_lr}")
        
        # Fine-tuning callbacks with different monitoring strategy
        fine_tune_callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',  # Monitor accuracy for fine-tuning
                patience=6,
                restore_best_weights=True,
                verbose=1,
                mode='max',
                min_delta=0.005  # Expect at least 0.5% improvement
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,  # More aggressive reduction
                patience=3,
                min_lr=1e-8,
                verbose=1,
                cooldown=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(self.config.trained_model_path).replace('.h5', '_fine_tuned_best.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1,
                mode='max'
            )
        ]
        
        print(f"   Starting fine-tuning for {fine_tune_epochs} epochs...")
        print("-" * 50)
        
        # Fine-tune the model
        history = self.model.fit(
            self.train_generator,
            epochs=fine_tune_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            callbacks=fine_tune_callbacks,
            class_weight=self.class_weights,
            verbose=1
        )
        
        # Save the final fine-tuned model
        fine_tuned_path = str(self.config.trained_model_path).replace('.h5', '_fine_tuned.h5')
        self.save_model(path=Path(fine_tuned_path), model=self.model)
        
        # Print fine-tuning summary
        if len(history.history['accuracy']) > 0:
            final_train_acc = history.history['accuracy'][-1]
            final_val_acc = history.history['val_accuracy'][-1]
            best_val_acc = max(history.history['val_accuracy'])
            
            print("\n" + "="*50)
            print("    FINE-TUNING COMPLETED!")
            print("="*50)
            print(f"   Final Results:")
            print(f"   Training accuracy: {final_train_acc:.4f} ({(final_train_acc*100):.2f}%)")
            print(f"   Validation accuracy: {final_val_acc:.4f} ({(final_val_acc*100):.2f}%)")
            print(f"   Best validation accuracy: {best_val_acc:.4f} ({(best_val_acc*100):.2f}%)")
            print(f"   Models saved:")
            print(f"   Final: {fine_tuned_path}")
            print(f"   Best: {fine_tuned_path.replace('.h5', '_best.h5')}")
        
        return history