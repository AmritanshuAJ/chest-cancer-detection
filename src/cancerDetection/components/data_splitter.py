import os
import shutil
import random
from glob import glob
from cancerDetection import logger
from cancerDetection.entity.config_entity import DataSplitConfig

class DataSplitter:
    def __init__(self, config: DataSplitConfig):
        self.config = config
        random.seed(42) # for reproducibility

    def split(self):
        source_dir = os.path.join(self.config.raw_data_dir) 
        split_dir = self.config.split_data_dir

        for class_name in sorted(os.listdir(source_dir)): # Sort to ensure consistent order
            class_dir = os.path.join(source_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            images = os.listdir(class_dir)
            random.shuffle(images)

            n_total = len(images)
            n_train = int(n_total * self.config.train_split)
            n_val = int(n_total * self.config.val_split)
            n_test = n_total - n_train - n_val

            splits = {
                "train": images[:n_train],
                "val": images[n_train:n_train + n_val],
                "test": images[n_train + n_val:]
            }

            for split_name, split_images in splits.items():
                split_class_dir = os.path.join(split_dir, split_name, class_name)
                os.makedirs(split_class_dir, exist_ok=True)
                for img_name in split_images:
                    src = os.path.join(class_dir, img_name)
                    dst = os.path.join(split_class_dir, img_name)
                    shutil.copy(src, dst)

        logger.info(len(glob(os.path.join(split_dir, "train", "**", "*.*"), recursive=True)))
        logger.info(len(glob(os.path.join(split_dir, "val", "**", "*.*"), recursive=True)))
        logger.info(len(glob(os.path.join(split_dir, "test", "**", "*.*"), recursive=True)))
