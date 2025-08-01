import os
import hashlib
from collections import defaultdict
from cancerDetection import logger
from glob import glob

class DuplicateImageCleaner:
    def __init__(self, directory: str):

        self.directory = directory

    def _file_hash(self, filepath):
        hasher = hashlib.md5()
        with open(filepath, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()

    def find_and_remove_duplicates(self):
        print(f"Scanning for duplicate images in: {self.directory}")
        hashes = defaultdict(list)
        initial_file_count = len(glob(os.path.join(self.directory, "**", "*.*"), recursive=True))
        logger.info(f"Initial number of files: {initial_file_count}")

        for root, _, files in os.walk(self.directory):
            for fname in files:
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    path = os.path.join(root, fname)
                    filehash = self._file_hash(path)
                    hashes[filehash].append(path)

        duplicates = [paths for paths in hashes.values() if len(paths) > 1]
        print(f"Found {sum(len(p) - 1 for p in duplicates)} duplicate images")

        for dup_list in duplicates:
            for dup_path in dup_list[1:]:  # Keep only the first occurrence
                os.remove(dup_path)
                print(f"Removed duplicate: {dup_path}")

        print("Duplicate cleaning complete.")
        final_file_count = len(glob(os.path.join(self.directory, "**", "*.*"), recursive=True))
        logger.info(f"Number of files after duplicate removal: {final_file_count}")
