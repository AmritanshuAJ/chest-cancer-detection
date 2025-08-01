from cancerDetection.config.configuration import ConfigurationManager
from cancerDetection.components.data_splitter import DataSplitter
from cancerDetection import logger

STAGE_NAME = "Data Split Stage"

class DataSplitPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_split_config = config.get_data_split_config()
        splitter = DataSplitter(config=data_split_config)
        logger.info("Data Splitter initiated.")
        splitter.split()
        logger.info("Data split completed.")

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataSplitPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n")
    except Exception as e:
        logger.exception(e)
        raise e