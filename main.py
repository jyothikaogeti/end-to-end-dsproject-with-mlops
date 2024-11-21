from src.pipeline.data_ingestion_pipeline import DataIngestionPipeline
from src.pipeline.data_validation_pipeline import DataValidationPipeline
from src import logger

STAGE_NAME = "Data Ingestion stage"
try:
    logger.info(f"stage {STAGE_NAME} started")
    data_ingestion = DataIngestionPipeline()
    data_ingestion.initialize_data_ingestion()
    logger.info(f"stage {STAGE_NAME} complated")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Validation Stage"
try:
    logger.info(f"stage {STAGE_NAME} started")
    data_validation = DataValidationPipeline()
    data_validation.initialize_data_validation()
    logger.info(f"stage {STAGE_NAME} completed")
except Exception as e:
    logger.exception(e)
    raise e