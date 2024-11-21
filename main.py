from src.pipeline.data_ingestion_pipeline import DataIngestionPipeline
from src.pipeline.data_validation_pipeline import DataValidationPipeline
from src.pipeline.data_transformation_pipeline import DataTransformationPipeline
from src.pipeline.model_training_pipeline import ModelTrainingPipeline
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

STAGE_NAME = "Data Transformation Stage"
try:
    logger.info(f"stage {STAGE_NAME} started")
    data_transformation = DataTransformationPipeline()
    data_transformation.initialize_data_transformation()
    logger.info(f"stage {STAGE_NAME} completed")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Training Stage"
try:
    logger.info(f"stage {STAGE_NAME} started")
    model_trainer = ModelTrainingPipeline()
    model_trainer.initialize_model_training()
    logger.info(f"stage {STAGE_NAME} completed")
except Exception as e:
    logger.exception(e)
    raise e