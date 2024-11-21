from src.config.configuration import ConfigurationManager
from src.components.model_training import ModelTrainer
from src import logger

STAGE_NAME = "Model Training Stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def initialize_model_training(self):
        config=ConfigurationManager()
        model_training_config = config.get_model_training_config()
        model_training = ModelTrainer(config=model_training_config)
        model_training.train_model()
            
        
if __name__ == "__main__":
    try:
        logger.info(f"stage {STAGE_NAME} started")
        obj = ModelTrainingPipeline()
        obj.initialize_model_training()
        logger.info(f"stage {STAGE_NAME} completed")
    except Exception as e:
        logger.exception(e)
        raise e
