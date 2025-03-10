import sys
from components import DataIngestion, DataTransformation, ModelTrainer
from logger import logging
from exception import CustomException


class TrainingPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def start_training_pipeline(self):
        try:
            logging.info("Training Pipeline Started")
            logging.info("Data Ingestion Started")
            train_data_path, test_data_path = self.data_ingestion.start_data_ingestion()
            logging.info("Data Ingestion Completed")
            
            logging.info("Data Transformation Started")
            train_data_trf, test_data_trf = self.data_transformation.start_data_transformation(train_data_path, test_data_path)
            logging.info("Data Transformation Completed")

            logging.info("Model Training Started")
            best_model_name, best_recall = self.model_trainer.start_model_trainer(train_data_trf, test_data_trf)
            logging.info(f"Model Training Completed | Best Model: {best_model_name} | Recall: {best_recall}")

            logging.info("Training Pipeline Completed")
        except Exception as e:
            raise CustomException("Training Pipeline Failed", sys)


if __name__ == "__main__":
    try:
        training_pipeline = TrainingPipeline()
        training_pipeline.start_training_pipeline()
    except Exception as e:
        raise CustomException("Training Pipeline Failed", sys)