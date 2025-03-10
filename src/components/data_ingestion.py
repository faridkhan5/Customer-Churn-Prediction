from config import DataIngestionConfig
from logger import logging
from exception import CustomException
import sys

from sklearn.model_selection import train_test_split


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def start_data_ingestion(self):
        try:
            raw_data = self.config.raw_data_pathg
            train_set, test_set =  train_test_split(raw_data, test_size=0.2, random_state=42)
            train_set.to_csv(self.config.train_data_path, index=False, header=True)
            test_set.to_csv(self.config.test_data_path, index=False, header=True)
            return (self.config.train_data_path, self.config.test_data_path)
        except Exception as e:
            raise CustomException("Data ingestion failed", sys)