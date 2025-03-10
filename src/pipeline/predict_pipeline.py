import sys
import os
import pandas as pd
from exception import CustomException
from utils import load_object
from config import PredictPipelineConfig


class PredictPipeline:
    def __init__(self):
        self.config = PredictPipelineConfig()

    def predict(self, input_data: pd.DataFrame):
        try:
            preprocessor = load_object(self.config.preprocessor_file_path)
            model = load_object(self.config.model_file_path)
            print(f"Preprocessor: {preprocessor}")
            print(f"Model: {model}")

            input_data_trf = preprocessor.transform(input_data)
            prediction = model.predict(input_data_trf)
            return prediction
        except Exception as e:
            raise CustomException("Prediction Failed", sys)