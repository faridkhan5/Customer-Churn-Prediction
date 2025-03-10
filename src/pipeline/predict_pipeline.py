import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.config import PredictPipelineConfig


class PredictPipeline:
    def __init__(self):
        self.config = PredictPipelineConfig()

    def predict(self, input_data: pd.DataFrame):
        try:
            preprocessor = load_object(self.config.preprocessor_path)
            model = load_object(self.config.model_path)
            print(f"Preprocessor: {preprocessor}")
            print(f"Model: {model}")
            encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]

            input_data_trf = preprocessor.transform(input_data)
            prediction = model.predict(input_data_trf)
            return prediction
        except Exception as e:
            raise CustomException("Prediction Failed", sys)
        

class CustomData:
    def __init__(self,
                gender: str,
                SeniorCitizen: str,
                Partner: str,
                Dependents: str,
                PhoneService: str,
                MultipleLines: str,
                InternetService: str,
                OnlineSecurity: str,
                OnlineBackup: str,
                DeviceProtection: str,
                TechSupport: str,
                StreamingTV: str,
                StreamingMovies: str,
                Contract: str,
                PaperlessBilling: str,
                PaymentMethod: str,
                tenure: int,
                MonthlyCharges: float,
                TotalCharges: float):
        self.gender = gender
        self.SeniorCitizen = SeniorCitizen
        self.Partner = Partner
        self.Dependents = Dependents
        self.PhoneService = PhoneService
        self.MultipleLines = MultipleLines
        self.InternetService = InternetService
        self.OnlineSecurity = OnlineSecurity
        self.OnlineBackup = OnlineBackup
        self.DeviceProtection = DeviceProtection
        self.TechSupport = TechSupport
        self.StreamingTV = StreamingTV
        self.StreamingMovies = StreamingMovies
        self.Contract = Contract
        self.PaperlessBilling = PaperlessBilling
        self.PaymentMethod = PaymentMethod
        self.tenure = tenure
        self.MonthlyCharges = MonthlyCharges
        self.TotalCharges = TotalCharges


    def get_data_as_data_frame(self):
        '''returns the given html input as a dataframe'''
        try:
            custom_data_input_dict = {
                'gender': [self.gender],
                'SeniorCitizen': [self.SeniorCitizen],
                'Partner': [self.Partner],
                'Dependents': [self.Dependents],
                'PhoneService': [self.PhoneService],
                'MultipleLines': [self.MultipleLines],
                'InternetService': [self.InternetService],
                'OnlineSecurity': [self.OnlineSecurity],
                'OnlineBackup': [self.OnlineBackup],
                'DeviceProtection': [self.DeviceProtection],
                'TechSupport': [self.TechSupport],
                'StreamingTV': [self.StreamingTV],
                'StreamingMovies': [self.StreamingMovies],
                'Contract': [self.Contract],
                'PaperlessBilling': [self.PaperlessBilling],
                'PaymentMethod': [self.PaymentMethod],
                'tenure': [self.tenure],
                'MonthlyCharges': [self.MonthlyCharges],
                'TotalCharges': [self.TotalCharges]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException('cannot covert html input to dataframe', sys)