from config import DataTransformationConfig
from logger import logging
from exception import CustomException
from utils import save_object
import sys

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.under_sampling import RandomUnderSampler

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_preprocessor(self):
        try:
            cat_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
            num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

            num_pipeline = Pipeline([
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline([
                ('encoder', OneHotEncoder())
            ])

            preprocessor = ColumnTransformer([
                ('num', num_pipeline, num_cols),
                ('cat', cat_pipeline, cat_cols)
            ])
            return preprocessor
        
        except Exception as e:
            raise CustomException("Error in get_preprocessor", sys)
        

    def start_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            preprocessor = self.get_preprocessor()

            target_feature = 'Churn'
            
            X_train, y_train = train_df.drop(target_feature, axis=1), train_df[target_feature]
            X_test, y_test = test_df.drop(target_feature, axis=1), test_df[target_feature]

            X_train_trf = preprocessor.fit_transform(X_train)
            X_test_trf = preprocessor.transform(X_test)

            X_train_down, y_train_down = RandomUnderSampler().fit_resample(X_train_trf, y_train)

            train_data_trf = np.c_(X_train_down, y_train_down)
            test_data_trf = np.c_(X_test_trf, y_test)

            save_object(self.config.preprocessor_file_path, preprocessor)
            logging.info("Preprocessor saved at: {}".format(self.config.preprocessor_file_path))

            return (train_data_trf, test_data_trf)
            
        except Exception as e:
            raise CustomException("Error in start_data_transformation", sys)