import os
import sys
from src.exception import CustomException

import numpy as np
import pandas as pd
import dill
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedir(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(file_path, obj):
    try:
        with open(file_path, 'rb') as file:
            obj = dill.load(file)
            return obj
    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(models, X_train, y_train, X_test, y_test):
    try:
        model_report = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            model_report[model_name] = {'accuracy': accuracy, 'recall': recall}
        return model_report
    except Exception as e:
        raise CustomException(e, sys)