from src.config import ModelTrainerConfig
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_models
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def start_model_trainer(self, train_data, test_data):
        try:
            X_train, y_train = train_data[:, :-1], train_data[:, -1]
            X_test, y_test = test_data[:, :-1], test_data[:, -1]

            # 1. train and evaluate multiple models
            models = {
                'LogisticRegression': LogisticRegression(),
                'RandomForest': RandomForestClassifier(),
                'XGBoost': XGBClassifier(),
            }

            model_report = evaluate_models(models, X_train, y_train, X_test, y_test)

            # 2. select the best model based on accuracy and recall
            best_model_name = None
            max_recall = 0
            for model_name, report in model_report.items():
                if report['accuracy'] > 0.5 and report['recall'] > max_recall:
                    max_recall = report['recall']
                    best_model_name = model_name
            
            if best_model_name is None:
                raise CustomException("No model found with accuracy > 0.5")

            # 3. tune the best model
            if best_model_name == 'LogisticRegression':
                model = LogisticRegression()
                param_grid = {
                    'C': [0.1, 1, 10, 100],
                    'solver': ['newton-cg', 'lbfgs', 'liblinear']
                }
            elif best_model_name == 'RandomForest':
                model = RandomForestClassifier()
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [5, 10, 20],
                    'max_samples': [0.5, 0.75, 0.9]
                }

            elif best_model_name == 'XGBoost':
                model = XGBClassifier()
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [5, 10],
                    'learning_rate': [0.01, 0.1]
                }
            
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='recall')
            grid_search.fit(X_train, y_train)

            best_tuned_model = grid_search.best_estimator_
            best_recall = grid_search.best_score_

            print(f"Best model: {best_tuned_model}")
            print(f"Best recall: {best_recall}")
            
            # 4. save the best tuned model
            save_object(self.config.trained_model_path, best_tuned_model)
            logging.info(f"Best model saved at: {self.config.trained_model_path}")

            return (best_tuned_model, best_recall)

        except Exception as e:
            raise CustomException("Error in start_model_trainer", sys)
