import os 
import sys
from dataclasses import dataclass
# Basic Import 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
# Modelling 
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso 
from sklearn.model_selection import RandomizedSearchCV 
# from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.utils import save_object
from src.logger import logging
from src.utils import evaluate_model
import warnings

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
            print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "KNeighbors Regressor": KNeighborsRegressor(),
                "Decision Tree Regressor": DecisionTreeRegressor(),
                # "XGBoost Regressor": XGBRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params = {
                "Linear Regression": {},
                "Random Forest": {
                    "n_estimators": [100, 200, 500],
                    "max_depth": [5, 10, 15, None],
                    "min_samples_split": [2, 5, 10],
                },
                "Gradient Boosting": {
                    "n_estimators": [100, 200, 500],
                    "learning_rate": [0.1, 0.05, 0.01],
                    "max_depth": [3, 5, 10],
                },
                "KNeighbors Regressor": {
                    "n_neighbors": [3, 5, 10, 20],
                },
                "Decision Tree Regressor": {
                    "max_depth": [5, 10, 15, None],
                    "min_samples_split": [2, 5, 10],
                },
                # "XGBoost Regressor": {
                #     "n_estimators": [100, 200, 300],
                #     "learning_rate": [0.1, 0.05, 0.01],
                #     "max_depth": [3, 5, 10],
                # },
                "AdaBoost Regressor": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.1, 0.05, 0.01],
                },
            }

            # Call evaluate_model, which should handle training and parameter tuning
            model_report: dict = evaluate_model(
                x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, models=models, params=params
            )

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No suitable model found with r2_score > 0.6")

            logging.info(f"Best model found: {best_model_name} with r2_score: {best_model_score}")

            # Refit the best model with training data
            best_model.fit(x_train, y_train)

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Predict and evaluate
            predicted = best_model.predict(x_test)
            r2 = r2_score(y_test, predicted)
            return r2

        except Exception as e:
            raise CustomException(e, sys)
