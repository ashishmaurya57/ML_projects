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
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor ,GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso 
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error 
from sklearn.model_selection import RandomizedSearchCV 
# from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.utils import save_object
from src.logger import logging
import warnings
from src.utils import evaluate_model
@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts', "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
            print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

            models={
                "Linear Regression":LinearRegression(),
                "lasso":Lasso(),
                "Ridge":Ridge(),
                "Random Forest":RandomForestRegressor(),
                "Gradient boosting":GradientBoostingRegressor(),
                "KNeighborsRegressor":KNeighborsRegressor(),
                "Decision TreeRegressor":DecisionTreeRegressor(),
                "random forest regresson":RandomForestRegressor(),
                "XGBRegressor":XGBRegressor(),
                "AdaBoostRegressor":AdaBoostRegressor()
            }
            # Corrected function call for evaluate_model()
            model_report:dict = evaluate_model(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, models=models)


            best_model_Score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_Score)
            ]
            best_model=models[best_model_name]
            if best_model_Score<0.6:
                raise CustomException("No best model found")
            
            logging.info(f"Best found model on noth training and testing datset ")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted=best_model.predict(x_test)
            r2=r2_score(y_test, predicted)
            return r2
        except Exception as e:
            raise CustomException(e, sys)        