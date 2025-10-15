import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model
from dataclasses import dataclass
from catboost import CatBoostRegressor
 

@dataclass
class ModelTrainerConfig:
    trainer_model_file_path=os.path.join('artifact','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("train test split")
            x_train,y_train,x_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1])
            models={"LinearRegression": LinearRegression(), 
                    "Ridge":Ridge(), 
                    "Lasso":Lasso(),
                    "KNN":KNeighborsRegressor(),
                    "DT":DecisionTreeRegressor(),
                    "RF":RandomForestRegressor(),
                    "AdaBoost Regressor": AdaBoostRegressor(),
                    "SVR":SVR(),
                    "CatBoostRegressor":CatBoostRegressor(verbose=False)}
            
            params = {
                "LinearRegression": {},
                "Ridge": {
                    "alpha": [0.1, 1.0, 10.0],
                    "solver": ["auto", "cholesky", "lsqr"]
                        },
                "Lasso": {
                    "alpha": [0.001, 0.01, 0.1, 1.0]
                    },
                "KNN": {
                    "n_neighbors": [3, 5, 7]
                    },
                "DT": {
                    "criterion": ["squared_error", "friedman_mse"],
                    "max_depth": [5, 10, None]
                    },
                "RF": {
                    "n_estimators": [50, 100],
                    "max_depth": [5, 10, None]
                    },
                "AdaBoost Regressor": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1, 1]
                    },
                "SVR": {
                    "kernel": ["linear", "rbf"],
                    "C": [1, 10]
                    },
                "CatBoostRegressor": {
                    "depth": [6, 8],
                    "learning_rate": [0.01, 0.1],
                    "iterations": [100, 200]
                    }}


            


            model_report:dict=evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models,params=params)
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model=models[best_model_name]
            if best_model_score<0.6:
                raise CustomException("no best model")
            logging.info(f"best model on train and test")

            save_object(file_path=self.model_trainer_config.trainer_model_file_path, obj=best_model)

            predicted=best_model.predict(x_test)
            r2=r2_score(y_test,predicted)
            logging.info(f"{r2} is the best score for model {best_model_name}")
            return r2
        except Exception as e:
            raise CustomException(e,sys)

