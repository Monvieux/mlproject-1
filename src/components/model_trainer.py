import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import ( AdaBoostRegressor, GradientBoostingRegressor,RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from exception import CustomException
from logger import logging

from utils import (save_object,evaluate_models)

@dataclass
class ModelTrainerConfig:
    trainer_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array, test_array,preprocessor_path=""):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1],)
            models = {
                "Random Forest Regressor" : RandomForestRegressor(),
                "Decision Tree Regressor" : DecisionTreeRegressor(),
                "Gradient Boosting Regressor" : GradientBoostingRegressor(),
                "Linear Regression" : LinearRegression(),
                "K-Neighbors Regressor" : KNeighborsRegressor(),
                "XGB Regressor" : XGBRegressor(),
                "CatBoosting Regressor" : CatBoostRegressor(verbose=False),
                "AdaBoost Regressor" : AdaBoostRegressor()
            }
            
            params = {
                "Random Forest Regressor":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Decision Tree Regressor": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                 "Gradient Boosting Regressor":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "K-Neighbors Regressor": {},
                "XGB Regressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
            }
            
            model_report:dict=evaluate_models(X_train, y_train, X_test, y_test,models,params)

            # To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # To gest best model name from dict
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.6 :
                raise CustomException("No best model found")
            logging.info(f"Best found model for dataset analysis :'{best_model}'")

            save_object(
                file_path = self.model_trainer_config.trainer_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return best_model_name, r2_square

        except Exception as e:
            raise CustomException(e,sys)