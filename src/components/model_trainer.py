# main aim of this file is to fit the appropriate model to our transformed data
import os
import sys
from dataclasses import dataclass

# from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models


# this function will have the path to which we can save our model as pickle file
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifact", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            # spilt the train and test arrays into X and y's where y will be the last column
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )
            # write all models in dictionary
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Nearest Neighbors": KNeighborsRegressor(),
                "XG-Boost Classsifier": XGBRegressor(),
                # "CatBoosting Classifier": CatBoostRegressor(),
                "AdaBoost Classifier": AdaBoostRegressor(),
            }

            #Hyperparameter tuning
            params = {
                "Decision Tree": {
                    "criterion": [
                        "squared_error",
                        "friedman_mse",
                        "absolute_error",
                        "poisson",
                    ],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest": {
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features':['sqrt','log2',None],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "Linear Regression": {},
                "K-Nearest Neighbors": {},
                "XG-Boost Classsifier": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                # "CatBoosting Classsifier": {
                #     "depth": [6, 8, 10],
                #     "learning_rate": [0.01, 0.05, 0.1],
                #     "iterations": [30, 50, 100],
                # },
                "AdaBoost Classifier": {
                    "learning_rate": [0.1, 0.01, 0.5, 0.001],
                    # 'loss':['linear','square','exponential'],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
            }

            # call the function evaluate_models in utils.py which takes input of
            # X_train, y_train, X_test, y_test, models and params
            # and return dictionary of all the R2 score for each model
            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params,
            )
            # maximum R2 score
            best_model_score = max(sorted(model_report.values()))
            # name of the model with the maximum R2 score
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            # model with the maximum R2 score
            best_model = models[best_model_name]
            # Raise an exception if maximum R2 score is less than 0.6 (all models performed poorly)
            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info("Found best model on both training and test dataset")
            # save the model as a pickle file
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )
            # return the R2 score of the best model
            return {best_model_name: best_model_score}
        except Exception as e:
            raise CustomException(e, sys)
