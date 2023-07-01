import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model
from dataclasses import dataclass
import sys
import os


@dataclass()
class ModelTrainerConfig:
    training_model_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Splitting dependent and independent variables from train and test data")
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            models = {
                "LinearRegression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "ElasticNet": ElasticNet()
            }

            model_report: dict = evaluate_model(x_train, y_train, x_test, y_test, models)
            print(model_report)
            print("\n---------------------------------------------------------------------")
            logging.info(f"Model Report: {model_report}")

            # to get best model score from dictionary
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            print(f"best model found, model name: {best_model_name} R2 score: {best_model_score}")
            logging.info(f"best model found, model name: {best_model_name} R2 score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.training_model_path,
                obj=best_model
            )
        except Exception as e:
            logging.info("An error has occurred at model training")
            raise CustomException(e, sys)