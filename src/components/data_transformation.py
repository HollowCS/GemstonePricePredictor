import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass()
class DataTransformationConfig:
    preprocessor_obj_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info("Data Transformation initiated")
            # define which column should be ordinal encoded and which one should be scaled

            categorical_cols = ['cut', 'color', 'clarity']
            numerical_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']

            # define the custom ranking for each ordinal variable

            cut_categories = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
            color_categories = ["D", "E", "F", "G", "H", "I", "J"]
            clarity_categories = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

            # numerical pipeline

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # categorical pipeline

            cat_pipeline = Pipeline(
                steps=[
                    ("Imputer", SimpleImputer(strategy="most_frequent")),
                    ("OrdinalEncoder",
                     OrdinalEncoder(categories=[cut_categories, color_categories, clarity_categories])),
                    ("Scaler", StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_cols),
                    ("cat_pipeline", cat_pipeline, categorical_cols)
                ]
            )
            return preprocessor
            logging.info("Pipeline Completed")

        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Reading train and test data
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info("Reading Train and Test data has completed")
            logging.info(f"Train dataframe Overview :\n {train_data.head().to_string()}")
            logging.info(f"Test dataframe Overview :\n {test_data.head().to_string()}")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = "price"
            drop_columns = [target_column_name, "Unnamed: 0", "index", "id"]

            input_feature_train_data = train_data.drop(columns=drop_columns, axis=1)
            target_feature_train_data = train_data[target_column_name]

            input_feature_test_data = test_data.drop(columns=drop_columns, axis=1)
            target_feature_test_data = test_data[target_column_name]

            logging.info("Applying preprocessing object on training and testing datasets.")

            # transforming using preprocessing object

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_data)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_data)

            logging.info("Applying preprocessing on training and testing datasets")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_data)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_data)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_path,
                obj=preprocessing_obj
            )
            logging.info("preprocessor file is saved")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_path
            )
        except Exception as e:
            logging.info("AN error has occurred in Initiate Data Transformation")
            raise CustomException(e, sys)