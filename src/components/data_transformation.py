# main aim: feature engineering, data cleaning, categorical features converted to numerical features
import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer # used to create pipeline
from sklearn.impute import SimpleImputer # for missing values
from sklearn.pipeline import Pipeline # to implement the pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

# will give the inputs we require in data transformation
@dataclass
class DataTransformationConfig:
    # to save the preprocessor object in a pickle file in artifact folder
    preprocessor_obj_file_path = os.path.join("artifact", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        # will get the particular variable from the above class as data_transformation_config
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        
        # This function is responsible for data transformation
        
        try:
            numerical_columns = ["writing score", "reading score"]
            categorical_columns = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course",
            ]
            # pipeline for numerical variables
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")), # replace missing values with median
                    ("scaler", StandardScaler(with_mean=False)), # standardize the data
                ]
            )
            # pipeline for categorical variables
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")), # replace missing values with mode
                    ("one_hot_encoder", OneHotEncoder()), # one hot encoding of the variable
                    ("scaler", StandardScaler(with_mean=False)), # standardize the data
                ]
            )
            
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")
            
            # Combine the above two pipelines and name them num_pipeline and cat_pipeline and give them the required columns
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        # get train_path and test_path from data_ingestion.py
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read training and test dataset")
            logging.info("Getting the preprocessing object")
            
            # preprocessing object from previous function
            preprocessing_obj = self.get_data_transformer_object()
            
            # target column
            target_column_name = "math score"

            # input training data
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            # target training data
            target_feature_train_df = train_df[target_column_name]
            # input test data
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            # target test data
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and test data")

            # applying preprocessing to input training and test data
            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # concatinate the arrays derived above with the respective target value
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            logging.info("Saved preprocessing object.")

            # save_object function from utils help to save preprocsser into pickle file
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path, # file path where we want to save the preprocessor (mentioned in config function)
                obj=preprocessing_obj, # preprocessor object we want to save
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
