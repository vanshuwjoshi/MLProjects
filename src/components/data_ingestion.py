# main aim of this file is to read the data from various data sources like local storage, mongodb, hadoop and many more
import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass  # used to create class variables


# for data ingestion we need input like the location for saving the training, test and complete data
# this class will help with that
@dataclass  # directly able to define class variable without __init__ (this class does not have any other functions)
class DataIngestionConfig:
    # paths for saving the training, test and raw data after data ingestion
    # we will create a atrifact folder which will contain our data
    train_data_path: str = os.path.join("artifact", "train.csv")
    test_data_path: str = os.path.join("artifact", "test.csv")
    raw_data_path: str = os.path.join("artifact", "data.csv")


# we do not define @dataclass as we have other funcitons inside this class
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        # the above three paths in DataIngestionConfig gets saved in this variable

    # this function will read the data from the data source
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        try:
            df = pd.read_csv("notebook\data\StudentsPerformance.csv")
            # we only change the above line if we have any other data source

            logging.info("Read the dataset as dataframe")

            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True
            )
            # this should create the directory only if it does not already exists

            # To save the raw data as csv
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Train test split initiated")

            # Create the train test split
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=43)
            # To save the train data as csv
            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )
            # To save the test data as csv
            test_set.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )

            logging.info("Data Ingestion is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
