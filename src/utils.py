import os
import sys
import dill

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True) # make the directory and does not do anything if the directory already exists 
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj) # save the obj
    except Exception as e:
        raise CustomException(e, sys)
