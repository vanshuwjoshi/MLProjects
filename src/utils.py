import os
import sys
import dill

from src.exception import CustomException
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True) # make the directory and does not do anything if the directory already exists 
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj) # save the obj
    except Exception as e:
        raise CustomException(e, sys)

# this model will fit all the models in the dictionary models to out train and test data and return the R2 scores of each model
def evaluate_models(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X_train,y_train)
            y_test_pred = model.predict(X_test)
            test_model_score = r2_score(y_test,y_test_pred)
            report[list(models.keys())[i]] = test_model_score
        return report
    except Exception as e:
        raise CustomException(e,sys)