# End-to-End Machine Learning Project: Predicting Math Scores

This repository contains a comprehensive machine-learning project aimed at predicting the math scores of students. The project covers data ingestion, data transformation, model training, and deployment of a web application using Flask for making custom predictions.

## Project Structure

The project is organized into the following components and pipelines:

### Components

#### Data Ingestion (src/components/data_ingestion.py)

In this component, we read data from our local storage. We split the data into training and test sets and save these datasets in the "artifact" folder.

#### Data Transformation (src/components/data_transformation.py)

The data transformation component focuses on feature engineering, data cleaning, and converting categorical features into numerical ones. We create a data preprocessing pipeline and return the transformed training and test data arrays, with the target variable as the last column. The preprocessing pipeline is saved as a pickle file in the "artifact" folder.

#### Model Trainer (src/components/model_trainer.py)

Here, we fit various machine learning models to the transformed data from the data transformation component. We split the data into features (X) and the target (y), apply multiple models in a pipeline, and select the best model based on R-squared. Hyperparameter tuning is also performed for each model, and the selected model is saved as a pickle file in the "artifact" folder.

### Pipeline

#### Prediction Pipeline (src/pipeline/predict_pipeline.py)

The prediction pipeline is responsible for mapping all the inputs from the HTML to the backend application loading the pickle file for the preprocessor and model and applying it to the mapped data. This pipeline returns the predicted math value. 

### Web Application (app.py)

To create a user-friendly interface for making custom predictions, we've developed a web application using Flask. Users can input their data, and the application will predict their math scores based on the trained machine learning model.
