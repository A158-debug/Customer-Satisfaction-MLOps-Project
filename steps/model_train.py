import logging
import mlflow
import pandas as pd
from zenml import step

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.model_dev import ( LinearRegressionModel)

from sklearn.base import RegressorMixin
from .config import ModelNameConfig

@step
def train_model(
    X_train:pd.DataFrame,
    X_test:pd.DataFrame,
    y_train:pd.Series,
    y_test:pd.Series,
    config: ModelNameConfig
    ) -> RegressorMixin:
    
    """
       Train the model on the ingested data.
       Args :
            X_train : Training data
            X_test : Testing data
            y_train : Training labels
            y_test : Testing labels
    """
    try:
        model = None
        tuner = None

        if config.model_name == "LinearRegression":
            mlflow.lightgbm.autolog()
            model = LinearRegressionModel()
        # elif config.model_name == "randomforest":
        #     mlflow.sklearn.autolog()
        #     model = RandomForestModel()
        # elif config.model_name == "xgboost":
        #     mlflow.xgboost.autolog()
        #     model = XGBoostModel()
        # elif config.model_name == "lightgbm":
        #     mlflow.sklearn.autolog()
        #     model = LightGBMModel()
        else:
            raise ValueError("Model name not supported")
        
        # tuner = HyperparameterTuner(model, X_train, y_train, X_test, y_test)
        
        if config.fine_tuning:
            best_params = tuner.optimize()
            trained_model = model.train(X_train, y_train, **best_params)
        else:
            trained_model = model.train(X_train, y_train)
        return trained_model
            
    except Exception as e:
        logging.error(f"Error in training model")
        logging.error(e)
        raise e        
        
    