import logging
import mlflow
import pandas as pd
from zenml import step

from src.model_dev import LinearRegressionModel
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
        if(config.model_name == "LinearRegression"):
            mlflow.sklearn.autolog()  # automatic logging all the things
            model = LinearRegressionModel()
            trained_model = model.train(X_train,y_train)
            return trained_model
        else:
            ValueError("Model {} not supported".format(config.model_name))
            
    except Exception as e:
        logging.error(f"Error in training model")
        logging.error(e)
        raise e        
        
    