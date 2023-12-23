import logging
import pandas as pd
from zenml import step
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.data_cleaning import DataCleaning, DataDrivenStrategy, DataPreProcessingStrategy
from typing_extensions import Annotated
from typing import Tuple

@step
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame,"X_train"],
    Annotated[pd.DataFrame,"X_test"],
    Annotated[pd.Series,"y_train"],
    Annotated[pd.Series,"y_test"]
]:
    """
    clean the data and divide it into train and test
    Args --> pandas dataframe
    Return : 
        X_train : Training data
        X_test : Testing data
        y_train : Training labels
        y_test : Testing labels
    
    """
    
    try:
        process_strategy = DataPreProcessingStrategy() 
        data_cleaning = DataCleaning(df, process_strategy)
        processed_data = data_cleaning.handle_data()
        
        divide_strategy = DataDrivenStrategy()
        data_cleaning = DataCleaning(processed_data, divide_strategy)
        X_train,X_test,y_train,y_test = data_cleaning.handle_data()
        
        logging.info("Data cleaning completed")
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logging.error("Error in cleaning data : {}".format(e))
        logging.error(e)
        raise e    
        
      