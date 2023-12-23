import logging
from abc import ABC, abstractmethod
from typing import Union

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    """
        Abstract class for data cleaning strategy.
    """
    @abstractmethod
    def handle_data(self, data:pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass
    
class DataPreProcessingStrategy(DataStrategy):
    """
        strategy for preprocessing data
    """
    def handle_data(self, data:pd.DataFrame) -> pd.DataFrame:
        """
          preprocess data
        """
        try:
            data = data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                    "order_item_id",
                    "customer_zip_code_prefix"
                ], 
                axis=1)
            data["product_weight_g"].fillna(data["product_weight_g"].median(), inplace=True)
            data["product_length_cm"].fillna(data["product_length_cm"].median(), inplace=True)
            data["product_height_cm"].fillna(data["product_height_cm"].median(), inplace=True)
            data["product_width_cm"].fillna(data["product_width_cm"].median(), inplace=True)
            
            data = data.select_dtypes(include=[np.number])
            return data
        
        except Exception as e:
            logging.error(f"Error in preprocessing data")
            logging.error(e)
            raise e
        
        
class DataDrivenStrategy(DataStrategy):
    """
       Diving data into train and test data
    """
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
                
        try:
            X = data.drop(["review_score"], axis=1)
            y = data["review_score"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        
        except Exception as e:
            logging.error(f"Error in splitting data")
            logging.error(e)
            raise e    
            
            
class DataCleaning:
    """
       class for cleaning data which process the data and divides it into train and test data
    """            
    def __init__(self,data:pd.DataFrame, strategy:DataStrategy) -> None:
        self.data = data
        self.strategy = strategy
    
    def handle_data(self)-> Union[pd.DataFrame, pd.Series]:
        try:
            return self.strategy.handle_data(self.data)  
        except Exception as e:
            logging.error("Error in handling data : {}".format(e))
            logging.error(e)
            raise e  