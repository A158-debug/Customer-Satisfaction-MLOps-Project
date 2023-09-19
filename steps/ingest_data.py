import logging
import pandas as pd
from zenml import step

class IngestData:
    """
       Ingesting data from a source and return a data frame.
    """
    def __init__(self, csv_path: str):
        self.csv_path = csv_path

    def get_data(self):
        logging.info(f"Reading data from {self.csv_path}")
        return pd.read_csv(self.csv_path)

@step
def ingest_df(csv_path: str) -> pd.DataFrame:
    """
       Ingest data from the csv_path.
       Args --> path of the data
       Returns --> pandas dataframe
    """
    
    try:
        return IngestData(csv_path).get_data()
    except Exception as e:
        logging.error(f"Error in reading data from {csv_path}")
        logging.error(e)
        raise e
        
    
    # return IngestData(csv_path).get_data()