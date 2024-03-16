import logging
import pandas as pd

class IngestData:
    """
    Class to ingests data from source and returns a DataFrame.
    """
    def __init__(self) -> None:
        pass
    
    def get_data(self) -> pd.DataFrame:
        df = pd.read_csv('./data/insurance.csv')
        return df

def ingest_data() -> pd.DataFrame:
    """
    Args:
        None

    Returns:
        pd.DataFrame: DataFrame of data.
    """
    try:
        ingest_data = IngestData()
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f'Exception occurred when ingesting data.\nException message: {str(e)}')
        raise e