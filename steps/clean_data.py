import logging
from typing import Tuple
from typing_extensions import Annotated
import pandas as pd
from model.data_cleaning import (
    DataPreprocessStrategy, 
    DataEncodeStrategy, 
    DataSplitStrategy, 
    DataCleaning
)

def clean_data(data: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, 'scaled_X_train'],
    Annotated[pd.DataFrame, 'scaled_X_test'],
    Annotated[pd.DataFrame, 'y_train'],
    Annotated[pd.DataFrame, 'y_test']
]:
    """
    Removes duplicate data, encodes categorical data, splits it into train/test data and scale the data.
    
    Args:
        data (pd.DataFrame): DataFrame of data.

    Returns:
        pd.DataFrame: Train/Test split DataFrame.
    """
    try:
        preprocess_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(data, preprocess_strategy)
        preprocessed_data = data_cleaning.handle_data()
        
        encode_strategy = DataEncodeStrategy()
        data_cleaning = DataCleaning(preprocessed_data, encode_strategy)
        encoded_data = data_cleaning.handle_data()
        
        splitting_strategy = DataSplitStrategy()
        data_cleaning = DataCleaning(encoded_data, splitting_strategy)
        scaled_X_train, scaled_X_test, y_train, y_test = data_cleaning.handle_data()
        return scaled_X_train, scaled_X_test, y_train, y_test
    except Exception as e:
        logging.error(f'Exception occurred when cleaning data.\nException message: {str(e)}')
        raise e