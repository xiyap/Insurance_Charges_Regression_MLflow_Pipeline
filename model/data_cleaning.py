import logging
from abc import ABC, abstractmethod
from typing import Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataStrategy(ABC):
    """
    Class blueprint to define data handling strategy.
    """
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Processes the data according to method chosen.

        Args:
            data (pd.DataFrame): DataFrame of data.

        Returns:
            Union[pd.DataFrame, pd.Series]: Processed DataFrame.
        """
        pass
    
class DataPreprocessStrategy(DataStrategy):
    """
    Class to preprocess the data.
    """
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Removes duplicated rows.

        Args:
            data (pd.DataFrame): DataFrame of data.

        Returns:
            pd.DataFrame: Processed DataFrame.
        """
        try:
            logging.info('Dropping duplicate data...')
            data = data.drop_duplicates()
            return data
        except Exception as e:
            logging.error(f'Exception occurred in handle_data method of DataPreprocessStrategy class.\nException message: {str(e)}')
            raise e
        
class DataEncodeStrategy(DataStrategy):
    """
    Class to encode the data.
    """
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Encodes categorical data.

        Args:
            data (pd.DataFrame): DataFrame of data.

        Returns:
            pd.DataFrame: Encoded DataFrame.
        """
        try:
            logging.info('Encoding the data...')
            label_enc = LabelEncoder()
            categorical_column = ['sex', 'smoker']
            for cat in categorical_column:
                data[cat] = label_enc.fit_transform(data[cat])
                
            data = pd.get_dummies(data)
            return data
        except Exception as e:
            logging.error(f'Exception occurred in handle_data method of DataEncodeStrategy class.\nException message: {str(e)}')
            raise e
        
class DataSplitStrategy(DataStrategy):
    """
    Class to split data into train/test data and scales them.
    """
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Splits the data into train/test data and scales them.

        Args:
            data (pd.DataFrame): DataFrame of data.

        Returns:
            pd.DataFrame: Train/Test split DataFrame.
        """
        try:
            logging.info('Splitting the data...')
            X = data.drop('charges', axis = 1)
            y = data['charges']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size = 0.3, random_state = 101
                )
            
            logging.info('Scaling the feature data...')
            scaler = StandardScaler()
            scaled_X_train = scaler.fit_transform(X_train)
            scaled_X_test = scaler.transform(X_test)
            
            return scaled_X_train, scaled_X_test, y_train, y_test
        except Exception as e:
            logging.error(f'Exception occurred in handle_data method of DataSplitStrategy class.\nException message: {str(e)}')
            raise e

class DataCleaning:
    """
    Class to perform cleaning of data.
    """
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy) -> None:
        self.data = data
        self.strategy = strategy
    
    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        Handle data based on strategy specified.

        Returns:
            Union[pd.DataFrame, pd.Series]: Data with strategy applied.
        """
        return self.strategy.handle_data(self.data)