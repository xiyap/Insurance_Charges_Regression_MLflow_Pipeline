import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

class Evaluation(ABC):
    """
    Class blueprint to define result evaluation strategy.
    """
    @abstractmethod
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates scores of prediction.

        Args:
            y_true (np.ndarray): True value of y.
            y_pred (np.ndarray): Predicted value of y.

        Returns:
            float: Prediction metric score.
        """
        pass
    
class MAE(Evaluation):
    """
    Class to calculate MAE of prediction.
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates MAE of prediction.

        Args:
            y_true (np.ndarray): True value of y.
            y_pred (np.ndarray): Predicted value of y.

        Returns:
            float: MAE score.
        """
        try:
            mae = mean_absolute_error(y_true, y_pred)
            logging.info(f'The MAE is {str(mae)}.')
            return mae
        except Exception as e:
            logging.error(f'Exception occurred in calculate_score method of MAE class.\nException message: {str(e)}')
            raise e

class MAPE(Evaluation):
    """
    Class to calculate MAPE of prediction.
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates MAPE of prediction.

        Args:
            y_true (np.ndarray): True value of y.
            y_pred (np.ndarray): Predicted value of y.

        Returns:
            float: MAPE score.
        """
        try:
            mape = mean_absolute_percentage_error(y_true, y_pred)
            logging.info(f'The MAPE is {str(mape)}.')
            return mape
        except Exception as e:
            logging.error(f'Exception occurred in calculate_score method of MAPE class.\nException message: {str(e)}')
            raise e

class RMSE(Evaluation):
    """
    Class to calculate RMSE of prediction.
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates RMSE of prediction.

        Args:
            y_true (np.ndarray): True value of y.
            y_pred (np.ndarray): Predicted value of y.

        Returns:
            float: RMSE score.
        """
        try:
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            logging.info(f'The RMSE is {str(rmse)}.')
            return rmse
        except Exception as e:
            logging.error(f'Exception occurred in calculate_score method of RMSE class.\nException message: {str(e)}')
            raise e

class R2(Evaluation):
    """
    Class to calculate R2-Score of prediction.
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates R2-Score of prediction.

        Args:
            y_true (np.ndarray): True value of y.
            y_pred (np.ndarray): Predicted value of y.

        Returns:
            float: R2-Score score.
        """
        try:
            r2 = r2_score(y_true, y_pred)
            logging.info(f'The R2-Score is {str(r2)}.')
            return r2
        except Exception as e:
            logging.error(f'Exception occurred in calculate_score method of R2 class.\nException message: {str(e)}')
            raise e