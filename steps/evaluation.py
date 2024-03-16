import logging
import mlflow
import pandas as pd
import numpy as np
from typing import Tuple
from typing_extensions import Annotated
from sklearn.base import RegressorMixin
from model.result_eval import MAE, MAPE, RMSE, R2

def evaluation(
    model: RegressorMixin, scaled_X_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[
    Annotated[float, 'mae'],
    Annotated[float, 'mape'],
    Annotated[float, 'rmse'],
    Annotated[float, 'r2']
]:
    """
    Calculates metric scores of predictions.
    
    Args:
        model (RegressorMixin): Regression model.
        scaled_X_test (pd.DataFrame): Scaled testing data.
        y_test (pd.Series): Testing target.
        
    Returns:
        float: Metric scores.
    """
    try:
        prediction = model.predict(scaled_X_test)
        
        mae_class = MAE()
        mae = mae_class.calculate_score(y_test, prediction)
        mlflow.log_metric('test_mae', mae)
        
        mape_class = MAPE()
        mape = mape_class.calculate_score(y_test, prediction)
        mlflow.log_metric('test_mape', mape)
        
        rmse_class = RMSE()
        rmse = rmse_class.calculate_score(y_test, prediction)
        mlflow.log_metric('test_rmse', rmse)
        
        r2_class = R2()
        r2 = r2_class.calculate_score(y_test, prediction)
        mlflow.log_metric('test_r2', r2)
        
        return mae, mape, rmse, r2
        
    except Exception as e:
        logging.error(f'Exception occurred when evaluating predictions.\nException message: {str(e)}')
        raise e