import logging
import mlflow
import pandas as pd
from sklearn.base import RegressorMixin


from model.model_dev import(
    LinearRegressionModel,
    SVRModel,
    RandomForestRegressorModel,
    XGBRegressorModel,
    HyperparameterTuner
)

def model_train(
    scaled_X_train: pd.DataFrame,
    scaled_X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    model_name: str,
    tuning_bool: bool,
    n_trials: int
) -> RegressorMixin:
    """
    Trains the model.

    Args:
        scaled_X_train (pd.DataFrame): Scaled training data.
        scaled_X_test (pd.DataFrame): Scaled testing data.
        y_train (pd.DataFrame): Training target.
        y_test (pd.DataFrame): Testing target.
        model_name (str): Model name.
        tuning_bool (bool): Hyperparameter tuning choice.
        n_trials (int): Number of trials for hyperparameter tuning.

    Returns:
        RegressorMixin: Trained model.
    """
    try:
        if model_name == 'linearregression':
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
        elif model_name == 'svr':
            mlflow.sklearn.autolog()
            model = SVRModel()
        elif model_name == 'randomforest':
            mlflow.sklearn.autolog()
            model = RandomForestRegressorModel()
        elif model_name == 'xgboost':
            mlflow.sklearn.autolog()
            model = XGBRegressorModel()
        else:
            raise ValueError('Model choice not supported. Choose from one below:\nlinearregression, svr, randomforest, xgboost')
        
        tuner = HyperparameterTuner(model, scaled_X_train, scaled_X_test, y_train, y_test, n_trials)
        
        if tuning_bool == True:
            best_params = tuner.optimize()
            trained_model = model.train(scaled_X_train, y_train, **best_params)
        else:
            trained_model = model.train(scaled_X_train, y_train)
        return trained_model
    except Exception as e:
        logging.error(f'Exception occurred when training model.\nException message: {str(e)}')
        raise e