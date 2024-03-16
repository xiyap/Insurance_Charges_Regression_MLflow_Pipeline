import logging
from abc import ABC, abstractmethod
import optuna
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

class Model(ABC):
    """
    Class blueprint for regression models.
    """
    @abstractmethod
    def train(self, scaled_X_train, y_train):
        """
        Trains the model on training data.

        Args:
            scaled_X_train (_type_): Scaled training data.
            y_train (_type_): Training target.
        """
        pass
    
    @abstractmethod
    def optimize(self, trial, scaled_X_train, scaled_X_test, y_train, y_test):
        """
        Optimizes hyperparameters of model.

        Args:
            trial: Optuna trial object.
            scaled_X_train: Training data.
            scaled_X_test: Testing data.
            y_train: Training target.
            y_test: Testing target.
        """
        pass
    
class LinearRegressionModel(Model):
    """
    Linear regression model class.
    """
    def train(self, scaled_X_train, y_train, **kwargs):
        lr = LinearRegression(**kwargs)
        lr.fit(scaled_X_train, y_train)
        return lr
        
    def optimize(self, trial, scaled_X_train, scaled_X_test, y_train, y_test):
        lr = self.train(scaled_X_train, y_train)
        return lr.score(scaled_X_test, y_test)
    
class SVRModel(Model):
    """
    SVR model class.
    """
    def train(self, scaled_X_train, y_train, **kwargs):
        svr = SVR(**kwargs)
        svr.fit(scaled_X_train, y_train)
        return svr
        
    def optimize(self, trial, scaled_X_train, scaled_X_test, y_train, y_test):
        C = trial.suggest_categorical('C', [600, 1000, 5000, 10000])
        kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly', 'sigmoid'])
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
        degree = trial.suggest_categorical('degree', [2, 3, 4])
        epsilon = trial.suggest_categorical('epsilon', [0.01, 0.1, 1, 2])
        svr = self.train(scaled_X_train, y_train, C = C, kernel = kernel, gamma = gamma, degree = degree, epsilon = epsilon)
        return svr.score(scaled_X_test, y_test)
    
class RandomForestRegressorModel(Model):
    """
    RandomForestRegressor model class.
    """
    def train(self, scaled_X_train, y_train, **kwargs):
        rfr = RandomForestRegressor(**kwargs)
        rfr.fit(scaled_X_train, y_train)
        return rfr
        
    def optimize(self, trial, scaled_X_train, scaled_X_test, y_train, y_test):
        n_estimators = trial.suggest_categorical('n_estimators', [64, 100, 128, 200])
        max_depth = trial.suggest_categorical('max_depth', [None, 10, 20, 30, 40, 50])
        min_samples_split = trial.suggest_categorical('min_samples_split', [2, 5, 10, 20])
        min_samples_leaf = trial.suggest_categorical('min_samples_leaf', [1, 2, 4, 8])
        max_features = trial.suggest_categorical('max_features', ["sqrt", "log2", None])
        bootstrap = trial.suggest_categorical('bootstrap', [True, False])
        oob_score = trial.suggest_categorical('oob_score', [True, False])
        rfr = self.train(scaled_X_train, y_train, n_estimators = n_estimators, max_depth = max_depth, min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf, max_features = max_features, bootstrap = bootstrap, oob_score = oob_score)
        return rfr.score(scaled_X_test, y_test)
    
class XGBRegressorModel(Model):
    """
    XGBRegressor model class.
    """
    def train(self, scaled_X_train, y_train, **kwargs):
        xgb = XGBRegressor(**kwargs)
        xgb.fit(scaled_X_train, y_train)
        return xgb
        
    def optimize(self, trial, scaled_X_train, scaled_X_test, y_train, y_test):
        learning_rate = trial.suggest_categorical('learning_rate', [0.01, 0.1, 0.2])
        n_estimators = trial.suggest_categorical('n_estimators', [50, 100, 200])
        max_depth = trial.suggest_categorical('max_depth', [3, 5, 7])
        min_child_weight = trial.suggest_categorical('min_child_weight', [1, 3, 5])
        subsample = trial.suggest_categorical('subsample', [0.8, 0.9, 1.0])
        colsample_bytree = trial.suggest_categorical('colsample_bytree', [0.8, 0.9, 1.0])
        xgb = self.train(scaled_X_train, y_train, learning_rate = learning_rate, n_estimators = n_estimators, max_depth = max_depth, min_child_weight = min_child_weight, subsample = subsample, colsample_bytree = colsample_bytree)
        return xgb.score(scaled_X_test, y_test)
    
class HyperparameterTuner:
    """
    Class for tuning hyperparameters if hyperparameter tuning is true.
    """
    def __init__(self, model, scaled_X_train, scaled_X_test, y_train, y_test, n_trials):
        self.model = model
        self.scaled_X_train = scaled_X_train
        self.scaled_X_test = scaled_X_test
        self.y_train = y_train
        self.y_test = y_test
        self.n_trials = n_trials
        
    def optimize(self):
        study = optuna.create_study(direction = 'maximize')
        study.optimize(lambda trial: self.model.optimize(trial, self.scaled_X_train, self.scaled_X_test, self.y_train, self.y_test), n_trials = self.n_trials)
        return study.best_trial.params