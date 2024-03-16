from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.model_train import model_train
from steps.evaluation import evaluation

def train_pipeline(model_name, tuning_bool, n_trials):
    """
    Model training pipeline.

    Args:
        ingest_data (DataClass)
        clean_data (DataClass)
        model_train (DataClass)
        evaluation (DataClass)
        model_name (string)
        tuning_bool (boolean)
        
    Returns:
        mae (float)
        mape (float)
        rmse (float)
        r2 (float)
    """
    df = ingest_data()
    scaled_X_train, scaled_X_test, y_train, y_test = clean_data(df)
    model = model_train(scaled_X_train, scaled_X_test, y_train, y_test, model_name, tuning_bool, n_trials)
    mae, mape, rmse, r2 = evaluation(model, scaled_X_test, y_test)