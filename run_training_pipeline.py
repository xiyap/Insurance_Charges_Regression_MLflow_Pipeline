import sys
import logging
import mlflow

from pipelines.training_pipeline import train_pipeline

def run_training_pipeline():
    mlflow.set_tracking_uri('http://127.0.0.1:5000')
    mlflow.set_experiment('insurance_charge_prediction')
    
    if len(sys.argv) < 2:
        error_message = 'Did not provide enough arguments. Pipeline should be ran as follow: python run_training_pipeline.py (model_name: STR) (hyperparameter_tuning: BOOL) (n_trials: INT)'
        logging.error(error_message)
        raise ValueError(error_message)

    model_name = sys.argv[1]
    
    if len(sys.argv) < 3:
        tuning_bool = False
        n_trials = 0
        print('No hyperparameter tuning performed.')
    else:
        if sys.argv[2].lower() == 'true' and len(sys.argv) < 4:
            tuning_bool = True
            n_trials = 10
            print(f'Defaulting n_trials to {n_trials} for hyperparameter tuning.')
        elif sys.argv[2].lower() == 'true' and len(sys.argv) < 5:
            tuning_bool = True
            n_trials_input = sys.argv[3]
            if n_trials_input.isdigit():
                n_trials = int(sys.argv[3])
                print(f'n_trials of {n_trials} used for hyperparameter tuning.')
            else:
                error_message = 'n_trials must be an integer.'
                logging.error(error_message)
                raise ValueError(error_message)
        else:
            tuning_bool = False
            n_trials = 0
            print('No hyperparameter tuning performed.')

    training = train_pipeline(
        model_name,
        tuning_bool,
        n_trials
    )

if __name__ == '__main__':
    run_training_pipeline()