import os

os.environ['UNIMOL_WEIGHT_DIR'] ='/share/cxy/release/fixed_hyper_param_random_weight'


import torch
import joblib
import pandas as pd
from unimol_tools import MolTrain, MolPredict

torch.cuda.set_device(0)


train_csv_path = 'train.csv'
test_csv_path = 'test.csv'
test_cleaned_csv_path = 'test_cleaned.csv'
current_directory = os.getcwd()
current_folder_name = os.path.basename(current_directory)


params = {
    'amp': True,
    'anomaly_clean': True,
    'batch_size': 1,
    'cuda': True,
    'data_type': 'molecule',
    'epochs': 350,
    'freeze_layers': 25,
    'freeze_layers_reversed': 25,
    'kfold': 5,
    'learning_rate': 8.5e-05,
    'load_model_dir': None,
    'logger_level': 1,
    'max_epochs': 100,
    'max_norm': 12.0,
    'metrics': 'mse',
    'model_name': 'unimolv1',
    'model_size': '84m',
    'num_classes': 1,
    'patience': 60,
    'remove_hs': False,
    'seed': 42,
    'smi_strict': True,
    'smiles_col': 'SMILES',
    'split': 'random',
    'split_group_col': 'scaffold',
    'split_seed': 42,
    'target_col_prefix': 'TARGET',
    'target_cols': 'TARGET',
    'target_normalize': 'auto',
    'task': 'regression',
    'use_amp': True,
    'use_cuda': True,
    'warmup_ratio': 0.03,
    'early_stopping': 80,
    'train_from_scratch': False,
    'save_dir': './fixed_hyper_param_random_weight',
}

def predict_and_save(clf, csv_path, file_name, save_path):
    result_path = os.path.join(save_path, file_name)
    if not os.path.exists(result_path):
        prediction = clf.predict(csv_path)
        joblib.dump(prediction, result_path)
        return prediction
    return joblib.load(result_path)

def test_hyper_parms_objective(**params) -> float:

    save_path = params['save_dir']

    metric_path = os.path.join(save_path, 'metric.result')
    if os.path.exists(metric_path):
        metric = joblib.load(metric_path)
        return metric['mse']


    clf = MolTrain(save_path=save_path, **params)
    clf.fit(train_csv_path)
    mse = clf.model.cv['metric']['mse']
    r2 = clf.model.cv['metric']['r2']


    clf = MolPredict(load_model=save_path)

    predict_and_save(clf, test_csv_path, 'predicted_test_delta_PCE.data', save_path)
    predict_and_save(clf, train_csv_path, 'predicted_train_delta_PCE.data', save_path)
    predict_and_save(clf, test_cleaned_csv_path, 'predicted_test_cleaned_delta_PCE.data', save_path)


test_hyper_parms_objective(**params)