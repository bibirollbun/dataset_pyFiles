# Import libraries for Tabnet
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau


!pip install pytorch-tabnet


# Import libraries for Tabnet
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetRegressor
from pytorch_tabnet.tab_model import TabNetClassifier


# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
from copy import deepcopy
from functools import partial

# Import sklearn classes for model selection, cross validation, and performance evaluation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold, GroupShuffleSplit, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from category_encoders import OneHotEncoder

# Import libraries for Hypertuning
import optuna
optuna.logging.set_verbosity(optuna.logging.ERROR)

# Import libraries for gradient boosting
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoost, CatBoostRegressor, CatBoostClassifier
from catboost import Pool

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


df_train = pd.read_csv('/kaggle/input/pistachio-cv-training/train (3).csv')
df_test = pd.read_csv('/kaggle/input/pistachio-cv-training/test.csv')
df_train['is_generated'] = 1
df_test['is_generated'] = 1
target_col = 'target'


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed = 42
seed_everything(seed)


# Concatenate train and original dataframes, and prepare train and test sets
X_train = df_train.drop([f'{target_col}'],axis=1).reset_index(drop=True)
y_train = df_train[f'{target_col}'].reset_index(drop=True)
X_test = df_test.reset_index(drop=True)
print(f"X_train shape :{X_train.shape} , y_train shape :{y_train.shape}")
print(f"X_test shape :{X_test.shape}")
# Delete the train and test dataframes to free up memory
#del df_train, df_test


# Learning Parameters
batch_size = 64 # 512
max_epochs = 500 # 500
patience = 100
num_workers = os.cpu_count()
device = "cuda" if torch.cuda.is_available() else "cpu"


"""def get_categorical_features(X_train, categorical_columns):
    # Get the feature columns in X_train
    feature_cols = [col for col in X_train.columns]
    
    # Create a dictionary to store the number of unique values for each categorical feature
    categorical_dims = {}
    for col in categorical_columns:
        categorical_dims[col] = X_train[col].nunique()
    
    # Get the indices of the categorical features in the feature columns list
    cat_idxs = [i for i, f in enumerate(feature_cols) if f in categorical_columns]
    
    # Get the number of unique values for each categorical feature
    cat_dims = [categorical_dims[f] for i, f in enumerate(feature_cols) if f in categorical_columns]
    
    return feature_cols, cat_idxs, cat_dims

# Get the feature columns, categorical feature indices, and number of unique values for each categorical feature
categorical_columns = ['is_generated']
feature_cols, cat_idxs, cat_dims = get_categorical_features(X_train, categorical_columns)

print('feature_cols:', feature_cols)
print('cat_idxs:', cat_idxs, ', cat_dims:', cat_dims)"""


# Create a list comprehension to find columns in feature_cols that are not in categorical_columns
numerical_columns = [col for col in X_train.columns]

# Initialize StandardScaler
scaler = StandardScaler()

# Scale the numerical features in X_train and X_test using StandardScaler
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])


def split_data(X, y, random_state, test_size=0.2, use_stratified_kfold=False, n_splits=6, n_repeats=5):
    if use_stratified_kfold:
        skf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        for train_index, val_index in skf.split(X, y):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
            yield X_train, X_val, y_train, y_val
    else:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
        yield X_train, X_val, y_train, y_val

use_stratified_kfold = True
n_splits = 5 # 5
n_repeats = 1

for X_train_, X_val, y_train_, y_val in split_data(X_train, y_train, random_state=seed, test_size=0.3, use_stratified_kfold=False, n_splits=n_splits, n_repeats=n_repeats):
    print('Set data for Optuna')


def get_params(params):
    optimizer_fn = torch.optim.Adam
    optimizer_params = {"lr": 2e-2, "weight_decay": 1e-5}

    scheduler_params = {
        "mode": "min",
        "patience": 5,
        "min_lr": 1e-5,
        "factor": 0.9
    }
    scheduler_fn = torch.optim.lr_scheduler.ReduceLROnPlateau

    return {
        "n_d": params['n_da'],
        "n_a": params['n_da'],
        "n_steps": params['n_steps'],
        "gamma": params['gamma'],
        "lambda_sparse": params['lambda_sparse'],
        "mask_type": params['mask_type'],
        "n_shared": params['n_shared'],
        "optimizer_fn": optimizer_fn,
        "optimizer_params": optimizer_params,
        "scheduler_fn": scheduler_fn,
        "scheduler_params": scheduler_params,
        "device_name": device,
        "seed": seed,
        "verbose": 50
    }


class TabNetOpt:
    def __init__(self, batch_size, max_epochs, from_unsupervised=None):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.from_unsupervised = from_unsupervised
        
    def pretrainer_objective(self, trial):
        """
        Pre-trains a TabNet model and returns the best cost.
        """
        params = self.get_params(trial)
        pretrainer = TabNetPretrainer(**params)
        pretrainer.fit(X_train=X_train.values, eval_set=[X_train.values],
                       max_epochs=self.max_epochs, batch_size=self.batch_size) 
        return pretrainer.best_cost

    def objective(self, trial):
        """
        Trains a TabNet model and returns the score on the validation set.
        """
        params = self.get_params(trial)
        model = TabNetClassifier(**params)
        #params['cat_idxs'] = cat_idxs # Comment out when unsupervised
        #params['cat_dims'] = cat_dims # Comment out when unsupervised
        #params['cat_emb_dim'] = 1     # Comment out when unsupervised
                
        model.fit(X_train=X_train_.values, y_train=y_train_.values,
                  eval_name=["train", "valid"], eval_metric=["accuracy"],
                  eval_set=[(X_train_.values, y_train_.values), (X_val.values, y_val.values)],
                  batch_size=self.batch_size,
                  max_epochs=self.max_epochs,
                  from_unsupervised=pretrainer)
        score = accuracy_score(y_val.values, model.predict(X_val.values))
        return score

    def get_params(self, trial):
        """
        Returns a dictionary of hyperparameters for the TabNet model.
        """
        mask_type = trial.suggest_categorical("mask_type", ["entmax", "sparsemax"])
        n_da = trial.suggest_int("n_da", 8, 64, step=8)
        n_steps = trial.suggest_int("n_steps", 1, 10, step=3)
        gamma = trial.suggest_float("gamma", 1.0, 2.0, step=0.2)
        n_shared = trial.suggest_int("n_shared", 1, 3)
        lambda_sparse = trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True)

        optimizer_fn = torch.optim.Adam
        optimizer_params = {"lr": 2e-2, "weight_decay": 1e-5}

        scheduler_params = {
            "mode": "min",
            "patience": 5,
            "min_lr": 1e-5,
            "factor": 0.9
        }
        scheduler_fn = torch.optim.lr_scheduler.ReduceLROnPlateau

        return {
            "n_d": n_da, 
            "n_a": n_da, 
            "n_steps": n_steps, 
            "gamma": gamma,
            "lambda_sparse": lambda_sparse, 
            "mask_type": mask_type, 
            "n_shared": n_shared,
            "optimizer_fn": optimizer_fn,
            "optimizer_params": optimizer_params,
            "scheduler_fn": scheduler_fn,
            "scheduler_params": scheduler_params,
            "device_name": device,
            "seed": seed,
            "verbose": 0
        }


# If you use optuna
n_trials = 1
study = optuna.create_study(direction='minimize')
study.optimize(TabNetOpt(batch_size=batch_size, max_epochs=max_epochs).pretrainer_objective, n_trials=n_trials)
tabnet_params = get_params(study.best_trial.params)
print(study.best_trial.params)

# Default Parameters
tabnet_params = {
    'mask_type': 'entmax', 'n_da': 8, 'n_steps': 3, 'gamma': 1.3, 'n_shared': 2, 'lambda_sparse': 1e-3, 'seed': seed
}

# Parameters determined by Optuna
# tabnet_params = {
#     'mask_type': 'entmax', 'n_da': 16, 'n_steps': 1, 'gamma': 1.6, 'n_shared': 2, 'lambda_sparse': 0.00011313516037838907
# }
tabnet_params = get_params(tabnet_params)

print(tabnet_params)


pretrainer = TabNetPretrainer(**tabnet_params)
fit_params = {
    "X_train": X_train.values,
    "eval_set": [X_train.values],
    "max_epochs": max_epochs,
    "patience": patience, 
    "batch_size": batch_size, 
    "virtual_batch_size": batch_size // 2,
    "num_workers": num_workers, 
    "drop_last": True
}
pretrainer.fit(**fit_params)


# If you use optuna
n_trials = 50 # Optuna 
study = optuna.create_study(direction='maximize')
study.optimize(TabNetOpt(batch_size=batch_size, max_epochs=max_epochs, from_unsupervised=pretrainer).objective, n_trials=n_trials)
tabnet_params = get_params(study.best_trial.params)
print(study.best_trial.params)

# Default Parameters
# tabnet_params = {
#     'mask_type': 'entmax', 'n_da': 8, 'n_steps': 3, 'gamma': 1.3, 'n_shared': 2, 'lambda_sparse': 1e-3, 'seed': seed
# }
# tabnet_params = get_params(tabnet_params)

# Parameters determined by Optuna
# tabnet_params = {
#     'mask_type': 'entmax', 'n_da': 32, 'n_steps': 1, 'gamma': 1.8, 'n_shared': 2, 'lambda_sparse': 9.234209012319759e-06
# }
# tabnet_params = get_params(tabnet_params)

print(tabnet_params)


#tabnet_params['cat_idxs'] = cat_idxs # Comment out when unsupervised
#tabnet_params['cat_dims'] = cat_dims # Comment out when unsupervised
#tabnet_params['cat_emb_dim'] = 1     # Comment out when unsupervised

model = TabNetClassifier(**tabnet_params)

tabnet_models = []
for X_train_, X_val, y_train_, y_val in split_data(X_train, y_train, random_state=seed, test_size=0.2, use_stratified_kfold=use_stratified_kfold, n_splits=n_splits, n_repeats=n_repeats):
    
    fit_params = {
        "X_train": X_train_.values, 
        "y_train": y_train_.values,
        "eval_set": [
            (X_train_.values, y_train_.values), 
            (X_val.values, y_val.values)
        ],
        "eval_name": ["train", "valid"],
        "eval_metric": ["accuracy"],
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "patience": patience,
        "drop_last": True,
        "num_workers": num_workers, 
        "pin_memory": True,
        "from_unsupervised": pretrainer # comment out when Unsupervised
    }
    
    model.fit(**fit_params)
    tabnet_models.append(deepcopy(model))


def plot_history(models):
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    for j, param in enumerate(['loss', 'lr', 'valid_accuracy']):
        for i, model in enumerate(models):
            axs[j].plot(model.history[param], label=f'model {i}')
        axs[j].set_title(param)
        axs[j].set_xlabel('epoch')
        axs[j].grid()
        axs[j].legend()
    plt.show()
    
plot_history(tabnet_models)


# Hyperparameters
n_estimators = 5000 # 5000
device = "gpu" if torch.cuda.is_available() else "cpu"

xgb_params = {
    'n_estimators': n_estimators,
    'learning_rate': 0.05,
    'max_depth': 7,
    'subsample': 1.0,
    'colsample_bytree': 1.0,
    'n_jobs': -1,
    'eval_metric': 'logloss',
    'objective': 'binary:logistic',
    'verbosity': 0,
    'random_state': seed,
}
if device == 'gpu':
    xgb_params['tree_method'] = 'gpu_hist'
    xgb_params['predictor'] = 'gpu_predictor'

lgb_params = {
    'n_estimators': n_estimators,
    'max_depth': 7,
    'learning_rate': 0.05,
    'subsample': 0.20,
    'colsample_bytree': 0.56,
    'reg_alpha': 0.25,
    'reg_lambda': 5e-08,
    'objective': 'binary',
    'metric': 'binary_error',
    'boosting_type': 'gbdt',
    'device': device,
    'random_state': seed
}

cb_params = {
    'iterations': n_estimators,
    'depth': 7,
    'learning_rate': 0.1,
    'l2_leaf_reg': 0.7,
    'random_strength': 0.2,
    'max_bin': 200,
    'od_wait': 65,
    'one_hot_max_size': 70,
    'grow_policy': 'Depthwise',
    'bootstrap_type': 'Bayesian',
    'od_type': 'Iter',
    'eval_metric': 'Logloss',
    'loss_function': 'Logloss',
    'task_type': device.upper(),
    'random_state': seed
}



from sklearn.metrics import accuracy_score
def train_classifier(classifier, X_train, y_train, X_val, y_val, classifier_params, early_stopping_rounds=300):
    classifier = classifier(**classifier_params)
    eval_set = [(X_val, y_val)]
    classifier.fit(X_train, y_train, early_stopping_rounds=early_stopping_rounds, eval_set=eval_set, verbose=False)
    val_preds = classifier.predict(X_val)
    accuracy = accuracy_score(y_val, val_preds)
    return classifier, accuracy

xgb_models, lgb_models, cb_models = [], [], []
for X_train_, X_val, y_train_, y_val in split_data(X_train, y_train, random_state=seed, test_size=0.2, use_stratified_kfold=use_stratified_kfold, n_splits=n_splits, n_repeats=n_repeats):
    xgb_model, xgb_accuracy= train_classifier(xgb.XGBClassifier, X_train_, y_train_, X_val, y_val, xgb_params)
    lgb_model, lgb_accuracy = train_classifier(lgb.LGBMClassifier, X_train_, y_train_, X_val, y_val, lgb_params)
    cb_model, cb_accuracy = train_classifier(CatBoostClassifier, X_train_, y_train_, X_val, y_val, cb_params)
    xgb_models.append(deepcopy(xgb_model)), lgb_models.append(deepcopy(lgb_model)), cb_models.append(deepcopy(cb_model))


rf_params = {
    'n_estimators': 1000,
    'max_depth': 10,
    'min_samples_leaf': 4,
    'min_samples_split': 5
}

# Random Forest Classifier Model
rf_model = RandomForestClassifier(random_state=seed, **rf_params)
rf_models = []
X_vals, y_vals = [], []
for X_train_, X_val, y_train_, y_val in split_data(X_train, y_train, random_state=seed, test_size=0.2, use_stratified_kfold=use_stratified_kfold, n_splits=n_splits, n_repeats=n_repeats):
    rf_model.fit(X_train_, y_train_)
    rf_models.append(deepcopy(rf_model))
    X_vals.append(X_val), y_vals.append(y_val)


def evaluate_ensemble(models, X_vals, y_vals):
    ensemble_accuracy = []
    for model, X_val, y_val in zip(models, X_vals, y_vals):
        X_val, y_val = X_val.values, y_val.values
        oof_pred = model.predict(X_val)
        score = accuracy_score(y_val, oof_pred)
        ensemble_accuracy.append(score)
    mean_score = np.mean(ensemble_accuracy)
    std_score = np.std(ensemble_accuracy)
    return f'{mean_score:.4f} ± {std_score:.4f}'


print('TabNet', evaluate_ensemble(tabnet_models, X_vals, y_vals))
#print('XGBoost AUC score', evaluate_ensemble(xgb_models, X_vals, y_vals))
print('LightGBM AUC score', evaluate_ensemble(lgb_models, X_vals, y_vals))
print('CatBoost AUC score', evaluate_ensemble(cb_models, X_vals, y_vals))
print('RandomForest AUC score', evaluate_ensemble(rf_models, X_vals, y_vals))


def visualize_importance(models, feature_cols, title='TabNet'):
    importances = []
    feature_importance = pd.DataFrame()
    for i, model in enumerate(models):
        _df = pd.DataFrame()
        _df["importance"] = model.feature_importances_
        _df["feature"] = pd.Series(feature_cols)
        _df["fold"] = i
        feature_importance = pd.concat([feature_importance, _df], axis=0, ignore_index=True)
        
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    plt.figure(figsize=(12, 4))
    sns.barplot(x='importance', y='feature', data=feature_importance, color='skyblue', errorbar='sd')
    plt.xlabel('Importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.title(f'{title} Feature Importance', fontsize=18)
    plt.grid(True, axis='x')
    plt.show()
feature_cols = [col for col in X_train.columns]
visualize_importance(tabnet_models, feature_cols, 'TabNet')
#visualize_importance(xgb_models, list(X_train.columns), 'XGBoost')
visualize_importance(lgb_models, list(X_train.columns), 'LightGBM')
visualize_importance(cb_models, list(X_train.columns), 'CatBoost')
visualize_importance(rf_models, list(X_train.columns), 'RandomForest')


class OptunaWeights:
    def __init__(self, seed):
        self.study = None
        self.weights = None
        self.seed = seed

    def _objective(self, trial, y_true, y_preds):
        # Define the weights for the predictions from each model
        weights = [trial.suggest_float(f"weight{n}", 0, 1) for n in range(len(y_preds))]

        # Calculate the weighted prediction
        weighted_pred = np.average(np.array(y_preds).T, axis=1, weights=weights)

        # Calculate the ROC AUC score for the weighted prediction
        score = accuracy_score(y_true, weighted_pred)
        return score

    def fit(self, y_true, y_preds, n_trials=2000):
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        sampler = optuna.samplers.CmaEsSampler(seed=self.seed)
        self.study = optuna.create_study(sampler=sampler, study_name="OptunaWeights", direction='maximize')
        objective_partial = partial(self._objective, y_true=y_true, y_preds=y_preds)
        self.study.optimize(objective_partial, n_trials=n_trials)
        self.weights = [self.study.best_params[f"weight{n}"] for n in range(len(y_preds))]

    def predict(self, y_preds):
        assert self.weights is not None, 'OptunaWeights error, must be fitted before predict'
        weighted_pred = np.average(np.array(y_preds).T, axis=1, weights=self.weights)
        return weighted_pred

    def fit_predict(self, y_true, y_preds, n_trials=2000):
        self.fit(y_true, y_preds, n_trials=n_trials)
        return self.predict(y_preds)
    
    def weights(self):
        return self.weights


models = np.array([tabnet_models, #xgb_models, lgb_models, cb_models, rf_models
                  ]).T.tolist()
names = ['TabNet', 
         #'LightGBM', 'CatBoost', 'RandomForest'
        ]

# Initialize variables
test_pred = np.zeros(X_test.shape[0])
zero_test_pred = np.zeros(X_test.shape[0])
ensemble_auc = []
weights = []

# Loop through the folds
i = 0
for _model, (X_train_, X_val, y_train_, y_val) in zip(models, split_data(X_train, y_train, random_state=seed, test_size=0.2, use_stratified_kfold=use_stratified_kfold, n_splits=n_splits, n_repeats=n_repeats)):
    val_probas, test_probas, zero_test_probas = [], [], []
    for model in _model:
        X_val, y_val = X_val.values, y_val.values
        oof_pred = model.predict(X_val)
        test_proba = model.predict(X_test)
        zero_test_proba = model.predict(X_test)
        val_probas.append(oof_pred)
        test_probas.append(test_proba)
        zero_test_probas.append(zero_test_proba)
        
    # Use Optuna to find the best ensemble weights
    optweights = OptunaWeights(seed=seed)
    val_proba = optweights.fit_predict(y_val, val_probas)
    score = accuracy_score(y_val, val_proba)
    print(f'[FOLD{i}] accuracy score {score:.5f}')
    ensemble_auc.append(score)
    weights.append(optweights.weights)
    
    # Predict on the test set using the optimized ensemble weights
    test_pred += optweights.predict(test_probas) / (n_splits * n_repeats)
    zero_test_pred += optweights.predict(zero_test_probas) / (n_splits * n_repeats)
    i += 1
    
# Calculate the mean AUC score of the ensemble
mean_score = np.mean(ensemble_auc)
std_score = np.std(ensemble_auc)
print(f'Ensemble accuracy score {mean_score:.5f} ± {std_score:.5f}')

# Print the mean and standard deviation of the ensemble weights for each model
print('--- Model Weights ---')
mean_weights = np.mean(weights, axis=0)
std_weights = np.std(weights, axis=0)
for name, mean_weight, std_weight in zip(names, mean_weights, std_weights):
    print(f'{name} {mean_weight:.5f} ± {std_weight:.5f}')


sub = pd.read_csv(os.path.join(filepath, 'sample_submission.csv'))
sub[f'{target_col}'] = test_pred
sub.to_csv('submission.csv', index=False)
sub.head(5)


df.describe()


df = pd.DataFrame(test_pred,columns=['target'])
df['target_binary'] = np.where(df['target']>=0.5, 1, 0)
df = df.drop('target', axis=1)
df.to_csv('out.csv', index=  False, header = False)


sns.histplot(df.target)





df





models = [tabnet_models]
names = ['TabNet']

def save_submission(X_test, models, name):
    print(name)
    X_test = X_test.values
    test_pred = np.zeros(X_test.shape[0])
    for model in models:
        test_pred += model.predict_proba(X_test)[:, 1].reshape(-1) / (n_splits * n_repeats)
        
    sub = pd.read_csv(os.path.join(filepath, 'sample_submission.csv'))
    sub[f'{target_col}'] = test_pred
    sub.to_csv('submission_tabnet.csv', index=False)
    
for _model, name in zip(models, names):
    save_submission(X_test, _model, name)


sub.head(5)


sns.histplot(sub.target)

