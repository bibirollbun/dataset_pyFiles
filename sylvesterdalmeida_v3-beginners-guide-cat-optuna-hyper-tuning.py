import torch

## -- System tools --
import sys
import os
import gc

import joblib
import pickle

## -- DATA MANIPUALATION --
import numpy as np
import pandas as pd
import random

## -- VISUALISATION --
from IPython.display import display, Image
import matplotlib.pyplot as plt
import seaborn as sns
import shap

## -- FUNCTIONAL TOOLS --
from itertools import combinations
from tqdm.auto import tqdm
import time

## -- SCIKIT-LEARN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score

## -- CATBoost || Optuna --
try:
    import catboost as catb
    import optuna
except:
    %pip install -qq -U catboost optuna
    import catboost as catb
    import optuna

import warnings


###################### --- GLOBAL SETTINGS --- ######################
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

pd.options.mode.copy_on_write = True
pd.set_option('display.max_columns', 1000)
# plt.style.use("ggplot")
sns.set_style("whitegrid")

## -- Set Global Seed --
SEED = 42
def set_global_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)

set_global_seed()


## -- Import datasets --
train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
submit = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")

## -- Drop IDs --
train.drop('id', axis=1, inplace=True)
test.drop('id', axis=1, inplace=True)


train.head()


test.head()


## -- Manual Label Encoding -- ##
target_map = {'Yes': 1, 'No': 0}
for col in ['Stage_fear', 'Drained_after_socializing']:
    train[col] = train[col].map(target_map)
    test[col] = test[col].map(target_map)

## -- Replace all NaNs with value -1 --
train.fillna(-1, inplace=True)
test.fillna(-1, inplace=True)

## -- Encode target labels --
le = LabelEncoder()
train['Personality'] = le.fit_transform(train['Personality']).astype('int8')

## -- Downcast numeric values --
int_64 = test.select_dtypes(include=['int']).columns.tolist()
float_64 = test.select_dtypes(include=['float']).columns.tolist()

for df in [train, test]:
    df[int_64] = df[int_64].astype('int8')
    df[float_64] = df[float_64].astype('float16')



train.head()


train.info()


## -- Define X, y --
X = train.drop('Personality', axis=1)
y = train['Personality']

## -- Split data --
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

## -- Define ansi color codes --
YELLOW_TXT = '\033[93m'
RESET_TXT = '\033[0m'


## -- Set lgb.Dataset() container for effective data handling --
cat_train = catb.Pool(X, label=y)

def objective(trial):
    param = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'depth': trial.suggest_int('depth', 3, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
        'random_strength': trial.suggest_float('random_strength', 0.1, 1.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.1, 2.0),
        'loss_function': 'Logloss',
        'eval_metric': 'Logloss',
        'random_state': SEED,
        'task_type': "GPU" if torch.cuda.is_available() else "cpu",
        'verbose': False,
        'allow_writing_files': False,
        'thread_count': 3,
    }

    ## -- Train LGBM model with cross validation --
    cv_results = catb.cv(
            cat_train,
            param,
            num_boost_round=1000, # max iterations -> alias n_estimators
            nfold=10, # number of folds
            stratified=True,
            shuffle=True, # shuffle after each fold
            early_stopping_rounds=100, # stop iterations when score doesn't improve after 100 iterations
            verbose_eval=100, # print result every 100 iterations
            seed=SEED
    )

    ## -- Extract the best mean validation logloss --
    best_score = cv_results['test-Logloss-mean'].min()
    best_iteration = cv_results['test-Logloss-mean'].idxmin() + 1 # Add 1 because index is 0-based


    ## -- Save best_iteration for retrieval -- 
    trial.set_user_attr('best_iteration', best_iteration)

    ## -- Optional: Log for debugging and get best iteration (to be used as n_estimators) --
    print(f"{YELLOW_TXT}Trial {trial.number} finished with best logloss {best_score} at iteration {best_iteration}{RESET_TXT}")

    return best_score


## -- Create an Optuna study --
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())

## -- Run the optimization --
study.optimize(objective, n_trials=30, show_progress_bar=True)


## -- Print optimization results --
print(f"{'='*100}")
best_trial = study.best_trial.number
best_score = study.best_trial.value
best_iteration = study.best_trial.user_attrs.get('best_iteration')
best_params = study.best_trial.params

print(f"{'â–ˆ'*10}| Best Trial: {YELLOW_TXT}{best_trial}{RESET_TXT}")
print(f"{'â–ˆ'*10}| Best Score: {YELLOW_TXT}{best_score}{RESET_TXT}")
print(f"{'â–ˆ'*10}| Best Iteration: {YELLOW_TXT}{best_iteration}{RESET_TXT}")
print(f"{'-'*100}")
print(f"{'â–ˆ'*10}| Best parameters:")
print(f"{YELLOW_TXT}{best_params}{RESET_TXT}")


## -- Score across trials --
optuna.visualization.plot_optimization_history(study)


## -- Visualize the variable score indicators --
optuna.visualization.plot_slice(study)


## -- Observe the most important parameters during optimization --
optuna.visualization.plot_param_importances(study)


## -- Compute accuracy score with best parameters --
catb_model = catb.CatBoostClassifier(
                    **best_params,
                    iterations=best_iteration,
                    loss_function='Logloss',
                    eval_metric='Logloss',
                    random_state=SEED,
                    task_type="GPU" if torch.cuda.is_available() else "cpu",
                    verbose=False,
                    allow_writing_files=False,
                    thread_count=3,
)

catb_model.fit(X_train, y_train)
y_preds = catb_model.predict(X_test)
accuracy_score = accuracy_score(y_test, y_preds)
print(f"{'â–ˆ'*10}| Accuracy Score after Optuna search: {YELLOW_TXT}{accuracy_score}{RESET_TXT}")


## -- Train on the entire training dataset --
final_catb_model = catb.CatBoostClassifier(
                    **best_params,
                    iterations=best_iteration,
                    loss_function='Logloss',
                    eval_metric='Logloss',
                    random_state=SEED,
                    task_type="GPU" if torch.cuda.is_available() else "cpu",
                    verbose=False,
                    allow_writing_files=False,
                    thread_count=3,
)

final_catb_model.fit(X, y)

## -- Predict on the test dataset
test_predictions = final_catb_model.predict(test)

final_test_predictions = le.inverse_transform(test_predictions)

## -- Add predictions to the submission file
submit['Personality'] = final_test_predictions

## -- Save the submission file
submit.to_csv('submission_CATB_tuned.csv', index=False)

print("\nCATBOOST: Final model trained and predictions generated.")
submit.head()




