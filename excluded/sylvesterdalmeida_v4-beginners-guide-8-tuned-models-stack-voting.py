# %load_ext cuml.accel


import torch

import sys
import os
import gc

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
from tqdm.auto import tqdm
import time

## -- SCIKIT-LEARN
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
            HistGradientBoostingClassifier,
            RandomForestClassifier,
            StackingClassifier,
            VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import (
            train_test_split,
            cross_val_score,
            StratifiedKFold,
            RepeatedStratifiedKFold
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score, make_scorer

## -- GBDTs --
import xgboost as xgb
import lightgbm as lgb

try:
    import catboost as cgb
    import optuna
except:
    %pip install -qq -U catboost optuna
    import catboost as cgb
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

## -- Set Device-Agnostic code --
USE_cud = "cuda" if torch.cuda.is_available() else "cpu"
USE_gpu = "gpu" if torch.cuda.is_available() else "cpu"
USE_GPU = "GPU" if torch.cuda.is_available() else "CPU"


## -- Import datasets --
train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
submit = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")

## -- Drop IDs --
train.drop('id', axis=1, inplace=True)
test.drop('id', axis=1, inplace=True)


train.head()


train.info()


test.info()


## -- Manual Label Encoding --
target_map = {'Yes': 1, 'No': 0}
for col in ['Stage_fear', 'Drained_after_socializing']:
    train[col] = train[col].map(target_map)
    test[col] = test[col].map(target_map)

## -- Replace all NaNs with value -1 --
train.fillna(-1, inplace=True)
test.fillna(-1, inplace=True)

## -- Encode target labels --
le = LabelEncoder()
train['Personality'] = le.fit_transform(train['Personality'])

## -- Convert all values to integer --
train = train.astype('int8')
test = test.astype('int8')


train.head()


train.info()


## -- Define X, y --
X = train.drop('Personality', axis=1)
y = train['Personality']

## -- Split data --
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

## -- Set kfold --
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

## -- Define ansi color codes --
YELLOW_TXT = '\033[93m'
RESET_TXT = '\033[0m'


## -- Optuna tuned parameters for models --
xgb_params = {'n_estimators': 62, 'learning_rate': 0.08815547031597082, 'max_depth': 4, 'min_child_weight': 4,
              'subsample': 0.7070983848106323, 'colsample_bytree': 0.6615310251499813, 'gamma': 0.23241671892589388,
              'alpha': 0.23638093309785654, 'lambda': 0.008785477733691375}
lbgm_params = {'learning_rate': 0.019503916497879536, 'max_depth': 5, 'min_child_weight': 5,
               'subsample': 0.4439459957095353, 'colsample_bytree': 0.36342076704095766,
               'alpha': 4.81820228636352e-05, 'lambda': 0.004831021494600727}
cat_params = {'learning_rate': 0.04044142944125684, 'depth': 7, 'min_child_samples': 6, 'l2_leaf_reg': 1.139180935263684,
              'random_strength': 0.9977255667629478, 'bagging_temperature': 1.5213907509520852}
hist_params = {'max_iter': 859, 'max_depth': 4, 'learning_rate': 0.008008755369320521, #'max_features': 0.102043313096188,
               'min_samples_leaf': 3, 'l2_regularization': 0.0015185336359667173, 'tol': 5.1931120618638194e-05}
rf_params = {'criterion': 'gini', 'n_estimators': 382, 'max_depth': 7, 'min_samples_split': 8,
             'min_samples_leaf': 4, 'max_features': 0.3368674966521711}
log_params = {'solver': 'lbfgs', 'max_iter': 948, 'C': 0.2672575206927923, 'tol': 0.000995498233811136}
mlp_params = {'solver': 'adam', 'max_iter': 300, 'alpha': 0.08613243088409207, 'tol': 0.00015513583987775698}
knn_params = {'n_neighbors': 494, 'weights': 'uniform', 'algorithm': 'brute', 'leaf_size': 51}
svc_params = {'max_iter': 994, 'C': 1.13691406273983, 'tol': 0.00011374810887059884}


## -- Instantiate diverse models --
models = {
    'XGBoost': xgb.XGBClassifier(**xgb_params,
                            objective='binary:logistic', eval_metric='logloss',
                            device=USE_cud, verbosity=0, n_jobs=-1,
                            random_state=SEED,
    ),

    'LightGBM': lgb.LGBMClassifier(**lbgm_params,
                            objective='binary', metric='binary_logloss',
                            device=USE_gpu, n_jobs=-1, verbose=-1,
                            random_state=SEED,
    ),

    'CATBoost': cgb.CatBoostClassifier(**cat_params,
                            loss_function='Logloss', eval_metric='Logloss',
                            task_type='CPU',
                            allow_writing_files=False,
                            random_state=SEED, verbose=0, thread_count=-1,
    ),
    'HGBoost': HistGradientBoostingClassifier(**hist_params,
                            scoring='neg_log_loss',
                            random_state=SEED, verbose=0
    ),
    'RF': RandomForestClassifier(**rf_params,
                            random_state=SEED, n_jobs=-1,
    ),
    'LOGistic': LogisticRegression(**log_params,
                                   random_state=SEED, n_jobs=-1,
    ),
    'NN_MLP': MLPClassifier(**mlp_params,
                            random_state=SEED
    ),
    'KNN': KNeighborsClassifier(**knn_params,
                                n_jobs=-1,
    ),
    'SVC': SVC(**svc_params,
               probability=True,
               random_state=SEED,
    )
}

print(f"Number of models: {len(models)}")
print(models.keys())


## -- Meta-Models --
meta_models = {
    'XGBoost': xgb.XGBClassifier(
                        n_estimators=1000, learning_rate=0.01, max_depth=3,
                        objective='binary:logistic', eval_metric='logloss',
                        reg_alpha=1.0, reg_lambda=0.01, device=USE_cud,
                        verbosity=0, n_jobs=-1, random_state=SEED,
    ),
    'Logistic': LogisticRegression(
                        solver='lbfgs', max_iter=1000, C=1.0, tol=1e-4,
                        random_state=SEED, n_jobs=-1,
    ),
    'NN_MLP': MLPClassifier(
                        max_iter=1000,
                        random_state=SEED,
    ),
}


# @title
## -- Function to plot results --
def plot_model_results(scores: list, times: list, model_names: list):
    df_results = pd.DataFrame({'Accuracy': scores, 'CV_Time(secs)': times}, index=model_names)
    df_results = df_results.sort_values(by='Accuracy', ascending=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))

    sns.barplot(data=df_results, x=df_results.index, y="Accuracy", ax=ax1)
    ax1.set_title('Model Scores (Accuracy)')
    ax1.set_xlabel('Model Name')
    ax1.set_ylabel('Score (Accuracy)')
    for i, score in enumerate([x for x in df_results['Accuracy']]):
        ax1.text(i, score + 0.01, str(score), ha='center', va='baseline')

    sns.barplot(data=df_results, x=df_results.index, y="CV_Time(secs)", ax=ax2)
    ax2.set_title('Training Time (seconds)')
    ax2.set_xlabel('Model Name')
    ax2.set_ylabel('Time (seconds)')
    for i, time in enumerate([x for x in df_results['CV_Time(secs)']]):
        ax2.text(i, time + max(times) * 0.01, str(time), ha='center', va='baseline')

    plt.tight_layout()
    plt.show()

    return df_results


## -- Ensembling by Stacking :: Use each of the 8 models as meta-estimator --
stacked_scores = []
stacked_times = []
stacked_models = []
stacked_model_names = []
stacked_predictions = {}

# List of tuples (name, estimator) for the StackingClassifier
estimator_list = [(name, model) for name, model in models.items()]

for name, model in tqdm(meta_models.items(), total=len(meta_models), desc="Ensembling Models"):
    print(f"{'â–ˆ'*5}| Stacking with {name} {'='*50}")
    tik = time.time()

    stacked_model = StackingClassifier(
                    estimators=estimator_list,
                    final_estimator=model,
                    cv=kfold,
                    stack_method='auto',
                    n_jobs=-1,
    )

    ## -- Use pipeline to streamline data processing --
    pipe_model = make_pipeline(StandardScaler(), stacked_model)

    pipe_model.fit(X_train, y_train)
    stacked_y_preds = pipe_model.predict(X_test)
    stacked_score = accuracy_score(y_test, stacked_y_preds)

    final_preds = pipe_model.predict(test)

    ## -- Record time --
    tok = time.time()
    stacked_timer = (tok-tik)

    ## -- Append all variables --
    stacked_predictions[name] = final_preds
    stacked_models.append(pipe_model)
    stacked_model_names.append(name)
    stacked_scores.append(stacked_score)
    stacked_times.append(np.round(stacked_timer, 2))

    print(f"{YELLOW_TXT}{'â–ˆ'*10}| Accuracy Score: {stacked_score} || Rendered in {stacked_timer:.2f}secs{RESET_TXT}\n")


stacked_results = plot_model_results(stacked_scores, stacked_times, stacked_model_names)
stacked_results.style.background_gradient(cmap='Blues')


## -- Preview first 5 predictions --
pd.DataFrame(stacked_predictions).head()


## -- Make Submission --
print(f"=== Saving submissions as: {'='*50}")
for k, v in stacked_predictions.items():
    submit['Personality'] = le.inverse_transform(v)
    submit.to_csv(f'submission_stacktuned_{k}_LV2.csv', index=False)
    print(f"{YELLOW_TXT}{'â–ˆ'*10}| 'submission_stacktuned_{k}_LV2.csv' || Complete!{RESET_TXT}")


# Define the Voting Classifier using soft voting
# 'soft' voting uses predicted probabilities, 'hard' voting uses predicted class labels
tik = time.time()

voting_models = VotingClassifier(
                    estimators = estimator_list,
                       weights = [3, 3, 3, 2, 2, 1, 1, 1, 1],
                        voting = 'soft',
                        n_jobs = -1
)

# Create features and scale with pipeline
pipe_model = make_pipeline(StandardScaler(), voting_models)

## -- Train the voting model on the training data --
print(f"=== Training Voting Ensemble {'='*50}")
pipe_model.fit(X_train, y_train)

## -- Evaluate the voting model --
voting_y_preds = pipe_model.predict(X_test)
voting_score = accuracy_score(y_test, voting_y_preds)

tok = time.time()
voting_timer = tok - tik

print(f"{YELLOW_TXT}{'â–ˆ'*10}| Voting Accuracy Score: {voting_score:.6f} || Rendered in {voting_timer:.2f}secs{RESET_TXT}")


## -- Make predictions on the test --
voting_final_preds = pipe_model.predict(test)
submit['Personality'] = le.inverse_transform(voting_final_preds)

file_name = 'submission_votingtuned_LV2.csv'
submit.to_csv(file_name, index=False)
print(f"{YELLOW_TXT}{'â–ˆ'*10}| Saving file as '{file_name}' || Complete!{RESET_TXT}\n")
submit.head()




