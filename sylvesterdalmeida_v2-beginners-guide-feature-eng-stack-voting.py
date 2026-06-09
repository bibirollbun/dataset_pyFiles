import torch

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
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
            HistGradientBoostingClassifier, RandomForestClassifier,
            StackingClassifier, VotingClassifier
)
from sklearn.linear_model import (
            LogisticRegression, Lasso, LassoCV, Ridge, RidgeCV, ElasticNet, ElasticNetCV
)
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import (
            train_test_split, cross_val_score, cross_val_predict,
            KFold, StratifiedKFold, RepeatedStratifiedKFold
)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.preprocessing import (
            StandardScaler, RobustScaler, PolynomialFeatures, LabelEncoder
)
from sklearn.metrics import log_loss,accuracy_score, roc_auc_score, make_scorer
from sklearn.feature_selection import (
                    SelectFromModel, SelectKBest, chi2, f_classif,
                    mutual_info_classif, SequentialFeatureSelector as SFS1
)
from mlxtend.feature_selection import SequentialFeatureSelector as SFS2

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


# ## -- Univariate relationships between target and features --
# mutual_info = mutual_info_classif(train.drop('Personality', axis=1), train['Personality'], random_state=SEED)
# mutual_dict = dict(zip(train.drop('Personality', axis=1).columns.to_list(), list(mutual_info)))
# mutual_info_df = pd.Series(mutual_dict, index=mutual_dict.keys()).sort_values()
# mutual_info_df


# # plt.figure(figsize=(20, 5))
# mutual_info_df.plot(kind='barh', color='c',figsize=(15, 5))
# plt.title("Univariate relationships between Features and Target", fontsize=15)
# plt.tight_layout()


train.info()


## -- Define X, y --
X = train.drop('Personality', axis=1)
y = train['Personality']

## -- Split data --
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=SEED, stratify=y)

## -- Set kfold --
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

## -- Define ansi color codes --
YELLOW_TXT = '\033[93m'
RESET_TXT = '\033[0m'


## -- Instantiate diverse models --
models = {
    'XGBoost': xgb.XGBClassifier(
                            objective='binary:logistic', eval_metric='logloss',
                            device=USE_cud, verbosity=0, n_jobs=-1,
                            random_state=SEED,
    ),

    'LightGBM': lgb.LGBMClassifier(
                            objective='binary', metric='binary_logloss',
                            device=USE_gpu, n_jobs=-1, verbose=-1,
                            random_state=SEED,
    ),

    'CATBoost': cgb.CatBoostClassifier(
                            loss_function='Logloss', eval_metric='Logloss',
                            task_type=USE_GPU, allow_writing_files=False,
                            random_state=SEED, verbose=0, thread_count=-1,
    ),
    'HGBoost': HistGradientBoostingClassifier(scoring='neg_log_loss', random_state=SEED, verbose=0),
    'LOGistic': LogisticRegression(max_iter=1000, random_state = SEED, n_jobs=-1),
    'RF': RandomForestClassifier(random_state=SEED, n_jobs=-1),
    'NN_MLP': MLPClassifier(max_iter=500, random_state=SEED),
    'KNN': KNeighborsClassifier(n_jobs=-1),
    'SVC': SVC(probability=True, random_state=SEED)
}

print(f"Number of models: {len(models)}")
print(models.keys())


## -- Training thw base models --
scores = []
times = []
model_names = []
trained_models = []
test_predictions = {}

## -- Instantiate Polyniomial Feature generator --
poly = PolynomialFeatures(degree=3)

for name, model in tqdm(models.items(), total=len(models), desc="Training PolyFeature Models"):
    print(f"=== Training Base {name} {'='*10}")
    t_s = time.time()

    # Create features and scale with pipeline
    pipe_model = make_pipeline(StandardScaler(), poly, model)

    # Calculate CV score
    score = cross_val_score(pipe_model, X, y, cv=kfold, scoring='accuracy', n_jobs=-1)

    pipe_model.fit(X, y)
    preds = pipe_model.predict(test)

    # Record time
    t_e = time.time()
    timer = (t_e-t_s)

    ## -- Append all variables --
    test_predictions[name] = preds
    trained_models.append(pipe_model)
    model_names.append(name)
    scores.append(np.round(np.mean(score), 6))
    times.append(np.round(timer, 2))

    print(f"{YELLOW_TXT}{'â–ˆ'*10}| Accuracy Score: {np.mean(score):.6f} || Rendered in {timer:.2f}secs{RESET_TXT}\n")


## -- Uncomment below code to see generated features --
# trained_models[0][1].get_feature_names_out(input_features=X.columns.to_list())


## -- Function to plot results --
def plot_model_results(scores: list, times: list, model_names: list):
    df_results = pd.DataFrame({'Accuracy': scores, 'CV_Time(secs)': times}, index=model_names)
    df_results = df_results.sort_values(by='Accuracy', ascending=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))

    sns.barplot(data=df_results, x=df_results.index, y="Accuracy", ax=ax1)
    ax1.set_title('PolyFeatures Model Scores (Accuracy)')
    ax1.set_xlabel('Model Name')
    ax1.set_ylabel('Score (Accuracy)')
    for i, score in enumerate([x for x in df_results['Accuracy']]):
        ax1.text(i, score + 0.01, str(score), ha='center', va='baseline')

    sns.barplot(data=df_results, x=df_results.index, y="CV_Time(secs)", ax=ax2)
    ax2.set_title('PolyFeatures Training Time (seconds)')
    ax2.set_xlabel('Model Name')
    ax2.set_ylabel('Time (seconds)')
    for i, time in enumerate([x for x in df_results['CV_Time(secs)']]):
        ax2.text(i, time + max(times) * 0.01, str(time), ha='center', va='baseline')

    plt.tight_layout()
    plt.show()

    return df_results



df_results = plot_model_results(scores, times, model_names)
df_results.style.background_gradient(cmap='Blues')


## -- Preview first 5 predictions --
pd.DataFrame(test_predictions).head()


## -- Make submission for each model --
# print(f"=== Saving submission: {'='*50}")
# for k, v in test_predictions.items():
#     submit['Personality'] = le.inverse_transform(v)
#     submit.to_csv(f'submission_{k}.csv', index=False)
#     print(f"{YELLOW_TXT}||||| {k} base... Complete!{RESET_TXT}")


## -- Ensembling by Stacking :: Use each of the 8 models as meta-estimator --
stacked_scores = []
stacked_times = []
stacked_models = []
stacked_model_names = []
stacked_predictions = {}

# List of tuples (name, estimator) for the StackingClassifier
estimator_list = [(name, model) for name, model in models.items()]

for name, model in tqdm(models.items(), total=len(models), desc="Ensembling PolyFeature Models"):
    print(f"=== Stacking with {name} {'='*10}")
    tik = time.time()
    
    s_model = StackingClassifier(
                    estimators=estimator_list,
                    final_estimator=model,
                    cv=kfold,
                    stack_method='auto', n_jobs=-1,
    )
    
    ## -- Create features and scale with pipeline --
    pipe_model = make_pipeline(StandardScaler(), poly, s_model)

    pipe_model.fit(X_train, y_train)
    s_y_preds = pipe_model.predict(X_test)
    s_score = accuracy_score(y_test, s_y_preds)

    final_preds = pipe_model.predict(test)

    ## -- Record time --
    tok = time.time()
    s_timer = (tok-tik)

    ## -- Append all variables --
    stacked_predictions[name] = final_preds
    stacked_models.append(pipe_model)
    stacked_model_names.append(name)
    stacked_scores.append(np.round(s_score, 6))
    stacked_times.append(np.round(s_timer, 2))

    print(f"{YELLOW_TXT}{'â–ˆ'*10}| Accuracy Score: {s_score:.6f} || Rendered in {s_timer:.2f}secs{RESET_TXT}\n")


stacked_results = plot_model_results(stacked_scores, stacked_times, stacked_model_names)
stacked_results.style.background_gradient(cmap='Blues')


## -- Preview first 5 predictions --
pd.DataFrame(stacked_predictions).head()


## -- Make Submission --
print(f"=== Saving submissions as: {'='*50}")
for k, v in stacked_predictions.items():
    submit['Personality'] = le.inverse_transform(v)
    submit.to_csv(f'submission_stack_{k}_LV2.csv', index=False)
    print(f"{YELLOW_TXT}{'â–ˆ'*10}| 'submission_stack_{k}_LV2.csv' || Complete!{RESET_TXT}")


# Define the Voting Classifier using soft voting
# 'soft' voting uses predicted probabilities, 'hard' voting uses predicted class labels
tik = time.time()

v_model = VotingClassifier(
                    estimators = estimator_list,
                        voting = 'soft',
                        n_jobs = -1
)
# Make new features with Polyniomial
poly = PolynomialFeatures(degree=3)

# Create features and scale with pipeline
pipe_model = make_pipeline(StandardScaler(), poly, v_model)

## -- Train the voting model on the training data --
print(f"=== Training Voting Ensemble {'='*50}")
pipe_model.fit(X_train, y_train)

## -- Evaluate the voting model --
v_y_preds = pipe_model.predict(X_test)
v_score = accuracy_score(y_test, v_y_preds)

tok = time.time()
v_timer = tok - tik

print(f"{YELLOW_TXT}{'â–ˆ'*10}| Voting Accuracy Score: {v_score:.6f} || Rendered in {v_timer:.2f}secs{RESET_TXT}")


## -- Make predictions on the test --
v_final_preds = pipe_model.predict(test)
submit['Personality'] = le.inverse_transform(v_final_preds)

file_name = 'submission_voting_LV2.csv'
submit.to_csv(file_name, index=False)
print(f"{YELLOW_TXT}{'â–ˆ'*10}| Saving file as '{file_name}' || Complete!{RESET_TXT}\n")
submit.head()




