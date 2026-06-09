# Checking CUDA version and NVIDIA driver pairs (for CUDF/RAPIDS)
!nvidia-smi


# Loading in necessary libraries & packages

# Cell magic command to use RAPIDS cuDF for all Pandas
%load_ext cudf.pandas

# Fundamental libraries
import pandas as pd
import numpy as np

# Tracking time
from time import time

# Hiding warnings
import warnings
warnings.filterwarnings("ignore")

# Data viz
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style = 'white', palette = 'Set2')
pal = sns.color_palette('Set2')

# Sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score, classification_report

# CatBoost (for Adversarial Validation)
import catboost
from catboost import Pool, CatBoostClassifier
from catboost.utils import eval_metric

# XGBoost
import xgboost as xgb
xgb.set_config(verbosity=1)
xgb_device = "cuda"

# Optuna
import optuna
from optuna.samplers import TPESampler

# Shap
import shap
shap.initjs()


# Loading in the Kaggle datasets
df_train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")


# Viewing first 5 entries of 'df_train'
df_train.head()


# Viewing summary info of 'df_train'
df_train.info()


# Examining summary statistics of each numeric column in 'df_train'
df_train.describe()


# Viewing first 5 entries of 'df_test'
df_test.head()


# Viewing summary info of 'df_test'
df_test.info()


# Examining summary statistics of each numeric column in 'df_test'
df_test.describe()


# Creating countplot for target variable 'Personality'
ax = sns.countplot(x='y', data=df_train, palette='Set2')
for label in ax.containers:
  ax.bar_label(label)
ax.set_ylabel('Count')
ax.set_xlabel('Target Variable: y')
ax.set_title('Target Variable Distribution')
ax.set_ylim(0, 700000)
plt.show()


# Checking the % proportion of each target class
print(df_train['y'].value_counts(normalize=True))


# Declaring 'object' type columns to be converted to 'category' datatype
objs = df_train.select_dtypes(exclude=np.number).columns.tolist()

# Converting to 'category'
for col in objs:
    df_train[col] = df_train[col].astype('category')


# Verifying changes to datatypes
df_train.info()


# Splitting the data into feature (X) and target (y) arrays
X = df_train.drop(['y'], axis=1)
y = df_train['y']

# Splitting the arrays into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


# Fitting the baseline XGBoost model to the training data
baseline_model = xgb.XGBClassifier(objective='binary:logistic', device=xgb_device, eval_metric='auc', 
                                   random_state=42, enable_categorical=True)
baseline_model.fit(X_train, y_train)


# Evaluating the baseline model's performance (auc score) on the validation data
y_val_pred_proba = baseline_model.predict_proba(X_valid)[:, 1]
auc_score = roc_auc_score(y_valid, y_val_pred_proba)
print(f'Baseline Model ROC-AUC Score: {auc_score:.4f}')


# Plotting the AUC-ROC curve of the baseline model
fpr, tpr, thresholds = roc_curve(y_valid, y_val_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("AUC-ROC: Baseline Model")
plt.legend()
plt.show()


# Plotting feature importance of the baseline model
xgb.plot_importance(baseline_model, max_num_features=12)


# Loading in fresh copies of the Kaggle datasets
df_train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")


# Extracting ids from 'df_test' for later submission
ids = df_test['id']


# Dropping 'id' column from both datasets
df_train.drop(['id'], axis=1, inplace=True)
df_test.drop(['id'], axis=1, inplace=True)


# Steps taken if including the original dataset

run = 0

if run == 1:

    # Loading in the original dataset
    df_original = pd.read_csv("/kaggle/input/bank-marketing-dataset-full/bank-full.csv", sep=";")

    # Mapping target variable as 1/0 (int)
    df_original['y'] = df_original['y'].map({'yes': 1, 'no': 0})

    # Combining the 'df_train' and 'df_original' datasets
    df_train = pd.concat([df_train, df_original], axis=0, ignore_index=True)


# Declaring numeric and categorical columns

# Numeric (not including id or target variable 'y')
num_cols = ['age', 'balance', 'day', 'duration', 
            'campaign', 'pdays', 'previous']

# Categorical
cat_cols = ['job', 'marital', 'education', 'default',
           'housing', 'loan', 'contact', 'month', 'poutcome']


# Creating function to convert 'object' type columns into 'category'
def convert_object(df):
    for col in cat_cols:
        df[col] = df[col].astype('category')

# Applying 'convert_object' function to both datasets
convert_object(df_train)
convert_object(df_test)


# Creating function to remove outliers by applying the Inter Quartile method
# Removes outliers from numeric columns
def remove_outliers(df):

    # Initializing deleted row count at 0
    rows_deleted_total = 0

    # Calculating IQR range
    for column in num_cols:
        Q1 = df[column].quantile(0.005)
        Q3 = df[column].quantile(0.995)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Filtering the data, including all data between the lower and upper bounds
        filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        rows_deleted = len(df) - len(filtered_df)
        rows_deleted_total += rows_deleted
        # Printing results of outlier removal
        print(f"{column}: {rows_deleted}")
        # Updating dataframe with removed outliers
        df = filtered_df
    # Printing total number of rows deleted
    print(f"Total rows deleted: {rows_deleted_total}")

# Applying 'remove_outlier' function to datasets
remove_outliers(df_train)
remove_outliers(df_test)


# Creating feature engineering function
def feature_eng(df):    
    df['balance_positive'] = (df['balance'] > 0).astype(int)
    df['has_previous'] = (df['previous'] > 0).astype(int)
    df['duration_long'] = (df['duration'] > 300).astype(int)
    df['campaign_multiple'] = (df['campaign'] > 2).astype(int)
    df['sqrt_age'] = np.sqrt(df['age'])
    # Log features
    df['duration_log']=np.log1p(df['duration'])
    df['campaign_log']=np.log1p(df['campaign'])
    df['pdays_log']=np.log1p(df['pdays']+2)
    df['previous_log']=np.log1p(df['previous']+1)


# Applying 'feature_eng' function to both datasets
feature_eng(df_train)
feature_eng(df_test)


# Verifying changes
df_train.info()


# Creating final versions of each preprocessed dataset
train = df_train.copy()
test = df_test.copy()


# Splitting the data into feature (X) and target (y) arrays
X = train.drop(['y'], axis=1)
y = train['y']

# Splitting the arrays into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, 
                                                      random_state=42, stratify=y)


# Calculating weight for target classes
Class_0 = (train['y'] == 0).sum()
Class_1 = (train['y'] == 1).sum()
scale_pos_weight = Class_1 / Class_0
print(f"scale_pos_weight: {scale_pos_weight:.4f}")


# Creating 'objective' function which will trial different parameter values and combinations
def objective(trial):
    params = {
        'objective': 'binary:logistic',
        'device': xgb_device,
        'eval_metric': 'auc',
        'random_state': 42,
        'enable_categorical': True,
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
        #'scale_pos_weight': scale_pos_weight,
        'n_estimators': 10000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'max_depth': trial.suggest_int('max_depth', 1, 15),
        'subsample': trial.suggest_float('subsample', 0.25, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),  
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True) 
    }

    # Fitting XGBoost model with parameters from the trials
    model = xgb.XGBClassifier(**params)
    
    model.fit(X_train, y_train)

    # Making predictions on the validation set
    y_pred_proba = model.predict_proba(X_valid)[:, 1]
    score = roc_auc_score(y_valid, y_pred_proba)
    print('ROC-AUC:', score)
    return score

# When set to 1, optuna will create a study to find the optimal parameters for the model
run=0

if run==1:

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    print('Best trial:')
    trial = study.best_trial

    print('Value: {}'.format(trial.value))
    print('Params: ')
    for key, value in trial.params.items():
        print(' {}: {}'.format(key, value))


# Recording best parameters from trial #1
best_params = {
    'objective': 'binary:logistic',
    'device': xgb_device,
    'eval_metric': 'auc',
    'random_state': 42,
    'enable_categorical': True,
    'n_estimators': 839, 
    'learning_rate': 0.025193901676668747, 
    'colsample_bytree': 0.4692203584552837, 
    'max_depth': 12, 
    'subsample': 0.7737770422287007, 
    'min_child_weight': 7, 
    'reg_alpha': 0.008655641788467871, 
    'reg_lambda': 0.09225529592581996
}


# Recording best parameters from trial #2
best_params_2 = {
    'objective': 'binary:logistic',
    'device': xgb_device,
    'eval_metric': 'auc',
    'random_state': 42,
    'enable_categorical': True,
    'n_estimators': 10000, 
    'learning_rate': 0.025193901676668747, 
    'colsample_bytree': 0.4692203584552837, 
    'max_depth': 12, 
    'subsample': 0.7737770422287007, 
    'min_child_weight': 7, 
    'reg_alpha': 0.008655641788467871, 
    'reg_lambda': 0.09225529592581996
}


# Fitting the model with the best parameters!
final_model = xgb.XGBClassifier(**best_params_2)


# Fitting the model with the training data
final_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=40, verbose=False)


# Making predictions on the validation set
y_val_pred_proba = final_model.predict_proba(X_valid)[:, 1]


# Evaluating the performance of the final model
auc_score = roc_auc_score(y_valid, y_val_pred_proba)
print(f'Validation ROC-AUC Score: {auc_score:.4f}')


# Plotting the AUC-ROC curve of the final model
fpr, tpr, thresholds = roc_curve(y_valid, y_val_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("AUC-ROC: Final Model")
plt.legend()
plt.show()


# Plotting feature importance of the final model
xgb.plot_importance(final_model, max_num_features=12)


# Making final predictions on the test data
preds = final_model.predict_proba(test)[:,1]


# Creating 'submission' dataframe to store predictions with ids
submission = pd.DataFrame({'id': ids, 'y': preds})
submission


# Creating .csv file for submissions and scoring
run = 1

if run == 1:
    submission.to_csv('submission.csv', index=False)

