# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import warnings
warnings.filterwarnings('ignore')



import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import KNNImputer
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import optuna
import json


train_data = pd.read_csv(os.path.join(dirname, 'train.csv'))
test_data = pd.read_csv(os.path.join(dirname, 'test.csv'))
sample_data = pd.read_csv(os.path.join(dirname, 'sample_submission.csv'))


train_df = train_data.copy()
test_df = test_data.copy()


train_df.head()


train_df.drop(columns=['id'], axis=1, inplace=True)
train_df.info()


train_df.isna().sum()


num_cols = train_df.drop(columns=['Personality']).select_dtypes(include=['int64', 'float64']).columns
cat_cols = train_df.drop(columns='Personality').select_dtypes(include=['object']).columns


train_df[num_cols].describe().T


personality_map = {"Extrovert": 0,
                   "Introvert": 1}
train_df["Personality"] = train_df["Personality"].map(personality_map)


encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
train_df[cat_cols] = encoder.fit_transform(train_df[cat_cols])


corr = train_df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f');


fig, axes = plt.subplots(nrows=1, ncols=len(num_cols), figsize=(5*len(num_cols), 5))
for i, col in enumerate(num_cols):
    sns.boxplot(y=train_df[col], ax=axes[i])
    axes[i].set_title(col)
plt.show()


fig, axes = plt.subplots(nrows=1, ncols=len(list(cat_cols) + list(['Personality'])), figsize=(5*len(list(cat_cols) + list(['Personality'])), 5))
for i, col in enumerate(list(cat_cols) + list(['Personality'])):
    sns.barplot(x=train_df[col].value_counts().index, y=train_df[col].value_counts().values, ax=axes[i])
    axes[i].set_title(col)
plt.show()


train_df.isna().sum()


def fix_missing_values(df, num_cols, cat_cols, k=2, encoded_return=False):
    df_encode = df.copy()
    knn_imputer = KNNImputer(n_neighbors=k)

    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df_encode[cat_cols] = encoder.fit_transform(df_encode[cat_cols])

    minmax = MinMaxScaler()
    df_encode[num_cols] = minmax.fit_transform(df_encode[num_cols])

    df_encode[list(num_cols) + list(cat_cols)] = knn_imputer.fit_transform(df_encode[list(num_cols) + list(cat_cols)])
    df_encode[cat_cols] = df_encode[cat_cols].round(0)

    if encoded_return:
        return df_encode
    else:
        df_encode[cat_cols] = encoder.inverse_transform(df_encode[cat_cols])
        df_encode[num_cols] = minmax.inverse_transform(df_encode[num_cols])
        return df_encode



train_df = fix_missing_values(train_df, num_cols, cat_cols, encoded_return=True)
train_df.head()


train_df.isna().sum()


fig, axes = plt.subplots(nrows=1, ncols=len(num_cols), figsize=(5*len(num_cols), 5))
for i, col in enumerate(num_cols):
    sns.boxplot(y=train_df[col], ax=axes[i])
    axes[i].set_title(col)
    #plt.tight_layout()
plt.show()


def fix_outliers(dataset, numeric_cols, strategy='remove'):
    data = dataset.copy()
    for col in numeric_cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
    
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        if strategy == 'remove':
            data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
        elif strategy == 'replace':
            data[col] = np.where(data[col] < lower_bound, lower_bound, data[col])
            data[col] = np.where(data[col] > upper_bound, upper_bound, data[col])

    return data


train_df = fix_outliers(train_df, num_cols, strategy="replace")


fig, axes = plt.subplots(nrows=1, ncols=len(num_cols), figsize=(5*len(num_cols), 5))
for i, col in enumerate(num_cols):
    sns.boxplot(y=train_df[col], ax=axes[i])
    axes[i].set_title(col)
    #plt.tight_layout()
plt.show()


x = train_df.drop(columns=["Personality"], axis=1)
y = train_df["Personality"]


class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
weights = dict(zip(np.unique(y), class_weights))
weights


scale_class = y.value_counts()[0] / y.value_counts()[1]
scale_class


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


estimators = [
    ('xgb', xgb.XGBClassifier(random_state=42, scale_pos_weight=scale_class)),
    ('rfc', RandomForestClassifier(random_state=42, class_weight=weights)),
    ('lr', LogisticRegression(random_state=42, class_weight=weights, max_iter=1000))]
sclf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(random_state=42, class_weight=weights), cv=5)



sclf.fit(x_train, y_train)


y_pred_stacking = sclf.predict(x_test)
print(f"  Accuracy of Stacking Ensemble: {accuracy_score(y_test, y_pred_stacking):.4f}")


conf_matrix = confusion_matrix(y_test, y_pred_stacking)
cm_display = ConfusionMatrixDisplay(confusion_matrix = conf_matrix)
cm_display.plot()
plt.show()


class_report = classification_report(y_test, y_pred_stacking)
print("Classification Report:")
print(class_report)


def objective_xgb(trial):

    params = {
        'objective': 'binary:logistic',     
        'eval_metric': 'logloss',           
        'use_label_encoder': False,         
        'n_estimators': trial.suggest_int('n_estimators', 200, 2000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0), 
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0), 
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True), 
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),  
    }
    
    model = xgb.XGBClassifier(**params, random_state=42, scale_pos_weight=scale_class)
    score = cross_val_score(model, x_train, y_train, n_jobs=-1, cv=3, scoring='accuracy')

    return score.mean()


study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=50) 

best_trial_xgb = study_xgb.best_trial
print(f"Score: {best_trial_xgb.value:.4f}")



output_file = "best_params_xgb.json"
with open(output_file, "w") as f:
    json.dump(study_xgb.best_trial.params, f) # Using indent for readability

print(f"\nOptimized parameters successfully saved to {output_file}")


def objective_rfc(trial):
    params = {
        "n_estimators" : trial.suggest_int('n_estimators', 100, 1000, step=50),
        "max_depth" : trial.suggest_int('max_depth', 5, 50),
        "min_samples_split" : trial.suggest_int('min_samples_split', 2, 20),
        "min_samples_leaf" : trial.suggest_int('min_samples_leaf', 1, 20),
        "max_features" : trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        "criterion" : trial.suggest_categorical('criterion', ['gini', 'entropy'])
    }

    rf_model = RandomForestClassifier(**params, random_state=42, n_jobs=-1, class_weight=weights)

    score = cross_val_score(rf_model, x_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
    
    # 4. Return the mean of the cross-validation scores
    return score.mean()


study_rfc = optuna.create_study(direction='maximize')
study_rfc.optimize(objective_rfc, n_trials=50)

best_trial_rfc = study_rfc.best_trial
print(f"  Score: {best_trial_rfc.value:.4f}")


output_file = "best_params_rfc.json"
with open(output_file, "w") as f:
    json.dump(study_rfc.best_trial.params, f)

print(f"\nOptimized parameters successfully saved to {output_file}")


def objective_lr(trial):
    solver = trial.suggest_categorical('solver', ['liblinear', 'saga'])
    
    c = trial.suggest_float('C', 1e-4, 1e4, log=True)
    
    if solver == 'liblinear':
        penalty = trial.suggest_categorical('penalty0', ['l1', 'l2'])
    else: 
        penalty = trial.suggest_categorical('penalty1', ['l1', 'l2', 'elasticnet'])
        
    params = {
        'solver': solver,
        'penalty': penalty,
        'C': c}
    
    if penalty == 'elasticnet':
        params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.1, 0.9)
        
    lr_model = LogisticRegression(**params, random_state=42, class_weight=weights, max_iter=1000)
    score = cross_val_score(lr_model, x_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
    return score.mean()


study_lr = optuna.create_study(direction='maximize')
study_lr.optimize(objective_lr, n_trials=100)
best_trial_lr = study_lr.best_trial

print(f"  Score: {best_trial_lr.value:.4f}")


output_file = "best_params_lr.json"
with open(output_file, "w") as f:
    json.dump(study_lr.best_trial.params, f) # Using indent for readability

print(f"\nOptimized parameters successfully saved to {output_file}")


with open("best_params_lr.json", "r") as f:
        params_lr = json.load(f)
print("params_lr:", params_lr, "\n")
with open("best_params_rfc.json", "r") as f:
        params_rfc = json.load(f)
print("params_rfc:", params_rfc, "\n")
with open("best_params_xgb.json", "r") as f:
        params_xgb = json.load(f)
print("params_xgb:", params_xgb, "\n")




params_lr = study_lr.best_trial.params
val = params_lr.pop('penalty0')

params_lr['penalty'] = val
params_lr


estimators_optimized = [
    ('xgb_o', xgb.XGBClassifier(**params_xgb ,random_state=42, scale_pos_weight=scale_class)),
    ('rfc_o', RandomForestClassifier(**params_rfc, random_state=42, class_weight=weights)),
    ('lr_o', LogisticRegression(**params_lr, random_state=42, class_weight=weights, max_iter=1000))]
sclf_optimized = StackingClassifier(estimators=estimators_optimized, final_estimator=LogisticRegression(random_state=42, class_weight=weights), cv=5)



sclf_optimized.fit(x_train, y_train)
y_pred_stacking = sclf_optimized.predict(x_test)
print(f"  Accuracy of Stacking Ensemble: {accuracy_score(y_test, y_pred_stacking):.4f}")


conf_matrix = confusion_matrix(y_test, y_pred_stacking)
cm_display = ConfusionMatrixDisplay(confusion_matrix = conf_matrix)
cm_display.plot()
plt.show()


class_report = classification_report(y_test, y_pred_stacking)
print("Classification Report:")
print(class_report)


inverse_personality_map = {0: "Extrovert",
                           1: "Introvert"}


test_df.isna().sum()


test_df = fix_missing_values(test_df, num_cols, cat_cols, encoded_return=True)
test_df.head()


x_ = test_df.drop(columns=["id"], axis=1)
predict = sclf_optimized.predict(x_)


submission = test_df[["id"]]
submission["Personality"] = predict


submission


submission["Personality"] = submission["Personality"].map(inverse_personality_map)
submission


submission.to_csv('submission.csv', index=False)




