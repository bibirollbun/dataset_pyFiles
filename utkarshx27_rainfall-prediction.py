import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score, mean_squared_error,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from statistics import mean, stdev

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter("ignore", RuntimeWarning)


train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
sub = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")
original = pd.read_csv("/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv")
original.columns = original.columns.str.strip()
test.drop(columns = "id", inplace = True)
train.drop(columns = "id", inplace= True)


num_columns = ['day', 'pressure', 'maxtemp', 'temparature', 'mintemp',
       'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection',
       'windspeed']

def summary(merged_data):
    print(f'data shape: {merged_data.shape}')

    summ = pd.DataFrame(columns=['dtype', 'missing', 'missing[%]', 'unique', 'min', 'max', 'median', 'std', 'outliers', 'lower_bound', 'upper_bound'])

    for col in num_columns:
        summ.loc[col, 'dtype'] = merged_data[col].dtype
        summ.loc[col, 'missing'] = merged_data[col].isnull().sum()
        summ.loc[col, 'missing[%]'] = merged_data[col].isnull().sum() / len(merged_data) * 100
        summ.loc[col, 'unique'] = merged_data[col].nunique()
        summ.loc[col, 'min'] = merged_data[col].min()
        summ.loc[col, 'max'] = merged_data[col].max()
        summ.loc[col, 'median'] = merged_data[col].median()
        summ.loc[col, 'std'] = merged_data[col].std()

        q1 = merged_data[col].quantile(0.25)
        q3 = merged_data[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = merged_data[(merged_data[col] < lower_bound) | (merged_data[col] > upper_bound)][col]
        summ.loc[col, 'outliers'] = outliers.count()
        summ.loc[col, 'lower_bound'] = lower_bound
        summ.loc[col, 'upper_bound'] = upper_bound

    return summ


summary(train)


summary(original)


summary(test)


original['rainfall'] = original['rainfall'].map({"yes": 1, "no": 0})
imputer = SimpleImputer(strategy='mean')
original_imputed = pd.DataFrame(imputer.fit_transform(original), columns=original.columns)
merged_data = pd.concat([train, original_imputed], ignore_index=True)
test_imputed = pd.DataFrame(imputer.fit_transform(test), columns=test.columns)
summary(merged_data)


X = merged_data.drop(columns=['rainfall']) 
y = merged_data['rainfall'] 


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def objective(trial):
    C = trial.suggest_loguniform('C', 1e-4, 1e2)
    solver = trial.suggest_categorical("solver", ["liblinear", "lbfgs"])
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in skf.split(X_scaled, y):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = LogisticRegression(C=C, solver=solver, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_val)[:, 1]
        scores.append(roc_auc_score(y_val, y_pred))
    
    return np.mean(scores)


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)


final_model = LogisticRegression(solver='newton-cg', penalty='none', max_iter=10000, random_state=43, C=1.0)
final_model.fit(X_scaled, y)


test_imputed_scaled = scaler.transform(test_imputed)
test_predictions = final_model.predict_proba(test_imputed_scaled)[:, 1]


y_train_pred = final_model.predict_proba(X_scaled)[:, 1]
y_train_pred_labels = final_model.predict(X_scaled)

roc_auc = roc_auc_score(y, y_train_pred)
accuracy = accuracy_score(y, y_train_pred_labels)
precision = precision_score(y, y_train_pred_labels)
recall = recall_score(y, y_train_pred_labels)
f1 = f1_score(y, y_train_pred_labels)

print(f"ROC AUC Score: {roc_auc}")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")


fpr, tpr, _ = roc_curve(y, y_train_pred)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


plt.figure(figsize=(6, 4))
conf_matrix = confusion_matrix(y, y_train_pred_labels)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


sub['rainfall'] = test_predictions
sub.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")

