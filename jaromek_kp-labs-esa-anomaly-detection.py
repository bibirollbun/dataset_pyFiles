!pip install imbalanced-learn==0.11.0 xgboost


import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from xgboost import XGBClassifier

import joblib



os.mkdir('/kaggle/working/models')


os.mkdir('/kaggle/working/submission')


DATA_PATH = "/kaggle/input/esa-adb-challenge/"
SAMPLE_SUBMISSION = f'{DATA_PATH}' + 'sample_submission.parquet'
TARGET_CHANNELS = f'{DATA_PATH}' + 'target_channels.csv'
TRAIN_DATA = f'{DATA_PATH}' + 'train.parquet'
TEST_DATA = f'{DATA_PATH}' + 'test.parquet'


sample_df = pd.read_parquet(SAMPLE_SUBMISSION)
train_df = pd.read_parquet(TRAIN_DATA)
test_df = pd.read_parquet(TEST_DATA)
target_df = pd.read_csv(TARGET_CHANNELS)


sample_df.head()


train_df.describe()


np.array(train_df.columns)


len(train_df.columns)


train_df.isnull().values.any()


dropped_cols = []


test_df.columns


test_df.head()


threshold = 1e-3
stds = train_df.std(numeric_only=True)
low_devuation_cols = stds[stds <= threshold].index

print(f'Dropping columns: {np.array(low_devuation_cols)}')
train_df.drop(columns=low_devuation_cols, inplace=True)
test_df.drop(columns=low_devuation_cols, inplace=True)


len(train_df.columns)


def column_entropy(col: pd.Series, base: float = 2) -> float:
    
    counts = col.value_counts(dropna=False)
    probs = counts / counts.sum()
    
    return -np.sum(probs * np.log2(probs))


threshold = 0.1
entropies = train_df.apply(column_entropy)
low_entropy_cols = entropies[entropies < threshold].index.tolist()
print(f'Columns to drop: {low_entropy_cols}')


train_df.drop(columns=low_entropy_cols, inplace=True)
test_df.drop(columns=low_entropy_cols, inplace=True)


len(train_df.columns)


corr_matrix = train_df.corr()


plt.figure(figsize=(10, 7))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt=".2f")
plt.show()


corr_matrix = corr_matrix.abs()
mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
upper = corr_matrix.where(mask)

threshold = 0.90
high_corr = (
    upper.stack()
         .loc[lambda x: x > threshold]
         .sort_values(ascending=False)
)

print("Highly correlated columns: (|corr| > 0.90):")
print(high_corr)


high_corellated_cols = []
for (col1, col2), corr_value in high_corr.items():
    high_corellated_cols.append(col2)

train_df.drop(columns=high_corellated_cols, inplace=True)
test_df.drop(columns=high_corellated_cols, inplace=True)


dropped_cols.append(high_corellated_cols)


len(train_df.columns)


train_df.head()


dataframe_no_id = train_df.drop(columns='id')
test_no_id = test_df.drop(columns='id')


def half_f1(precision, recall):
    return (1+(0.5)^2 * precision * recall)/((0.5)^2 * precision + recall)

def half_f1(precision, recall):
    beta = 0.5
    return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)



def evaluate_model(y_test, y_pred, model_name):
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
    f1_half = half_f1(precision, recall)

    print(f"\nMetryki dla modelu: {model_name}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 score:  {f1:.4f}")
    print(f"F1_0.5 score:  {f1_half:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Macierz pomyłek: {model_name}')
    plt.tight_layout()
    plt.show()


X = dataframe_no_id.drop(columns='is_anomaly')
y = dataframe_no_id['is_anomaly']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


rf_model = RandomForestClassifier(n_estimators=100, random_state=42, verbose=1)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

joblib.dump(rf_model, '/kaggle/working/models/random_forest_model.pkl')


xgb_model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42, verbose=1)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

joblib.dump(xgb_model, '/kaggle/working/models/xgboost_model.pkl')


evaluate_model(y_test, y_pred_rf, "Random Forest")


evaluate_model(y_test, y_pred_xgb, "XGBoost")


y_test_rf = rf_model.predict(test_no_id)


y_test_xgb = xgb_model.predict(test_no_id)


random_forest_submission = sample_df


y_test_rf = pd.Series(y_test_rf, name='is_anomaly')
random_forest_submission['is_anomaly'] = y_test_rf


random_forest_submission.head()


random_forest_submission.to_csv('/kaggle/working/submission/random_forest_submission.csv')


xgboost_submission = sample_df


y_test_xgb = pd.Series(y_test_xgb, name='is_anomaly')
xgboost_submission['is_anomaly'] = y_test_xgb


xgboost_submission.head()


xgboost_submission.to_csv('/kaggle/working/submission/xgb_submission.csv')

