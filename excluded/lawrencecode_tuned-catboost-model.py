import os
import logging
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, f1_score, recall_score, 
    roc_curve, auc, ConfusionMatrixDisplay
)

import shap
from catboost import CatBoostClassifier

plt.style.use('ggplot')
%matplotlib inline

ISKAGGLE = os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "") != ""

if not ISKAGGLE:
    from kaggle.api.kaggle_api_extended import KaggleApi

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")



class TelcoDataProcessor:
    def __init__(self, path="customer-churn-prediction-2020"):
        self.path = path
        self.column_transformer = None
        self.is_kaggle = ISKAGGLE

    def download_and_extract(self):
        if not self.is_kaggle and not Path(self.path).exists():
            zip_path = Path(f"{self.path}.zip")
            if not zip_path.exists():
                import kaggle
                kaggle.api.competition_download_cli('customer-churn-prediction-2020')
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(self.path)
            print("Dataset extracted successfully.")
        
        if self.is_kaggle:
            self.path = '/kaggle/input/customer-churn-prediction-2020'  # Set Kaggle path explicitly

    def load_data(self, filename):
        try:
            return pd.read_csv(f"{self.path}/{filename}")
        except FileNotFoundError:
            print(f"File {filename} not found in path {self.path}.")
            return None

    def preprocess(self, df):
        df = df.copy()  # Make a copy at the start

        # Feature Engineering
        df['total_charge_per_minute'] = df['total_day_charge'] / df['total_day_minutes']
        df['international_charge_per_minute'] = df['total_intl_charge'] / df['total_intl_minutes']
        df['night_charge_per_minute'] = df['total_night_charge'] / df['total_night_minutes']
        df['eve_charge_per_minute'] = df['total_eve_charge'] / df['total_eve_minutes']
        df['day_charge_per_minute'] = df['total_day_charge'] / df['total_day_minutes']

        # Calculate averages
        avg_total_charge_per_minute = df['total_charge_per_minute'].mean()
        avg_intl_charge_per_minute = df['international_charge_per_minute'].mean()
        avg_night_charge_per_minute = df['night_charge_per_minute'].mean()
        avg_eve_charge_per_minute = df['eve_charge_per_minute'].mean()
        avg_day_charge_per_minute = df['day_charge_per_minute'].mean()

        # Calculate over average features
        df['total_charge_per_minute_over_average'] = df['total_charge_per_minute'] / avg_total_charge_per_minute
        df['international_charge_per_minute_over_average'] = df['international_charge_per_minute'] / avg_intl_charge_per_minute
        df['night_charge_per_minute_over_average'] = df['night_charge_per_minute'] / avg_night_charge_per_minute
        df['eve_charge_per_minute_over_average'] = df['eve_charge_per_minute'] / avg_eve_charge_per_minute
        df['day_charge_per_minute_over_average'] = df['day_charge_per_minute'] / avg_day_charge_per_minute

        # Calculate over average by area code
        df['total_charge_per_minute_over_average_by_area_code'] = df.groupby('area_code')['total_charge_per_minute'].transform(lambda x: x / x.mean())
        df['international_charge_per_minute_over_average_by_area_code'] = df.groupby('area_code')['international_charge_per_minute'].transform(lambda x: x / x.mean())
        df['night_charge_per_minute_over_average_by_area_code'] = df.groupby('area_code')['night_charge_per_minute'].transform(lambda x: x / x.mean())
        df['eve_charge_per_minute_over_average_by_area_code'] = df.groupby('area_code')['eve_charge_per_minute'].transform(lambda x: x / x.mean())
        df['day_charge_per_minute_over_average_by_area_code'] = df.groupby('area_code')['day_charge_per_minute'].transform(lambda x: x / x.mean())

        # Calculate over average by state
        df['total_charge_per_minute_over_average_by_state'] = df.groupby('state')['total_charge_per_minute'].transform(lambda x: x / x.mean())
        df['international_charge_per_minute_over_average_by_state'] = df.groupby('state')['international_charge_per_minute'].transform(lambda x: x / x.mean())
        df['night_charge_per_minute_over_average_by_state'] = df.groupby('state')['night_charge_per_minute'].transform(lambda x: x / x.mean())
        df['eve_charge_per_minute_over_average_by_state'] = df.groupby('state')['eve_charge_per_minute'].transform(lambda x: x / x.mean())
        df['day_charge_per_minute_over_average_by_state'] = df.groupby('state')['day_charge_per_minute'].transform(lambda x: x / x.mean())

        # Convert object columns to category
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].astype('category')

        return df  # No need for an extra processed_df variable



processor = TelcoDataProcessor()
processor.download_and_extract()


train_raw_df = processor.load_data("train.csv")
test_raw_df = processor.load_data("test.csv")
submission_raw_df = processor.load_data("sampleSubmission.csv")


train_raw_df.select_dtypes(include='integer')


train_raw_df.select_dtypes(include='float')


sorted_columns = train_raw_df.columns.to_series().groupby(train_raw_df.dtypes).groups
sorted_columns = [col for dtype in sorted_columns.keys() for col in sorted_columns[dtype]]
train_raw_df = train_raw_df[sorted_columns]
train_raw_df.info()


train_raw_df.describe()


train_raw_df.describe(include='object')


shared_features = np.intersect1d(train_raw_df.columns, test_raw_df.columns)


for feature in train_raw_df.select_dtypes(include='object').columns:
        print(f'{feature}: {train_raw_df[feature].unique()}')


train_raw_df['churn'].value_counts().plot(kind='bar')
plt.xlabel('Target')
plt.ylabel('Count')
plt.title('Bar Chart of Target Column Counts')
plt.show()


target = 'churn'

features = [
    'state',
    'account_length',
    'area_code',
    'international_plan',
    'voice_mail_plan',
    'number_vmail_messages',
    'total_day_minutes',
    'total_day_calls',
    'total_day_charge',
    'total_eve_minutes',
    'total_eve_calls',
    'total_eve_charge',
    'total_night_minutes',
    'total_night_calls',
    'total_night_charge',
    'total_intl_minutes',
    'total_intl_calls',
    'total_intl_charge',
    'number_customer_service_calls',
    'total_charge_per_minute',
    'total_charge_per_minute_over_average',
    'total_charge_per_minute_over_average_by_area_code',
    'total_charge_per_minute_over_average_by_state',
    'international_charge_per_minute',
    'international_charge_per_minute_over_average',
    'international_charge_per_minute_over_average_by_area_code',
    'international_charge_per_minute_over_average_by_state',
    'night_charge_per_minute',
    'night_charge_per_minute_over_average',
    'night_charge_per_minute_over_average_by_area_code',
    'night_charge_per_minute_over_average_by_state',
    'eve_charge_per_minute',
    'eve_charge_per_minute_over_average',
    'eve_charge_per_minute_over_average_by_area_code',
    'eve_charge_per_minute_over_average_by_state',
    'day_charge_per_minute',
    'day_charge_per_minute_over_average',
    'day_charge_per_minute_over_average_by_area_code',
    'day_charge_per_minute_over_average_by_state'
]

# Prepare the data
X = train_raw_df.drop(columns=[target])
y = train_raw_df[target]

# Convert target variable to category
y = y.astype('category').cat.codes

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

X_train_processed = processor.preprocess(X_train)
X_test_processed = processor.preprocess(X_test)


accuracy = []
recall = []
roc_auc = []
precision = []
f1_scores = []
model_names = []

categorical_features_indices = np.where(X_train_processed.dtypes == 'category')[0]

class_weights = y_train.value_counts()[0] / y_train.value_counts()[1]

tuned_params = {
    "iterations": 501,
    "depth": 5,
    "learning_rate": 0.05738634862977308,
    "scale_pos_weight": class_weights,
    "bagging_temperature": 0.38138547122967104,
    "l2_leaf_reg": 1.11829190438502,
    "border_count": 76,
    "task_type": "GPU" if os.environ.get("CUDA_VISIBLE_DEVICES") else "CPU",
    "devices": "",
    "eval_metric": "F1",
    "verbose": False
}

catboost_base = CatBoostClassifier(**tuned_params)
catboost_base.fit(X_train_processed, y_train, cat_features=categorical_features_indices)


y_pred = catboost_base.predict(X_test_processed)

accuracy.append(round(accuracy_score(y_test, y_pred), 4))
recall.append(round(recall_score(y_test, y_pred), 4))
roc_auc.append(round(roc_auc_score(y_test, y_pred), 4))
precision.append(round(precision_score(y_test, y_pred, zero_division=0), 4))
f1_scores.append(round(f1_score(y_test, y_pred), 4))

model_names.append('Catboost_normal_weight')

result_df1 = pd.DataFrame({'Accuracy': accuracy, 'Recall': recall, 'Roc_Auc': roc_auc, 'Precision': precision, 'F1 Score': f1_scores}, index=model_names)
result_df1


print(result_df1)

fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues')
plt.title('Confusion Matrix')
plt.show()


explainer = shap.TreeExplainer(catboost_base)

shap_values = explainer.shap_values(X_test_processed)

shap.summary_plot(shap_values, X_test_processed, plot_type="bar")

shap.summary_plot(shap_values, X_test_processed)

shap.dependence_plot("number_customer_service_calls", shap_values, X_test_processed)

shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0,:], X_test_processed.iloc[0,:])


class_weights = y_train.value_counts()[0] / y_train.value_counts()[1]

tuned_params = {
    "iterations": 501,
    "depth": 5,
    "learning_rate": 0.05738634862977308,
    "scale_pos_weight": class_weights,
    "bagging_temperature": 0.38138547122967104,
    "l2_leaf_reg": 1.11829190438502,
    "border_count": 76,
    "task_type": "GPU" if os.environ.get("CUDA_VISIBLE_DEVICES") else "CPU",
    "devices": "",
    "eval_metric": "F1",
    "verbose": False
}

X_processed = processor.preprocess(X)

categorical_features_indices = np.where(X_processed.dtypes == 'category')[0]

catboost_submission = CatBoostClassifier(**tuned_params)
catboost_submission.fit(X_processed, y, cat_features=categorical_features_indices)


X_submission = test_raw_df[X.columns]
X_submission_processed = processor.preprocess(X_submission)
y_submission = catboost_submission.predict(X_submission_processed)
submission_df = pd.DataFrame({'id': test_raw_df['id'], 'churn': y_submission})


submission_df['churn'] = submission_df['churn'].map({1: 'yes', 0: 'no'})
submission_raw_df.drop('churn', axis = 1).merge(submission_df, on='id').to_csv('submission.csv', index=False)

