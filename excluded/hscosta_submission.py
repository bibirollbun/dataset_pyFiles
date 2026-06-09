pip install venn-abers==1.4.6 -q


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score, log_loss, precision_recall_curve
from sklearn.calibration import calibration_curve
import shap

from venn_abers import VennAbersCalibrator
from catboost import CatBoostClassifier

import seaborn as sns
import matplotlib.pyplot as plt


def clf_metric_report(y_score, y_true):
    print('Evaluating the model...')
    roc_auc = roc_auc_score(y_true, y_score)
    brier = brier_score_loss(y_true, y_score)
    avg_precision = average_precision_score(y_true, y_score)
    logloss = log_loss(y_true, y_score)
    print(f'ROC AUC: {roc_auc}')
    print(f'Brier Score: {brier}')
    print(f'Average Precision: {avg_precision}')
    print(f'Log Loss: {logloss}')

def plot_calibration_curve(y_score, y_true, title='Calibration Curve'):
    prob_true, prob_pred = calibration_curve(y_score, y_true, n_bins=10)
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker='.')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability')
    plt.title(title)
    plt.show()

def plot_pr_calib_curve(y_score, y_true):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=10)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')

    plt.subplot(1, 2, 2)
    plt.plot(prob_pred, prob_true, marker='.')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability')
    plt.title('Calibration Curve')

    plt.tight_layout()
    plt.show()

def plot_dis_probs(y_score, y_true):
    plt.figure(figsize=(10, 6))
    sns.histplot(y_score[y_true == 1], bins=50, color='red', label='Churn', kde=True, stat='density')
    sns.histplot(y_score[y_true == 0], bins=50, color='blue', label='Non-Churn', kde=True, stat='density')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.title('Distribution of Predicted Probabilities for Churn vs Non-Churn')
    plt.legend()
    plt.show()

def plot_shap_values(shap_values, X_test, figsize=(10, 12), type='bar'):
    #plt.figure(figsize=figsize)
    shap.summary_plot(shap_values, X_test, plot_type='bar', max_display=25)


df = pd.read_parquet("../input/neo-bank-feature-dataset/feature_engineering_dataset.parquet")


test = pd.read_parquet("../input/neo-bank-non-sub-churn-prediction/test.parquet")


(test.date.max() - pd.DateOffset(days=3*420))


train_max_date = '2023-07-20'
validation_max_date = '2024-01-01'

test_start_date = '2024-01-01'
target = 'churn'


features = [
    'interest_rate',
    'tenure',
    'prior_crypto_balance',
    'prior_mean_balance',
    'prior_sum_days_between',
    'prior_std_days_between',
    'prior_mean_days_between',
    'prior_max_days_between',
    'prior_mean_bank_transfer_in',
    'prior_mean_bank_transfer_out',
    'prior_mean_crypto_in',
    'prior_mean_crypto_out',
    'prior_mean_bank_transfer_in_volume',
    'prior_mean_crypto_in_volume',
    'prior_mean_crypto_out_volume',
    'prior_sum_crypto_in_volume',
    'prior_sum_crypto_out_volume',
    'prior_10D_mean_bank_transfer_out_volume',
    'prior_10D_mean_balance',
    'prior_90D_sum_days_between',
    'prior_90D_mean_days_between',
    'prior_90D_min_days_between',
    'prior_90D_mean_bank_transfer_out',
    'prior_90D_mean_balance',
    'prior_180D_sum_days_between',
    'prior_180D_mean_days_between',
    'prior_180D_max_days_between',
    'prior_180D_min_days_between',
    'prior_180D_mean_bank_transfer_in_volume',
    'prior_365D_sum_days_between',
    'prior_365D_mean_days_between',
    'prior_450D_sum_days_between',
    'prior_450D_mean_days_between',
    'prior_450D_std_days_between',
    'prior_450D_min_days_between',
    'prior_450D_mean_bank_transfer_out',
    'prior_450D_mean_crypto_out_volume',
    'prior_450D_mean_balance',
    'country',
    'broad_job_category'
]

cat_features = ['country', 'broad_job_category']


train_df = df.loc[df['date'] < train_max_date]
test_df = df.loc[df['date'] >= test_start_date]
validation_df = df.loc[(df['date'] >= train_max_date) & (df['date'] < validation_max_date)]

validation_df, calibration_df = train_test_split(validation_df, test_size=0.20, random_state=42, stratify=validation_df[target])

print('Train Shape: ', train_df.shape, train_df.churn.mean())
print('Test Shape: ', test_df.shape, test_df.churn.mean())
print('Validation shape: ', validation_df.shape, validation_df.churn.mean())
print('Calibration shape: ', calibration_df.shape, calibration_df.churn.mean())


X_train, y_train = train_df.loc[:, features], train_df.loc[:, target]
X_test, y_test = test_df.loc[:, features], test_df.loc[:, target]
X_validation, y_validation= validation_df.loc[:, features], validation_df.loc[:, target]
X_calibration, y_calibration = calibration_df.loc[:, features], calibration_df.loc[:, target]


params = {
    'iterations': 3000,
    'depth': 6, 
    'l2_leaf_reg': 1.0, 
    'learning_rate': 0.007,
    'bagging_temperature': 0.998925260111212, 
    'random_strength': 2.6226030049202764, 
    'auto_class_weights': 'SqrtBalanced',
    'cat_features': cat_features,
    'verbose': 10,
    'task_type': 'GPU',
    'eval_metric': 'Logloss'
}

model = CatBoostClassifier(**params)
model.fit(X_train, y_train, eval_set=(X_validation, y_validation), early_stopping_rounds=100)

y_pred = model.predict_proba(X_validation)[:, 1]

clf_metric_report(y_pred, y_validation)


p_cal = model.predict_proba(X_calibration)
p_val = model.predict_proba(X_validation)

va = VennAbersCalibrator()
va_prefit_prob = va.predict_proba(p_cal=p_cal, y_cal=y_calibration.values, p_test=p_val)
y_pred_va = va_prefit_prob[:, 1]


y_calibration.values.shape


p_val.shape


print("\nVenn-Abers Calibration Metrics:")
clf_metric_report(y_pred_va, y_validation)


p_test = model.predict_proba(X_test)
va = VennAbersCalibrator()
test_prefit_prob = va.predict_proba(p_cal=p_cal, y_cal=y_calibration.values, p_test=p_test)
y_pred_test = test_prefit_prob[:, 1]


print("\nVenn-Abers Calibration Metrics:")
clf_metric_report(y_pred_test, y_test)


submission = test_df[["Id"]].copy()
submission["churn"] = y_pred_test


submission.to_csv('submission.csv', index=False)

