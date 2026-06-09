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


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, f1_score
import xgboost as xgb
from catboost import CatBoostClassifier


# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

# Load and preprocess data
df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
dt = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")

# Original datasets
dataset1 = pd.read_csv("/kaggle/input/extrovert-vs-introvert-behavior-data-backup/personality_dataset.csv")
dataset2 = pd.read_csv("/kaggle/input/extrovert-vs-introvert-behavior-data/personality_dataset.csv")
dataset3 = pd.read_csv("/kaggle/input/personality-prediction-data-introvert-extrovert/personality_dataset.csv")


# Data processing
df = df.drop(columns=['id'])
dt = dt.drop(columns=['id'])

# Combine and clean original datasets
org = pd.concat([dataset1, dataset2], ignore_index=True)
org = org.rename(columns={"Personality": "P2"})
org = org.drop_duplicates(subset=[
    'Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
    'Going_outside', 'Drained_after_socializing', 'Friends_circle_size',
    'Post_frequency'
])

# Merge with original data
df = df.merge(org, how='left')
dt = dt.merge(org, how='left')

# Encode target variable
le = LabelEncoder()
df["Personality"] = le.fit_transform(df["Personality"])
X = df.drop(columns=["Personality"])
y = df["Personality"]
X_test = dt.copy()


# Prepare combined data for encoding
combined = pd.concat([X, X_test], axis=0)
cat_cols = ['Stage_fear','Drained_after_socializing','P2']
encoder = OrdinalEncoder()
combined[cat_cols] = encoder.fit_transform(combined[cat_cols])

X = combined.iloc[:len(X)].reset_index(drop=True)
X_test = combined.iloc[len(X):].reset_index(drop=True)


# Ensemble modeling
N_SPLITS = 6
N_REPEATS = 1

# Store predictions
oof_xgb = np.zeros(len(X))
test_xgb = np.zeros(len(X_test))
oof_cat = np.zeros(len(X))
test_cat = np.zeros(len(X_test))

# XGBoost Parameters
xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 8,
    "eta": 0.01,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
}


# Cross-validation
skf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=42)




# XGBoost Training
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)

    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=1000,
        evals=[(dval, "valid")],
        early_stopping_rounds=50,
        verbose_eval=False
    )
    
    oof_xgb[val_idx] += model.predict(dval) / N_REPEATS
    test_xgb += model.predict(dtest) / (N_REPEATS * N_SPLITS)



# CatBoost Training
skf = RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.09,
        depth=9,
        eval_metric='Logloss',
        random_seed=42,
        verbose=0,
        early_stopping_rounds=50
    )

    model.fit(X_train, y_train, eval_set=(X_val, y_val))

    oof_cat[val_idx] += model.predict_proba(X_val)[:, 1] / 2
    test_cat += model.predict_proba(X_test)[:, 1] / (10 * 2)



# Final Ensemble
weight_xgb = 0.7
weight_cat = 0.3

oof_preds = weight_xgb * oof_xgb + weight_cat * oof_cat
test_preds = weight_xgb * test_xgb + weight_cat * test_cat


plt.style.use('seaborn')
sns.set_palette("pastel")

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Training data distribution
train_counts = df['Personality'].value_counts()
train_counts.index = le.inverse_transform(train_counts.index)
train_plot = sns.barplot(x=train_counts.index, y=train_counts.values, ax=ax1, 
                        palette=["#3498db", "#e74c3c"])
ax1.set_title('Распределение в тренировочных данных', fontsize=14, pad=15)
ax1.set_xlabel('Тип личности', fontsize=12)
ax1.set_ylabel('Количество', fontsize=12)
ax1.grid(axis='y', linestyle='--', alpha=0.7)


for p in train_plot.patches:
    ax1.annotate(f'{p.get_height():,.0f}', 
                (p.get_x() + p.get_width()/2., p.get_height()), 
                ha='center', va='center', 
                fontsize=11, color='black',
                xytext=(0, 8), 
                textcoords='offset points')


sub_counts = submission['Personality'].value_counts()
sub_plot = sns.barplot(x=sub_counts.index, y=sub_counts.values, ax=ax2,
                      palette=["#2ecc71", "#f39c12"])
ax2.set_title('Распределение предсказаний', fontsize=14, pad=15)
ax2.set_xlabel('Тип личности', fontsize=12)
ax2.set_ylabel('Количество', fontsize=12)
ax2.grid(axis='y', linestyle='--', alpha=0.7)


for p in sub_plot.patches:
    ax2.annotate(f'{p.get_height():,.0f}', 
                (p.get_x() + p.get_width()/2., p.get_height()), 
                ha='center', va='center', 
                fontsize=11, color='black',
                xytext=(0, 8), 
                textcoords='offset points')


plt.suptitle('Сравнение распределения типов личности', y=1.02, fontsize=16)

plt.tight_layout()
plt.show()

# Metrics visualization with proper imports
metrics = {
    'Log Loss': log_loss(y, oof_preds),
    'ROC AUC': roc_auc_score(y, oof_preds),
    'Accuracy': accuracy_score(y, (oof_preds > 0.50).astype(int)),
    'F1 Score': f1_score(y, (oof_preds > 0.50).astype(int))
}

plt.figure(figsize=(12, 6))
metric_plot = sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), 
                         palette=["#9b59b6", "#1abc9c", "#e67e22", "#34495e"])

plt.title('Метрики качества модели', fontsize=16, pad=15)
plt.xlabel('Метрика', fontsize=12)
plt.ylabel('Значение', fontsize=12)
plt.ylim(0, 1.15)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value annotations with different formatting for Log Loss
for i, (metric, value) in enumerate(metrics.items()):
    color = 'white' if value > 0.7 else 'black'
    plt.text(i, value + 0.03, f"{value:.4f}", 
             ha='center', va='center', 
             fontsize=11, color=color,
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

plt.tight_layout()
plt.show()

# Final output
print("\nРезультаты оценки модели:")
print(f"Log Loss: {metrics['Log Loss']:.4f}")
print(f"ROC AUC: {metrics['ROC AUC']:.4f}")
print(f"Accuracy: {metrics['Accuracy']:.4f}")
print(f"F1 Score: {metrics['F1 Score']:.4f}")


final_preds = (test_preds > 0.50).astype(int)
submission["Personality"] = le.inverse_transform(final_preds)
submission.to_csv("submission.csv", index=False)
print("submission.csv created")




