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


!pip install optuna

import lightgbm
import xgboost
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import category_encoders as ce
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
import optuna

import warnings
warnings.filterwarnings('ignore')


data_train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
data_test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")

df_train = pd.DataFrame(data_train)
df_test = pd.DataFrame(data_test)

df_train.head()


print("="*50)
print(df_train.info())
print("="*50)
print(df_train.isnull().sum())
print("="*50)
print(df_train.describe())


print("="*50)
print(df_test.info())
print("="*50)
print(df_test.isnull().sum())
print("="*50)
print(df_test.describe())


target_col = 'loan_paid_back'

num_features = df_train.select_dtypes(include = object).columns.drop(target_col,errors = 'ignore' )

n_features = len(num_features)
n_cols = 2

n_rows = (n_features + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 3))

axes = axes.flatten()

for i, col in enumerate(num_features):
    sns.countplot(x = df_train[col], ax=axes[i])
    axes[i].set_title(f"Count plot of {col}", fontsize = 10)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('count')

for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()


num_target2 = 'loan_paid_back'

# Select numeric columns except target
num_features2 = df_train.select_dtypes(include=np.number).columns.drop(num_target2, errors='ignore')

n_features2 = len(num_features2)
n_cols2 = 2
n_rows2 = (n_features2 + n_cols2 - 1) // n_cols2  # correct formula

# Create subplots
fig2, axes2 = plt.subplots(n_rows2, n_cols2, figsize=(12, n_rows2 * 3))
axes2 = axes2.flatten()

# Plot histograms for each feature
for i, col in enumerate(num_features2):
    sns.histplot(df_train[col], kde=True, ax=axes2[i], color='skyblue', bins = 20)
    axes2[i].set_title(f"Distribution of {col}", fontsize=10)
    axes2[i].set_xlabel(col)
    axes2[i].set_ylabel('Count')

# Hide empty subplots
for j in range(i + 1, len(axes2)):
    axes2[j].axis('off')

plt.tight_layout()
plt.show()


df_train.head()


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

df_train_encoded = df_train.copy()
target_col = "loan_paid_back"

df_train_encoded[target_col] = pd.to_numeric(df_train_encoded[target_col], errors='coerce')

cat_cols = df_train_encoded.select_dtypes(include='object').columns.tolist()

global_mean = df_train_encoded[target_col].mean()

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for col in cat_cols:
    df_train_encoded[col + '_enc'] = np.nan  
    for train_idx, val_idx in kf.split(df_train_encoded):
        train_fold = df_train_encoded.iloc[train_idx]
        val_fold = df_train_encoded.iloc[val_idx]
        
        means = train_fold.groupby(col)[target_col].mean()
        
        df_train_encoded.loc[val_idx, col + '_enc'] = val_fold[col].map(means)
    
    df_train_encoded[col + '_enc'].fillna(global_mean, inplace=True)

encoded_features = [col + '_enc' for col in cat_cols]

num_cols = df_train_encoded.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c not in encoded_features + [target_col]]

final_df = df_train_encoded[num_cols + encoded_features + [target_col]]

print("âœ… Final encoded dataset shape:", final_df.shape)
print(final_df.head())


means_dict = {}

for col in cat_cols:
    means_dict[col] = df_train_encoded.groupby(col)[target_col].mean()

global_mean = df_train_encoded[target_col].mean()

df_test_encoded = df_test.copy()

for col in cat_cols:
    df_test_encoded[col + '_enc'] = df_test_encoded[col].map(means_dict[col])
    
    df_test_encoded[col + '_enc'].fillna(global_mean, inplace=True)

df_test_final = df_test_encoded.drop(columns=cat_cols)

print("Encoded test data shape:", df_test_final.shape)
print(df_test_final.head())


plt.figure(figsize=(12,6))
sns.heatmap(
    final_df.corr(),
    annot = True,
    fmt = '.2g',
    center = 0,
    cmap = 'coolwarm'
)
plt.tight_layout()
plt.show()


plt.figure(figsize=(12,6))
sns.heatmap(
    df_test_final.corr(),
    annot = True,
    fmt = '.2g',
    center = 0,
    cmap = 'coolwarm'
)
plt.tight_layout()
plt.show()


X = final_df.drop(['loan_paid_back'], axis = 1)
y = final_df['loan_paid_back']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 42)


lgbm = LGBMClassifier(
        boosting_type='gbdt',
        num_leaves=228,
        max_depth=6,
        learning_rate=0.07818212260126128,
        n_estimators=638,
        subsample=0.6788195590022161,
        colsample_bytree=0.7709246021708535,
        min_child_samples=94,
        objective='binary',
        device='gpu',
        gpu_platform_id=0,
        gpu_device_id=0,
        verbose=-1,
        random_state=42,
        scale_pos_weight=0.550155015451764,
        lambda_l1 = 0.9135614161946477,
        lambda_l2 = 0.18702316713039244
)

lgbm.fit(X_train, y_train)


lgbm_train_preds = lgbm.predict(X_train)
lgbm_accuracy_train = accuracy_score(y_train, lgbm_train_preds)
print(f"Training accuracy of LightGBM : {lgbm_accuracy_train}")

print("="*50)
lgbm_valid_preds = lgbm.predict(X_val)
lgbm_valid_accuracy = accuracy_score(y_val, lgbm_valid_preds)
print(f"Validation accuracy of LightGBM : {lgbm_valid_accuracy}")
print("="*50)
print(f"Classification Report of LightGBM \n {classification_report(y_val, lgbm_valid_preds)}")
print("="*50)
print(f"Confusion Matrix of LightGBM \n {confusion_matrix(y_val, lgbm_valid_preds)}")
print("="*50)

cv_score = cross_val_score(lgbm, X_train, y_train, cv=5, scoring = 'accuracy')
print(f"Cross val score : {cv_score}")
print(f"Cross val score : {cv_score.mean()}")
print(f"Cross val score : {cv_score.std()}")


xgb = XGBClassifier(
        tree_method='gpu_hist',
        predictor='gpu_predictor',
        gpu_id=0,
        max_depth=8,
        learning_rate=0.10192483716710252,
        n_estimators=770,
        subsample=0.865758093102577,
        colsample_bytree=0.856500325208692,
        gamma=0.21261422434306043,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        verbosity=0,
        scale_pos_weight=0.5330503466838368,
        min_child_weight=10,
        alpha = 0.7433222866855071
            
)


xgb.fit(X_train, y_train)


xgb_train_preds = xgb.predict(X_train)
xgb_accuracy_train = accuracy_score(y_train, xgb_train_preds)
print(f"Training accuracy of XGBoost : {xgb_accuracy_train}")

print("="*50)
xgb_valid_preds = xgb.predict(X_val)
xgb_valid_accuracy = accuracy_score(y_val, xgb_valid_preds)
print(f"Validation accuracy of XGBoost : {xgb_valid_accuracy}")
print("="*50)
print(f"Classification Report of XGBoost \n {classification_report(y_val, xgb_valid_preds)}")
print("="*50)
print(f"Confusion Matrix of XGBoost \n {confusion_matrix(y_val, xgb_valid_preds)}")


feature_importances = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': lgbm.feature_importances_
}).sort_values(by='Importance', ascending=False)

print(feature_importances.head(20))

plt.figure(figsize=(10,6))
plt.barh(feature_importances['Feature'][:20][::-1],
         feature_importances['Importance'][:20][::-1])
plt.title("Important Features (LightGBM)")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()


import optuna

def objective(trial):
    # Define hyperparameter search space
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 5.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 5.0),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 1.5),
        'random_state': 42
    }

    # Train model
    model = LGBMClassifier(**params,
                          device = 'gpu', gpu_platform_id=0,gpu_device_id=0)
    model.fit(X_train, y_train)

    # Predict probabilities
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    # Try an adaptive threshold (can also optimize it)
    threshold = trial.suggest_float('threshold', 0.4, 0.7)
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Custom metric: Precision for class 1 (reduces FPs)
    precision = precision_score(y_val, y_pred, pos_label=1)

    # Optional: penalize false positives explicitly
    cm = confusion_matrix(y_val, y_pred)
    fp = cm[0, 1]  # false positives
    score = precision - (fp / len(y_val)) * 0.1  # penalize FP slightly

    return score

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print("Best Parameters:", study.best_params)
print("Best Score:", study.best_value)



def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'lambda': trial.suggest_float('lambda', 0.0, 5.0),
        'alpha': trial.suggest_float('alpha', 0.0, 5.0),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 1.5),
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'random_state': 42
    }

    model = XGBClassifier(**params, tree_method='gpu_hist',predictor='gpu_predictor',gpu_id=0)
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_val)[:, 1]
    threshold = trial.suggest_float('threshold', 0.4, 0.7)
    y_pred = (y_pred_proba >= threshold).astype(int)

    precision = precision_score(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred)
    fp = cm[0, 1]
    score = precision - (fp / len(y_val)) * 0.1

    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
print("Best Parameters:", study.best_params)


test_ids = df_test_final['id']

X_test = df_test_final[X_train.columns]

predictions = lgbm.predict(X_test)
submission = pd.DataFrame({
    "id": test_ids,
    "loan_paid_back": predictions
})

submission.to_csv("submission.csv", index=False)


submission

