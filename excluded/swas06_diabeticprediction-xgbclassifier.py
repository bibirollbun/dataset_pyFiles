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
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve
import xgboost as xgb
import optuna
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import chi2_contingency
import plotly.figure_factory as f



train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")


train.head(3)


train['total_hdl_ratio'] = train['cholesterol_total'] / train['hdl_cholesterol']
train['trig_hdl_ratio'] = train['triglycerides'] / train['hdl_cholesterol']


train['pulse_pressure'] = train['systolic_bp'] - train['diastolic_bp']
train['map'] = train['diastolic_bp'] + (train['pulse_pressure'] / 3)


train['bmi_category'] = pd.cut(train['bmi'], bins=[0,18.5,25,30,100], labels=[0,1,2,3])


train['screen_sleep_ratio'] = train['screen_time_hours_per_day'] / train['sleep_hours_per_day']




test['total_hdl_ratio'] = test['cholesterol_total'] / test['hdl_cholesterol']
test['trig_hdl_ratio'] = test['triglycerides'] / test['hdl_cholesterol']


test['pulse_pressure'] = test['systolic_bp'] - test['diastolic_bp']
test['map'] = test['diastolic_bp'] + (test['pulse_pressure'] / 3)


test['bmi_category'] = pd.cut(test['bmi'], bins=[0,18.5,25,30,100], labels=[0,1,2,3])


test['screen_sleep_ratio'] = test['screen_time_hours_per_day'] / test['sleep_hours_per_day']



train.drop(['cholesterol_total', 'ldl_cholesterol', 'triglycerides', 
            'diastolic_bp', 'bmi'], 
           axis=1, inplace=True, errors='ignore')


test.drop(['cholesterol_total', 'ldl_cholesterol', 'triglycerides', 
            'diastolic_bp', 'bmi'], 
           axis=1, inplace=True, errors='ignore')


train.head(3)


train.info()


X = train.drop('diagnosed_diabetes', axis='columns')
y = train['diagnosed_diabetes']

from sklearn.preprocessing import MinMaxScaler

cols_to_scale = X.select_dtypes(['int64', 'float64']).columns

scaler = MinMaxScaler()

X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])
X.describe()


from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(data):
    vif_df = pd.DataFrame()
    vif_df['Column'] = data.columns
    vif_df['VIF'] = [variance_inflation_factor(data.values,i) for i in range(data.shape[1])]
    return vif_df


calculate_vif(X[cols_to_scale])


def calculate_woe_iv(df, feature, target):
    grouped = df.groupby(feature)[target].agg(['count','sum'])
    grouped = grouped.rename(columns={'count': 'total', 'sum': 'good'})
    grouped['bad']=grouped['total']-grouped['good']
    
    total_good = grouped['good'].sum()
    total_bad = grouped['bad'].sum()
    
    grouped['good_pct'] = grouped['good'] / total_good
    grouped['bad_pct'] = grouped['bad'] / total_bad
    grouped['woe'] = np.log(grouped['good_pct']/ grouped['bad_pct'])
    grouped['iv'] = (grouped['good_pct'] -grouped['bad_pct'])*grouped['woe']
    
    grouped['woe'] = grouped['woe'].replace([np.inf, -np.inf], 0)
    grouped['iv'] = grouped['iv'].replace([np.inf, -np.inf], 0)
    
    total_iv = grouped['iv'].sum()
    
    return grouped, total_iv


iv_values = {}

for feature in X.columns:
    if X[feature].dtype == 'object':
        _, iv = calculate_woe_iv(pd.concat([X, y],axis=1), feature, 'diagnosed_diabetes' )
    else:
        X_binned = pd.cut(X[feature], bins=10, labels=False)
        _, iv = calculate_woe_iv(pd.concat([X_binned, y],axis=1), feature, 'diagnosed_diabetes' )
    iv_values[feature] = iv
        
iv_values


def interpret_iv(iv):
    if iv < 0.02:
        return 'Not useful'
    elif iv < 0.1:
        return 'Weak'
    elif iv < 0.3:
        return 'Medium'
    elif iv < 0.5:
        return 'Strong'
    else:
        return 'Suspiciously Predictive'

# Create summary
for feature, iv in iv_values.items():
    print(f"{feature:20} | IV = {iv:.2f} | {interpret_iv(iv)}")


train['education_level'] = train['education_level'].map({
    'No formal': 0,
    'Highschool': 1,
    'Graduate': 2,
    'Postgraduate': 3
})
test['education_level'] = test['education_level'].map({
    'No formal': 0,
    'Highschool': 1,
    'Graduate': 2,
    'Postgraduate': 3
})

# Income Level (clear progression)
income_order = {'Low': 0, 'Lower-Middle': 1, 'Middle': 2, 'Upper-Middle': 3, 'High': 4}
train['income_level'] = train['income_level'].map(income_order)
test['income_level'] = test['income_level'].map(income_order)

# Smoking Status (health impact order)
smoking_order = {'Never': 0, 'Former': 1, 'Current': 2}
train['smoking_status'] = train['smoking_status'].map(smoking_order)
test['smoking_status'] = test['smoking_status'].map(smoking_order)


# Gender - One-Hot (3 categories, no order)
train = pd.get_dummies(train, columns=['gender'], prefix='gender', drop_first=False)
test = pd.get_dummies(test, columns=['gender'], prefix='gender', drop_first=False)

# Ethnicity - One-Hot (5 categories, no order)
train = pd.get_dummies(train, columns=['ethnicity'], prefix='ethnicity', drop_first=False)
test = pd.get_dummies(test, columns=['ethnicity'], prefix='ethnicity', drop_first=False)

# Employment Status - One-Hot (4 categories, no clear order)
train = pd.get_dummies(train, columns=['employment_status'], prefix='employment', drop_first=False)
test = pd.get_dummies(test, columns=['employment_status'], prefix='employment', drop_first=False)


train = train.drop(columns=["id"])
test = test.drop(columns=["id"])


X = train.drop(columns=['diagnosed_diabetes'])
y = train['diagnosed_diabetes']

X_test = test[X.columns]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)


train.info()


scale_pos_weight = y_train.value_counts()[0.0] / y_train.value_counts()[1.0]


def objective(trial):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "scale_pos_weight": scale_pos_weight,  # XGBoost's way of handling class imbalance
        "n_estimators": trial.suggest_int("n_estimators", 300, 5000),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "max_leaves": trial.suggest_int("max_leaves", 16, 256),  # XGBoost equivalent to num_leaves
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 200),  # Similar to min_child_samples
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "random_state": 42,
        "tree_method": "hist",  # Faster training
        "verbosity": 0
    }
    
    # Yes, you can change n_splits - common values: 3, 5, 10
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Changed to 5
    auc_scores = []
    
    for train_idx, valid_idx in skf.split(X_train, y_train):
        X_tr, X_va = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[valid_idx]
        
        model = XGBClassifier(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False
        )
        
        preds = model.predict_proba(X_va)[:, 1]
        auc_scores.append(roc_auc_score(y_va, preds))
    
    return np.mean(auc_scores)


best_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "scale_pos_weight": scale_pos_weight,
    "tree_method": "gpu_hist",  # GPU acceleration
    "gpu_id": 0,  # GPU device ID
    "enable_categorical": True,
    "n_estimators": 4133,
    "learning_rate": 0.02068171689337402,
    "max_leaves": 28,  # XGBoost equivalent to num_leaves
    "max_depth": 5,
    "min_child_weight": 24,  # Equivalent to min_child_samples
    "subsample": 0.7412048932901383,
    "colsample_bytree": 0.965256994834274,
    "reg_alpha": 2.5659237069355467,
    "reg_lambda": 1.028840143860832,
    "verbosity": 0,
    "random_state": 42  # Add for reproducibility
}

# Train the model
#model = XGBClassifier(**best_params)
#model.fit(X_train, y_train)


from sklearn.metrics import roc_auc_score, roc_curve
from xgboost import XGBClassifier
import numpy as np
import matplotlib.pyplot as plt

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test))
fold_aucs = []

plt.figure(figsize=(12, 8))

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
    X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]
    
    model = XGBClassifier(**best_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False  # XGBoost uses verbose instead of eval_metric in fit
    )
    
    y_val_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_val_pred)
    fold_aucs.append(auc)
    print(f"Fold {fold} AUC: {auc:.6f}")
    
    # Accumulate test predictions
    test_preds += model.predict_proba(X_test)[:, 1] / skf.n_splits
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_val, y_val_pred)
    plt.plot(fpr, tpr, label=f'Fold {fold} (AUC={auc:.3f})')

print("\nMean AUC:", np.mean(fold_aucs))
print("Std AUC:", np.std(fold_aucs))

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('XGBoost ROC Curves per Fold')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.show()


submission_df = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')


submission = pd.DataFrame({
    'id': submission_df.id,  
    'prediction': test_preds
})

submission.to_csv('submission.csv', index=False)

