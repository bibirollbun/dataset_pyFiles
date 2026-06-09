# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from catboost import CatBoostClassifier



train_df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')

# Keep raw copies (GOOD PRACTICE)
train_raw = train_df.copy()
test_raw = test_df.copy()

print("DATASET SHAPES:")
print(f"  Training set: {train_df.shape}")
print(f"  Test set: {test_df.shape}")
print(f"  Sample submission: {sample_submission.shape}")


print("DATASET INFO:")
print(train_df.info())

print("DATASET DESCRIPTION:")
print(train_df.info())


print("\nTARGET VARIABLE DISTRIBUTION:")
target_counts = train_df['diagnosed_diabetes'].value_counts()
target_percentage = train_df['diagnosed_diabetes'].value_counts(normalize=True) * 100

print(f"  Class 0 (No Diabetes): {target_counts[0]:,} ({target_percentage[0]:.2f}%)")
print(f"  Class 1 (Diabetes): {target_counts[1]:,} ({target_percentage[1]:.2f}%)")


#Seperating Numerical and categorical values
numerical_features = train_df.select_dtypes(include=['int64','float64']).columns.tolist()
numerical_features.remove('id')

if 'diagnosed_diabetes' in numerical_features:
    numerical_features.remove('diagnosed_diabetes')

categorical_features = train_df.select_dtypes(include=['object']).columns.tolist()

#CORRELATION
print("Correlation of features with target variable")
correlations = train_df[numerical_features + ['diagnosed_diabetes']].corr()['diagnosed_diabetes'].sort_values(ascending=False)

print(correlations)

top_features = correlations.drop('diagnosed_diabetes').abs().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
colors = ['green' if correlations[feat] > 0 else 'red' for feat in top_features.index]
plt.barh(range(len(top_features)), [correlations[feat] for feat in top_features.index], color=colors)
plt.yticks(range(len(top_features)), top_features.index)
plt.xlabel('Correlation with Diabetes')
plt.title('Top 10 Features Correlated with Diabetes', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

print("\nTOP 10 PREDICTIVE FEATURES:")
for i, (feature, corr_value) in enumerate(top_features.items(), 1):
    direction = "positive" if correlations[feature] > 0 else "negative"
    print(f"  {i:2d}. {feature:40s}: {correlations[feature]:7.4f} ({direction})")


import math
import matplotlib.pyplot as plt
import seaborn as sns

features = [
    'age',
    'bmi',
    'systolic_bp',
    'ldl_cholesterol',
    'triglycerides',
    'cholesterol_total',
    'waist_to_hip_ratio',
    'hdl_cholesterol'
]

n_features = len(features)
n_cols = 2
n_rows = math.ceil(n_features / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
axes = axes.flatten()

for i, col in enumerate(features):
    sns.boxplot(
        x=train_df[col].sample(5000, random_state=42),
        ax=axes[i]
    )
    axes[i].set_title(col)

# Remove unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()



print("OUTLIER DETECTION (IQR Method):\n")
outlier_summary = {}
for col in numerical_features:
    Q1 = train_df[col].quantile(0.25)
    Q3 = train_df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = train_df[(train_df[col] < lower_bound) | (train_df[col] > upper_bound)]
    outlier_percentage = (len(outliers) / len(train_df)) * 100
    if outlier_percentage > 1:
        outlier_summary[col] = outlier_percentage
        print(f"   {col:40s}: {outlier_percentage:5.2f}% outliers")

if not outlier_summary:
    print(" No significant outliers detected (< 1% threshold)")


def feature_engineering(df):
    df = df.copy()

    # Risk score
    df['metabolic_risk_score'] = (
        (df['bmi'] > 30).astype(int) +
        (df['systolic_bp'] > 130).astype(int) +
        (df['ldl_cholesterol'] > 130).astype(int) +
        (df['triglycerides'] > 150).astype(int)
    )

    # Age-normalized
    df['bmi_age_ratio'] = df['bmi'] / df['age']
    df['bp_age_ratio'] = df['systolic_bp'] / df['age']

    # Lifestyle
    df['activity_screen_ratio'] = (
        df['physical_activity_minutes_per_week'] /
        (df['screen_time_hours_per_day'] * 7 + 1)
    )

    df['sleep_deviation'] = abs(df['sleep_hours_per_day'] - 7)

    # BMI category
    df['bmi_category'] = df['bmi'].apply(
        lambda bmi: 'underweight' if bmi < 18.5 else
                    'normal' if bmi < 25 else
                    'overweight' if bmi < 30 else
                    'obese'
    )

    # Blood pressure category
    df['bp_category'] = pd.cut(
        df['systolic_bp'],
        bins=[0, 120, 130, 140, 300],
        labels=['normal', 'elevated', 'stage1', 'stage2']
    )

    # Combined history
    df['any_cardiac_history'] = (
        df['hypertension_history'] |
        df['cardiovascular_history']
    ).astype(int)

    # Interactions
    df['age_activity_interaction'] = (
        df['age'] * df['physical_activity_minutes_per_week']
    )

    df['smoking_bmi_interaction'] = (
        df['bmi'] * (df['smoking_status'] == 'Current').astype(int)
    )

    return df


train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)


X = train_df.drop(['id', 'diagnosed_diabetes'], axis=1)
y = train_df['diagnosed_diabetes']

X_test = test_df.drop(['id'], axis=1)
test_ids = test_df['id']


from pandas.api.types import is_numeric_dtype

for col in X.columns:
    if is_numeric_dtype(X[col]):
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)



X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)



cat_features = [
    'gender',
    'ethnicity',
    'education_level',
    'income_level',
    'smoking_status',
    'employment_status',
    'bmi_category',
    'bp_category'
]

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np
from catboost import CatBoostClassifier

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cat_oof = np.zeros(len(X))
cat_test_preds = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nðŸ”¥ CatBoost Fold {fold+1}")

    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    model = CatBoostClassifier(
        iterations=800,
        depth=6,
        learning_rate=0.03,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=42,
        verbose=200
    )

    model.fit(
        X_tr, y_tr,
        cat_features=cat_features,
        eval_set=(X_val, y_val),
        use_best_model=True
    )

    cat_oof[val_idx] = model.predict_proba(X_val)[:, 1]
    cat_test_preds += model.predict_proba(X_test)[:, 1] / skf.n_splits

print("âœ… CatBoost CV AUC:", roc_auc_score(y, cat_oof))



val_preds = model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, val_preds)
print(f"\nValidation AUC: {val_auc:.5f}")



model.fit(
    X,
    y,
    cat_features=cat_features,
    verbose=200
)



catboost_test_preds = model.predict_proba(X_test)[:, 1]

submission_cb = sample_submission.copy()
submission_cb['diagnosed_diabetes'] = catboost_test_preds
submission_cb.to_csv('submission_catboost.csv', index=False)

print("CatBoost submission saved")



submission_cb.head()


 categorical_cols = [
    'gender',
    'ethnicity',
    'education_level',
    'income_level',
    'smoking_status',
    'employment_status',
    'bmi_category',
    'bp_category'
]



X = train_df.drop(['id', 'diagnosed_diabetes'], axis=1)
y = train_df['diagnosed_diabetes']
X_test = test_df.drop(['id'], axis=1)



from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)



from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import numpy as np

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nFold {fold+1}")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    X_train_enc = pd.get_dummies(
        X_train,
        columns=categorical_cols,
        drop_first=True
    )

    X_val_enc = pd.get_dummies(
        X_val,
        columns=categorical_cols,
        drop_first=True
    )

    X_test_enc = pd.get_dummies(
        X_test,
        columns=categorical_cols,
        drop_first=True
    )

    X_train_enc, X_val_enc = X_train_enc.align(
        X_val_enc, join='left', axis=1, fill_value=0
    )
    X_train_enc, X_test_enc = X_train_enc.align(
        X_test_enc, join='left', axis=1, fill_value=0
    )

    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='auc',
        tree_method='hist',
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_enc, y_train)

    val_pred = model.predict_proba(X_val_enc)[:, 1]
    oof_preds[val_idx] = val_pred

    test_preds += model.predict_proba(X_test_enc)[:, 1] / skf.n_splits

    auc = roc_auc_score(y_val, val_pred)
    print(f"Fold AUC: {auc:.5f}")



cv_auc = roc_auc_score(y, oof_preds)
print(f"\nXGBoost CV AUC: {cv_auc:.5f}")



xgb_test_preds = test_preds.copy()

submission_xgb = sample_submission.copy()
submission_xgb['diagnosed_diabetes'] = xgb_test_preds
submission_xgb.to_csv('submission_xgboost.csv', index=False)

print("âœ… XGBoost submission saved")




final_preds = (
    0.7 * catboost_test_preds +
    0.3 * xgb_test_preds
)



final_submission = sample_submission.copy()
final_submission['diagnosed_diabetes'] = final_preds

final_submission.to_csv('submission.csv', index=False)

print("FINAL submission saved as submission.csv")



from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import numpy as np



categorical_cols = [
    'gender',
    'ethnicity',
    'education_level',
    'income_level',
    'smoking_status',
    'employment_status',
    'bmi_category',
    'bp_category'
]



X = train_df.drop(['id', 'diagnosed_diabetes'], axis=1)
y = train_df['diagnosed_diabetes']

X_test = test_df.drop(['id'], axis=1)



skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)



lgb_oof_preds = np.zeros(len(X))
lgb_test_preds = np.zeros(len(X_test))



for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nðŸŒ¿ LightGBM Fold {fold+1}")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    X_train_enc = pd.get_dummies(
        X_train,
        columns=categorical_cols,
        drop_first=True
    )

    X_val_enc = pd.get_dummies(
        X_val,
        columns=categorical_cols,
        drop_first=True
    )

    X_test_enc = pd.get_dummies(
        X_test,
        columns=categorical_cols,
        drop_first=True
    )

    X_train_enc, X_val_enc = X_train_enc.align(
        X_val_enc, axis=1, fill_value=0
    )
    X_train_enc, X_test_enc = X_train_enc.align(
        X_test_enc, axis=1, fill_value=0
    )

    train_data = lgb.Dataset(X_train_enc, label=y_train)
    val_data = lgb.Dataset(X_val_enc, label=y_val)

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbosity': -1,
        'seed': 42
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(100)]
    )

    val_preds = model.predict(
        X_val_enc,
        num_iteration=model.best_iteration
    )

    lgb_oof_preds[val_idx] = val_preds

    lgb_test_preds += model.predict(
        X_test_enc,
        num_iteration=model.best_iteration
    ) / skf.n_splits

    fold_auc = roc_auc_score(y_val, val_preds)
    print(f"Fold AUC: {fold_auc:.5f}")



lgb_cv_auc = roc_auc_score(y, lgb_oof_preds)
print(f"\nâœ… LightGBM CV AUC: {lgb_cv_auc:.5f}")


