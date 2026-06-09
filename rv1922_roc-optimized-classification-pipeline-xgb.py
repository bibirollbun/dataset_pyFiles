import pandas as pd
import numpy as np
import warnings
import time
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from xgboost import XGBClassifier


train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')
orig = pd.read_csv('/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv')


train.head()


TARGET = 'diagnosed_diabetes'

CATS = ['gender', 'ethnicity', 'education_level', 'income_level',
        'smoking_status', 'employment_status']

NUMS = ['age', 'alcohol_consumption_per_week', 'physical_activity_minutes_per_week', 
        'diet_score', 'sleep_hours_per_day', 'screen_time_hours_per_day', 'bmi',
        'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate',
        'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol',
        'triglycerides', 'family_history_diabetes', 'hypertension_history',
        'cardiovascular_history']


BASE = [col for col in train.columns if col not in ['id', TARGET]]
ORIG = [] 

BASE = [c for c in BASE if c in train.columns and c in orig.columns]


for col in BASE:
    # --- MEAN ENCODING ---
    if col in orig.columns:
        mean_map = orig.groupby(col)[TARGET].mean().reset_index()
        new_mean_col_name = f"orig_mean_{col}"
        
        # Rename the target column to the new feature name for merging
        mean_map = mean_map.rename(columns={TARGET: new_mean_col_name})
        
        train = train.merge(mean_map, on=col, how='left')
        test = test.merge(mean_map, on=col, how='left')
        ORIG.append(new_mean_col_name)

    if col in orig.columns:
        new_count_col_name = f"orig_count_{col}"
        count_map = orig.groupby(col).size().reset_index(name=new_count_col_name)
        
        train = train.merge(count_map, on=col, how='left')
        test = test.merge(count_map, on=col, how='left')
        ORIG.append(new_count_col_name)

print(f'{len(ORIG)} Orig Features Created!!')
FEATURES = BASE + ORIG
print(f'{len(FEATURES)} Total Features.')


def engineer_features(df):
    df = df.copy()

    df['alcohol_log'] = np.log1p(df['alcohol_consumption_per_week'])
    df['screen_time_log'] = np.log1p(df['screen_time_hours_per_day'])
    df['triglycerides_log'] = np.log1p(df['triglycerides'])

    df['age_sq'] = df['age'] ** 2
    df['bmi_sq'] = df['bmi'] ** 2
    df['whr_sq'] = df['waist_to_hip_ratio'] ** 2
    df['sbp_sq'] = df['systolic_bp'] ** 2

    df['age_bmi'] = df['age'] * df['bmi']
    df['htn_sbp'] = df['hypertension_history'] * df['systolic_bp']
    df['fh_trig'] = df['family_history_diabetes'] * df['triglycerides_log']

    df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
    df['mean_arterial_pressure'] = (df['systolic_bp'] + 2 * df['diastolic_bp']) / 3

    df['bmi_x_age'] = df['bmi'] * df['age']
    df['waist_x_bmi'] = df['waist_to_hip_ratio'] * df['bmi']

    df['chol_hdl_ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 1e-5)
    df['ldl_hdl_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + 1e-5)
    df['non_hdl_cholesterol'] = df['cholesterol_total'] - df['hdl_cholesterol']

    df['activity_x_diet'] = df['physical_activity_minutes_per_week'] * df['diet_score']

    return df


train = engineer_features(train)
test = engineer_features(test)


FEATURES = BASE + ORIG 
FEATURES = [f for f in FEATURES if f in train.columns]

print(len(FEATURES), "Total features.")


train.head()


X = train[FEATURES].copy()
y = train[TARGET]
X_test = test[FEATURES].copy()


for col in CATS:
    if col in X.columns:
        X[col] = X[col].astype('category')
        X_test[col] = X_test[col].astype('category')


%%time
N_SPLITS = 10
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 5,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'n_estimators': 10000,
    'learning_rate': 0.01,
    'early_stopping_rounds': 100,
    'random_state': 42,
    'n_jobs': -1,
    'device': 'cuda', 
    'enable_categorical': True,
}

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test))

print("\nStarting Training...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = XGBClassifier(**params)
    
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=1000)

    val_preds = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_preds
    
    fold_score = roc_auc_score(y_val, val_preds)
    print(f'--- Fold {fold}/{N_SPLITS} AUC: {fold_score:.5f} ---')
    
    test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS

overall_auc = roc_auc_score(y, oof_preds)
print(f'\n==================================')
print(f'Overall OOF AUC: {overall_auc:.5f}')
print(f'==================================')


plt.figure(figsize=(12, 5))

# Plot 1: ROC Curve (OOF)
plt.subplot(1, 2, 1)
fpr, tpr, thresholds = roc_curve(y, oof_preds)
plt.plot(fpr, tpr, color='blue', label=f'Overall AUC = {overall_auc:.4f}')
plt.plot([0, 1], [0, 1], color='red', linestyle='--') 
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (OOF Predictions)')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)

# Plot 2: Histogram of Test Predictions
plt.subplot(1, 2, 2)
sns.histplot(test_preds, bins=50, kde=True, color='green')
plt.title('Distribution of Test Predictions')
plt.xlabel('Predicted Probability (loan_paid_back)')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


submission = pd.DataFrame({
    "id": submission.id,
    "loan_paid_back": test_preds
})


submission.to_csv("submission.csv", index=False)
print("File Saved")


submission.head()

