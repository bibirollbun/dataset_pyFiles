import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')
orig = pd.read_csv('/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv')


train.head()


TARGET = 'diagnosed_diabetes'

CATS = ['gender', 'ethnicity', 'education_level', 'income_level',
        'smoking_status', 'employment_status']

BASE = [col for col in train.columns if col not in ['id', TARGET]]


for col in BASE:
    # Check if the feature exists in the external dataset
    if col in orig.columns:
        # --- Mean Encoding ---
        mean_map = orig.groupby(col)[TARGET].mean().reset_index()
        new_mean_col_name = f"orig_mean_{col}"
        mean_map = mean_map.rename(columns={TARGET: new_mean_col_name})
        
        train = train.merge(mean_map, on=col, how='left')
        test = test.merge(mean_map, on=col, how='left')

        # --- Count Encoding ---
        new_count_col_name = f"orig_count_{col}"
        count_map = orig.groupby(col).size().reset_index(name=new_count_col_name)
        
        train = train.merge(count_map, on=col, how='left')
        test = test.merge(count_map, on=col, how='left')

print('External Mean/Count Encoding Complete.')


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


FEATURES = [col for col in train.columns if col not in ['id', TARGET]]
print(f'{len(FEATURES)} Total Features used for training.')


train.head()


X = train[FEATURES].copy()
y = train[TARGET]
X_test = test[FEATURES].copy()


final_cats = [c for c in CATS if c in X.columns]
for col in final_cats:
    X[col] = X[col].astype('category')
    X_test[col] = X_test[col].astype('category')


N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# Use 'params' to match variable name below
params = {
    'objective': 'binary',
    'n_estimators': 20000,
    'colsample_bytree': 0.47467103707530733,
    'colsample_bynode': 0.34677470146824885,
    'subsample': 0.8847606391870514,
    'learning_rate': 0.0075,
    'lambda_l1': 1.5417558006311665,
    'lambda_l2': 6.720810365899034,
    'max_depth': 6,
    'num_leaves': 769, 
    'min_data_in_leaf': 140,
    'metric': 'auc',
    'device': 'cpu', 
    'verbose': -1,
    'extra_trees': True
}

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test))
feature_importance_df = pd.DataFrame()

print("\nStarting LightGBM Training...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # CRITICAL FIX: Changed lgb_params to params
    model = LGBMClassifier(**params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[
            early_stopping(stopping_rounds=100, verbose=False),
            log_evaluation(period=1000)
        ]
    )

    val_preds = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_preds
    
    # Store Feature Importance
    fold_importance = pd.DataFrame()
    fold_importance["feature"] = FEATURES
    fold_importance["importance"] = model.feature_importances_
    fold_importance["fold"] = fold
    feature_importance_df = pd.concat([feature_importance_df, fold_importance], axis=0)
    
    fold_score = roc_auc_score(y_val, val_preds)
    print(f'--- Fold {fold}/{N_SPLITS} AUC: {fold_score:.5f} ---')
    
    test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS

overall_auc = roc_auc_score(y, oof_preds)
print(f'\n==================================')
print(f'Overall OOF AUC: {overall_auc:.5f}')
print(f'==================================')


plt.figure(figsize=(10, 6))
fpr, tpr, thresholds = roc_curve(y, oof_preds)
plt.plot(fpr, tpr, color='blue', label=f'Overall AUC = {overall_auc:.4f}')
plt.plot([0, 1], [0, 1], color='red', linestyle='--') 
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (LGBM OOF)')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()


plt.figure(figsize=(10, 6))
sns.histplot(test_preds, bins=50, kde=True, color='purple')
plt.title('Distribution of Test Predictions')
plt.xlabel('Predicted Probability')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)
plt.show()


plt.figure(figsize=(10, 8))
cols = (feature_importance_df[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:20].index)
best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]
sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
plt.title('Top 20 Feature Importance (Avg over folds)')
plt.tight_layout()
plt.show()


submission[TARGET] = test_preds
submission.to_csv('submission.csv', index=False)
print("Submission file saved successfully!")


submission_out.to_csv('submission_diabetes.csv', index=False)
print("Submission saved successfully.")


submission_out.head()

