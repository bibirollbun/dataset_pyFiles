import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt


df_train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
df_test_raw = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')


def perform_feature_engineering(data):
    df_fe = data.copy()
    if 'id' in df_fe.columns: df_fe = df_fe.drop(columns=['id'])
    
    # LIPID PROFILE 
    df_fe['tg_hdl_ratio'] = df_fe['triglycerides'] / (df_fe['hdl_cholesterol'] + 1e-5)
    # Non-HDL Cholesterol 
    df_fe['non_hdl_cholesterol'] = df_fe['cholesterol_total'] - df_fe['hdl_cholesterol']
    # Cholesterol Ratio
    df_fe['total_hdl_ratio'] = df_fe['cholesterol_total'] / (df_fe['hdl_cholesterol'] + 1e-5)
    # CARDIOVASCULAR METRICS
    df_fe['pulse_pressure'] = df_fe['systolic_bp'] - df_fe['diastolic_bp']
    # Mean Arterial Pressure (MAP)
    df_fe['mean_arterial_pressure'] = (df_fe['systolic_bp'] + 2 * df_fe['diastolic_bp']) / 3
    # OBESITY & BODY COMPOSITION
    df_fe['body_fat_index'] = df_fe['bmi'] * df_fe['waist_to_hip_ratio']
    # Obesity (BMI > 30)
    df_fe['is_obese'] = (df_fe['bmi'] >= 30).astype(int)
    # LIFESTYLE DYNAMICS
    df_fe['activity_to_screen_ratio'] = (df_fe['physical_activity_minutes_per_week'] / 60) / (df_fe['screen_time_hours_per_day'] * 7 + 1e-5)
    # Total Lifestyle Score
    df_fe['health_score'] = df_fe['diet_score'] + (df_fe['sleep_hours_per_day'] / 2) + (df_fe['physical_activity_minutes_per_week'] / 100)
    # 6. RISK AGING
    df_fe['age_family_risk'] = df_fe['age'] * (df_fe['family_history_diabetes'] + 1)
    
    return df_fe

df_train_ready = perform_feature_engineering(df_train)
cat_cols = ['gender', 'ethnicity', 'education_level', 'income_level', 'smoking_status', 'employment_status']
df_final = pd.get_dummies(df_train_ready, columns=cat_cols, drop_first=True)

X = df_final.drop('diagnosed_diabetes', axis=1)
y = df_final['diagnosed_diabetes']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

df_test_ready = perform_feature_engineering(df_test_raw)
df_test_final = pd.get_dummies(df_test_ready, columns=cat_cols, drop_first=True)
df_test_final = df_test_final.reindex(columns=X_train.columns, fill_value=0)


model_xgb = XGBClassifier(
    n_estimators=2000,
    learning_rate=0.005, 
    max_depth=10,
    subsample=0.9,
    colsample_bytree=0.9,
    tree_method='hist',
    objective='binary:logistic', 
    eval_metric='auc',          
    early_stopping_rounds=100,    # Stop When AUC not improved
    n_jobs=-1
)

model_xgb.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=100
)



y_prob_val = model_xgb.predict_proba(X_val)[:, 1]

# AUC-ROC Evaluation
auc_score = roc_auc_score(y_val, y_prob_val)
print(f"\n AUC-ROC Score on Validation: {auc_score:.4f}")

fpr, tpr, thresholds = roc_curve(y_val, y_prob_val)
plt.plot(fpr, tpr, label=f'XGBoost (AUC = {auc_score:.4f})')
plt.plot([0, 1], [0, 1], 'k--') 
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


test_probabilities = model_xgb.predict_proba(df_test_final)[:, 1]


submission = pd.DataFrame({
    'id': df_test_raw['id'],
    'diagnosed_diabetes': test_probabilities # Ini adalah angka 0-1 (probabilitas)
})

submission.to_csv('submission_auc.csv', index=False)
print('done')

