import pandas as pd
import numpy as np
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

TARGET_COL = "diagnosed_diabetes"
ID_COL = "id"
SEED = 42

df_train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

# Mapping
genderMap = {"Female": 0, "Male": 1, "Other": 2}
ethnicityMap = {"Hispanic": 0 , "White": 1, "Asian": 2, "Black": 3, "Other": 4}
eduLvlMap = {"Highschool": 0, "Graduate":1 , "Postgraduate": 2, "No formal": 3}
incomeLvlMap = {"Lower-Middle": 0 , "Upper-Middle":1 , "Low":2 , "Middle": 3, "High": 4}
smokingMap = {"Never": 0, "Current": 1, "Former": 2}
employmentMap = {"Employed": 0, "Retired": 1, "Student": 2, "Unemployed": 3}

def process_data(df):
    df = df.copy()
    if 'gender' in df.columns: df["gender"] = df["gender"].map(genderMap).fillna(-1)
    if 'ethnicity' in df.columns: df["ethnicity"] = df["ethnicity"].map(ethnicityMap).fillna(-1)
    if 'education_level' in df.columns: df["education_level"] = df["education_level"].map(eduLvlMap).fillna(-1)
    if 'income_level' in df.columns: df["income_level"] = df["income_level"].map(incomeLvlMap).fillna(-1)
    if 'employment_status' in df.columns: df["employment_status"] = df["employment_status"].map(employmentMap).fillna(-1)
    if 'smoking_status' in df.columns: df["smoking_status"] = df["smoking_status"].map(smokingMap).fillna(-1)
    
    bool_cols = df.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        df[col] = df[col].astype(int)

    if 'systolic_bp' in df.columns and 'diastolic_bp' in df.columns:
        df['MAP'] = (df['systolic_bp'] + 2 * df['diastolic_bp']) / 3 
        df['Pulse_Pressure'] = df['systolic_bp'] - df['diastolic_bp']
        
    if 'cholesterol_total' in df.columns and 'hdl_cholesterol' in df.columns:
        df["Non_HDL"] = df["cholesterol_total"] - df["hdl_cholesterol"]
        df["Chol_HDL_Ratio"] = df["cholesterol_total"] / (df["hdl_cholesterol"] + 1e-5)
        
    if 'bmi' in df.columns and 'waist_to_hip_ratio' in df.columns:
        df["Metabolic_Index"] = df["bmi"] * df["waist_to_hip_ratio"]
        
    if 'age' in df.columns and 'bmi' in df.columns:
        df['Age_BMI_Interact'] = df['age'] * df['bmi']

    return df

df_train = process_data(df_train)
df_test = process_data(df_test)

features = [c for c in df_train.columns if c not in [ID_COL, TARGET_COL]]
print(f"Features len: {len(features)}")
print(f"NaN in train: {df_train[features].isna().sum().sum()}")

X = df_train[features]
y = df_train[TARGET_COL]
X_test = df_test[features]

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

print("\n--- Training XGBoost ---")
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

xgb_model = xgb.XGBClassifier(
    n_estimators=3000,
    learning_rate=0.01,
    max_depth=8,
    subsample=0.7,
    colsample_bytree=0.7,
    scale_pos_weight=scale_pos_weight,
    random_state=SEED,
    n_jobs=-1,
    early_stopping_rounds=100,
    tree_method='hist'
)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
xgb_preds = xgb_model.predict_proba(X_val)[:, 1]
print(f"XGBoost AUC: {roc_auc_score(y_val, xgb_preds):.5f}")

print("\n--- Training LightGBM ---")
lgb_model = LGBMClassifier(
    n_estimators=3000,
    learning_rate=0.01,
    num_leaves=80,
    class_weight='balanced',
    random_state=SEED,
    n_jobs=-1,
    verbose=-1 
)

from lightgbm import early_stopping
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='auc',
    callbacks=[early_stopping(stopping_rounds=100)]
)
lgb_preds = lgb_model.predict_proba(X_val)[:, 1]
print(f"LightGBM AUC: {roc_auc_score(y_val, lgb_preds):.5f}")

ensemble_preds = (xgb_preds + lgb_preds) / 2
print(f"\n>>> ENSEMBLE FINAL AUC: {roc_auc_score(y_val, ensemble_preds):.5f} <<<")

print("Generazione file submission...")
xgb_test = xgb_model.predict_proba(X_test)[:, 1]
lgb_test = lgb_model.predict_proba(X_test)[:, 1]
final_test = (xgb_test + lgb_test) / 2

submission = pd.DataFrame({
    ID_COL: df_test[ID_COL],
    TARGET_COL: final_test
})
submission.to_csv("submission.csv", index=False)


submission.head()




