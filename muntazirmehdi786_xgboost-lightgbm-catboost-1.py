import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import os

# Load Test Data
test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")

# Ensure model directory exists
MODEL_DIR = "/kaggle/input/my-new-dataset-model"

# Feature Engineering (Same as Training)
def add_features(df):
    df['donor_age_hct_diff'] = df['donor_age'] - df['age_at_hct']
    df['comorbidity_karnofsky_ratio'] = df['comorbidity_score'] / (df['karnofsky_score'] + 1)
    df['year_hct_adjusted'] = df['year_hct'] - 2000
    df['is_cyto_score_same'] = (df['cyto_score'] == df['cyto_score_detail']).astype(int)
    return df

test = add_features(test)

# Encode categorical features (Same as Training)
categorical_cols = test.select_dtypes(include=['object', 'category']).columns
for col in categorical_cols:
    test[col] = test[col].astype('category').cat.codes  # Convert to integer encoding

FEATURES = [col for col in test.columns if col not in ["ID"]]

# Load Models & Make Predictions
final_preds = np.zeros(len(test))

for fold in range(15):
    print(f"Loading Models for Fold {fold}")

    model_xgb = joblib.load(f"{MODEL_DIR}/xgb_fold{fold}.pkl")
    model_lgb = joblib.load(f"{MODEL_DIR}/lgb_fold{fold}.pkl")
    model_cat = cb.CatBoostRegressor()
    model_cat.load_model(f"{MODEL_DIR}/cat_fold{fold}.cbm")

    final_preds += model_xgb.predict(test[FEATURES]) * 0.4 / 15
    final_preds += model_lgb.predict(test[FEATURES]) * 0.4 / 15
    final_preds += model_cat.predict(test[FEATURES]) * 0.2 / 15

# Save Submission File
submission = pd.DataFrame({"ID": test["ID"], "prediction": final_preds})
submission.to_csv("submission.csv", index=False)
print("Inference complete. Submission saved.")




