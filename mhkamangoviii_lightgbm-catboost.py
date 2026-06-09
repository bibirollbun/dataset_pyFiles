import numpy as np
import pandas as pd
import joblib
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Load Test Data
test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")

# Preprocess Test Data 
def feature_engineering(df):
    df['donor_age_diff'] = df['donor_age'] - df['age_at_hct']
    df['score_interaction'] = df['comorbidity_score'] * df['karnofsky_score']
    df['age_group'] = df['age_at_hct'] // 10
    df['dri_score_NA'] = df['dri_score'].apply(lambda x: int('N/A' in str(x)))
    return df

test = feature_engineering(test)

# Encode categorical features (Same as Training)
for col in test.select_dtypes(include=['object', 'category']).columns:
    test[col] = test[col].astype('category').cat.codes

# Load Trained Models
lgb_models, cat_models = joblib.load("/kaggle/input/dataset-model/ensemble_models.pkl")

# Ensure Test Data Has the Same Features as Training
train_features = set(lgb_models[0].feature_name_) 
test_features = set(test.columns)

missing_in_test = train_features - test_features
extra_in_test = test_features - train_features

# Add missing columns with default values
for col in missing_in_test:
    if col == "weight":
        test[col] = 1.0 
    else:
        test[col] = 0


# Ensure column order matches training
test = test[list(train_features)]

# Make Predictions
lgb_preds = np.mean([model.predict(test) for model in lgb_models], axis=0)
cat_preds = np.mean([model.predict(test) for model in cat_models], axis=0)

# Weighted Ensemble
final_preds = 0.725 * lgb_preds + 0.275 * cat_preds

# Save Submission File
submission = pd.DataFrame({
    "ID": test["ID"],  # Ensure test dataset has 'ID' column
    "prediction": final_preds
})
submission.to_csv("submission.csv", index=False)

print("Inference complete! Submission file saved as submission.csv")




