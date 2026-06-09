import pandas as pd
import os

# File paths
train_q_path = "/kaggle/input/widsdatathon2025/TRAIN_OLD/TRAIN_QUANTITATIVE_METADATA.xlsx"
train_c_path = "/kaggle/input/widsdatathon2025/TRAIN_OLD/TRAIN_CATEGORICAL_METADATA.xlsx"
test_q_path = "/kaggle/input/widsdatathon2025/TEST/TEST_QUANTITATIVE_METADATA.xlsx"
test_c_path = "/kaggle/input/widsdatathon2025/TEST/TEST_CATEGORICAL.xlsx"
labels_path = "/kaggle/input/widsdatathon2025/TRAIN_OLD/TRAINING_SOLUTIONS.xlsx"
sample_submission_path = "/kaggle/input/widsdatathon2025/SAMPLE_SUBMISSION.xlsx"

# Load data
train_q = pd.read_excel(train_q_path)
train_c = pd.read_excel(train_c_path)
test_q = pd.read_excel(test_q_path)
test_c = pd.read_excel(test_c_path)
labels = pd.read_excel(labels_path).set_index("participant_id")

# Merge datasets
train_combined = pd.merge(train_q, train_c, on="participant_id", how="left").set_index("participant_id")
test_combined = pd.merge(test_q, test_c, on="participant_id", how="left").set_index("participant_id")

# Display data info
train_combined.info(), test_combined.info()



from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

# Impute missing values using median strategy
imputer = SimpleImputer(strategy="median")
train_imputed = pd.DataFrame(imputer.fit_transform(train_combined), columns=train_combined.columns, index=train_combined.index)
test_imputed = pd.DataFrame(imputer.transform(test_combined), columns=test_combined.columns, index=test_combined.index)

# Scale features with RobustScaler
scaler = RobustScaler()
train_scaled = pd.DataFrame(scaler.fit_transform(train_imputed), columns=train_imputed.columns, index=train_imputed.index)
test_scaled = pd.DataFrame(scaler.transform(test_imputed), columns=test_imputed.columns, index=test_imputed.index)

# Check preprocessed data
train_scaled.info(), test_scaled.info()



from xgboost import XGBClassifier

# Define target variables
y_adhd = labels["ADHD_Outcome"]
y_sex = labels["Sex_F"]

# Train XGBoost model for ADHD Outcome prediction
xgb_model = XGBClassifier(
    n_estimators=50, max_depth=3, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
    random_state=42, use_label_encoder=False, eval_metric="logloss", verbosity=0
)
xgb_model.fit(train_scaled, y_adhd)
test_predictions = xgb_model.predict_proba(test_scaled)[:, 1]

# Train XGBoost model for Sex_F prediction
xgb_sex_model = XGBClassifier(
    n_estimators=50, max_depth=3, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
    random_state=42, use_label_encoder=False, eval_metric="logloss", verbosity=0
)
xgb_sex_model.fit(train_scaled, y_sex)
test_sex_predictions = xgb_sex_model.predict_proba(test_scaled)[:, 1]

# Prepare submission file
sample_submission = pd.read_excel(sample_submission_path)
submission = pd.DataFrame({
    "participant_id": test_combined.index,
    "ADHD_Outcome": test_predictions,
    "Sex_F": test_sex_predictions
})
submission = submission[sample_submission.columns]

# Save submission
submission_file_path = "/kaggle/working/submission_raw.csv"
submission.to_csv(submission_file_path, index=False)
submission.head()




# Convert probabilities to binary values (0 or 1) using a threshold of 0.5
submission["ADHD_Outcome"] = (submission["ADHD_Outcome"] >= 0.5).astype(int)
submission["Sex_F"] = (submission["Sex_F"] >= 0.5).astype(int)

# Save the corrected submission file
corrected_submission_file_path = "/kaggle/working/submission.csv"
submission.to_csv(corrected_submission_file_path, index=False)
submission.head()


