!pip install autogluon.tabular


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from autogluon.tabular import TabularPredictor

# --- 1. Load your dataset ---
df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")  

# --- 2. Separate target and features ---
target_col = "y"  
y = df[target_col]
X = df.drop(columns=[target_col])

# --- 3. Identify categorical columns ---
cat_cols = X.select_dtypes(include="object").columns.tolist()

# --- 4. Split into training and validation sets ---
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 5. Combine features and target for AutoGluon ---
train_data = pd.concat([X_train, y_train], axis=1)
val_data = pd.concat([X_val, y_val], axis=1)

# --- 6. Define and train the AutoGluon model ---
predictor = TabularPredictor(
    label=target_col,
    problem_type='binary',
    eval_metric='roc_auc'
).fit(
    train_data=train_data,
    tuning_data=val_data,  # Validation data for early stopping and model selection
    time_limit=600,  # 10 minute time limit (in seconds)
    presets='medium_quality',  # Balance between quality and speed
    verbosity=2  # Detailed logging
)

# --- 7. Evaluation ---
y_pred_proba = predictor.predict_proba(val_data, as_multiclass=False)
y_pred = predictor.predict(val_data)

print("ROC AUC:", roc_auc_score(y_val, y_pred_proba))
print(classification_report(y_val, y_pred))

# Check model performance on leaderboard
leaderboard = predictor.leaderboard(val_data)
print(leaderboard)

# --- 8. Optional: Predict on test data ---
test_df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
test_pred_proba = predictor.predict_proba(test_df, as_multiclass=False)

submit = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')
submit['y'] = test_pred_proba
submit.to_csv('submission.csv', index=False)
display(submit)

