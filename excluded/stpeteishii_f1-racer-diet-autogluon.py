!pip install autogluon.tabular


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from autogluon.tabular import TabularPredictor

# --- 1. Load your dataset ---
df = pd.read_csv("/kaggle/input/f-1-racer-diet-planning/train.csv")  

# --- 2. Separate target and features ---
target_col = "Calories"  # Continuous target variable
y = df[target_col]
X = df.drop(columns=[target_col])

# --- 3. Identify categorical columns ---
cat_cols = X.select_dtypes(include="object").columns.tolist()

# --- 4. Split into training and validation sets ---
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
    # Removed stratify=y for regression (not applicable)
)

# --- 5. Combine features and target for AutoGluon ---
train_data = pd.concat([X_train, y_train], axis=1)
val_data = pd.concat([X_val, y_val], axis=1)

# --- 6. Define and train the AutoGluon model ---
predictor = TabularPredictor(
    label=target_col,
    problem_type='regression',  # Changed to regression
    eval_metric='root_mean_squared_error'  # Changed to regression metric
).fit(
    train_data=train_data,
    tuning_data=val_data,  # Validation data for early stopping and model selection
    time_limit=600,  # 10 minute time limit (in seconds)
    presets='medium_quality',  # Balance between quality and speed
    verbosity=2  # Detailed logging
)

# --- 7. Make predictions and evaluate ---
y_pred = predictor.predict(val_data)
print(f"RMSE: {np.sqrt(mean_squared_error(y_val, y_pred))}")
print(f"MAE: {mean_absolute_error(y_val, y_pred)}")
print(f"R²: {r2_score(y_val, y_pred)}")

# --- 7. Check model performance on leaderboard ---
leaderboard = predictor.leaderboard(val_data)
print(leaderboard)

# --- 8. Make predictions on validation set for evaluation ---
y_pred = predictor.predict(val_data.drop(columns=[target_col]))
print("\nValidation Set Performance:")
print(f"RMSE: {np.sqrt(mean_squared_error(y_val, y_pred))}")
print(f"MAE: {mean_absolute_error(y_val, y_pred)}")
print(f"R²: {r2_score(y_val, y_pred)}")

# --- 9. Predict on test data ---
test_df = pd.read_csv("/kaggle/input/f-1-racer-diet-planning/test.csv")

# For regression, use predict() instead of predict_proba()
test_predictions = predictor.predict(test_df)

# Prepare submission
submit = pd.read_csv('/kaggle/input/f-1-racer-diet-planning/sample_submission.csv')

# Assuming the target column in submission is 'Calories'
# Update based on actual column name in sample_submission.csv
submit['Calories'] = test_predictions  # or use the actual target column name

submit.to_csv('submission.csv', index=False)
print("\nSubmission preview:")
display(submit.head())




