import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


# Load and prepare the dataset
df_train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')

# Remove ID columns as they are not needed for prediction
df_train.drop('id', axis=1, inplace=True)
df_test.drop('id', axis=1, inplace=True)

# Separate features and target variable
features = df_train.drop('BeatsPerMinute', axis=1)
target = df_train['BeatsPerMinute']


# Standardize the features for better model performance
feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(features)
test_scaled = feature_scaler.transform(df_test)


# Create validation split for model evaluation
X_train, X_val, y_train, y_val = train_test_split(X_scaled, target, test_size=0.2, random_state=42)



# Define optimized hyperparameters for ensemble models
lightgbm_config = {
    'colsample_bytree': 1.0,
    'learning_rate': 0.01,
    'max_depth': 15,
    'min_child_samples': 50,
    'n_estimators': 200,
    'num_leaves': 31,
    'reg_alpha': 0.1,
    'reg_lambda': 0.5,
    'subsample': 0.6
}

catboost_config = {
    'depth': 11,
    'learning_rate': 0.007886860469042982,
    'l2_leaf_reg': 1,
    'bagging_temperature': 2.0,
    'random_strength': 0.1,
    'border_count': 64,
    'min_data_in_leaf': 10,
    'grow_policy': 'Depthwise'
}



# Build ensemble model using stacking approach
model_1 = LGBMRegressor(**lightgbm_config, random_state=42)
model_2 = CatBoostRegressor(**catboost_config, random_state=42, verbose=0)

# Create the stacking ensemble
ensemble_models = [
    ('lightgbm_regressor', model_1),
    ('catboost_regressor', model_2)
]

ensemble_stack = StackingRegressor(
    estimators=ensemble_models,
    final_estimator=Ridge(alpha=1.0),
    cv=5,
    n_jobs=-1,
    passthrough=False
)

# Train the ensemble model on full dataset
print("Training ensemble model...")
ensemble_stack.fit(X_scaled, target)

# Evaluate on validation set
print("Generating predictions...")
validation_pred = ensemble_stack.predict(X_val)

# Calculate performance metric
rmse_score = np.sqrt(mean_squared_error(validation_pred, y_val))
print(f"Validation RMSE: {rmse_score:.4f}")



# Generate predictions for test set
test_predictions = ensemble_stack.predict(test_scaled)

# Prepare submission file
submission_df = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")
submission_df["BeatsPerMinute"] = test_predictions
submission_df.to_csv("submission.csv", index=False)

print("Submission file created successfully!")




