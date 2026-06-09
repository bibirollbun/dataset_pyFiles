# # Step 1: Import necessary libraries
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# from xgboost import XGBRegressor
# import matplotlib.pyplot as plt

# # Step 2: Load and explore the data
# print("Loading data...")
# data = pd.read_csv('/kaggle/input/california-homelessness-prediction-challenge/train.csv')

# print(f"Data shape: {data.shape}")
# print("\nFirst 5 rows:")
# print(data.head())

# print("\nColumn names:")
# print(data.columns.tolist())

# print("\nBasic statistics:")
# print(data.describe())

# # Step 3: Prepare the data for training
# # Separate features (X) and target variable (y)
# X = data.drop(columns=['ID', 'HOMELESS_RATE'])  # Features
# y = data['HOMELESS_RATE']  # Target variable

# print(f"\nFeatures shape: {X.shape}")
# print(f"Target variable shape: {y.shape}")

# # Step 4: Split data into training and validation sets
# X_train, X_val, y_train, y_val = train_test_split(
#     X, y, 
#     test_size=0.2, 
#     random_state=42
# )

# print(f"\nTraining set: {X_train.shape[0]} samples")
# print(f"Validation set: {X_val.shape[0]} samples")

# # Step 5: Initialize and train the XGBoost model
# print("\nTraining XGBoost model...")
# model = XGBRegressor(
#     n_estimators=1000,        # Number of trees
#     learning_rate=0.05,       # Step size shrinkage
#     max_depth=6,              # Maximum tree depth
#     subsample=0.8,            # Fraction of samples used for each tree
#     colsample_bytree=0.8,     # Fraction of features used for each tree
#     random_state=42,          # For reproducibility
#     early_stopping_rounds=50, # Stop if no improvement for 50 rounds
#     eval_metric='rmse'        # Evaluation metric
# )

# # Train the model
# model.fit(
#     X_train, y_train,
#     eval_set=[(X_val, y_val)],
#     verbose=100  # Print progress every 100 trees
# )

# # Step 6: Evaluate the model
# print("\nEvaluating model...")
# # Predict on validation set
# val_predictions = model.predict(X_val)

# # Calculate Mean Squared Error (competition metric)
# mse = mean_squared_error(y_val, val_predictions)
# print(f"Validation Mean Squared Error: {mse:.6f}")

# # Calculate RMSE for easier interpretation
# rmse = np.sqrt(mse)
# print(f"Validation Root Mean Squared Error: {rmse:.6f}")

# # Step 7: Feature importance analysis
# print("\nTop 20 most important features:")
# feature_importance = pd.DataFrame({
#     'feature': X.columns,
#     'importance': model.feature_importances_
# })
# feature_importance = feature_importance.sort_values('importance', ascending=False)

# print(feature_importance.head(20))

# # Plot feature importance
# plt.figure(figsize=(12, 8))
# plt.barh(feature_importance['feature'][:20], feature_importance['importance'][:20])
# plt.xlabel('Importance')
# plt.title('Top 20 Feature Importance')
# plt.gca().invert_yaxis()  # Most important at top
# plt.tight_layout()
# plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
# plt.show()

# # Step 8: Prepare for submission (you'll need test.csv from Kaggle)
# print("\nTo make submission predictions:")
# print("1. Download test.csv from Kaggle")
# print("2. Use the following code:")
# print("""
# test_data = pd.read_csv('test.csv')
# test_ids = test_data['ID']
# X_test = test_data.drop(columns=['ID'])
# test_predictions = model.predict(X_test)

# submission = pd.DataFrame({
#     'ID': test_ids,
#     'HOMELESS_RATE': test_predictions
# })
# submission.to_csv('submission.csv', index=False)
# print("Submission file created!")
# """)

# print("\nTraining completed! Check feature_importance.png to see which factors")
# print("are most correlated with homelessness rates in your model.")


# # Let's try some hyperparameter tuning and feature engineering
# from sklearn.model_selection import GridSearchCV

# # Create some new features that might be meaningful
# X_engineered = X.copy()

# # Create age group ratios that might be meaningful
# X_engineered['SENIOR_TO_YOUNG_RATIO'] = (X['AGE_60_61_PCT'] + X['AGE_55_59_PCT']) / (X['AGE_25_34_PCT'] + 1e-6)
# X_engineered['FAMILY_DENSITY'] = X['FAMILY_HH_TOTAL'] / X['TOTAL_HOUSEHOLDS_PCT']

# # Try a more optimized parameter set
# optimized_model = XGBRegressor(
#     n_estimators=500,
#     learning_rate=0.03,
#     max_depth=4,
#     subsample=0.7,
#     colsample_bytree=0.7,
#     random_state=42
# )

# # Train the optimized model
# X_train_eng, X_val_eng, y_train, y_val = train_test_split(
#     X_engineered, y, test_size=0.2, random_state=42
# )

# optimized_model.fit(X_train_eng, y_train)
# val_pred_optimized = optimized_model.predict(X_val_eng)

# print(f"Optimized Model MSE: {mean_squared_error(y_val, val_pred_optimized):.8f}")
# print(f"Optimized Model RMSE: {np.sqrt(mean_squared_error(y_val, val_pred_optimized)):.6f}")



# Improved XGBoost Pipeline
# ==========================

# 1ï¸�âƒ£ Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# 2ï¸�âƒ£ Set file paths (Kaggle environment)
TRAIN_PATH = '/kaggle/input/california-homelessness-prediction-challenge/train.csv'
TEST_PATH = '/kaggle/input/california-homelessness-prediction-challenge/test.csv'
SAMPLE_SUB_PATH = '/kaggle/input/california-homelessness-prediction-challenge/sample_submission.csv'

# 3ï¸�âƒ£ Load data
train_data = pd.read_csv(TRAIN_PATH)
test_data = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

print(f"Train data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")
print(f"Sample submission shape: {sample_sub.shape}")

# 4ï¸�âƒ£ Quick data exploration
print("\nMissing values in train data:")
print(train_data.isnull().sum())
print("\nTrain data columns:", train_data.columns.tolist())

# 5ï¸�âƒ£ Prepare features and target
X = train_data.drop(columns=['ID', 'HOMELESS_RATE'])
y = train_data['HOMELESS_RATE']

# Optional: handle categorical variables (if any)
# X = pd.get_dummies(X, drop_first=True)
# test_data_encoded = pd.get_dummies(test_data.drop(columns=['ID']), drop_first=True)
# Ensure same columns
# X_test = test_data_encoded.reindex(columns=X.columns, fill_value=0)

X_test = test_data.drop(columns=['ID'])

# 6ï¸�âƒ£ Train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)
print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")

# 7ï¸�âƒ£ Train XGBoost model
model = XGBRegressor(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='rmse',
    tree_method='hist'  # Use CPU
)


# Early stopping with validation set
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=100
)

# 8ï¸�âƒ£ Evaluate model
val_preds = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_preds))
print(f"\nValidation RMSE: {rmse:.6f}")

# 9ï¸�âƒ£ Feature importance
feat_imp = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values(by='importance', ascending=False)

print("\nTop 10 features:")
print(feat_imp.head(10))

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feat_imp.head(20))
plt.title("Top 20 Feature Importances")
plt.tight_layout()
plt.show()

# ğŸ”Ÿ Make predictions on test data
test_preds = model.predict(X_test)

# 1ï¸�âƒ£1ï¸�âƒ£ Create submission file
submission = pd.DataFrame({
    'ID': test_data['ID'],
    'HOMELESS_RATE': test_preds
})

submission_path = '/kaggle/working/submission.csv'
submission.to_csv(submission_path, index=False)
print(f"\nSubmission saved to: {submission_path}")

# 1ï¸�âƒ£2ï¸�âƒ£ Preview submission
print("\nSubmission preview:")
print(submission.head())
print("\nâœ… Pipeline completed successfully!")


