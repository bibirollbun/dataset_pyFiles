import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')


# Load data
train = pd.read_csv('/kaggle/input/um-game-playing-strength-of-mcts-variants/train.csv')
test = pd.read_csv('/kaggle/input/um-game-playing-strength-of-mcts-variants/test.csv')


print(len(train.columns))
print(len(test.columns))


set(train.columns) - set(test.columns)


# Define target and columns to remove
TARGET = 'utility_agent1'
drop_cols = ['num_draws_agent1', 'num_losses_agent1', 'num_wins_agent1', TARGET]
y = train[TARGET]
X = train.drop(columns=drop_cols)
X_test = test.copy()


col_types = []
for columns in train.columns:
    col_types.append(train[columns].dtype)

np.unique(col_types)


for columns in train.columns:
    if train[columns].dtype == 'O':
        print(columns)


# Categorical columns to encode
cat_cols = ['GameRulesetName', 'agent1', 'agent2', 'EnglishRules', 'LudRules']


# Robust encoding (handles unseen categories)
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X[cat_cols] = encoder.fit_transform(X[cat_cols])
X_test[cat_cols] = encoder.transform(X_test[cat_cols])

# Align columns (train and test)
X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)

# Split train into train/validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define LightGBM regressor
model = XGBRegressor(
        n_estimators=1000, learning_rate=0.05, max_depth=6,
        tree_method='gpu_hist', predictor='gpu_predictor', verbosity=0)

# Train model with early stopping
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=0
)

# Evaluate RMSE
val_pred = model.predict(X_val)
rmse = mean_squared_error(y_val, val_pred, squared=False)
print(f'Validation RMSE: {rmse:.4f}')

# Feature importance
importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Features:")
print(importance.head(10))

# Generate test predictions
test_pred = model.predict(X_test)

# Create submission file
submission = pd.DataFrame({'Id': test['Id'], 'utility_agent1': test_pred})
submission.to_csv('submission.csv', index=False)
print("\nPredictions saved to submission.csv")


# Compare predictions vs actual values
comparison_df = pd.DataFrame({
    'Actual': y_val.values,
    'Predicted': val_pred
})

# Preview first few rows
print("\nValidation Predictions vs Actual:")
print(comparison_df.head(10))


pd.read_csv('/kaggle/working/submission.csv')

