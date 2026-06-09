# Basic imports
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
import warnings
warnings.filterwarnings("ignore")


# Load the datasets
train_df = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)


train_df.isna().sum()


test_df.isna().sum()


# Define the target and features.
target = "Listening_Time_minutes"
# Exclude id and target from features
features = [col for col in train_df.columns if col not in ["id", target]]


# Basic preprocessing:
# Fill missing values
train_df.fillna(-999, inplace=True)
test_df.fillna(-999, inplace=True)



# Convert object type features to category (if any)
for col in train_df.select_dtypes(include=["object"]).columns:
    train_df[col] = train_df[col].astype("category")
    if col in test_df.columns:
        test_df[col] = test_df[col].astype("category")

# Identify categorical features for LightGBM
cat_features = [col for col in features if train_df[col].dtype.name == "category"]


# Split the training data into training and validation sets
X = train_df[features]
y = train_df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Prepare LightGBM datasets
lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
lgb_eval = lgb.Dataset(X_test, label=y_test, categorical_feature=cat_features, reference=lgb_train)


# Set model parameters
# params = {
#     "objective": "regression",
#     "metric": "rmse",
#     "boosting_type": "gbdt",
#     "learning_rate": 0.05,
#     "num_leaves": 31,
#     "seed": 42,
#     "verbose": -1,
# }


# IN-DEPTH PARAMETER FOR FUTURE TUNING 
params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,         # Lower learning rate for better convergence
    "num_leaves": 31,              # More leaves for capturing complex patterns
    "max_depth": 8,               # Deeper trees for more flexibility
    "min_data_in_leaf": 20,        # Avoid overfitting with too small leaf size
    "feature_fraction": 0.9,       # Randomly select features per iteration
    "bagging_fraction": 0.8,       # Randomly sample data per iteration
    "bagging_freq": 5,             # Frequency of bagging
    "lambda_l1": 1.0,              # L1 regularization
    "lambda_l2": 1.0,              # L2 regularization
    "verbosity": -1,
    "seed": 42,
    "n_jobs": -1,
}



# Train the model with early stopping using callbacks
print("Training LightGBM model...")
# model = lgb.train(params, train_set, num_boost_round=3000, 
#                   valid_sets=[val_set], early_stopping_rounds=200)

model = lgb.train(
    params,
    lgb_train,
    num_boost_round=1800,
    valid_sets=[lgb_train, lgb_eval],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)


# Evaluate the model on the validation set
val_preds = model.predict(X_test, num_iteration=model.best_iteration)
y_pred = model.predict(X_test)
# rmse = mean_squared_error(y_test, y_pred, squared=False)
# r2 = r2_score(y_test, y_pred)


rmse = np.sqrt(mean_squared_error(y_test, val_preds))
r2 = r2_score(y_test, val_preds)

print(f"Validation RMSE: {rmse:.4f}")
print(f'Final Model R² Score: {r2}')






# Generate predictions for the test set.
# Ensure that the test set features match the training set.
test_features = [col for col in test_df.columns if col != "id"]
test_preds = model.predict(test_df[test_features], num_iteration=model.best_iteration)


# Create submission DataFrame
submission = pd.DataFrame({
    "id": test_df["id"],
    "Listening_Time_minutes": test_preds})

print(submission.shape)


# Save submission to a CSV file
submission.to_csv("submission.csv", index=False)
print("Submission Saved!")

