# Install AutoGluon (if not installed)
!pip install autogluon --quiet


# Import required libraries
import pandas as pd
import time
import os
from autogluon.tabular import TabularPredictor

# Track start time
start_time = time.time()

# Load datasets
train_df = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
extra_train_df = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")

# Combine train datasets
train_df = pd.concat([train_df, extra_train_df], ignore_index=True)

# Drop ID column from train set (AutoGluon handles it automatically)
test_ids = test_df["id"]
train_df.drop(columns=["id"], inplace=True)
test_df.drop(columns=["id"], inplace=True)

# Define target variable
target_col = "Price"

# Train AutoGluon Model
predictor = TabularPredictor(label=target_col, eval_metric="root_mean_squared_error").fit(
    train_data=train_df,
    time_limit=10 * 60 * 60,  # 10 hours
    presets="best_quality",  # Best quality models (Auto-stacking enabled)
    num_stack_levels=3,  # Increase stack levels for better performance
    num_bag_folds=5,  # Cross-validation
    ag_args_fit={"num_gpus": 1}  # Use GPU
)

predictor


# Get leaderboard (Top 10 models)
leaderboard = predictor.leaderboard(silent=True)
top_10_models = leaderboard.head(10)

# Print Top 10 Model Details
print("\nðŸ”· Top 10 Models from AutoGluon:")
print(top_10_models)


# Save leaderboard to CSV
top_10_models.to_csv("autogluon_leaderboard.csv", index=False)

# Generate predictions using the best model
preds = predictor.predict(test_df)

# Save submission file
submission = pd.DataFrame({"id": test_ids, "Price": preds})
submission.to_csv("submission_autogluon.csv", index=False)

