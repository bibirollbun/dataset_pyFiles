# libraries
import numpy as np
import pandas as pd

# Define the weights dictionary
weights = {
    "/kaggle/input/predicting-road-accident-risk-vault/submission.csv": 1.2,  # strong vault base
    "/kaggle/input/predicting-road-accident-risk-vault/submission (1).csv": 0.6}  # variant}
print("done")


# # Normalize a weight map to sum to 1.0
# def normalize_weights(weight_map):
#     # Compute sum of weights
#     total = sum(weight_map.values())
#     # Validate total is non-zero
#     if total == 0:
#         # Raise an error for zero-sum weights
#         raise ValueError("Weights sum to zero.")
#     # Return normalized weights
#     return {k: v / total for k, v in weight_map.items()}

# # Infer the prediction column name
# def infer_prediction_column(df):
#     # Define candidate column names
#     candidates = ["accident_risk"]
#     # Return the first candidate that exists
#     for c in candidates:
#         if c in df.columns:
#             return c
#     # Fallback to first numeric column
#     numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
#     # Validate that a numeric column exists
#     if not numeric_cols:
#         # Raise error when nothing numeric is found
#         raise ValueError("No numeric columns available to infer predictions.")
#     # Return the first numeric column as a fallback
#     return numeric_cols[0]

# # Load a CSV and return the frame and its prediction column
# def load_csv(path):
#     # Read the CSV
#     df = pd.read_csv(path)
#     # Infer the prediction column
#     pred_col = infer_prediction_column(df)
#     # Return the frame and prediction column name
#     return df, pred_col

# # Minimal EDA just for submission columns
# def minimal_submission_eda(name, df, pred_col):
#     # Print file header
#     print(f"\n=== {name} ===")
#     # Print shape
#     print("Shape:", df.shape)
#     # Print prediction column
#     print("Prediction column:", pred_col)
#     # Print missing values for prediction
#     print("Missing in prediction:", df[pred_col].isna().sum())
#     # Print simple numeric stats for prediction
#     print(df[pred_col].describe())


import pandas as pd
import numpy as np

def normalize_weights(weight_map):
    """Normalize a weight dictionary to sum to 1.0 with validation."""
    total = sum(weight_map.values())
    if not total:
        raise ValueError("Weights sum to zero.")
    return {k: v / total for k, v in weight_map.items()}


def infer_prediction_column(df):
    """Infer prediction column name from known candidates or numeric columns."""
    candidates = ["accident_risk"]
    for c in candidates:
        if c in df.columns:
            return c
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if numeric_cols.empty:
        raise ValueError("No numeric columns found for predictions.")
    return numeric_cols[0]


def load_csv(path):
    """Load CSV and automatically detect prediction column."""
    df = pd.read_csv(path)
    return df, infer_prediction_column(df)


def minimal_submission_eda(name, df, pred_col):
    """Print concise submission diagnostics for quick validation."""
    pred = df[pred_col]
    print(f"\n=== {name} ===")
    print(f"Shape: {df.shape} | Prediction column: {pred_col}")
    print(f"Missing: {pred.isna().sum()} | Duplicates: {df.duplicated().sum()}")
    print(pred.describe().to_string())


# Normalize weights
norm_weights = normalize_weights(weights)

# Prepare containers
dfs = {}
pred_cols = {}
pred_series = {}

# Iterate through the files in the weight map
for path, w in norm_weights.items():
    # Load the CSV and infer the prediction column
    df, pred_col = load_csv(path)
    # Store the DataFrame
    dfs[path] = df
    # Store the prediction column name
    pred_cols[path] = pred_col
    # Store the prediction Series
    pred_series[path] = df[pred_col]

# Display first few rows for a quick glance
for path, df in dfs.items():
    # Show a small preview
    display(df.head(3))


# Run minimal EDA for each submission
for path, df in dfs.items():
    # Retrieve the prediction column for this file
    pred_col = pred_cols[path]
    # Execute minimal EDA
    minimal_submission_eda(path, df, pred_col)


# Initialize blended prediction as zeros of the first series shape for stability
first_path = next(iter(pred_series))
blended = pd.Series(0, index=pred_series[first_path].index, dtype=float)

# Accumulate weighted sums
for path, w in norm_weights.items():
    blended += pred_series[path].astype(float) * float(w)

# Normalize final blend to ensure valid probability range if applicable
if blended.min() < 0 or blended.max() > 1:
    blended = (blended - blended.min()) / (blended.max() - blended.min())

# Attach blended result to a base DataFrame
base_path = next(iter(dfs))
out_df = dfs[base_path].copy()
out_df["accident_risk"] = blended

# Display preview
print("Blending complete. Preview of output:")
display(out_df.head(10))

# # Initialize blended series
# blended = None

# # Iterate through paths and normalized weights
# for path, w in norm_weights.items():
#     # Retrieve the current prediction series
#     s = pred_series[path].astype(float)
#     # Initialize blended or accumulate weighted sum
#     if blended is None:
#         # Start with weighted base
#         blended = s * float(w)
#     else:
#         # Add weighted component
#         blended = blended + s * float(w)

# # Choose a base DataFrame to attach the blended column
# base_path = list(dfs.keys())[0]

# # Create a copy as the output DataFrame
# out_df = dfs[base_path].copy()

# # Assign the blended prediction
# out_df["accident_risk"] = blended

# # Show a small preview
# display(out_df.head(10))


# Define output path
output_path = "/kaggle/working/submission.csv"

# Save without index
out_df.to_csv(output_path, index=False)

# Print confirmation
print(f"Saved final: {output_path}")

