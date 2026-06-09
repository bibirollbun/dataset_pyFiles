import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler



df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
target = df.columns.tolist()[-1]
print(df.shape)
df.head()


def create_frequency_features(train_df, test_df, cols, num, cat):
    """
    Add frequency and binning features to the dataset.
    - For each column, create <col>_freq = how often each value appears in train data.
    - For numeric columns, split values into 5 and 10 quantile bins (groups) to show rank or range.
    """
    
    # Make copies of the input DataFrames to prevent modifying the originals
    train, test = train_df.copy(), test_df.copy()
    
    # Iterate over each column specified in 'cols'
    for col in cols:
        # Compute the normalized frequency (proportion) of each unique value in the training data for the current column
        freq = train[col].value_counts(normalize=True)
        
        # Map the computed frequencies to the corresponding values in the training DataFrame
        train[f"{col}_freq"] = train[col].map(freq)
        
        # Map the frequencies to the test DataFrame and fill any missing values with the mean frequency from the training data
        test[f"{col}_freq"] = test[col].map(freq).fillna(train[f"{col}_freq"].mean())
        
        # Check if the current column is a numeric feature; if so, apply binning
        if col in num:
            # Loop over the specified quantile values for binning (adjusted to match docstring: 5 and 10)
            for q in [5, 10]:
                try:
                    # Use qcut to bin the training data into 'q' quantiles, returning bin labels and boundaries; drop duplicate bins if any
                    train[f"{col}_bin{q}"], bins = pd.qcut(train[col], q=q, labels=False, retbins=True, duplicates="drop")
                    
                    # Apply the same bin boundaries to the test data using cut, ensuring the lowest value is included
                    test[f"{col}_bin{q}"] = pd.cut(test[col], bins=bins, labels=False, include_lowest=True)
                except Exception:
                    # If binning fails (e.g., due to too few unique values), assign 0 to the bin column in both DataFrames
                    train[f"{col}_bin{q}"] = 0
                    test[f"{col}_bin{q}"] = 0
    
    # Generate an updated list of numeric columns by excluding categorical columns and the target variable from the training DataFrame's columns
    new_num = train.drop(columns=cat + [target]).columns.tolist()
    
    # Return the modified training and test DataFrames along with the updated list of numeric features
    return train, test, new_num


# Identify feature
cols = df.drop(columns=target).columns.tolist()

# Categorical features
cat = [col for col in cols if df[col].dtype in ["object","category"] and col != target]

# Numerical features
num = [col for col in cols if df[col].dtype not in ["object","category","bool"] and col not in ["id", target]]

# Creating new features based on the frequency of numerical features
df, df_test, new_num = create_frequency_features(df, df_test.copy(), cols, num, cat)

# Preparing categorical features
df[cat], df_test[cat] = df[cat].astype("category"), df_test[cat].astype("category")

# Mapping a column
map_col = "num_reported_accidents"
map_num_reported = {0:0, 1:0, 2:0, 3:2, 4:4, 5:3, 6:1, 7:0}
df[map_col] = df[map_col].map(map_num_reported)
df_test[map_col] = df_test[map_col].map(map_num_reported)

# Dropping unnecessary columns
remove = ["time_of_day", "num_lanes", "road_type", "road_signs_present", "id_freq"]
df = df.drop(columns=remove)
df_test = df_test.drop(columns=remove)

# Dropping ID and duplicates
df.drop(columns="id", inplace=True)
df.drop_duplicates(inplace=True)


print(df.columns.tolist())


df.head()


# --- Prepare DMatrix for XGBoost ---
X = df.drop(columns=target)
y = df[target]

dtrain = xgb.DMatrix(
    data=X,
    label=y,
    enable_categorical=True
)

# --- Define XGBoost parameters ---
xgb_params = {
    "max_depth": 11,
    "learning_rate": 0.011,
    "subsample": 0.82,
    "colsample_bytree": 0.81,
    "min_child_weight": 3,
    "gamma": 0.011,
    "reg_alpha": 0.12,
    "reg_lambda": 0.4,
    "max_delta_step": 1,
    "colsample_bylevel": 0.86,
    "colsample_bynode": 0.88,
    "scale_pos_weight": 0.36,
    "max_bin": 512,
    "tree_method": "hist",
    "device": "cuda",
    "eval_metric": "rmse",
    "random_state": 42,
}

# --- Run Cross-Validation ---
cv_results = xgb.cv(
    params=xgb_params,
    dtrain=dtrain,
    nfold=5,
    num_boost_round=2000,
    metrics="rmse",
    early_stopping_rounds=50,
    verbose_eval=100,
    seed=42
)

# --- Display results ---
print("\nğŸ“Š Last few CV results:")
print(cv_results.tail())

# --- Extract best boosting round ---
best_round = cv_results["test-rmse-mean"].idxmin()
best_rmse = cv_results.loc[best_round, "test-rmse-mean"]

print(f"\nâœ… Best round: {best_round}")
print(f"ğŸ�† Best CV RMSE: {best_rmse:.7f}")



# putting the n_estimator at the average early stopping point to avoid overfitting
last_round = len(cv_results) - 1
xgb_params["n_estimators"] = last_round + 10


# Final Training and Submission Section

# Split the training data into features and target variable
X_train = df.drop(columns=target)

# Assign the target column to y_train
y_train = df[target]

# Detect numerical columns that require scaling (integer and float types)
numerical_cols = [col for col in X_train.columns if X_train[col].dtype in ['int64', 'float64']]

# Create an instance of StandardScaler for feature normalization
scaler = StandardScaler()

# Fit the scaler on the training numerical features and transform them
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])

# Transform the corresponding numerical features in the test data using the fitted scaler
df_test[numerical_cols] = scaler.transform(df_test[numerical_cols])

# Instantiate the XGBoost regressor with the specified parameters and support for categorical variables
model = XGBRegressor(**xgb_params, enable_categorical=True)

# Train the model using the scaled training features and target
model.fit(X_train, y_train)

# Perform predictions on the test features (excluding the 'id' column)
pred = model.predict(df_test.drop(columns="id"))

# Construct the submission DataFrame with 'id' and the predicted values
sub = pd.DataFrame({
    "id": df_test["id"],
    target: pred
})


# Write the submission DataFrame to a CSV file, excluding the index column



# Write the submission DataFrame to a CSV file, excluding the index column
sub.to_csv("submission.csv", index=False)


import os

# Check if the file exists in the working directory
if os.path.exists("submission.csv"):
    print("âœ… submission.csv has been created successfully!")
    print(f"File size: {os.path.getsize('submission.csv') / (1024 * 1024):.2f} MB")  # Size in MB
else:
    print("â�Œ submission.csv was not created. Check for errors above.")

# Optional: Preview the first few rows of the file
import pandas as pd
try:
    sub_preview = pd.read_csv("submission.csv")
    print("\nPreview of submission.csv:")
    print(sub_preview.head())
    print(f"Total rows: {len(sub_preview)}")  # Should match test set rows (~345k for this comp)
except FileNotFoundError:
    print("File not foundâ€”re-run the submission cell.")

