# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from joblib import Parallel, delayed

### Load Data ###
print("Loading data...")

def load_data():
    sales_train = pd.read_csv(
        "/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv",
        parse_dates=["date"]
    )
    sales_test = pd.read_csv(
        "/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv",
        parse_dates=["date"]
    )
    inventory = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/inventory.csv")
    calendar = pd.read_csv(
        "/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv",
        parse_dates=["date"]
    )
    test_weights = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/test_weights.csv")

    # Filter sales_train to only keep data from the last two years
    last_date = sales_train["date"].max()
    two_years_ago = last_date - pd.DateOffset(years=2)
    sales_train = sales_train[sales_train["date"] >= two_years_ago]

    return sales_train, sales_test, inventory, calendar, test_weights

sales_train, sales_test, inventory, calendar, test_weights = load_data()
print("Data loaded.")

### Reduce Memory Usage (if needed) ###
def reduce_memory_usage(df):
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and col_type != "datetime64[ns]":
            if str(col_type).startswith("int"):
                df[col] = pd.to_numeric(df[col], downcast="integer")
            elif str(col_type).startswith("float"):
                df[col] = pd.to_numeric(df[col], downcast="float")
    return df

sales_train = reduce_memory_usage(sales_train)
sales_test = reduce_memory_usage(sales_test)
inventory = reduce_memory_usage(inventory)
calendar = reduce_memory_usage(calendar)
test_weights = reduce_memory_usage(test_weights)
print("Memory reduction complete.")

### Ensure All Dates with Zero Sales for Missing Dates ###
print("\nFilling missing dates with zero sales for each (unique_id, warehouse) pair...")

def ensure_all_dates(data, calendar, is_train):
    # Get unique (unique_id, warehouse) combinations from the data
    unique_pairs = data[["unique_id", "warehouse"]].drop_duplicates()
    # Get a sorted array of all unique dates from the calendar
    all_dates = pd.DataFrame({"date": np.sort(calendar["date"].unique())})
    
    # Create a cross-join between unique pairs and all dates
    unique_pairs["key"] = 1
    all_dates["key"] = 1
    full_date_range = unique_pairs.merge(all_dates, on="key").drop("key", axis=1)
    
    # Merge the full date range with the original data on unique_id, warehouse, and date
    data = full_date_range.merge(data, on=["unique_id", "warehouse", "date"], how="left")
    
    # For training data, fill missing sales with zero.
    if is_train:
        data["sales"].fillna(0, inplace=True)
    
    # Forward-fill sell_price_main and fill discount columns with 0.
    data["sell_price_main"].fillna(method="ffill", inplace=True)
    discount_cols = [col for col in data.columns if "discount" in col]
    data[discount_cols] = data[discount_cols].fillna(0)
    
    # Sort for consistency
    data = data.sort_values(["unique_id", "warehouse", "date"])
    return data

# Apply to training and test data (for test, we do not have a real sales column)
sales_train = ensure_all_dates(sales_train, calendar, is_train=True)

# For sales_test, add a placeholder sales column so that lags can be computed later if needed.
sales_test["sales"] = np.nan  
sales_test = ensure_all_dates(sales_test, calendar, is_train=False)
sales_test.drop(columns=["sales"], inplace=True)  # Remove the placeholder

print("Missing dates filled.")

### Merge Datasets ###
print("\nMerging sales with calendar on 'date' and 'warehouse'...")
def merge_data(sales, calendar):
    return sales.merge(calendar, on=["date", "warehouse"], how="left")

sales_train = merge_data(sales_train, calendar)
sales_test = merge_data(sales_test, calendar)
print("Merge complete.")

### Feature Engineering ###
print("\nCreating features...")

# Date-based features
for df in [sales_train, sales_test]:
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["is_weekend"] = df["day_of_week"].isin([5,6]).astype(int)

# Discount features (if discount columns exist)
discount_cols = [col for col in sales_train.columns if "discount" in col]
if discount_cols:
    sales_train["max_discount"] = sales_train[discount_cols].max(axis=1)
    sales_test["max_discount"] = sales_test[discount_cols].max(axis=1)
    sales_train["price_discount_ratio"] = sales_train["sell_price_main"] / (1 - sales_train["max_discount"]).replace({0: 1})
    sales_test["price_discount_ratio"] = sales_test["sell_price_main"] / (1 - sales_test["max_discount"]).replace({0: 1})
    print("Discount features added.")
else:
    print("No discount columns found.")

# Adding Lag Features (1-day, 7-day, 30-day) by unique_id and warehouse
print("\nAdding lag features (1-day, 7-day, 30-day)...")
sales_train = sales_train.sort_values(["unique_id", "warehouse", "date"])
for lag in [1, 7, 30]:
    sales_train[f"sales_lag_{lag}"] = sales_train.groupby(["unique_id", "warehouse"])["sales"].shift(lag)
sales_train.fillna(0, inplace=True)
print("Lag features added in sales_train.")
# (Optional: similar lag features can be generated for sales_test if desired using a similar approach.)

### FFT Feature Extraction ###
print("\nExtracting FFT features (by unique_id and warehouse)...")
from scipy.fft import rfft
from joblib import Parallel, delayed

def compute_fft_for_series(unique_id, warehouse, sales_series):
    # Compute the FFT for the given sales series (ensure it's not empty)
    if len(sales_series) == 0:
        return {"unique_id": unique_id, "warehouse": warehouse,
                "fft_dominant_freq": np.nan,
                "fft_top_freq_1": np.nan,
                "fft_top_freq_2": np.nan,
                "fft_top_freq_3": np.nan,
                "fft_energy_top": np.nan}
    
    fft_result = rfft(sales_series)
    fft_magnitudes = np.abs(fft_result)
    
    # Exclude the DC component (index 0)
    if len(fft_magnitudes) <= 1:
        return {"unique_id": unique_id, "warehouse": warehouse,
                "fft_dominant_freq": np.nan,
                "fft_top_freq_1": np.nan,
                "fft_top_freq_2": np.nan,
                "fft_top_freq_3": np.nan,
                "fft_energy_top": np.nan}
    
    # Get dominant frequency index (offset by 1 because we skip DC)
    dominant_idx = np.argmax(fft_magnitudes[1:]) + 1
    dominant_freq = dominant_idx
    
    # Sort indices (excluding DC) in descending order by magnitude
    sorted_indices = np.argsort(-fft_magnitudes[1:]) + 1
    # Ensure we have at least 3 frequencies (if not, fill with NaN)
    top_freqs = sorted_indices[:3]
    fft_top_freq_1 = top_freqs[0] if len(top_freqs) >= 1 else np.nan
    fft_top_freq_2 = top_freqs[1] if len(top_freqs) >= 2 else np.nan
    fft_top_freq_3 = top_freqs[2] if len(top_freqs) >= 3 else np.nan
    
    # Compute total energy (excluding DC component)
    total_energy = np.sum(fft_magnitudes[1:] ** 2)
    # Energy ratio for the dominant frequency
    energy_top = (fft_magnitudes[dominant_idx] ** 2) / total_energy if total_energy > 0 else np.nan

    return {"unique_id": unique_id,
            "warehouse": warehouse,
            "fft_dominant_freq": dominant_freq,
            "fft_top_freq_1": fft_top_freq_1,
            "fft_top_freq_2": fft_top_freq_2,
            "fft_top_freq_3": fft_top_freq_3,
            "fft_energy_top": energy_top}

# Apply FFT feature extraction in parallel grouped by unique_id and warehouse
fft_features = Parallel(n_jobs=-1)(
    delayed(compute_fft_for_series)(uid, wh, group["sales"].fillna(0).values)
    for (uid, wh), group in sales_train.groupby(["unique_id", "warehouse"])
)
fft_features_df = pd.DataFrame(fft_features)

# Merge FFT features into both training and test data
sales_train = sales_train.merge(fft_features_df, on=["unique_id", "warehouse"], how="left")
sales_test = sales_test.merge(fft_features_df, on=["unique_id", "warehouse"], how="left")
print("FFT features merged.")

# For debugging, show a sample of the new FFT features
print("\nSample FFT features:")
print(fft_features_df.head())

print("\nSample of sales_train after feature engineering:")
print(sales_train.head())

### Data Preprocessing ###
print("\nPreprocessing data...")

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Define features for preprocessing
numerical_cols = ["sell_price_main", "total_orders", "max_discount", "fft_dominant_freq","fft_top_freq_1",
                  "sales_lag_1", "sales_lag_7", "sales_lag_30","fft_top_freq_2","fft_top_freq_3", "fft_energy_top"]
categorical_cols = ["warehouse"]

num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])
cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, numerical_cols),
        ("cat", cat_transformer, categorical_cols)
    ]
)

# Save original test dataframe (for submission info) before dropping columns
test_processed = sales_test.copy()

# Drop columns not needed for modeling
X_train = sales_train.drop(columns=["sales", "date", "unique_id", "availability"])
y_train = sales_train["sales"]
X_test = sales_test.drop(columns=["date", "unique_id"])
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("Preprocessing complete.")
print("X_train_processed shape:", X_train_processed.shape)
print("X_test_processed shape:", X_test_processed.shape)

### Model Training ###
print("\nTraining XGBoost Model...")

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

X_train_split, X_val, y_train_split, y_val = train_test_split(X_train_processed, y_train, test_size=0.2, random_state=42)
print("Training and validation splits created.")

xgb_model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=300,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    n_jobs=-1
)
print("Fitting the XGBoost model...")
xgb_model.fit(X_train_split, y_train_split)
print("Model training complete.")

y_val_pred = xgb_model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
print(f"Validation RMSE: {rmse:.4f}")

y_test_pred = xgb_model.predict(X_test_processed)
print("Test set predictions generated.")

### Create Submission File ###
print("\nGenerating final submission file...")
# We'll use the saved test_processed DataFrame (which has unique_id and date)
submission_df = test_processed[["unique_id", "date"]].copy()
submission_df["sales_hat"] = y_test_pred

# If predictions are misaligned or missing, align to the test file
sales_test = pd.read_csv(
        "/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv",
        parse_dates=["date"])
submission_df = sales_test.merge(submission_df, on=["unique_id", "date"], how="left")


submission_df["id"] = submission_df["unique_id"].astype(str) + "_" + submission_df["date"].dt.strftime('%Y-%m-%d')
submission_df = submission_df[["id", "sales_hat"]]

submission_file = "submission.csv"
submission_df.to_csv(submission_file, index=False)
print(f"Final submission file saved as {submission_file}")




print(submission_df.shape)

