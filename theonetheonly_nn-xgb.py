import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from sklearn.utils.random import check_random_state

# === Data Type Optimization ===
def optimize_dataframe(df):
    for col in df.columns:
        col_type = df[col].dtype
        if pd.api.types.is_numeric_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()
            if pd.api.types.is_integer_dtype(col_type):
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min >= np.finfo(np.float16).min and c_max <= np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    return df

# === Load datasets ===
train = pd.read_parquet("/kaggle/input/drw-crypto-market-prediction/train.parquet")
test = pd.read_parquet("/kaggle/input/drw-crypto-market-prediction/test.parquet")
submission = pd.read_csv("/kaggle/input/drw-crypto-market-prediction/sample_submission.csv")

# === Extract periodic time features BEFORE dropping timestamp (from index)
timestamp = pd.to_datetime(train.index)
dayth = timestamp.dayofyear
minute = timestamp.hour * 60 + timestamp.minute

# Encode sin/cos for daily and yearly periodicity
train["day_sin"] = np.sin(2 * np.pi * dayth / 365)
train["day_cos"] = np.cos(2 * np.pi * dayth / 365)
train["minute_sin"] = np.sin(2 * np.pi * minute / 1440)
train["minute_cos"] = np.cos(2 * np.pi * minute / 1440)

# Store target for timestamp prediction
y_time = train[["day_sin", "day_cos", "minute_sin", "minute_cos"]].copy()

# Drop timestamp only from test (train uses it as index)
test.drop(columns=["timestamp"], inplace=True, errors="ignore")


from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
import pandas as pd
import numpy as np

# === Ensure top_features is defined
top_features = pd.read_csv("/kaggle/input/shapfeature/shap_selected_features.csv")["feature"].tolist()

# === Inputs and targets for time prediction model
X_time = train[top_features]
y_time_target = y_time[["day_sin", "day_cos", "minute_sin", "minute_cos"]]

# === Split into train/val
X_tr, X_val, y_tr, y_val = train_test_split(X_time, y_time_target, test_size=0.2, random_state=42)

# === Confirm GPU usage
print("ğŸš€ Available devices:", tf.config.list_physical_devices())
print("ğŸ§  Using GPU:" if tf.config.list_physical_devices('GPU') else "âš ï¸� GPU not found, running on CPU")

# === Define the model
def build_time_predictor(input_dim):
    model = models.Sequential([
        layers.Dense(256, activation='relu', input_shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(4)  # Predict: day_sin, day_cos, minute_sin, minute_cos
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

model_time = build_time_predictor(X_tr.shape[1])

# === Train the model
model_time.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=20, batch_size=1024, verbose=1)

# === Evaluate performance
y_val_pred = model_time.predict(X_val)
mae = mean_absolute_error(y_val, y_val_pred)
cos_sim = cosine_similarity(y_val, y_val_pred).diagonal().mean()

print(f"\nğŸ“Š Time Feature Model Evaluation")
print(f"ğŸ”¹ MAE: {mae:.5f}")
print(f"ğŸ”¹ Avg. Cosine Similarity: {cos_sim:.5f}")

# === Predict time features for test set
test_time_pred = model_time.predict(test[top_features])

# === Append predicted features to test set
test["day_sin"] = test_time_pred[:, 0]
test["day_cos"] = test_time_pred[:, 1]
test["minute_sin"] = test_time_pred[:, 2]
test["minute_cos"] = test_time_pred[:, 3]


!pip install gplearn

import xgboost as xgb
from scipy.stats import pearsonr
from gplearn.genetic import SymbolicTransformer

# === Add time columns to feature set
all_features = top_features + ["day_sin", "day_cos", "minute_sin", "minute_cos"]

# === Redefine target (if lost)
y = train["label"]

# === Optimize types again
X = optimize_dataframe(train[all_features])
test = optimize_dataframe(test[all_features])

# === Clip to prevent symbolic overflow
X_clipped = X.clip(lower=-1e3, upper=1e3)
test_clipped = test.clip(lower=-1e3, upper=1e3)

# === gplearn symbolic transformation
function_set = ['add', 'sub', 'mul', 'sqrt', 'abs', 'neg']
gp = SymbolicTransformer(
    generations=10,
    population_size=1000,
    hall_of_fame=100,
    n_components=10,
    function_set=function_set,
    parsimony_coefficient=0.01,
    max_samples=0.9,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

gp.fit(X_clipped, y)
gp_features_train = gp.transform(X_clipped)
gp_features_test = gp.transform(test_clipped)

# === Final feature matrices
X_full = np.hstack([X, gp_features_train])
test_full = np.hstack([test, gp_features_test])

# === Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X_full, y, test_size=0.2, random_state=42)

# === XGBoost training
model = xgb.XGBRegressor(
    tree_method="hist",
    device="cuda",
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.6,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42
)

model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          early_stopping_rounds=20,
          verbose=True)

# === Evaluate
y_pred_val = model.predict(X_val)
pearson_corr, _ = pearsonr(y_val, y_pred_val)

print(f"\nğŸ“Š Final GP+Time+SHAP Model Evaluation")
print(f"ğŸ“ˆ Validation Pearson Correlation: {pearson_corr:.5f}")

# === Predict and Save submission
y_test_pred = model.predict(test_full)
submission["prediction"] = y_test_pred
submission.to_csv("submission.csv", index=False)
print("ğŸ“¦ Saved: submission.csv")



print("ğŸ“‹ Final XGBoost Feature Columns (train):", X.shape[1])
print(X.columns.tolist() if isinstance(X, pd.DataFrame) else "â�¡ï¸� Not a DataFrame â€” likely NumPy array after hstack")

print("\nğŸ“‹ Final XGBoost Feature Columns (test):", test.shape[1])
print(test.columns.tolist() if isinstance(test, pd.DataFrame) else "â�¡ï¸� Not a DataFrame â€” likely NumPy array after hstack")



# Show first 5 rows of test data before symbolic features were added
print("ğŸ“Š Sample of test data (with SHAP + predicted time features):")
print(test.head())


