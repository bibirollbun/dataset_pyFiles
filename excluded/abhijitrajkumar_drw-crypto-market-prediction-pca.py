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


train_path = '/kaggle/input/drw-crypto-market-prediction/train.parquet'
test_path = '/kaggle/input/drw-crypto-market-prediction/test.parquet'


import gc

df_train = pd.read_parquet(train_path)
df_test = pd.read_parquet(test_path)

# Convert all numeric columns to float32
df_train = df_train.apply(lambda col: col.astype(np.float32) if np.issubdtype(col.dtype, np.number) else col)
df_test = df_test.apply(lambda col: col.astype(np.float32) if np.issubdtype(col.dtype, np.number) else col)

# âœ… Confirm the conversion
print(df_train.dtypes)
print(df_test.dtypes)
gc.collect()


# Show the first 5 rows of df_train
print("\n====== df_train.shape ======")
print(df_train.shape)

# Show the first 5 rows of df_test
print("\n====== df_test.shape ======")
print(df_test.shape)


import psutil
ram = psutil.virtual_memory()
print(f'Total RAM: {ram.total / (1024 ** 3):.2f} GB')
print(f'Available RAM: {ram.available / (1024 ** 3):.2f} GB')
print(f'Used RAM: {ram.used / (1024 ** 3):.2f} GB')
print(f'RAM Usage %: {ram.percent}%')



# import gc
# import numpy as np
# import pandas as pd
# from pandas.util import hash_pandas_object as hash_object

# def clean_dataset(df: pd.DataFrame, file_name: str = None) -> pd.DataFrame:
#     import time

#     print("\n==========================")
#     print(f"ğŸ§¹ Cleaning Data: {file_name if file_name else 'Unnamed'}")
#     print("==========================")

#     # 1ï¸�âƒ£ Null Columns Check
#     start_null = time.time()
#     null_counts = df.isnull().sum()
#     null_columns = null_counts[null_counts > 0]
#     if not null_columns.empty:
#         print(f"Columns with Null Values:\n{null_columns}")
#     else:
#         print("No columns contain null values.")
#     print(f"Null Check Time: {round(time.time() - start_null, 2)} seconds")

#     # 2ï¸�âƒ£ Drop Duplicate Columns (Hashing)
#     start_dup_cols = time.time()
#     hashes = df.apply(lambda col: hash_object(col).sum())
#     duplicates_mask = hashes.duplicated()
#     duplicate_columns = df.columns[duplicates_mask]
#     if len(duplicate_columns) > 0:
#         print(f"Dropping duplicate columns: {duplicate_columns.tolist()}")
#         df = df.drop(columns=duplicate_columns)
#     else:
#         print("No duplicate columns found.")
#     print(f"Duplicate Columns Check Time: {round(time.time() - start_dup_cols, 2)} seconds")

#     # 2ï¸�âƒ£.b Drop Duplicate Rows
#     start_row_dup = time.time()
#     initial_shape = df.shape
#     df.drop_duplicates(inplace=True)
#     final_shape = df.shape
#     rows_dropped = initial_shape[0] - final_shape[0]
#     if rows_dropped > 0:
#         print(f"Dropped {rows_dropped} duplicate rows.")
#     else:
#         print("No duplicate rows found.")
#     print(f"Duplicate Rows Check Time: {round(time.time() - start_row_dup, 2)} seconds")

#     # 3ï¸�âƒ£ Drop inf / -inf and Rows with NaNs
#     start_inf = time.time()
#     df.replace([np.inf, -np.inf], np.nan, inplace=True)
#     df.dropna(inplace=True)
#     print(f"Data shape after removing inf/-inf rows: {df.shape}")
#     print(f"Inf / -Inf Clean Time: {round(time.time() - start_inf, 2)} seconds")

#     # ğŸ”§ Convert all columns to float32
#     df = df.astype(np.float32)
#     print(f"âœ… Converted all columns to float32. Current memory usage: {df.memory_usage().sum() / 1e6:.2f} MB")

#     # 4ï¸�âƒ£ Save Cleaned Data (Optional)
#     if file_name:
#         save_path = f"/kaggle/working/{file_name}"
#         df.to_parquet(save_path, index=False)
#         print(f"âœ… Saved cleaned data to: {save_path}")

#     # 5ï¸�âƒ£ Delete DataFrame & Force GC
#     del df
#     gc.collect()
#     print("ğŸ—‘ï¸� DataFrame deleted from memory and garbage collected.")

#     # 6ï¸�âƒ£ Read Back if Needed
#     if file_name:
#         df = pd.read_parquet(f"/kaggle/working/{file_name}")
#         print("âœ… Reloaded DataFrame from saved file.")
#         return df
#     else:
#         print("âœ… Returned cleaned DataFrame.")
#         return df  # Instead of returning None



import gc
from pandas.util import hash_pandas_object as hash_object


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    import time

    print("\n==========================")
    print(f"ğŸ§¹ Cleaning Data")
    print("==========================")

    # 1ï¸�âƒ£ Null Columns Check
    start_null = time.time()
    null_counts = df.isnull().sum()
    null_columns = null_counts[null_counts > 0]
    if not null_columns.empty:
        print(f"Columns with Null Values:\n{null_columns}")
    else:
        print("No columns contain null values.")
    print(f"Null Check Time: {round(time.time() - start_null, 2)} seconds")

    # 2ï¸�âƒ£ Drop Duplicate Columns (Hashing)
    start_dup_cols = time.time()
    hashes = df.apply(lambda col: hash_object(col).sum())
    duplicates_mask = hashes.duplicated()
    duplicate_columns = df.columns[duplicates_mask]
    if len(duplicate_columns) > 0:
        print(f"Dropping duplicate columns: {duplicate_columns.tolist()}")
        df = df.drop(columns=duplicate_columns)
    else:
        print("No duplicate columns found.")
    print(f"Duplicate Columns Check Time: {round(time.time() - start_dup_cols, 2)} seconds")

    # 2ï¸�âƒ£.b Drop Duplicate Rows
    start_row_dup = time.time()
    initial_shape = df.shape
    df.drop_duplicates(inplace=True)
    final_shape = df.shape
    rows_dropped = initial_shape[0] - final_shape[0]
    if rows_dropped > 0:
        print(f"Dropped {rows_dropped} duplicate rows.")
    else:
        print("No duplicate rows found.")
    print(f"Duplicate Rows Check Time: {round(time.time() - start_row_dup, 2)} seconds")

    # 3ï¸�âƒ£ Drop inf / -inf and Rows with NaNs
    start_inf = time.time()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    print(f"Data shape after removing inf/-inf rows: {df.shape}")
    print(f"Inf / -Inf Clean Time: {round(time.time() - start_inf, 2)} seconds")

   
    print(f"âœ… Current memory usage: {df.memory_usage().sum() / 1e6:.2f} MB")

    # ğŸ”„ Garbage Collection (Optional)
    gc.collect()

    print("âœ… Returned cleaned DataFrame.")
    return df



# train_file = "cleaned_train.parquet"
# test_file = "cleaned_test.parquet"

# df_train_cleaned = clean_dataset(df_train, train_file)
# df_test_cleaned = clean_dataset(df_test, test_file)


df_train_cleaned = clean_dataset(df_train)
df_test_cleaned = clean_dataset(df_test)


# df_train_cleaned = pd.read_parquet("/kaggle/working/cleaned_train.parquet")
# df_test_cleaned = pd.read_parquet("/kaggle/working/cleaned_test.parquet")


print("\n====== df_train_cleaned.shape ======")
print(df_train_cleaned.shape)

# Show the first 5 rows of df_test
print("\n====== df_test_cleaned.shape ======")
print(df_test_cleaned.shape)


ram = psutil.virtual_memory()
print(f'Total RAM: {ram.total / (1024 ** 3):.2f} GB')
print(f'Available RAM: {ram.available / (1024 ** 3):.2f} GB')
print(f'Used RAM: {ram.used / (1024 ** 3):.2f} GB')
print(f'RAM Usage %: {ram.percent}%')



# import gc
# from sklearn.preprocessing import QuantileTransformer
# from xgboost import XGBRegressor

# #
# def prepare_train_data(df_train: pd.DataFrame, label_col: str = 'label', importance_threshold: float = 0.000001):
#     # 1ï¸�âƒ£ Clip Outliers
#     for col in df_train.drop(columns=[label_col]).columns:
#         lower = df_train[col].quantile(0.01)
#         upper = df_train[col].quantile(0.99)
#         df_train[col] = df_train[col].clip(lower, upper)

#     # 2ï¸�âƒ£ XGBoost Feature Importance
#     X_train = df_train.drop(columns=[label_col])
#     y_train = df_train[label_col]
#     model = XGBRegressor()
#     model.fit(X_train, y_train)

#     importance = pd.Series(model.feature_importances_, index=X_train.columns)
#     important_features = importance[importance > importance_threshold].index.tolist()

#     # 3ï¸�âƒ£ Quantile Scaling
#     scaler = QuantileTransformer(output_distribution='normal', random_state=42)
#     X_train_scaled = scaler.fit_transform(X_train[important_features])

#     df_train_cleaned = pd.DataFrame(X_train_scaled, columns=important_features)
#     df_train_cleaned[label_col] = y_train.reset_index(drop=True)

#     # 4ï¸�âƒ£ Explicit Memory Cleanup
#     del df_train, X_train, y_train, X_train_scaled, model, importance
#     gc.collect()

#     return df_train_cleaned, important_features, scaler



from sklearn.preprocessing import QuantileTransformer
from xgboost import XGBRegressor


def prepare_train_data(df_train: pd.DataFrame, label_col: str = 'label', importance_threshold: float = 1e-6):
    print("1ï¸�âƒ£ Clipping Outliers...")
    # 1ï¸�âƒ£ Clip Outliers
    for col in df_train.drop(columns=[label_col]).columns:
        low = df_train[col].quantile(0.01)
        high = df_train[col].quantile(0.99)
        df_train[col] = df_train[col].clip(lower=low, upper=high)
    print("âœ… Outliers clipped.")

    print("2ï¸�âƒ£ Fitting XGBoost Model for Feature Importance (GPU)...")
    # 2ï¸�âƒ£ XGBoost Feature Importance with GPU
    X_train = df_train.drop(columns=[label_col])
    y_train = df_train[label_col]
    model = XGBRegressor(tree_method='hist', device='cuda')
    model.fit(X_train, y_train)
    print("âœ… XGBoost Model Trained.")

    importance = pd.Series(model.feature_importances_, index=X_train.columns)
    important_features = importance[importance > importance_threshold].index.tolist()
    print(f"âœ… Selected {len(important_features)} important features: {important_features}")

    print("3ï¸�âƒ£ Applying Quantile Scaling...")
    # 3ï¸�âƒ£ Quantile Scaling (CPU)
    scaler = QuantileTransformer(output_distribution='normal', random_state=42)
    X_train_scaled = scaler.fit_transform(X_train[important_features])
    print("âœ… Quantile Scaling Completed.")

    df_train_cleaned = pd.DataFrame(X_train_scaled, columns=important_features)
    df_train_cleaned[label_col] = y_train.reset_index(drop=True)

    print("4ï¸�âƒ£ Cleaning Up Memory...")
    # 4ï¸�âƒ£ Cleanup
    del df_train, X_train, y_train, X_train_scaled, model, importance
    gc.collect()
    print("âœ… Memory cleaned.")

    print("ğŸ�‰ Data Preparation Completed.")
    return df_train_cleaned, important_features, scaler



def prepare_test_data(df_test: pd.DataFrame, important_features, scaler):
    print("1ï¸�âƒ£ Selecting Important Features from Test Data...")
    X_test = df_test[important_features]
    print("âœ… Selected important features.")

    print("2ï¸�âƒ£ Applying Quantile Scaling to Test Data...")
    X_test_scaled = scaler.transform(X_test)
    print("âœ… Quantile Scaling applied.")

    df_test_cleaned = pd.DataFrame(X_test_scaled, columns=important_features)
    print("âœ… Created cleaned test DataFrame.")

    print("3ï¸�âƒ£ Cleaning Up Memory...")
    # ğŸ”´ Explicit memory cleanup
    del df_test, X_test, X_test_scaled
    gc.collect()
    print("âœ… Memory cleaned.")

    print("ğŸ�‰ Test Data Preparation Completed.")
    return df_test_cleaned



df_train_cleaned, important_features, scaler = prepare_train_data(df_train_cleaned, label_col='label')
df_test_cleaned = prepare_test_data(df_test_cleaned, important_features, scaler)


print("\n====== df_train_cleaned.shape ======")
print(df_train_cleaned.shape)

# Show the first 5 rows of df_test
print("\n====== df_test_cleaned.shape ======")
print(df_test_cleaned.shape)


ram = psutil.virtual_memory()
print(f'Total RAM: {ram.total / (1024 ** 3):.2f} GB')
print(f'Available RAM: {ram.available / (1024 ** 3):.2f} GB')
print(f'Used RAM: {ram.used / (1024 ** 3):.2f} GB')
print(f'RAM Usage %: {ram.percent}%')



import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras import mixed_precision


# âš¡ Mixed Precision for faster GPU computation
print("âš¡ Enabling Mixed Precision for TensorFlow...")
mixed_precision.set_global_policy('mixed_float16')
print("âœ… Mixed Precision enabled.\n")

# ğŸ“¦ KFold Setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
print(f"ğŸ“¦ KFold Split: {kf.get_n_splits()} folds configured.\n")

# ğŸ�—ï¸� Model Architecture
def build_model(input_shape):
    print(f"ğŸ”§ Building Model for input shape: {input_shape}")
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(1, dtype='float32')  # Output must be float32
    ])
    model.compile(optimizer=AdamW(learning_rate=1e-3), loss='mse', metrics=['mae'])
    print("âœ… Model Compiled.\n")
    return model


# ğŸ› ï¸� Prepare Data
print("ğŸ› ï¸� Preparing Data...")
X_np = df_train_cleaned.drop(columns=['label']).values.astype(np.float32)
y_np = df_train_cleaned['label'].values.astype(np.float32)
X_test_np = df_test_cleaned.values.astype(np.float32)
print(f"âœ… Data prepared with shapes: X_train {X_np.shape}, y_train {y_np.shape}, X_test {X_test_np.shape}\n")

oof_preds = np.zeros(len(X_np))
test_preds = np.zeros(len(X_test_np))

# ğŸ�‹ï¸�â€�â™‚ï¸� Training Loop with EarlyStopping
early_stop = EarlyStopping(monitor='val_mae', patience=5, restore_best_weights=True, verbose=1)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_np)):
    print(f"\nğŸŒ€ Starting Fold {fold+1}/5...")

    X_train, X_val = X_np[train_idx], X_np[val_idx]
    y_train, y_val = y_np[train_idx], y_np[val_idx]
    print(f"ğŸ“Š Train shape: {X_train.shape}, Validation shape: {X_val.shape}")

    model = build_model(X_train.shape[1])
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=100,
                        batch_size=2048,  # T4 handles large batches better
                        callbacks=[early_stop],
                        verbose=1)

    print(f"ğŸ“ˆ Predicting on validation set (Fold {fold+1})...")
    oof_preds[val_idx] = model.predict(X_val, batch_size=2048).reshape(-1)
    print(f"ğŸ“ˆ Predicting on test set (Fold {fold+1})...")
    test_preds += model.predict(X_test_np, batch_size=2048).reshape(-1) / kf.n_splits

    print(f"ğŸ§¹ Cleaning memory for Fold {fold+1}...")
    del X_train, X_val, y_train, y_val, model, history
    gc.collect()
    print(f"âœ… Fold {fold+1} completed.\n")


# ğŸ“Š Evaluate Out-of-Fold Performance
mse = mean_squared_error(y_np, oof_preds)
print(f"\nâœ… Out-Of-Fold MSE: {mse:.5f}")

# ğŸ�¯ Final Test Predictions Sample
print("\nğŸ�¯ Final Test Predictions (first 10 rows):")
print(test_preds[:10])

# ğŸ’¾ Save Predictions if needed
print("\nğŸ’¾ Saving predictions to disk...")
np.save("oof_preds.npy", oof_preds)
np.save("test_preds.npy", test_preds)
print("âœ… Predictions saved as 'oof_preds.npy' and 'test_preds.npy'.")



ram = psutil.virtual_memory()
print(f'Total RAM: {ram.total / (1024 ** 3):.2f} GB')
print(f'Available RAM: {ram.available / (1024 ** 3):.2f} GB')
print(f'Used RAM: {ram.used / (1024 ** 3):.2f} GB')
print(f'RAM Usage %: {ram.percent}%')



mse = mean_squared_error(y_np, oof_preds)
print(f"\nâœ… OOF MSE: {mse:.6f}")

# Save to submission
submission = pd.DataFrame({
    "ID": range(1, len(test_preds)+1),  # or use actual test IDs if available
    "prediction": test_preds
})
submission.to_csv("submission.csv", index=False)
print("ğŸ“¦ Saved: submission.csv")



submission.tail()

