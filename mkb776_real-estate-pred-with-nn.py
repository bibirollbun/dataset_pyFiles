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


# === Imports ===
import os, gc
import numpy as np
import pandas as pd
import calendar
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# === 1. Load all datasets ===
DATA_DIR = "/kaggle/input/china-real-estate-demand-prediction"

def read_csv(path):
    return pd.read_csv(os.path.join(DATA_DIR, path))

train_paths = {
    "new": "train/new_house_transactions.csv",
    "new_nb": "train/new_house_transactions_nearby_sectors.csv",
    "pre": "train/pre_owned_house_transactions.csv",
    "pre_nb": "train/pre_owned_house_transactions_nearby_sectors.csv",
    "land": "train/land_transactions.csv",
    "land_nb": "train/land_transactions_nearby_sectors.csv",
    "city_idx": "train/city_indexes.csv",
    "city_search_index" : "train/city_search_index.csv",
    "s_POI" : "train/sector_POI.csv"
}
test_path = "test.csv"

df_new = read_csv(train_paths["new"])
df_new_nb = read_csv(train_paths["new_nb"])
df_pre = read_csv(train_paths["pre"])
df_pre_nb = read_csv(train_paths["pre_nb"])
df_land = read_csv(train_paths["land"])
df_land_nb = read_csv(train_paths["land_nb"])
poi = read_csv(train_paths["s_POI"])
city_idx = read_csv(train_paths["city_idx"])
city_search_idx = read_csv(train_paths["city_search_index"])
comp_test_df = read_csv(test_path)


# Extract Year, Month and Sector Number
df_new[["Year", "Month"]] = df_new["month"].str.split("-", expand=True)
df_new["Year"] = df_new["Year"].astype(int)
df_new["sector_num"] = df_new["sector"].str.extract(r'(\d+)').astype(int)

# Map months to numeric values
month_codes = {m: i for i, m in enumerate(calendar.month_abbr) if m}
df_new["Month_num"] = df_new["Month"].map(month_codes)

# Drop old string columns
df_new.drop(columns=["month", "sector", "Month"], inplace=True)


# Create complete grid
years = sorted(df_new["Year"].unique())
months = sorted(df_new["Month_num"].unique())
sectors = sorted(df_new["sector_num"].unique())

CUT_Y, CUT_M = 2024, 7

full_index = pd.DataFrame(
    [(y, m, s) for y in years for m in months for s in sectors if (y < CUT_Y) or (y == CUT_Y and m <= CUT_M)],
    columns=["Year", "Month_num", "sector_num"]
)

df_full = pd.merge(full_index, df_new, on=["Year", "Month_num", "sector_num"], how="left")
df_full = df_full.sort_values(["sector_num", "Year", "Month_num"])

# Add 12-month ahead target
df_full["target_12m_ahead"] = (
    df_full.groupby("sector_num")["amount_new_house_transactions"].shift(-12)
)



# Use only data until 2023-07 for training and validation
df_train = df_full[
    (df_full["Year"] < 2023) | ((df_full["Year"] == 2023) & (df_full["Month_num"] <= 7))
]
df_test = df_full[
    (df_full["Year"] > 2023) | ((df_full["Year"] == 2023) & (df_full["Month_num"] > 7))
].copy()
df_train_sorted = df_train.sort_values(["Year", "Month_num", "sector_num"]).reset_index(drop=True)

# Define target and drop columns
TARGET_COL = "target_12m_ahead"
drop_cols = [TARGET_COL]


# Split sector-wise (80/20 time-based split)
X_tr_list, X_val_list, y_tr_list, y_val_list = [], [], [], []

for sector, group in df_train_sorted.groupby("sector_num"):
    group = group.sort_values(["Year", "Month_num"]).reset_index(drop=True)
    X_grp = group.drop(columns=drop_cols)
    y_grp = group[TARGET_COL]
    
    # 80% train, 20% validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_grp, y_grp, test_size=0.20, shuffle=False
    )
    X_tr_list.append(X_tr)
    X_val_list.append(X_val)
    y_tr_list.append(y_tr)
    y_val_list.append(y_val)

X_tr = pd.concat(X_tr_list).reset_index(drop=True)
X_val = pd.concat(X_val_list).reset_index(drop=True)
y_tr = pd.concat(y_tr_list).reset_index(drop=True)
y_val = pd.concat(y_val_list).reset_index(drop=True)

# Show shapes
print(f"Train Shape: {X_tr.shape}, Validation Shape: {X_val.shape}")


from sklearn.preprocessing import StandardScaler

def clean_xy(X, y):
    y = pd.to_numeric(y, errors="coerce")
    y = y.replace([np.inf, -np.inf], np.nan)
    ok = y.notna()
    return X.loc[ok].copy(), y.loc[ok].copy()

# Check and clean target values
y_tr = np.nan_to_num(y_tr, nan=0.0, posinf=0.0, neginf=0.0)
y_val = np.nan_to_num(y_val, nan=0.0, posinf=0.0, neginf=0.0)

# Check and clean feature values
X_tr = X_tr.fillna(0).replace([np.inf, -np.inf], 0)
X_val = X_val.fillna(0).replace([np.inf, -np.inf], 0)

# Standardize the features
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_val_scaled = scaler.transform(X_val)


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Build the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_tr_scaled.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')  # Linear for regression
])


from tensorflow.keras.metrics import RootMeanSquaredError
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=[
        'mae',
        RootMeanSquaredError()
    ]
)

# Summary
model.summary()


from tensorflow.keras.callbacks import EarlyStopping

# Define early stopping
early_stopping = EarlyStopping(
    monitor='val_rmse',  # Monitors Root Mean Squared Error on validation set
    patience=10,         # Stop if no improvement for 10 epochs
    restore_best_weights=True,
    mode='min'           # We want to minimize RMSE
)

# Train the model
history = model.fit(
    X_tr_scaled, y_tr,
    validation_data=(X_val_scaled, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)


loss, mae, rmse = model.evaluate(X_val_scaled, y_val, verbose=0)
print(f"[NN Validation] RMSE: {rmse:.2f} | MAE: {mae:.2f}")


import matplotlib.pyplot as plt

# Plot training & validation loss values
plt.figure(figsize=(10, 4))

# --- Loss ---
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# --- MAE ---
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.title('MAE over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.legend()

plt.tight_layout()
plt.show()



# Drop the target column if it exists
df_test_nn = df_test.drop(columns=["target_12m_ahead"], errors="ignore")

# Scale test features
X_test_scaled = scaler.transform(df_test_nn)

# Predict with trained NN model
test_preds_nn = model.predict(X_test_scaled).flatten()

# Add predictions
df_test_nn["target_predicted"] = pd.Series(test_preds_nn, index=df_test_nn.index)




import calendar

# Step 1: Create id in required format
mabbr = {i: calendar.month_abbr[i] for i in range(1, 13)}

map_df = df_test_nn[["Year", "Month_num", "sector_num", "target_predicted"]].copy()
map_df["id"] = (
    (map_df["Year"].astype(int) + 1).astype(str) + " " +
    map_df["Month_num"].astype(int).map(mabbr) + "_sector " +
    map_df["sector_num"].astype(int).astype(str)
)

# Step 2: Get latest prediction for each id (to handle duplicates)
map_df = map_df.groupby("id", as_index=False, sort=False)["target_predicted"].last()

# Step 3: Merge with official test file (comp_test_df from earlier)
submission_nn = comp_test_df.merge(map_df, on="id", how="left")

# Step 4: Final formatting for submission
submission_nn["new_house_transaction_amount"] = (
    submission_nn["new_house_transaction_amount"]
    .fillna(submission_nn["target_predicted"])
)
submission_nn.drop(columns=["target_predicted"], inplace=True)
submission_nn = submission_nn.fillna(0)

# Step 5: Save to CSV
submission_nn.to_csv("submission_nn.csv", index=False)

# ✅ Final message
print("✅ Submission file created successfully: `submission_nn.csv`")





