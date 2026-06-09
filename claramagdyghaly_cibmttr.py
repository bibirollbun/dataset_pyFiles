import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ---------------------------
# 1. Load Data
# ---------------------------
train_path = "/kaggle/input/equity-post-HCT-survival-predictions/train.csv"
test_path = "/kaggle/input/equity-post-HCT-survival-predictions/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Save and drop the ID column
train_ids = train_df["ID"]
test_ids = test_df["ID"]
train_df.drop(columns=["ID"], inplace=True)
test_df.drop(columns=["ID"], inplace=True)

# ---------------------------
# 2. Define Target and Positive "Risk Score"
# ---------------------------
#   - 'efs_time' is the survival time.
#   - 'efs' is the event indicator (0 = censored, 1 = event).
#   - We'll define a "risk_score" so that higher values = shorter survival.
#     Example: risk_score = max_survival - efs_time
y_true_all = train_df[["efs_time", "efs"]]

max_survival = train_df["efs_time"].max()
risk_score = max_survival - train_df["efs_time"]  # Higher for shorter survival

# ---------------------------
# 3. Prepare Features
# ---------------------------
if "race_group" in train_df.columns:
    # Keep race_group for stratified evaluation if needed
    race_group_series = train_df["race_group"]
    X = train_df.drop(columns=["efs_time", "efs", "race_group"])
else:
    race_group_series = None
    X = train_df.drop(columns=["efs_time", "efs"])

# Drop race_group from test if present
if "race_group" in test_df.columns:
    test_df.drop(columns=["race_group"], inplace=True)

# Drop any remaining non-numeric columns
non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
X.drop(columns=non_numeric_cols, inplace=True)
test_df.drop(columns=test_df.select_dtypes(exclude=[np.number]).columns, inplace=True)

# Fill missing values
X.fillna(X.mean(), inplace=True)
test_df.fillna(test_df.mean(), inplace=True)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test_df)

# ---------------------------
# 4. Train-Test Split
# ---------------------------
X_train, X_val, y_train_risk, y_val_risk, y_train_true, y_val_true = train_test_split(
    X_scaled, risk_score, y_true_all, test_size=0.2, random_state=42
)

if race_group_series is not None:
    race_group_train, race_group_val = train_test_split(race_group_series, test_size=0.2, random_state=42)
else:
    race_group_val = None

# ---------------------------
# 5. Concordance Index Functions
# ---------------------------
def concordance_index(y_true, y_pred):
    """
    Compute the concordance index comparing survival times with predicted risk scores.
    Higher risk => shorter survival time.
    """
    y_true = y_true.to_numpy()
    times = y_true[:, 0]  # efs_time
    n_pairs = 0
    concordant_pairs = 0

    for i in range(len(times)):
        for j in range(i + 1, len(times)):
            if times[i] != times[j]:
                n_pairs += 1
                # With risk_score = max_survival - survival_time:
                # if times[i] > times[j], we expect y_pred[i] < y_pred[j].
                if (times[i] > times[j] and y_pred[i] < y_pred[j]) or \
                   (times[i] < times[j] and y_pred[i] > y_pred[j]):
                    concordant_pairs += 1

    return concordant_pairs / n_pairs if n_pairs > 0 else 0.5

def stratified_concordance_index(y_true, y_pred, race_group):
    """
    Compute stratified c-index = mean(c-index per race_group) - std(c-index per race_group).
    """
    unique_race_groups = np.unique(race_group)
    c_indices = []
    for r in unique_race_groups:
        idx = (race_group == r)
        if np.sum(idx) < 2:
            continue
        c_idx = concordance_index(y_true[idx], y_pred[idx])
        c_indices.append(c_idx)
    c_indices = np.array(c_indices)
    if len(c_indices) == 0:
        return 0.5
    return c_indices.mean() - c_indices.std()

# ---------------------------
# 6. Train 4 Models (NN, XGBoost, RF, CatBoost)
# ---------------------------
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Neural Network Regressor
nn_model = Sequential([
    Dense(256, activation="relu", input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),
    Dense(1, activation="linear")
])
nn_model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
nn_callbacks = [EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)]

nn_model.fit(X_train, y_train_risk, epochs=50, batch_size=32,
             validation_data=(X_val, y_val_risk),
             verbose=1, callbacks=nn_callbacks)

nn_val_preds = nn_model.predict(X_val).flatten()

from xgboost import XGBRegressor
xgb_model = XGBRegressor(learning_rate=0.001, random_state=42)
xgb_model.fit(X_train, y_train_risk)
xgb_val_preds = xgb_model.predict(X_val)

from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train_risk)
rf_val_preds = rf_model.predict(X_val)

from catboost import CatBoostRegressor
cat_model = CatBoostRegressor(verbose=0, random_state=42)
cat_model.fit(X_train, y_train_risk)
cat_val_preds = cat_model.predict(X_val)

# Evaluate each on validation set
nn_ci = concordance_index(y_val_true, nn_val_preds)
xgb_ci = concordance_index(y_val_true, xgb_val_preds)
rf_ci = concordance_index(y_val_true, rf_val_preds)
cat_ci = concordance_index(y_val_true, cat_val_preds)

print(f"NN c-index:  {nn_ci:.4f}")
print(f"XGB c-index: {xgb_ci:.4f}")
print(f"RF c-index:  {rf_ci:.4f}")
print(f"Cat c-index: {cat_ci:.4f}")

if race_group_val is not None:
    nn_strat_ci = stratified_concordance_index(y_val_true, nn_val_preds, race_group_val.to_numpy())
    xgb_strat_ci = stratified_concordance_index(y_val_true, xgb_val_preds, race_group_val.to_numpy())
    rf_strat_ci = stratified_concordance_index(y_val_true, rf_val_preds, race_group_val.to_numpy())
    cat_strat_ci = stratified_concordance_index(y_val_true, cat_val_preds, race_group_val.to_numpy())

    print(f"NN strat c-index:  {nn_strat_ci:.4f}")
    print(f"XGB strat c-index: {xgb_strat_ci:.4f}")
    print(f"RF strat c-index:  {rf_strat_ci:.4f}")
    print(f"Cat strat c-index: {cat_strat_ci:.4f}")

# ---------------------------
# 7. Predict on Test & SCALE to Smaller Range
# ---------------------------
# Predict risk on the test set
test_preds_nn = nn_model.predict(test_scaled).flatten()
test_preds_xgb = xgb_model.predict(test_scaled)
test_preds_rf = rf_model.predict(test_scaled)
test_preds_cat = cat_model.predict(test_scaled)

# Ensemble: average
final_test_preds = (test_preds_nn + test_preds_xgb + test_preds_rf + test_preds_cat) / 4

# EXAMPLE: Min-Max scale these ensemble scores to a smaller range, e.g. [0, 2].
# (Adjust the range if you want them around [0,1], [0,2], etc.)
min_pred = final_test_preds.min()
max_pred = final_test_preds.max()
scaled_test_preds = (final_test_preds - min_pred) / (max_pred - min_pred) * 2.0

# ---------------------------
# 8. Create Submission File
# ---------------------------
# "scaled_test_preds" now are smaller values, e.g. 0.5, 1.2, 0.8, ...
submission = pd.DataFrame({"ID": test_ids, "prediction": scaled_test_preds})
submission.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")





