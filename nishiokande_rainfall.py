import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


train_df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


train_df.head(5)


test_df.head(5)


# Compute the correlation matrix
corr_matrix = train_df.corr()

# Plot the heatmap for all correlation
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()


test_df["winddirection"].fillna(test_df["winddirection"].mode()[0], inplace=True)


train_df["humidity_cloud"] = train_df["humidity"] * train_df["cloud"]
test_df["humidity_cloud"] = test_df["humidity"] * test_df["cloud"]

train_df["temp_dew_diff"] = train_df["humidity"] - train_df["sunshine"]
test_df["temp_dew_diff"] = test_df["humidity"] - test_df["sunshine"]

train_df["pressure_change"] = train_df["humidity"].diff().fillna(0)
test_df["pressure_change"] = test_df["humidity"].diff().fillna(0)

train_df["wind_cos"] = np.cos(np.radians(train_df["winddirection"]))
train_df["wind_sin"] = np.sin(np.radians(train_df["winddirection"]))
test_df["wind_cos"] = np.cos(np.radians(test_df["winddirection"]))
test_df["wind_sin"] = np.sin(np.radians(test_df["winddirection"]))


train_df.head(5)


test_df.head(5)


# Compute the correlation matrix
corr_matrix = train_df.corr()

# Plot the heatmap for all correlation
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()


X = train_df.drop(["id", "rainfall"], axis = 1)
y = train_df["rainfall"]

X_test = test_df.drop(["id"], axis = 1)


import xgboost as xgb
import optuna
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


# Split data into training and testing sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


def mish(x):
    return x * K.tanh(K.softplus(x))


# XGBoost model
model_xgb = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)

# Train XGBoost
model_xgb.fit(X_train, y_train)

# Prediction with validation data
xgb_pred_proba = model_xgb.predict_proba(X_val)[:, 1]

# Normalize features for NN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Build Neural Network model
nn_model = keras.Sequential([
    keras.layers.Dense(128, activation=mish, input_shape=(X_train.shape[1],), kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation=mish, kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation=mish, kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(16, activation=mish, kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(8, activation=mish, kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation='sigmoid')
])

nn_model.compile(optimizer='adamw', loss='binary_crossentropy', metrics=['AUC'])

# callback
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
]

# Train NN model
nn_model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_data=(X_val_scaled, y_val), callbacks=callbacks, verbose=1)

# Prediction with NN
nn_pred_proba = nn_model.predict(X_val_scaled).flatten()

# Ensemble (average of XGBoost and NN predictions)
ensemble_pred_proba = (xgb_pred_proba + nn_pred_proba) / 2

# Evaluate ensemble model
auc_ensemble = roc_auc_score(y_val, ensemble_pred_proba)
print(f"Validation AUC (XGBoost): {roc_auc_score(y_val, xgb_pred_proba):.4f}")
print(f"Validation AUC (NN): {roc_auc_score(y_val, nn_pred_proba):.4f}")
print(f"Validation AUC (Ensemble): {auc_ensemble:.4f}")



# XGBoostの予測
xgb_test_pred = model_xgb.predict_proba(X_test)[:, 1]

# NNの予測（テストデータのスケーリングも忘れずに）
X_test_scaled = scaler.transform(X_test)
nn_test_pred = nn_model.predict(X_test_scaled).flatten()

# アンサンブル（単純平均）
ensemble_test_pred = (xgb_test_pred + nn_test_pred) / 2

# 提出用データフレーム作成
submission = pd.DataFrame({"id": test_df["id"], "rainfall": ensemble_test_pred})

# CSVとして保存
submission.to_csv("submission.csv", index=False)

