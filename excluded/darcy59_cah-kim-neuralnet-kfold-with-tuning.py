# 1. Imports & Setup
import os, re, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

from sklearn.model_selection import train_test_split, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler, RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error

RANDOM_STATE = 1337

def rmse(y, yhat):
    return np.sqrt(mean_squared_error(y, yhat))


# 2. Load Data
DATA_PATH = "/kaggle/input/california-homelessness-prediction-challenge"

train = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
test = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))
sample_sub = pd.read_csv(os.path.join(DATA_PATH, "sample_submission.csv"))


# 3: Enhanced Feature Engineering
# Extract both CountyCode (e.g., "CA") and AreaCode (e.g., "037") from ID

# Create CountyCode and AreaCode columns
train['CountyCode'] = train['ID'].astype(str).str.split("_", n=1).str[0]
# train['AreaCode'] = train['ID'].astype(str).str.split("_", n=1).str[1]

# Drop original ID column and extract target
target_col = 'HOMELESS_RATE'
y = train[target_col]
X = train.drop(columns=['ID', target_col])

# Verify the structure
X.shape
print()
X.head()



# 4. Preprocessing Pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.impute import SimpleImputer

# Identify updated column types
numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Compute skew again using updated features
from scipy.stats import skew
skew_vals = X[numeric_cols].apply(lambda x: skew(x.dropna()))
skew_df = pd.DataFrame(skew_vals, columns=["Skewness"]).reset_index().rename(columns={"index": "Feature"})

def recommend_transform(skew_val):
    if abs(skew_val) < 0.5:
        return "none"
    elif 0.5 <= skew_val < 1.5:
        return "log"
    elif skew_val >= 1.5 or skew_val <= -0.5:
        return "yeo"
    return "none"

skew_df["Transform"] = skew_df["Skewness"].apply(recommend_transform)

# Assign columns
log_cols = skew_df[skew_df["Transform"] == "log"]["Feature"].tolist()
yeo_cols = skew_df[skew_df["Transform"] == "yeo"]["Feature"].tolist()
none_cols = list(set(numeric_cols) - set(log_cols) - set(yeo_cols))

# Pipelines for each transformation type
log_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("log", FunctionTransformer(lambda x: np.log1p(x))),
    ("scale", StandardScaler())
])

yeo_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("yeo", PowerTransformer(method="yeo-johnson")),
    ("scale", StandardScaler())
])

none_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scale", StandardScaler())
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer([
    ("log", log_pipe, log_cols),
    ("yeo", yeo_pipe, yeo_cols),
    ("none", none_pipe, none_cols),
    ("cat", cat_pipe, categorical_cols)
])

# Apply preprocessing
X_processed = preprocessor.fit_transform(X)
X_processed.shape



# 5: Neural Network + KFold with Tuning
!pip install -q -U keras-tuner

import keras_tuner as kt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import numpy as np

# === Define Model Search Space ===
def build_model(hp):
    model = Sequential()
    model.add(Dense(
        hp.Int('units1', 64, 256, step=32),
        activation=hp.Choice('activation1', ['relu', 'tanh']),
        input_shape=(X_processed.shape[1],)
    ))
    model.add(Dropout(hp.Float('dropout1', 0.0, 0.5, step=0.1)))

    if hp.Boolean('second_layer'):
        model.add(Dense(hp.Int('units2', 32, 128, step=32),
                        activation=hp.Choice('activation2', ['relu', 'tanh'])))
        model.add(Dropout(hp.Float('dropout2', 0.0, 0.5, step=0.1)))

    model.add(Dense(1))  # Output layer
    model.compile(optimizer=Adam(hp.Float('lr', 1e-4, 1e-2, sampling='log')),
                  loss='mse')
    return model

# === Split for tuning (first fold only) ===
RANDOM_STATE = 1337
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
first_train_idx, first_val_idx = next(kf.split(X_processed))

X_tr, X_va = X_processed[first_train_idx], X_processed[first_val_idx]
y_tr, y_va = np.log1p(y.iloc[first_train_idx]), np.log1p(y.iloc[first_val_idx])

# === Run Tuner ===
tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=15,
    executions_per_trial=1,
    directory='kt_dir',
    project_name='homeless_tune'
)

tuner.search(X_tr, y_tr,
             validation_data=(X_va, y_va),
             epochs=100,
             batch_size=16,
             callbacks=[EarlyStopping(patience=10)],
             verbose=1)

best_hp = tuner.get_best_hyperparameters(1)[0]
print("Best Hyperparameters:", best_hp.values)

# === Apply Best Hyperparameters to K-Fold Training ===
train_mae, train_rmse, val_mae, val_rmse = [], [], [], []

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_processed), 1):
    X_fold_tr, X_fold_va = X_processed[tr_idx], X_processed[va_idx]
    y_fold_tr = np.log1p(y.iloc[tr_idx])
    y_fold_va = np.log1p(y.iloc[va_idx])

    model = tuner.hypermodel.build(best_hp)

    model.fit(X_fold_tr, y_fold_tr,
              validation_data=(X_fold_va, y_fold_va),
              epochs=100,
              batch_size=16,
              verbose=0,
              callbacks=[EarlyStopping(patience=10, restore_best_weights=True)])

    # Predict
    y_tr_pred = model.predict(X_fold_tr).flatten()
    y_va_pred = model.predict(X_fold_va).flatten()

    # Inverse transform
    y_tr_true = np.expm1(y_fold_tr)
    y_va_true = np.expm1(y_fold_va)
    y_tr_pred = np.expm1(y_tr_pred)
    y_va_pred = np.expm1(y_va_pred)

    # Metrics
    train_mae.append(mean_absolute_error(y_tr_true, y_tr_pred))
    train_rmse.append(np.sqrt(mean_squared_error(y_tr_true, y_tr_pred)))
    val_mae.append(mean_absolute_error(y_va_true, y_va_pred))
    val_rmse.append(np.sqrt(mean_squared_error(y_va_true, y_va_pred)))

    print(f"Fold {fold} – Train MAE: {train_mae[-1]:.4f}, RMSE: {train_rmse[-1]:.4f} | "
          f"Val MAE: {val_mae[-1]:.4f}, RMSE: {val_rmse[-1]:.4f}")

# === Final Averages ===
print("\n=== Average Metrics ===")
print(f"Train MAE : {np.mean(train_mae):.4f}")
print(f"Train RMSE: {np.mean(train_rmse):.4f}")
print(f"Val MAE   : {np.mean(val_mae):.4f}")
print(f"Val RMSE  : {np.mean(val_rmse):.4f}")



# === Block 7: Final Neural Network Fit + Full Data Evaluation ===
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Build final model using best tuned hyperparameters
final_model = tuner.hypermodel.build(best_hp)

# Fit model on all data (log-transformed target)
final_model.fit(
    X_processed, np.log1p(y),
    epochs=100,
    batch_size=16,
    verbose=0,
    callbacks=[EarlyStopping(patience=10, restore_best_weights=True)]
)

# Predict on full training set
full_pred_log = final_model.predict(X_processed).flatten()
full_pred = np.expm1(full_pred_log)

# Evaluate on full training data
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

results_compare = pd.DataFrame({
    "Dataset": ["Full Training Set"],
    "MAE": [mean_absolute_error(y, full_pred)],
    "RMSE": [rmse(y, full_pred)]
})

print("✅ Final Neural Network Performance on Full Training Data:")
display(results_compare)



# === Block 8: Test Prediction & Submission File ===
test = pd.read_csv(f"{DATA_PATH}/test.csv")
sample_sub = pd.read_csv(f"{DATA_PATH}/sample_submission.csv")

# Reapply feature engineering
test["CountyCode"] = test["ID"].astype(str).str.split("_", n=1).str[0]
# test["AreaCode"] = test["ID"].astype(str).str.split("_", n=1).str[1]
X_test = test.drop(columns=["ID"])

# Preprocess test data
X_test_proc = preprocessor.transform(X_test)

# Predict and inverse-transform
test_preds_log = final_model.predict(X_test_proc).flatten()
test_preds = np.expm1(test_preds_log)
test_preds = np.clip(test_preds, 0, 1)

# Prepare submission
submission = sample_sub.copy()
submission["HOMELESS_RATE"] = test_preds
submission.to_csv("submission_neuralnet_tuned.csv", index=False)

print("✅ Submission saved: submission_neuralnet_tuned.csv")
submission.head()


