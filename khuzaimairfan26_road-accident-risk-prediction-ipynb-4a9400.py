# SLOT 01 â€” Imports, Config & Setup
import os, gc, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Ridge
import lightgbm as lgb
from catboost import CatBoostRegressor

RANDOM_STATE = 42
N_FOLDS = 5
TARGET_COL = "accident_risk"
ID_COL = "id"

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

np.random.seed(RANDOM_STATE)



# SLOT 02 â€” Load Data
def load_data():
    for path in ["/kaggle/input", ".", "/mnt/data"]:
        for root, dirs, files in os.walk(path):
            if any("train" in f for f in files):
                train = pd.read_csv(os.path.join(root, [f for f in files if "train" in f][0]))
                test = pd.read_csv(os.path.join(root, [f for f in files if "test" in f][0]))
                sample = pd.read_csv(os.path.join(root, [f for f in files if "sample" in f][0]))
                print("âœ… Data Loaded!")
                print(train.shape, test.shape)
                return train, test, sample
    raise FileNotFoundError("Train/Test files not found!")

train, test, sample = load_data()



# SLOT 03 â€” EDA & Histograms
print(train.head())
print(train.describe())

plt.figure(figsize=(10,5))
sns.histplot(train[TARGET_COL], bins=30, kde=True, color="green")
plt.title("Distribution of Accident Risk")
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(train.corr(numeric_only=True), cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")
plt.show()



# SLOT 04 â€” Feature Engineering
def feature_engineering(train, test):
    full = pd.concat([train.drop(columns=[TARGET_COL]), test], axis=0).reset_index(drop=True)
    for col in full.select_dtypes("object").columns:
        le = LabelEncoder()
        full[col] = le.fit_transform(full[col].astype(str))
    for c in full.columns:
        if full[c].isnull().sum() > 0:
            full[c] = full[c].fillna(full[c].median())
    n_train = train.shape[0]
    train_fe = full.iloc[:n_train]
    test_fe = full.iloc[n_train:]
    train_fe[TARGET_COL] = train[TARGET_COL].values
    return train_fe, test_fe

train_fe, test_fe = feature_engineering(train, test)
print("Features ready:", train_fe.shape[1]-2)



# SLOT 05 â€” Correlation & Top Features
corr = train_fe.corr(numeric_only=True)[TARGET_COL].sort_values(ascending=False)
plt.figure(figsize=(8,5))
sns.barplot(x=corr.values[:10], y=corr.index[:10], palette="viridis")
plt.title("Top 10 Correlated Features with Accident Risk")
plt.show()

# ==============================
# ğŸ§© SLOT 5: FEATURE SELECTION & PREPARATION (FIXED FOR CATEGORICALS)
# ==============================
from sklearn.preprocessing import LabelEncoder

TARGET = "accident_risk"
ID_COL = "id"

# Separate features and target
X = train.drop(columns=[TARGET, ID_COL], errors="ignore")
y = train[TARGET]
X_test = test.drop(columns=[ID_COL], errors="ignore")

# Identify categorical columns (object or string types)
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
print(f"ğŸ”  Categorical columns detected: {cat_cols}")

# Encode all categorical columns using LabelEncoder
for col in cat_cols:
    le = LabelEncoder()
    combined_values = pd.concat([X[col], X_test[col]], axis=0).astype(str)
    le.fit(combined_values)
    X[col] = le.transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

# Ensure all columns are numeric now
print(f"âœ… All features are numeric: {X.dtypes.nunique() == 1}")
print(f"âœ… Training shape: {X.shape}, Test shape: {X_test.shape}")




## ==============================
# ğŸ§© SLOT 6: MODEL TRAINING 
# ==============================
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

# Cross-validation settings
N_FOLDS = 5
RANDOM_STATE = 42

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
oof_lgb = np.zeros(len(X))
preds_lgb = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\nğŸŸ¢ Fold {fold + 1}/{N_FOLDS}")
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    # LightGBM model definition
    model = LGBMRegressor(
        n_estimators=5000,
        learning_rate=0.03,
        num_leaves=31,
        colsample_bytree=0.7,
        subsample=0.7,
        random_state=RANDOM_STATE
    )

    # âœ… Use callback-style early stopping (works in LightGBM â‰¥4.0)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[
            early_stopping(stopping_rounds=100),
            log_evaluation(period=200)
        ]
    )

    # Out-of-fold and test predictions
    oof_lgb[val_idx] = model.predict(X_val)
    preds_lgb += model.predict(X_test) / N_FOLDS

# Overall CV performance
rmse = mean_squared_error(y, oof_lgb, squared=False)
print(f"\nâœ… Cross-validated RMSE: {rmse:.5f}")




# ==============================
# ğŸ§© SLOT 7: FEATURE IMPORTANCE
# ==============================
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Re-train model on full data for feature importance visualization
model_final = LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=31,
    colsample_bytree=0.7,
    subsample=0.7,
    random_state=42
)
model_final.fit(X, y)

# Extract feature importance
feature_imp = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model_final.feature_importances_
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_imp["Feature"][:20][::-1], feature_imp["Importance"][:20][::-1])
plt.title("Top 20 Feature Importances (LightGBM)")
plt.xlabel("Importance Score")
plt.ylabel("Feature Name")
plt.tight_layout()
plt.show()



# ==============================
# ğŸ§© SLOT 8: RESIDUAL & PREDICTION ANALYSIS
# ==============================
import seaborn as sns

residuals = y - oof_lgb
plt.figure(figsize=(8, 5))
sns.histplot(residuals, bins=40, kde=True, color="steelblue")
plt.title("Distribution of Residuals (y - Å·)")
plt.xlabel("Residual")
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(x=oof_lgb, y=y, alpha=0.4)
plt.title("Predicted vs Actual Accident Risk")
plt.xlabel("Predicted Risk")
plt.ylabel("Actual Risk")
plt.plot([0, 1], [0, 1], color="red", linestyle="--")
plt.show()



# ==============================
# ğŸ§© SLOT 9: CORRELATION HEATMAP
# ==============================
corr = train.select_dtypes(include=[np.number]).corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap")
plt.show()



# ==============================
# ğŸ§© SLOT 10: EVALUATION SUMMARY & RISK HISTOGRAM
# ==============================
print("âœ… Cross-validated RMSE:", round(rmse, 6))

plt.figure(figsize=(8, 5))
sns.histplot(oof_lgb, bins=30, color="darkorange", kde=True)
plt.title("Distribution of Predicted Accident Risk (OOF)")
plt.xlabel("Predicted Risk")
plt.ylabel("Count")
plt.show()



# ==============================
# ğŸ§© SLOT 11: CALIBRATION / BINNED PLOT
# ==============================
import pandas as pd

df_eval = pd.DataFrame({"actual": y, "pred": oof_lgb})
df_eval["pred_bin"] = pd.qcut(df_eval["pred"], q=10, duplicates="drop")

grouped = df_eval.groupby("pred_bin")[["actual", "pred"]].mean()
plt.figure(figsize=(7, 5))
plt.plot(grouped["pred"], grouped["actual"], "o-", color="purple")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.title("Calibration: Mean Actual vs Predicted Accident Risk")
plt.xlabel("Predicted")
plt.ylabel("Observed")
plt.show()



# ==========================
# ğŸŸ© SLOT 12: Final Submission Creation
# ==========================
import numpy as np
import pandas as pd

# âœ… Combine predictions from all models if available
# Example: preds_lgb, preds_xgb, preds_rf from previous slots
try:
    # Weighted ensemble example
    preds_final = (
        0.4 * preds_lgb +
        0.3 * preds_xgb +
        0.3 * preds_rf
    )
    print("âœ… Combined predictions from multiple models.")
except NameError:
    # If only one model's prediction exists
    print("âš ï¸� Using single model predictions (no ensemble detected).")
    if 'preds_lgb' in locals():
        preds_final = preds_lgb
    elif 'preds_xgb' in locals():
        preds_final = preds_xgb
    elif 'preds_rf' in locals():
        preds_final = preds_rf
    else:
        raise NameError("â�Œ No prediction variable found! Please ensure one of preds_lgb, preds_xgb, or preds_rf exists.")

# âœ… Ensure predictions are between 0 and 1
preds_final = np.clip(preds_final, 0, 1)

# âœ… Create submission DataFrame
submission = pd.DataFrame({
    "id": test["id"],          # change this to match your test fileâ€™s ID column name
    "accident_risk": preds_final
})

# âœ… Save to CSV
submission.to_csv("road_accident_risk_submission.csv", index=False)
print("âœ… 'road_accident_risk_submission.csv' created successfully!")

# âœ… Preview output
submission.head()


