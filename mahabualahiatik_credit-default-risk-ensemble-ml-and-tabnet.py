import pandas as pd

# Load all datasets
app_train = pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv')
app_test = pd.read_csv('/kaggle/input/home-credit-default-risk/application_test.csv')
bureau = pd.read_csv('/kaggle/input/home-credit-default-risk/bureau.csv')
bureau_balance = pd.read_csv('/kaggle/input/home-credit-default-risk/bureau_balance.csv')
prev = pd.read_csv('/kaggle/input/home-credit-default-risk/previous_application.csv')
pos = pd.read_csv('/kaggle/input/home-credit-default-risk/POS_CASH_balance.csv')
installments = pd.read_csv('/kaggle/input/home-credit-default-risk/installments_payments.csv')
cc = pd.read_csv('/kaggle/input/home-credit-default-risk/credit_card_balance.csv')

# Print full column names for each file
print("\nğŸ“„ application_train.csv columns:\n", list(app_train.columns))
print("\nğŸ“„ bureau.csv columns:\n", list(bureau.columns))
print("\nğŸ“„ bureau_balance.csv columns:\n", list(bureau_balance.columns))
print("\nğŸ“„ previous_application.csv columns:\n", list(prev.columns))
print("\nğŸ“„ POS_CASH_balance.csv columns:\n", list(pos.columns))
print("\nğŸ“„ installments_payments.csv columns:\n", list(installments.columns))
print("\nğŸ“„ credit_card_balance.csv columns:\n", list(cc.columns))



import pandas as pd
import numpy as np



# ------------------------------------------------------------------
# 2) BUREAU + BUREAU_BALANCE
# ------------------------------------------------------------------
bb_agg = (
    bureau_balance
      .groupby("SK_ID_BUREAU")
      .agg(
          MONTHS_BALANCE_MIN   = ("MONTHS_BALANCE", "min"),
          MONTHS_BALANCE_MAX   = ("MONTHS_BALANCE", "max"),
          MONTHS_BALANCE_SIZE  = ("MONTHS_BALANCE", "count"),
          STATUS_BAD           = ("STATUS", lambda x: (x.isin(["1","2","3","4","5"])).sum()),
      )
      .reset_index()
)

bureau = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")

bureau_agg = (
    bureau
      .groupby("SK_ID_CURR")
      .agg(
          BUREAU_COUNT                     = ("SK_ID_BUREAU", "count"),
          BUREAU_CREDIT_SUM_MEAN           = ("AMT_CREDIT_SUM", "mean"),
          BUREAU_CREDIT_SUM_MAX            = ("AMT_CREDIT_SUM", "max"),
          BUREAU_CREDIT_ACTIVE_COUNT       = ("CREDIT_ACTIVE", lambda x: (x == "Active").sum()),
          BUREAU_STATUS_BAD_SUM            = ("STATUS_BAD", "sum"),
          BUREAU_MONTHS_BALANCE_SIZE_MEAN  = ("MONTHS_BALANCE_SIZE", "mean"),
      )
      .reset_index()
)

# ------------------------------------------------------------------
# 3) PREVIOUS APPLICATION
# ------------------------------------------------------------------
prev_agg = (
    prev
      .groupby("SK_ID_CURR")
      .agg(
          PREV_COUNT              = ("SK_ID_PREV", "count"),
          PREV_AMT_APPLICATION_MEAN = ("AMT_APPLICATION", "mean"),
          PREV_AMT_APPLICATION_MAX  = ("AMT_APPLICATION", "max"),
          PREV_AMT_CREDIT_MEAN      = ("AMT_CREDIT", "mean"),
          PREV_APPROVED_COUNT       = ("NAME_CONTRACT_STATUS", lambda x: (x == "Approved").sum()),
          PREV_REFUSED_COUNT        = ("NAME_CONTRACT_STATUS", lambda x: (x == "Refused").sum()),
      )
      .reset_index()
)

# ------------------------------------------------------------------
# 4) POS-CASH BALANCE
# ------------------------------------------------------------------
pos_agg = (
    pos
      .groupby("SK_ID_CURR")
      .agg(
          POS_PREV_NUNIQUE              = ("SK_ID_PREV", "nunique"),
          POS_MONTHS_BALANCE_MEAN       = ("MONTHS_BALANCE", "mean"),
          POS_MONTHS_BALANCE_MAX        = ("MONTHS_BALANCE", "max"),
          POS_CNT_INSTALMENT_MEAN       = ("CNT_INSTALMENT", "mean"),
          POS_CNT_INSTALMENT_FUTURE_MEAN= ("CNT_INSTALMENT_FUTURE", "mean"),
          POS_SK_DPD_MAX                = ("SK_DPD", "max"),
          POS_SK_DPD_DEF_MAX            = ("SK_DPD_DEF", "max"),
      )
      .reset_index()
)

# ------------------------------------------------------------------
# 5) INSTALLMENTS PAYMENTS
# ------------------------------------------------------------------
inst = installments.copy()
inst["PAYMENT_PERC"] = inst["AMT_PAYMENT"] / inst["AMT_INSTALMENT"]
inst["PAYMENT_DIFF"] = inst["AMT_INSTALMENT"] - inst["AMT_PAYMENT"]

install_agg = (
    inst
      .groupby("SK_ID_CURR")
      .agg(
          INST_PREV_NUNIQUE        = ("SK_ID_PREV", "nunique"),
          INST_VERSION_NUNIQUE     = ("NUM_INSTALMENT_VERSION", "nunique"),
          INST_AMT_PAYMENT_MEAN    = ("AMT_PAYMENT", "mean"),
          INST_AMT_PAYMENT_MAX     = ("AMT_PAYMENT", "max"),
          INST_PAYMENT_PERC_MEAN   = ("PAYMENT_PERC", "mean"),
          INST_PAYMENT_DIFF_MEAN   = ("PAYMENT_DIFF", "mean"),
          INST_DAYS_INSTAL_MEAN    = ("DAYS_INSTALMENT", "mean"),
          INST_DAYS_ENTRY_MEAN     = ("DAYS_ENTRY_PAYMENT", "mean"),
      )
      .reset_index()
)

# ------------------------------------------------------------------
# 6) CREDIT CARD BALANCE
# ------------------------------------------------------------------
cc_agg = (
    cc
      .groupby("SK_ID_CURR")
      .agg(
          CC_PREV_NUNIQUE              = ("SK_ID_PREV", "nunique"),
          CC_MONTHS_BALANCE_MEAN       = ("MONTHS_BALANCE", "mean"),
          CC_AMT_BALANCE_MEAN          = ("AMT_BALANCE", "mean"),
          CC_AMT_BALANCE_MAX           = ("AMT_BALANCE", "max"),
          CC_AMT_CREDIT_LIMIT_MEAN     = ("AMT_CREDIT_LIMIT_ACTUAL", "mean"),
          CC_AMT_DRAWINGS_CURR_MEAN    = ("AMT_DRAWINGS_CURRENT", "mean"),
          CC_SK_DPD_MAX                = ("SK_DPD", "max"),
          CC_SK_DPD_DEF_MAX            = ("SK_DPD_DEF", "max"),
      )
      .reset_index()
)

# ------------------------------------------------------------------
# 7) MERGE EVERYTHING INTO app_train
# ------------------------------------------------------------------
train_full = app_train.copy()
for agg in [bureau_agg, prev_agg, pos_agg, install_agg, cc_agg]:
    train_full = train_full.merge(agg, on="SK_ID_CURR", how="left")

print(f"âœ… Final training shape: {train_full.shape}")
test_full = app_test.copy()
for agg in [bureau_agg, prev_agg, pos_agg, install_agg, cc_agg]:
    test_full = test_full.merge(agg, on="SK_ID_CURR", how="left")

print(f"âœ… Final test shape: {test_full.shape}")




train_full.head()


# -----------------------------------------------------------
#  Categorical profile & missing-value statistics
# -----------------------------------------------------------
cat_cols = train_full.select_dtypes(include=["object"]).columns.tolist()

print(f"ğŸŸ¦ Total categorical columns: {len(cat_cols)}\n")
print("ğŸŸ¦ Column names:")
for col in cat_cols:
    print("   â€¢", col)

# --- missing value % for each categorical column
missing_pct = (
    train_full[cat_cols]
      .isna()
      .mean()
      .mul(100)
      .sort_values(ascending=False)
)

print("\nğŸŸ¦ Missing-value % for categorical columns:")
print(missing_pct.to_string(float_format="%.2f"))

# --- highlight columns that are â€œmostly NaNâ€�
mostly_nan = missing_pct[missing_pct > 80]

print("\nğŸŸ¥ Categorical columns with >80 % NaNs:")
if mostly_nan.empty:
    print("   (None)")
else:
    for col, pct in mostly_nan.items():
        print(f"   â€¢ {col:35s}  â†’  {pct:.2f}% missing")






from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Step 1: Split before encoding
X = train_full.drop(columns=["TARGET"])
y = train_full["TARGET"]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 2: Encode only on training categorical features
cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

ohe = OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.uint8)
ohe.fit(X_train[cat_cols])

# Step 3: Transform both sets (safe because encoder is fit on train only)
X_train_ohe = pd.DataFrame(ohe.transform(X_train[cat_cols]), columns=ohe.get_feature_names_out(cat_cols), index=X_train.index)
X_valid_ohe = pd.DataFrame(ohe.transform(X_valid[cat_cols]), columns=ohe.get_feature_names_out(cat_cols), index=X_valid.index)

# Step 4: Drop old cats and concat encoded versions
X_train_final = pd.concat([X_train.drop(columns=cat_cols), X_train_ohe], axis=1)
X_valid_final = pd.concat([X_valid.drop(columns=cat_cols), X_valid_ohe], axis=1)

# Step 5: Apply same encoding to test_full
X_test_ohe = pd.DataFrame(ohe.transform(test_full[cat_cols]), columns=ohe.get_feature_names_out(cat_cols), index=test_full.index)
X_test_final = pd.concat([test_full.drop(columns=cat_cols), X_test_ohe], axis=1)

print(f"âœ… Final shapes: train {X_train_final.shape}, valid {X_valid_final.shape}, test {X_test_final.shape}")



# Collect all current feature names
all_features = X_train_final.columns.tolist()

# Define characters LightGBM rejects
illegal_chars = set('[]<>{}():=,"\\')  # confirmed from LightGBM JSON issues

# Check which features contain them
bad_features = [f for f in all_features if any(c in f for c in illegal_chars)]

print(f"â�Œ Found {len(bad_features)} feature names with illegal characters.")
if bad_features:
    for bf in bad_features[:10]:
        print("  â€¢", bf)
    if len(bad_features) > 10:
        print("  ... (showing first 10)")






def clean_feature_names(df):
    return df.rename(columns=lambda x: (
        x.replace('[', '_')
         .replace(']', '')
         .replace('<', '_lt_')
         .replace('>', '_gt_')
         .replace('{', '_')
         .replace('}', '')
         .replace(':', '_')     # â†� critical fix for your current issue
         .replace('=', '_eq_')
         .replace(',', '_')
         .replace('"', '')
         .replace('\\', '_')
         .replace('(', '_')
         .replace(')', '')
         .replace(' ', '_')
    ))


X_train_final = clean_feature_names(X_train_final)
X_valid_final = clean_feature_names(X_valid_final)
X_test_final = clean_feature_names(X_test_final)


# Clean up any infinities in training and validation sets
X_train_final = X_train_final.replace([np.inf, -np.inf], np.nan).fillna(0)
X_test_final = X_test_final.replace([np.inf, -np.inf], np.nan).fillna(0)
X_valid_final = X_valid_final.replace([np.inf, -np.inf], np.nan).fillna(0)



def add_handcrafted_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds domain-driven, leakage-safe features to the merged Home Credit table.
    Works for both train & test because it uses only contemporaneous columns.
    """

    df = df.copy()

    ## ----- Core ratios & interactions  ------------------------------------
    # EXT_SOURCE aggregates
    srcs = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    df["EXT_SOURCES_MEAN"]   = df[srcs].mean(axis=1)
    df["EXT_SOURCES_MAX"]    = df[srcs].max(axis=1)
    df["EXT_SOURCES_MIN"]    = df[srcs].min(axis=1)
    df["EXT_SOURCES_STD"]    = df[srcs].std(axis=1)

    # Credit & income relationships
    df["CREDIT_INCOME_RATIO"]      = df["AMT_CREDIT"]  / (df["AMT_INCOME_TOTAL"] + 1)
    df["ANNUITY_INCOME_RATIO"]     = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["ANNUITY_CREDIT_RATIO"]     = df["AMT_ANNUITY"] / (df["AMT_CREDIT"] + 1)
    df["GOODS_CREDIT_DIFF"]        = df["AMT_GOODS_PRICE"] - df["AMT_CREDIT"]

    # Age / employment dynamics  (days are negative)
    df["AGE_YEARS"]                = (-df["DAYS_BIRTH"])      / 365
    df["EMPLOYED_YEARS"]           = np.where(
                                        df["DAYS_EMPLOYED"] < 0,
                                        0,
                                        df["DAYS_EMPLOYED"] / 365
                                    )
    df["EMPLOYED_BIRTH_RATIO"]     = df["EMPLOYED_YEARS"] / (df["AGE_YEARS"] + 1e-3)

    # Family & children context
    df["CHILDREN_RATIO"]           = df["CNT_CHILDREN"] / (df["CNT_FAM_MEMBERS"] + 1)
    df["INCOME_PER_FAM"]           = df["AMT_INCOME_TOTAL"] / (df["CNT_FAM_MEMBERS"] + 1)

    ## ----- Aggregated-table signals  -------------------------------------
    # Installment discipline
    if {"INST_PAYMENT_DIFF_MEAN","INST_AMT_PAYMENT_MEAN"}.issubset(df.columns):
        df["INSTALL_PAYMENT_RATIO"] = df["INST_AMT_PAYMENT_MEAN"] / (
                                          df["INST_AMT_PAYMENT_MAX"] + 1
                                      )

    # Credit-card utilisation
    if {"CC_AMT_BALANCE_MEAN", "CC_AMT_CREDIT_LIMIT_MEAN"}.issubset(df.columns):
        df["CC_UTIL_MEAN"] = df["CC_AMT_BALANCE_MEAN"] / (
                                 df["CC_AMT_CREDIT_LIMIT_MEAN"] + 1
                             )

    # POS loan progress
    if {"POS_CNT_INSTALMENT_FUTURE_MEAN","POS_CNT_INSTALMENT_MEAN"}.issubset(df.columns):
        df["POS_REMAINING_PERC"] = df["POS_CNT_INSTALMENT_FUTURE_MEAN"] / (
                                       df["POS_CNT_INSTALMENT_MEAN"] + 1
                                   )

    ## ----- Fill any div/0 inf values with NaN -----------------------------
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df






X_train_final = add_handcrafted_features(X_train_final)
X_valid_final  = add_handcrafted_features(X_valid_final)
X_test_final  = add_handcrafted_features(X_test_final)






# -------------------------------------------------------
# 1) Fit LightGBM (baseline that gave best AUC â‰ˆ0.781)
# -------------------------------------------------------
from lightgbm import LGBMClassifier
import numpy as np, pandas as pd, sys, subprocess, warnings
warnings.filterwarnings("ignore")

lgbm = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    n_jobs=-1,
    random_state=42
)
lgbm.fit(X_train_final, y_train)

# -------------------------------------------------------
# 2) Install & import SHAP if missing
# -------------------------------------------------------
try:
    import shap
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "shap==0.41.0"])
    import shap

# -------------------------------------------------------
# 3) Compute SHAP values (sample 3 000 rows for speed)
# -------------------------------------------------------
sample_idx   = np.random.RandomState(42).choice(X_train_final.index, size=10000, replace=False)
X_sample     = X_train_final.loc[sample_idx]

explainer    = shap.TreeExplainer(lgbm)
shap_values  = explainer.shap_values(X_sample, check_additivity=False)[1]  # class-1 contributions

# -------------------------------------------------------
# 4) Mean |SHAP| importance for every feature
# -------------------------------------------------------
mean_abs     = np.abs(shap_values).mean(axis=0)
imp_df       = pd.DataFrame({
    "feature"        : X_train_final.columns,
    "mean_abs_shap"  : mean_abs
}).sort_values("mean_abs_shap", ascending=False)

# Print full table without truncation
pd.set_option("display.max_rows", None)
print("\nğŸ“�  SHAP feature importance (all features):\n")
print(imp_df.to_string(index=False))



threshold = 9e-3
irrelevant_feats = imp_df.loc[imp_df["mean_abs_shap"] < threshold, "feature"].tolist()

print(f"ğŸ”» Dropping {len(irrelevant_feats)} low-importance features")

X_train_final = X_train_final.drop(columns=irrelevant_feats)
X_valid_final = X_valid_final.drop(columns=irrelevant_feats)
X_test_final  = X_test_final.drop(columns=irrelevant_feats)








from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from lightgbm import LGBMClassifier
import pandas as pd

# LightGBM model
lgbm = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    n_jobs=-1
)

# Train
print("\nğŸ”§ Training LightGBM...")
lgbm.fit(X_train_final, y_train)

# Validation predictions
y_pred = lgbm.predict(X_valid_final)
y_proba = lgbm.predict_proba(X_valid_final)[:, 1]

# Evaluation
acc = accuracy_score(y_valid, y_pred)
auc = roc_auc_score(y_valid, y_proba)

print(f"âœ… LightGBM Accuracy:     {acc:.4f}")
print(f"âœ… LightGBM ROC-AUC:      {auc:.4f}")
print(f"ğŸ“Š LightGBM Classification Report:\n", classification_report(y_valid, y_pred))

# Test predictions
test_preds = lgbm.predict_proba(X_test_final)[:, 1]

# Create submission DataFrame
submission = pd.DataFrame({
    "SK_ID_CURR": test_full["SK_ID_CURR"],
    "TARGET": test_preds
})

# Save submission file
submission.to_csv("submission.csv", index=False)
print("âœ… Submission file saved as 'submission.csv'")



from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Optional: silence warnings
import warnings
warnings.filterwarnings("ignore")

models = {
    "XGBoost": XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1,
        verbosity=0
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        n_jobs=-1
    ),
    "CatBoost": CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        verbose=0
    )
}

# Train and evaluate
for name, model in models.items():
    print(f"\nğŸ”§ Training {name}...")
    model.fit(X_train_final, y_train)

    y_pred = model.predict(X_valid_final)
    y_proba = model.predict_proba(X_valid_final)[:, 1]

    acc = accuracy_score(y_valid, y_pred)
    auc = roc_auc_score(y_valid, y_proba)

    print(f"âœ… {name} Accuracy:     {acc:.4f}")
    print(f"âœ… {name} ROC-AUC:      {auc:.4f}")
    print(f"ğŸ“Š {name} Classification Report:\n", classification_report(y_valid, y_pred))


from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# Define base models
models = {
    "XGBoost": XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1,
        verbosity=0
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        n_jobs=-1
    ),
    "CatBoost": CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        verbose=0
    )
}

# Train and collect predictions
proba_valid = []
for name, model in models.items():
    print(f"\nğŸ”§ Training {name}...")
    model.fit(X_train_final, y_train)
    proba = model.predict_proba(X_valid_final)[:, 1]
    proba_valid.append(proba)

# Ensemble: average predicted probabilities
ensemble_proba = np.mean(proba_valid, axis=0)
ensemble_pred = (ensemble_proba >= 0.5).astype(int)

# Evaluation
acc = accuracy_score(y_valid, ensemble_pred)
auc = roc_auc_score(y_valid, ensemble_proba)

print(f"\nğŸ�¯ Ensemble Accuracy:     {acc:.4f}")
print(f"ğŸ�¯ Ensemble ROC-AUC:      {auc:.4f}")
print(f"ğŸ“Š Ensemble Classification Report:\n", classification_report(y_valid, ensemble_pred))



!pip install pytorch_tabnet


# ===============================================================
#  TabNet baseline on X_train_final / X_valid_final  (CPU only)
# ===============================================================
import sys, subprocess, warnings, types, numpy as np, pandas as pd
warnings.filterwarnings("ignore")


import torch
from pytorch_tabnet.tab_model import TabNetClassifier



# â”€â”€ 3. Standard-scale numeric features â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_final)
X_valid_scaled = scaler.transform(X_valid_final)

X_train_np = X_train_scaled.astype(np.float32)
X_valid_np = X_valid_scaled.astype(np.float32)
y_train_np = y_train.values
y_valid_np = y_valid.values

# â”€â”€ 4. Initialise & train TabNet â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
clf = TabNetClassifier(
    n_d=32, n_a=32, n_steps=5,
    gamma=1.5, lambda_sparse=1e-4,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params=dict(step_size=50, gamma=0.9),
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    mask_type="sparsemax",
    seed=42, verbose=10
)

clf.fit(
    X_train_np, y_train_np,
    eval_set=[(X_valid_np, y_valid_np)],
    eval_name=["valid"],
    eval_metric=["auc"],
    max_epochs=300,
    patience=50,
    batch_size=16384,
    virtual_batch_size=4096,
    num_workers=0,
    drop_last=False
)

# â”€â”€ 5. Validation metrics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
proba_valid = clf.predict_proba(X_valid_np)[:, 1]
pred_valid  = (proba_valid >= 0.5).astype(int)

auc  = roc_auc_score(y_valid_np, proba_valid)
acc  = accuracy_score(y_valid_np, pred_valid)

print(f"\nğŸ”� TabNet Validation AUC : {auc:.4f}")
print(f"ğŸ”� TabNet Validation ACC : {acc:.4f}")
print("\nğŸ“Š Classification Report:\n", classification_report(y_valid_np, pred_valid))














