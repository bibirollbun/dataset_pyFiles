# ğŸ“¦ Import all dependencies


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from category_encoders.target_encoder import TargetEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

print("âœ… Libraries imported successfully.")



# ğŸ§¹ Load the Dataset


train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')

print(f"âœ… Train shape: {train.shape}")
print(f"âœ… Test shape: {test.shape}")

# Preview the data
train.head()



train.info()

# Target column
target = 'loan_paid_back'

# Check target balance
sns.countplot(x=target, data=train)
plt.title("Target Distribution: Loan Paid Back (1) vs Not (0)")
plt.show()



cat_cols = ['gender', 'marital_status', 'education_level', 
            'employment_status', 'loan_purpose', 'grade_subgrade']

num_cols = ['annual_income', 'debt_to_income_ratio', 
            'credit_score', 'loan_amount', 'interest_rate']

print("Categorical cols:", cat_cols)
print("Numeric cols:", num_cols)



def kfold_target_encoding(train_df, test_df, target_ser, cat_columns, n_splits=5, seed=42):
    te_train, te_test = train_df.copy(), test_df.copy()
    global_mean = target_ser.mean()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    for col in cat_columns:
        te_train[col + "_te"] = np.nan
        for tr_idx, val_idx in skf.split(train_df, target_ser):
            means = train_df.iloc[tr_idx].groupby(col)[target_ser.name].mean()
            te_train.loc[val_idx, col + "_te"] = train_df.loc[val_idx, col].map(means)
        te_test[col + "_te"] = te_test[col].map(train_df.groupby(col)[target_ser.name].mean())
        te_train[col + "_te"].fillna(global_mean, inplace=True)
        te_test[col + "_te"].fillna(global_mean, inplace=True)
    return te_train, te_test



# ğŸ§© Apply Target Encoding 

from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

def kfold_target_encoding(train_df, test_df, target_ser, cat_columns, n_splits=5, seed=42):
    """
    Performs K-Fold Target Encoding safely (no leakage).
    """
    te_train = train_df.copy()
    te_test = test_df.copy()
    skf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    global_mean = target_ser.mean()

    for col in cat_columns:
        col_te = f"{col}_te"
        te_train[col_te] = np.nan
        print(f"Encoding column: {col} ...")

        # For each fold, compute mean target value
        for tr_idx, val_idx in skf.split(train_df):
            tr_fold = train_df.iloc[tr_idx]
            y_tr = target_ser.iloc[tr_idx]
            val_fold = train_df.iloc[val_idx]
            means = tr_fold.groupby(col)[y_tr.name].mean() if y_tr.name in tr_fold.columns else tr_fold.groupby(col).apply(lambda x: y_tr[x.index].mean())
            te_train.iloc[val_idx, te_train.columns.get_loc(col_te)] = val_fold[col].map(means).fillna(global_mean).values

        # Encode test using full data
        full_means = train_df.groupby(col).apply(lambda x: target_ser[x.index].mean())
        te_test[col_te] = test_df[col].map(full_means).fillna(global_mean).values

    return te_train, te_test


# ğŸ”§ Run Target Encoding


cat_cols = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']

print("Categorical columns:", cat_cols)

# Make sure X_raw, X_test_raw, y exist
assert 'X_raw' in locals(), "â�Œ X_raw not found"
assert 'X_test_raw' in locals(), "â�Œ X_test_raw not found"
assert 'y' in locals(), "â�Œ Target variable y not found"

X_enc, X_test_enc = kfold_target_encoding(X_raw, X_test_raw, y, cat_cols, n_splits=5)
print("\nâœ… Target Encoding applied successfully!")



for col in cat_cols:
    le = LabelEncoder()
    X_enc[col] = le.fit_transform(X_enc[col])
    X_test_enc[col] = le.transform(X_test_enc[col])

print("âœ… Label Encoding completed successfully.")



print("Train shape:", X_enc.shape)
print("Test shape:", X_test_enc.shape)
print("Target shape:", y.shape)

print("\nğŸ§  Preprocessing Complete â€” Data is ready for model training!")



# âš™ï¸� Model Setup

import lightgbm as lgb
import xgboost as xgb
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

RANDOM_SEED = 42
N_SPLITS = 5
print("âœ… Libraries loaded successfully!")



# ğŸ�¯ Optuna Objective Function for LightGBM

def objective(trial):
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 16, 256),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 5.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
    }

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    oof_preds = np.zeros(len(X_enc))

    for train_idx, valid_idx in cv.split(X_enc, y):
        X_train, X_valid = X_enc.iloc[train_idx], X_enc.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        dtrain = lgb.Dataset(X_train, label=y_train)
        dvalid = lgb.Dataset(X_valid, label=y_valid)

        model = lgb.train(
            params, dtrain,
            valid_sets=[dvalid],
            early_stopping_rounds=50,
            verbose_eval=False
        )

        preds = model.predict(X_valid)
        oof_preds[valid_idx] = preds

    auc = roc_auc_score(y, oof_preds)
    return auc

print("âœ… Optuna Objective Function ready!")



import lightgbm as lgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

N_SPLITS = 5
RANDOM_SEED = 42

def objective(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "lambda_l1": trial.suggest_float("lambda_l1", 0, 5),
        "lambda_l2": trial.suggest_float("lambda_l2", 0, 5),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "random_state": RANDOM_SEED
    }

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    auc_scores = []

    for train_idx, valid_idx in skf.split(X_enc, y):
        X_train, X_valid = X_enc.iloc[train_idx], X_enc.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = lgb.LGBMClassifier(**params)

        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(0)
            ]
        )

        preds = model.predict_proba(X_valid)[:, 1]
        auc = roc_auc_score(y_valid, preds)
        auc_scores.append(auc)

    return np.mean(auc_scores)


# ğŸ”§ Run Optuna optimization
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=25, show_progress_bar=True)

print("\nğŸ�† Best AUC Score:", study.best_value)
print("ğŸ�¯ Best Params:\n", study.best_params)



# ============================================================
# âš¡ Final Model Training
# ============================================================
best_params = study.best_params
best_params.update({
    "objective": "binary",
    "metric": "auc",
    "verbosity": -1,
    "boosting_type": "gbdt"
})

final_model = lgb.LGBMClassifier(**best_params, random_state=RANDOM_SEED)
final_model.fit(X_enc, y)

print("âœ… Final LightGBM model trained successfully!")



# ğŸ“Š Cross Validation (AUC)

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
cv_scores = []

for train_idx, valid_idx in cv.split(X_enc, y):
    X_train, X_valid = X_enc.iloc[train_idx], X_enc.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    final_model.fit(X_train, y_train)
    preds = final_model.predict_proba(X_valid)[:, 1]
    score = roc_auc_score(y_valid, preds)
    cv_scores.append(score)

print(f"âœ… Mean CV AUC: {np.mean(cv_scores):.5f} Â± {np.std(cv_scores):.5f}")



# ğŸ”§ Auto Feature Interaction Creation

# Assumes: X_enc, X_test_enc, y exist in the environment.
# If num_cols or cat TE cols were lost, we try to infer them.

import numpy as np
import pandas as pd

# Safety checks
if 'X_enc' not in globals() or 'X_test_enc' not in globals():
    raise RuntimeError("X_enc / X_test_enc not found. Run preprocessing first.")

# PARAMETERS (tune these to control feature explosion)
TOP_K_NUM = 4             # take top-K numeric features by variance for pairwise interactions
CORR_THRESHOLD = 0.98     # drop features with abs(corr) >= threshold (very high collinearity)
KEEP_CORR_WITH_TARGET = 0.02  # if newly created feature has near-zero corr with target, you may drop (optional)

# 1) Infer numeric columns if not available
num_cols = [c for c in X_enc.columns if np.issubdtype(X_enc[c].dtype, np.number)]
# remove any TE columns from num_cols if you'd like; here we keep numeric and TE both
# choose top-K by variance (most informative candidates)
num_var = X_enc[num_cols].var().sort_values(ascending=False)
top_num = num_var.index[:TOP_K_NUM].tolist()
print("Top numeric features selected for interactions:", top_num)

# 2) Identify target-encoded categorical columns (those ending with '_te')
te_cat_cols = [c for c in X_enc.columns if c.endswith('_te')]
print("Target-encoded categorical columns:", te_cat_cols)

new_features = []

# 3) Pairwise product and ratio among top numeric features
for i in range(len(top_num)):
    for j in range(i+1, len(top_num)):
        a = top_num[i]
        b = top_num[j]
        prod_name = f"{a}_x_{b}"
        div_name = f"{a}_div_{b}"
        # product
        X_enc[prod_name] = X_enc[a] * X_enc[b]
        X_test_enc[prod_name] = X_test_enc[a] * X_test_enc[b]
        new_features.append(prod_name)
        # ratio (safe)
        X_enc[div_name] = np.where(X_enc[b] != 0, X_enc[a] / (X_enc[b]), 0.0)
        X_test_enc[div_name] = np.where(X_test_enc[b] != 0, X_test_enc[a] / (X_test_enc[b]), 0.0)
        new_features.append(div_name)

# 4) Polynomial (square) for top numeric features
for a in top_num:
    sq_name = f"{a}_sq"
    X_enc[sq_name] = X_enc[a] ** 2
    X_test_enc[sq_name] = X_test_enc[a] ** 2
    new_features.append(sq_name)

# 5) Numeric Ã— target-encoded categorical interactions
for n in top_num:
    for te in te_cat_cols:
        inter_name = f"{n}_x_{te}"
        X_enc[inter_name] = X_enc[n] * X_enc[te]
        X_test_enc[inter_name] = X_test_enc[n] * X_test_enc[te]
        new_features.append(inter_name)

print(f"Created {len(new_features)} new interaction features.")

# 6) Optionally compute correlation with target and drop near-zero corr features (uncomment to use)
# importances = {}
# for f in new_features:
#     corr = np.corrcoef(X_enc[f].fillna(0), y)[0,1]
#     importances[f] = abs(corr)
# drop_low_corr = [f for f,v in importances.items() if abs(v) < KEEP_CORR_WITH_TARGET]
# print("Dropping low-corr features (optional):", drop_low_corr)
# X_enc.drop(columns=drop_low_corr, inplace=True); X_test_enc.drop(columns=drop_low_corr, inplace=True)
# new_features = [f for f in new_features if f not in drop_low_corr]

# 7) Remove features that are nearly duplicate (very high correlation) with existing features or each other
print("Pruning highly correlated features (abs corr >= {:.2f})...".format(CORR_THRESHOLD))
# Build correlation matrix on combined data for robust estimate (train only to avoid leakage)
corr_matrix = X_enc.corr().abs()

to_drop = set()
# iterate through upper triangle
cols = corr_matrix.columns
for i in range(len(cols)):
    if cols[i] in to_drop: 
        continue
    for j in range(i+1, len(cols)):
        if cols[j] in to_drop:
            continue
        if corr_matrix.iloc[i, j] >= CORR_THRESHOLD:
            # prefer to drop the feature that is newer (in new_features), else drop the j-th
            col_i = cols[i]
            col_j = cols[j]
            if col_j in new_features:
                to_drop.add(col_j)
            elif col_i in new_features:
                to_drop.add(col_i)
            else:
                # if neither new, drop the j-th (arbitrary)
                to_drop.add(col_j)

print("Number of features to drop due to high correlation:", len(to_drop))
if len(to_drop) > 0:
    X_enc.drop(columns=list(to_drop), inplace=True)
    X_test_enc.drop(columns=list(to_drop), inplace=True)
    # clean new_features list
    new_features = [f for f in new_features if f not in to_drop]

print(f"Final new features kept: {len(new_features)}")
print("Sample of new features:", new_features[:20])

# 8) Final shape summary
print("Final X_enc shape:", X_enc.shape)
print("Final X_test_enc shape:", X_test_enc.shape)



# ============================================================
# âš¡ Evaluate Feature Importances (Quick LightGBM)
# ============================================================

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np

# Safety check
if 'X_enc' not in globals() or 'y' not in globals():
    raise RuntimeError("â�Œ Data not found. Please run preprocessing & interaction cells first.")

# Quick 80-20 split for validation
X_train, X_valid, y_train, y_valid = train_test_split(
    X_enc, y, test_size=0.2, random_state=42, stratify=y
)

lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 3,
    "random_state": 42,
    "n_jobs": -1,
}

model = lgb.LGBMClassifier(**lgb_params)

# ğŸ‘‡ No 'verbose' argument; use callbacks instead
model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="auc",
    callbacks=[lgb.log_evaluation(0)]  # 0 â†’ silence all logs; 10 â†’ log every 10 rounds
)

# Get feature importance
importances = pd.DataFrame({
    'feature': X_enc.columns,
    'gain': model.booster_.feature_importance(importance_type='gain'),
    'split': model.booster_.feature_importance(importance_type='split')
})

importances['gain_norm'] = importances['gain'] / importances['gain'].sum()
importances.sort_values(by='gain', ascending=False, inplace=True)

# Show top 20 important features
print("ğŸ�† Top 20 Features by Gain:")
display(importances.head(20))

# Track which new features contributed most
new_feats_present = [f for f in importances['feature'] if f in new_features]
important_new_feats = [f for f in new_feats_present if f in importances.head(50)['feature'].values]
print(f"ğŸ“ˆ {len(important_new_feats)} interaction features are among the top 50 important ones.")



# ============================================================
# ğŸ§  Automatic Feature Selection
# ============================================================

# PARAMETERS
TOP_N_NEW = 30         # Keep top-N new interaction features by importance/gain
MIN_CORR = 0.03        # Minimum correlation threshold with target

# Compute correlations of new features with target (quick sanity filter)
corrs = {}
for f in new_features:
    if f in X_enc.columns:
        corr_val = np.corrcoef(X_enc[f].fillna(0), y)[0,1]
        corrs[f] = abs(corr_val)
corrs_df = pd.DataFrame(list(corrs.items()), columns=['feature','abs_corr']).sort_values(by='abs_corr', ascending=False)

# Merge correlation and gain-based importance
merged = importances.merge(corrs_df, on='feature', how='left').fillna(0)
merged['score'] = merged['gain_norm'] + merged['abs_corr']   # simple combined score
merged.sort_values(by='score', ascending=False, inplace=True)

# Keep top features
top_keep = merged.head(TOP_N_NEW)['feature'].tolist()
print(f"âœ… Keeping top {len(top_keep)} new interaction features by combined score.")

# Drop low-scoring interaction features (only drop from new_features)
drop_feats = [f for f in new_features if f not in top_keep]
X_enc.drop(columns=drop_feats, inplace=True, errors='ignore')
X_test_enc.drop(columns=drop_feats, inplace=True, errors='ignore')

print(f"ğŸš€ Final shape after feature selection: {X_enc.shape}")
print("Remaining top interaction features:", top_keep[:10])



# ============================================================
# ğŸ“Š Visualize Top 20 Feature Importances (Gain-based)
# ============================================================

import matplotlib.pyplot as plt

top_feats = importances.head(20).iloc[::-1]  # reverse for better plot order

plt.figure(figsize=(10, 6))
plt.barh(top_feats['feature'], top_feats['gain_norm'], height=0.6)
plt.title("ğŸ”� Top 20 Most Important Features (by Gain)", fontsize=14, weight='bold')
plt.xlabel("Normalized Gain Importance", fontsize=12)
plt.ylabel("Feature Name", fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()



# ğŸŒˆ Top 20 Feature Importances (Gain-based, Gradient Style)


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Prepare top 20 features (sorted by importance)
top_feats = importances.head(20).iloc[::-1]  # reverse order for horizontal bars
colors = sns.color_palette("coolwarm", n_colors=20)

# Create figure
plt.figure(figsize=(10, 6))

# Plot horizontal bars
bars = plt.barh(top_feats['feature'], top_feats['gain_norm'], height=0.6, color=colors)

# Add title and axis labels
plt.title("ğŸŒŸ Top 20 Most Important Features (by Gain)", fontsize=15, weight='bold', pad=15)
plt.xlabel("Normalized Gain Importance", fontsize=12)
plt.ylabel("Feature Name", fontsize=12)

# Add value labels to each bar
for i, v in enumerate(top_feats['gain_norm']):
    plt.text(v + 0.005, i, f"{v:.3f}", va='center', fontsize=10, color='black')

# Add light grid
plt.grid(axis='x', linestyle='--', alpha=0.5)

# Layout adjustment
plt.tight_layout()

# Show plot
plt.show()



# ğŸŒˆ Feature Importance Visualization â€” Gain vs Split


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Ensure feature importances DataFrame is ready
# (example: already created after training)
# importances = pd.DataFrame({
#     'feature': model.feature_name_,
#     'gain': model.feature_importances_,
#     'split': model.feature_importances_
# })

# Normalize gain and split
importances['gain_norm'] = importances['gain'] / importances['gain'].sum()
importances['split_norm'] = importances['split'] / importances['split'].sum()

# Select top 20 features by gain
top20 = importances.sort_values("gain_norm", ascending=False).head(20).iloc[::-1]

# Create two subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 8))

# ---------------------------
# Plot 1: Gain Importance
# ---------------------------
sns.barplot(
    y="feature", x="gain_norm", data=top20,
    palette="coolwarm", ax=axes[0]
)
axes[0].set_title("ğŸŒŸ Top 20 Features by Gain", fontsize=14, weight='bold', pad=15)
axes[0].set_xlabel("Normalized Gain Importance", fontsize=12)
axes[0].set_ylabel("Feature Name", fontsize=12)
axes[0].grid(axis='x', linestyle='--', alpha=0.4)

# ---------------------------
# Plot 2: Split Importance
# ---------------------------
sns.barplot(
    y="feature", x="split_norm", data=top20,
    palette="viridis", ax=axes[1]
)
axes[1].set_title("âš¡ Top 20 Features by Split", fontsize=14, weight='bold', pad=15)
axes[1].set_xlabel("Normalized Split Importance", fontsize=12)
axes[1].set_ylabel("")
axes[1].grid(axis='x', linestyle='--', alpha=0.4)

# Overall formatting
plt.suptitle("ğŸ”� LightGBM Feature Importance â€” Gain vs Split", fontsize=16, weight='bold', y=1.02)
plt.tight_layout()
plt.show()



# ğŸ§  SHAP Explainability â€” Understanding Model Predictions (Fast Version)
import shap
import matplotlib.pyplot as plt

# Use a subset of training data to speed up SHAP
sample_X = X_train.sample(800, random_state=42)

# Create SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sample_X)[1]  # binary classification, positive class

# 1ï¸�âƒ£ SHAP Summary Plot (Global Feature Importance)
plt.title("ğŸŒˆ SHAP Summary Plot â€” Global Feature Importance", fontsize=14)
shap.summary_plot(shap_values, sample_X, plot_type="dot", show=True)

# 2ï¸�âƒ£ SHAP Dependence Plot (For a Single Feature)
key_feature = "interest_rate"  # change to your top encoded feature if needed
plt.title(f"ğŸ“Š SHAP Dependence Plot â€” Impact of {key_feature}", fontsize=14)
shap.dependence_plot(key_feature, shap_values, sample_X, show=True)

# 3ï¸�âƒ£ SHAP Force Plot (For an Individual Prediction)
sample_index = 10
shap.initjs()
shap.force_plot(
    explainer.expected_value[1],
    shap_values[sample_index, :],
    sample_X.iloc[sample_index, :]
)



# 1ï¸�âƒ£ Test data loaded and preprocessed like training (numerical features etc.)
X_test_enc = X_test.copy()

# 2ï¸�âƒ£ Encode categorical features (paste this here)
from sklearn.preprocessing import LabelEncoder

cat_cols = ["gender", "marital_status", "education_level", "employment_status", "loan_purpose", "grade_subgrade"]

for col in cat_cols:
    le = LabelEncoder()
    le.fit(pd.concat([X_train[col], X_test_enc[col]], axis=0).astype(str))  # Fit on train + test
    X_test_enc[col] = le.transform(X_test_enc[col].astype(str))

# 3ï¸�âƒ£ Now you can safely make predictions
test_predictions = model.predict_proba(X_test_enc)[:, 1]

# 4ï¸�âƒ£ Create submission
submission = pd.DataFrame({
    "id": test["id"],
    "loan_paid_back": test_predictions
})
submission.to_csv("submission.csv", index=False)
print("âœ… Submission file created")


