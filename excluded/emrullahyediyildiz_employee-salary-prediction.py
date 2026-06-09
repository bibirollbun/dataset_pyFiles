import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="seaborn")


# Root definieren
DATA_DIR = "/kaggle/input/thapar-summer-school-employee-salary-prediction/"

# Dateien laden
train = pd.read_csv(DATA_DIR + "train.csv")
test  = pd.read_csv(DATA_DIR + "test.csv")
sample_submission = pd.read_csv(DATA_DIR + "sample-submission.csv")



# Shape and first rows
print("Train shape:", train.shape)
print("Test shape :", test.shape)
display(train.head(3))


# Column overview
print("\nColumn types:")
print(train.dtypes)


# Missing values (Top 15)
na_train = train.isna().sum().sort_values(ascending=False).head(15)
na_test  = test.isna().sum().sort_values(ascending=False).head(15)
print("\nMissing values (train):\n", na_train)
print("\nMissing values (test):\n", na_test)


train.describe()


# Target column check (here assumed as 'Salary')
target_col = "salary"  # adjust if different
if target_col in train.columns:
    print("\nTarget describe():")
    print(train[target_col].describe())

 # Histogram of target
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,4))
    train[target_col].hist(bins=40)
    plt.xlabel(target_col)
    plt.ylabel("Count")
    plt.title("Target Distribution")
    plt.show()
else:
    print("âš ï¸� Target column not found â€“ please check column names!")


# ==============================
# ğŸ“‘ Columns & basic splits
# ==============================
target_col = "salary"
id_col     = "id"

# features = all columns except id + target
features = [c for c in train.columns if c not in [id_col, target_col]]
cat_cols = [c for c in features if train[c].dtype == "object"]
num_cols = [c for c in features if c not in cat_cols]

print(f"Features: {len(features)} | Numeric: {len(num_cols)} | Categorical: {len(cat_cols)}")



# ==========================================
# ğŸ“Š Correlation heatmap (numeric features)
# ==========================================
# Uses seaborn if available, otherwise falls back to matplotlib.
import numpy as np, pandas as pd, matplotlib.pyplot as plt
try:
    import seaborn as sns
    USE_SNS = True
except Exception:
    USE_SNS = False

num_for_corr = train[[*num_cols, target_col]].copy()
corr = num_for_corr.corr(numeric_only=True)

plt.figure(figsize=(7,5))
if USE_SNS:
    sns.heatmap(corr, annot=False, cmap="viridis")
else:
    # simple matplotlib fallback
    plt.imshow(corr, cmap="viridis"); plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
plt.title("Correlation Heatmap (numeric)")
plt.tight_layout()
plt.show()



# ================================
# Pairplot (sampled for speed)
# ================================
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")


numeric_features = num_cols  # from earlier split
sampled_train = train.sample(2000, random_state=42)  # sample to keep plot fast
sns.pairplot(sampled_train[numeric_features + [target_col]], diag_kind="kde")
plt.suptitle("Pairwise Relationships of Numeric Features", y=1.02)
plt.show()

# ================================
# Boxplots for categorical features
# ================================
for col in cat_cols:
    plt.figure(figsize=(8,4))
    sns.boxplot(data=train, x=col, y=target_col)
    plt.title(f"Salary Distribution by {col}")
    plt.xticks(rotation=45)
    plt.show()



# Age difference
train["age_diff"] = train["age"] - train["age_when_joined"]
test["age_diff"]  = test["age"] - test["age_when_joined"]

# Bonus per year
train["bonus_per_year"] = train["annual_bonus"] / (train["years_in_the_company"] + 1)
test["bonus_per_year"]  = test["annual_bonus"] / (test["years_in_the_company"] + 1)

# Total experience
train["total_experience"] = train["prior_years_experience"] + train["years_in_the_company"]
test["total_experience"]  = test["prior_years_experience"] + test["years_in_the_company"]

# Experience ratio
train["experience_ratio"] = train["prior_years_experience"] / (train["years_in_the_company"] + 1)
test["experience_ratio"]  = test["prior_years_experience"] / (test["years_in_the_company"] + 1)

# Companyâ€“Department combo
train["company_department"] = train["company"] + "_" + train["department"]
test["company_department"]  = test["company"] + "_" + test["department"]

# Employment type (argmax across one-hot style cols)
train["employment_type"] = train[["full_time","part_time","contractor"]].idxmax(axis=1)
test["employment_type"]  = test[["full_time","part_time","contractor"]].idxmax(axis=1)



# =========================================
# ğŸš€ compact encode + 5 models (fixed)
# =========================================
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet
import matplotlib.pyplot as plt

# Optional boosters (skip gracefully if not installed)
XGB_OK = LGB_OK = True
try:
    from xgboost import XGBRegressor
except Exception:
    XGB_OK = False
try:
    import lightgbm as lgb
    from lightgbm import LGBMRegressor
except Exception:
    LGB_OK = False

# --- columns ---
target_col, id_col = "salary", "id"
features = [c for c in train.columns if c not in [target_col, id_col]]
cat_cols = [c for c in features if train[c].dtype == "object"]
num_cols = [c for c in features if c not in cat_cols]

# --- robust: replace inf ---
train = train.replace([np.inf, -np.inf], np.nan)
test  = test.replace([np.inf, -np.inf], np.nan)

X = train[features]
y = train[target_col].values
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- single encoder (fast): numeric impute + ordinal encode categoricals ---
prep = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
    ],
    remainder="drop"
)

# fit-transform once (speed!)
Xtr_enc = prep.fit_transform(X_train)
Xva_enc = prep.transform(X_val)
Xte_enc = prep.transform(test[features])

# --- models (lean configs) ---
models = {
    "RandomForest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=42),
    "Ridge":        Ridge(alpha=1.0, random_state=42),
    "ElasticNet":   ElasticNet(alpha=0.08, l1_ratio=0.5, max_iter=4000, random_state=42),
}
if XGB_OK:
    models["XGBoost"] = XGBRegressor(
        n_estimators=1200,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=150   # âœ… moved to constructor (no warning)
    )
if LGB_OK:
    models["LightGBM"] = LGBMRegressor(
        n_estimators=3000, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        objective="mae", random_state=42
    )

# --- linear models benefit from scaling ---
scaler = StandardScaler()
Xtr_scaled = scaler.fit_transform(Xtr_enc)
Xva_scaled = scaler.transform(Xva_enc)
Xte_scaled = scaler.transform(Xte_enc)

rows = []
pred_val_cache = {}

for name, model in models.items():
    if name in ["Ridge", "ElasticNet"]:
        model.fit(Xtr_scaled, y_train)
        pred = model.predict(Xva_scaled)
    elif name == "XGBoost" and XGB_OK:
        # âœ… early_stopping set in constructor; no kwarg in fit()
        model.fit(Xtr_enc, y_train, eval_set=[(Xva_enc, y_val)], verbose=False)
        pred = model.predict(Xva_enc)
    elif name == "LightGBM" and LGB_OK:
        model.fit(Xtr_enc, y_train,
                  eval_set=[(Xva_enc, y_val)],
                  eval_metric="mae",
                  callbacks=[lgb.early_stopping(150, verbose=False)])
        pred = model.predict(Xva_enc)
    else:
        model.fit(Xtr_enc, y_train)
        pred = model.predict(Xva_enc)

    rmse = mean_squared_error(y_val, pred, squared=False)
    mae  = mean_absolute_error(y_val, pred)
    rows.append({"Model": name, "Val RMSE": rmse, "Val MAE": mae})
    pred_val_cache[name] = pred

results_df = pd.DataFrame(rows).sort_values("Val MAE").reset_index(drop=True)

best_name = results_df.loc[0, "Model"]
print("âœ… Best by MAE:", best_name)
best_model = models[best_name]


#  fit best on full train 
# re-encode full train once
Xfull_enc = prep.fit_transform(X)
Xtest_enc = prep.transform(test[features])

if best_name in ["Ridge", "ElasticNet"]:
    scaler_full = StandardScaler().fit(Xfull_enc)
    Xfull_enc = scaler_full.transform(Xfull_enc)
    Xtest_enc = scaler_full.transform(Xtest_enc)

# final fit
if best_name == "XGBoost" and XGB_OK:
    best_model.fit(Xfull_enc, y, verbose=False)
elif best_name == "LightGBM" and LGB_OK:
    best_model.fit(Xfull_enc, y)
else:
    best_model.fit(Xfull_enc, y)


display(results_df)
print("âœ… Best by MAE:", best_name)



# predict test & build submission
test_pred = best_model.predict(Xtest_enc)
sub = sample_submission.copy()
target_cols = [c for c in sub.columns if c.lower() not in ["id", "employee_id", "emp_id"]]
assert len(target_cols) == 1, f"Ambiguous target in sample_submission: {list(sub.columns)}"
sub[target_cols[0]] = test_pred
sub.to_csv("submission.csv", index=False)
print("ğŸ’¾ Saved submission.csv")
display(sub.head())



plt.figure(figsize=(6,4))
plt.barh(results_df["Model"], results_df["Val MAE"], color="skyblue")
plt.xlabel("Validation MAE")
plt.title("Model Comparison (lower = better)")
plt.gca().invert_yaxis()
plt.show()



best_pred = best_model.predict(Xva_enc if best_name not in ["Ridge","ElasticNet"] else Xva_scaled)
plt.figure(figsize=(5,5))
plt.scatter(y_val, best_pred, alpha=0.3)
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title(f"{best_name}: Predicted vs Actual")
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], "r--")
plt.show()



import joblib

joblib.dump(best_model, f"{best_name}_model.pkl")
print(f"âœ… Saved {best_name}_model.pkl")





