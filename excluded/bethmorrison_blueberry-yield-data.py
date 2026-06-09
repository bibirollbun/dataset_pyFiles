!pip install kagglehub[pandas-datasets]


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="scipy")


# Install dependencies as needed:

import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load
file_path = "WildBlueberryPollinationSimulationData.csv"

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "shashwatwork/wild-blueberry-yield-prediction-dataset",
  file_path,
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

print("First 5 records:", df.head())


df.describe().loc[['count', 'mean', 'std', 'min', 'max']]


import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor


# Load and split the data
# df = pd.read_csv("WildBlueberryPollinationSimulationData.csv")
# Define features and target, dropping unused or redundant features
features = ['clonesize', 'honeybee', 'bumbles', 'andrena', 'osmia',
            'AverageOfUpperTRange', 'AverageOfLowerTRange', 'RainingDays',
           'MaxOfUpperTRange', 'MinOfUpperTRange', 'MaxOfLowerTRange', 'MinOfLowerTRange',
           'AverageRainingDays']


X = df[features]
y = df['yield']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Set up pipelines for each model
models = {
    "Linear": Pipeline([("scaler", StandardScaler()), ("regressor", LinearRegression())]),
    "RandomForest": Pipeline([("scaler", StandardScaler()), ("regressor", RandomForestRegressor(random_state=0))]),
    "XGBoost": Pipeline([("scaler", StandardScaler()), ("regressor", XGBRegressor(random_state=0, verbosity=0))])
}



# 1. Linear Regression (no hyperparameters to tune)
lin_scores = cross_val_score(models["Linear"], X_train, y_train, cv=5, scoring='r2')
lin_r2 = lin_scores.mean()


print(lin_scores)
print(lin_r2)


import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

# Ensure X is numeric and fill NaNs with 0
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

# Ensure y is numeric and drop rows where y is NaN
y = pd.to_numeric(y, errors='coerce')
mask = ~y.isna()
X = X[mask]
y = y[mask]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define pipelines
models = {
    "LinearRegression": Pipeline([
        ("scaler", StandardScaler()), 
        ("regressor", LinearRegression())
    ]),
    "RandomForest": Pipeline([
        ("regressor", RandomForestRegressor(random_state=0))
    ]),
    "XGBoost": Pipeline([
        ("regressor", XGBRegressor(random_state=0, verbosity=0))
    ])
}

# 1️⃣ Cross-validation on training set
print("=== Cross-validation R² on training set ===")
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    print(f"{name}: CV mean R² = {cv_scores.mean():.3f}")

# 2️⃣ Fit on full training set and evaluate on test set
print("\n=== Test set R² ===")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"{name}: Test R² = {r2:.3f}")

## LR test set is wrong because of some crazy values that LR is not robust against, doing the modeling correctly with robust ridge LR
## approach will give a value closer to 0.80 so I will say acceptable values are between 0.79 and 0.89


import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import pandas as pd

# Get the already-fitted XGBoost pipeline
xgb_pipeline = models["XGBoost"]   # this was fit in your test-eval loop

# If you didn't run the fit loop above, uncomment:
# xgb_pipeline.fit(X_train, y_train)

# Permutation importance on the held-out test set
result = permutation_importance(
    xgb_pipeline,
    X_test,
    y_test,
    scoring="r2",
    n_repeats=10,
    random_state=42,
    n_jobs=-1,
)

# Report test R² for reference
y_pred = xgb_pipeline.predict(X_test)
print("XGBoost Test R²:", r2_score(y_test, y_pred))

# Build a tidy importance table
importances = result.importances_mean
std = result.importances_std
feat_names = np.array(X_test.columns)

order = np.argsort(importances)[::-1]
imp_df = pd.DataFrame({
    "feature": feat_names[order],
    "perm_importance_mean": importances[order],
    "perm_importance_std": std[order],
})

# Show top 20 in text
print(imp_df.head(20).to_string(index=False))

# Plot
plt.figure(figsize=(10, 6))
plt.title("Permutation Feature Importance - XGBoost (Test Set)")
plt.bar(range(len(order)), importances[order], yerr=std[order], align="center")
plt.xticks(range(len(order)), feat_names[order], rotation=45, ha="right")
plt.tight_layout()
plt.show()



!pip install shap


# --- SHAP feature importances for the fitted XGBoost model (after permutation importance) ---

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 1) Get the already-fitted XGBoost pipeline and model
xgb_pipeline = models["XGBoost"]                # was fit earlier in your eval loop
# If you might not have run the fit loop, uncomment:
# xgb_pipeline.fit(X_train, y_train)

xgb_model = xgb_pipeline.named_steps["regressor"]

# 2) (Optional but recommended) Use a small background set for efficiency/stability
#    For tree models, SHAP can infer background, but a sample of training data is good practice
bg = X_train.sample(min(1000, len(X_train)), random_state=42)

# 3) Build the explainer and compute SHAP values on the test set
explainer = shap.Explainer(xgb_model, bg)   # Auto-selects TreeExplainer for XGB
shap_values = explainer(X_test)             # Keep as DataFrame to preserve feature names

# 4) Global importance = mean(|SHAP|) across test rows
#    shap_values.values may be deprecated in newer SHAP; .values works across versions.
vals = shap_values.values if hasattr(shap_values, "values") else shap_values.data
mean_abs = np.abs(vals).mean(axis=0)

imp_df = pd.DataFrame({
    "Feature": X_test.columns,
    "Mean |SHAP|": mean_abs
}).sort_values("Mean |SHAP|", ascending=False).reset_index(drop=True)

print("Top 20 features by mean |SHAP|:")
print(imp_df.head(20).to_string(index=False))

# 5) Plot (top 20) for readability
top = 20
plt.figure(figsize=(10, 6))
plt.title("SHAP Global Importance (Mean |SHAP|) - XGBoost (Test Set)")
order = np.argsort(mean_abs)[::-1][:top]
plt.bar(range(len(order)), mean_abs[order], align="center")
plt.xticks(range(len(order)), X_test.columns[order], rotation=45, ha="right")
plt.tight_layout()
plt.show()

# 6) (Optional) Classic SHAP plots
# shap.summary_plot(vals, X_test, feature_names=X_test.columns)              # dot summary
# shap.plots.bar(shap_values, max_display=20)                                # bar summary (new API)



# # --- Imports ---
# import kagglehub
# from kagglehub import KaggleDatasetAdapter

# import numpy as np
# import pandas as pd

# from sklearn.model_selection import train_test_split, KFold, cross_validate
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import RobustScaler
# from sklearn.linear_model import RidgeCV
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import r2_score
# from xgboost import XGBRegressor

# # --- Load data (your kagglehub call) ---
# file_path = "WildBlueberryPollinationSimulationData.csv"
# df = kagglehub.load_dataset(
#     KaggleDatasetAdapter.PANDAS,
#     "shashwatwork/wild-blueberry-yield-prediction-dataset",
#     file_path,
# )

# # --- Feature/target selection ---
# features = [
#     'clonesize','honeybee','bumbles','andrena','osmia',
#     'AverageOfUpperTRange','AverageOfLowerTRange','RainingDays',
#     'MaxOfUpperTRange','MinOfUpperTRange','MaxOfLowerTRange','MinOfLowerTRange',
#     'AverageRainingDays'
# ]

# X = df[features].apply(pd.to_numeric, errors='coerce')
# y = pd.to_numeric(df['yield'], errors='coerce')

# # Drop rows with NaN target only (feature NaNs handled by pipeline imputers)
# mask = ~y.isna()
# X, y = X.loc[mask], y.loc[mask]

# # --- Train/test split ---
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=0
# )

# # --- Outlier clipping based on TRAIN quantiles (winsorization) ---
# def fit_clip_bounds(X, lower=0.001, upper=0.999):
#     ql = X.quantile(lower)
#     qu = X.quantile(upper)
#     return ql, qu

# def apply_clip(X, lower_bounds, upper_bounds):
#     return X.clip(lower=lower_bounds, upper=upper_bounds, axis=1)

# lb, ub = fit_clip_bounds(X_train, lower=0.001, upper=0.999)
# X_train = apply_clip(X_train, lb, ub)
# # IMPORTANT: clip test using *train* bounds to avoid leakage
# X_test  = apply_clip(X_test,  lb, ub)

# # --- CV splitter (shuffled) ---
# kf = KFold(n_splits=5, shuffle=True, random_state=0)

# # --- Models (clean pipelines) ---
# models = {
#     "Linear_Ridge_Robust": Pipeline([
#         ("imputer", SimpleImputer(strategy="median")),
#         ("scaler", RobustScaler()),
#         ("regressor", RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], cv=kf))
#     ]),
#     "RandomForest": Pipeline([
#         ("imputer", SimpleImputer(strategy="median")),
#         ("regressor", RandomForestRegressor(
#             n_estimators=500, random_state=0, n_jobs=-1
#         ))
#     ]),
#     "XGBoost": Pipeline([
#         ("imputer", SimpleImputer(strategy="median")),
#         ("regressor", XGBRegressor(
#             objective="reg:squarederror",
#             n_estimators=600,
#             learning_rate=0.05,
#             max_depth=6,
#             subsample=0.9,
#             colsample_bytree=0.9,
#             random_state=0,
#             n_jobs=-1,
#             verbosity=0,
#             tree_method="hist"
#         ))
#     ])
# }

# # --- 1) Cross-validation (validation R² on training set) ---
# print("=== Cross-validation R² on training set (shuffled KFold) ===")
# for name, model in models.items():
#     cv = cross_validate(model, X_train, y_train, cv=kf, scoring="r2")
#     print(f"{name}: CV mean R² = {cv['test_score'].mean():.3f}  (± {cv['test_score'].std():.3f})")

# # --- 2) Fit on full training set and evaluate on test set ---
# print("\n=== Test set R² ===")
# for name, model in models.items():
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     print(f"{name}: Test R² = {r2_score(y_test, y_pred):.3f}")


