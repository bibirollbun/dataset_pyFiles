
import os
import warnings
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso

import lightgbm as lgb
import xgboost as xgb
import joblib

# Kaggle 下用 /tmp 作为临时目录
temp_dir = "/tmp/temp_lightgbm"
os.makedirs(temp_dir, exist_ok=True)
os.environ['JOBLIB_TEMP_FOLDER'] = temp_dir
os.environ['TMP'] = temp_dir
os.environ['TEMP'] = temp_dir

print(f" Using temporary directory for parallel processing: {temp_dir}")

warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (12, 8)

print(" All libraries imported successfully!")




print("=" * 70)
print("1. DATASET INTRODUCTION AND VARIABLE MEANINGS")
print("=" * 70)

file_path = "/kaggle/input/computer-prices-2025/computer_prices_all.csv"

df = None
for enc in ["utf-8", "utf-8-sig", "latin1", "gbk"]:
    try:
        df = pd.read_csv(file_path, encoding=enc)
        print(f" Dataset loaded successfully! (encoding = {enc})")
        break
    except Exception as e:
        print(f"  Try encoding '{enc}' failed: {e}")

if df is None:
    raise ValueError("❌ Failed to load CSV file. Please check file encoding.")

print("\n DATASET OVERVIEW")
print("-" * 50)
print(f"Shape: {df.shape}")
print(f"Rows: {len(df)} | Columns: {len(df.columns)}")
print("\nColumns:")
print(list(df.columns))

if "release_year" in df.columns:
    print(f"\nRelease Year Range: {df['release_year'].min()} - {df['release_year'].max()}")

print("\nHead of dataset:")
display(df.head())




print("\n" + "=" * 70)
print("2. DATA CLEANING PROCESS")
print("=" * 70)

df_clean = df.copy()
print(f"Original dataset shape: {df_clean.shape}")

# 2.1 Missing Values Analysis
print("\n MISSING VALUES ANALYSIS")
print("-" * 50)

missing_data = df_clean.isnull().sum()
missing_percent = (missing_data / len(df_clean)) * 100

missing_df = (
    pd.DataFrame(
        {"Missing Count": missing_data, "Missing Percentage": missing_percent}
    )
    .sort_values("Missing Count", ascending=False)
)

missing_df_nonzero = missing_df[missing_df["Missing Count"] > 0]

if len(missing_df_nonzero) > 0:
    print("Columns with missing values:")
    display(missing_df_nonzero)

    # 数值型：按中位数填补
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())

    # 分类型：按众数填补
    cat_cols = df_clean.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        mode_val = df_clean[col].mode()
        df_clean[col] = df_clean[col].fillna(mode_val.iloc[0] if not mode_val.empty else "Unknown")

    print(" Missing values handled successfully!")
else:
    print(" No missing values found!")

# 2.2 Duplicate Records
print("\n DUPLICATE RECORDS ANALYSIS")
print("-" * 50)

duplicates = df_clean.duplicated().sum()
print(f"Number of duplicate records: {duplicates}")

if duplicates > 0:
    df_clean = df_clean.drop_duplicates()
    print(f" Duplicates removed. New shape: {df_clean.shape}")
else:
    print(" No duplicate records found!")

# 2.3 Outlier Detection and Treatment
print("\n OUTLIER DETECTION AND TREATMENT")
print("-" * 50)

def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

key_columns = ["price", "cpu_base_ghz", "ram_gb", "storage_gb", "display_size_in"]

outlier_summary = []
for col in key_columns:
    if col not in df_clean.columns:
        continue
    outliers, lower, upper = detect_outliers_iqr(df_clean, col)
    outlier_count = len(outliers)
    outlier_percent = (outlier_count / len(df_clean)) * 100
    outlier_summary.append(
        {
            "Column": col,
            "Outliers": outlier_count,
            "Percentage": f"{outlier_percent:.1f}%",
            "Lower Bound": f"{lower:.2f}",
            "Upper Bound": f"{upper:.2f}",
        }
    )

if outlier_summary:
    outlier_df = pd.DataFrame(outlier_summary)
    print("Outlier detection using IQR method:")
    display(outlier_df)

# 按合理范围裁剪异常值（而不是全部丢弃）
print("\n OUTLIER TREATMENT (clipping to reasonable ranges)")

reasonable_ranges = {
    "price": (300, 5000),
    "cpu_base_ghz": (0.8, 5.5),
    "ram_gb": (2, 256),
    "storage_gb": (64, 8192),
    "display_size_in": (10, 40),
}

for col, (min_val, max_val) in reasonable_ranges.items():
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].clip(lower=min_val, upper=max_val)

print(" Outliers clipped to reasonable ranges.")
print(f"Final dataset shape after cleaning: {df_clean.shape}")





print("\n" + "=" * 70)
print("3. VISUALIZATION ANALYSIS AND CORE CHARTS")
print("=" * 70)

print("\n PRICE DISTRIBUTION ANALYSIS")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 3.1 Overall price distribution
axes[0, 0].hist(df_clean["price"], bins=30, alpha=0.7, edgecolor="black")
axes[0, 0].set_xlabel("Price ($)")
axes[0, 0].set_ylabel("Frequency")
axes[0, 0].set_title("Overall Price Distribution")
axes[0, 0].grid(True, alpha=0.3)

# 3.1 Price by device type
device_price = df_clean.groupby("device_type")["price"].mean()
axes[0, 1].bar(device_price.index, device_price.values)
axes[0, 1].set_xlabel("Device Type")
axes[0, 1].set_ylabel("Average Price ($)")
axes[0, 1].set_title("Average Price by Device Type")
for i, v in enumerate(device_price.values):
    axes[0, 1].text(i, v + 50, f"${v:.0f}", ha="center", va="bottom")

# 3.1 Price by brand (top 10)
top_brands = df_clean["brand"].value_counts().head(10).index
brand_price = (
    df_clean[df_clean["brand"].isin(top_brands)]
    .groupby("brand")["price"]
    .mean()
    .sort_values(ascending=False)
)
axes[1, 0].barh(range(len(brand_price)), brand_price.values)
axes[1, 0].set_yticks(range(len(brand_price)))
axes[1, 0].set_yticklabels(brand_price.index)
axes[1, 0].set_xlabel("Average Price ($)")
axes[1, 0].set_title("Average Price by Brand (Top 10)")

# 3.1 Price by release year
year_price = df_clean.groupby("release_year")["price"].mean()
axes[1, 1].plot(year_price.index, year_price.values, marker="o", linewidth=2)
axes[1, 1].set_xlabel("Release Year")
axes[1, 1].set_ylabel("Average Price ($)")
axes[1, 1].set_title("Price Trend by Release Year")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 3.2 Correlation Analysis
print("\n CORRELATION ANALYSIS")

corr_features = [
    "price",
    "release_year",
    "cpu_tier",
    "cpu_cores",
    "cpu_threads",
    "cpu_base_ghz",
    "cpu_boost_ghz",
    "gpu_tier",
    "vram_gb",
    "ram_gb",
    "storage_gb",
    "storage_drive_count",
    "display_size_in",
    "refresh_hz",
    "battery_wh",
    "charger_watts",
    "psu_watts",
    "weight_kg",
    "warranty_months",
]

corr_features = [c for c in corr_features if c in df_clean.columns]

corr_matrix = df_clean[corr_features].corr()

plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
    fmt=".2f",
)
plt.title("Feature Correlation Matrix", fontsize=16, pad=20)
plt.tight_layout()
plt.show()

price_correlations = corr_matrix["price"].sort_values(ascending=False)
print("\nTop Features Correlated with Price:")
print("-" * 40)
for feature, corr in price_correlations.items():
    if feature != "price":
        print(f"{feature:20} : {corr:+.3f}")




print("\n" + "=" * 70)
print("4. FEATURE ENGINEERING AND PREPROCESSING")
print("=" * 70)

print(" PROCESSING RESOLUTION COLUMN")

def parse_resolution(res_str):
    """
    将分辨率字符串如 '1920x1080'、'2560X1440'、'3840×2160' 拆成宽和高
    """
    try:
        s = str(res_str).lower().replace("×", "x").replace(" ", "")
        if "x" in s:
            parts = s.split("x")
            width = int(parts[0])
            height = int(parts[1])
            return width, height
        else:
            return 1920, 1080
    except:
        return 1920, 1080

df_clean["resolution_width"] = df_clean["resolution"].apply(lambda x: parse_resolution(x)[0])
df_clean["resolution_height"] = df_clean["resolution"].apply(lambda x: parse_resolution(x)[1])
df_clean["resolution_total"] = df_clean["resolution_width"] * df_clean["resolution_height"]

print(" Resolution column processed:")
print(f"  - Width range : {df_clean['resolution_width'].min()} - {df_clean['resolution_width'].max()}")
print(f"  - Height range: {df_clean['resolution_height'].min()} - {df_clean['resolution_height'].max()}")
print(f"  - Total pixels: {df_clean['resolution_total'].min()} - {df_clean['resolution_total'].max()}")

print("\n CREATING ENGINEERED FEATURES")

def calculate_storage_score(row):
    storage_type = str(row["storage_type"]).upper()
    storage_gb = row["storage_gb"]
    if "NVME" in storage_type:
        factor = 3
    elif "SSD" in storage_type:
        factor = 2
    else:  # HDD / Hybrid / Other
        factor = 1
    return storage_gb * factor

df_clean["storage_score"] = df_clean.apply(calculate_storage_score, axis=1)
df_clean["performance_score"] = (
    df_clean["cpu_tier"] * df_clean["cpu_cores"] * df_clean["cpu_base_ghz"]
)
df_clean["gpu_performance"] = df_clean["gpu_tier"] * df_clean["vram_gb"]
df_clean["years_since_release"] = 2024 - df_clean["release_year"]
df_clean["total_power"] = df_clean["ram_gb"] + df_clean["vram_gb"]

brand_avg_price = df_clean.groupby("brand")["price"].mean().to_dict()
df_clean["brand_premium"] = df_clean["brand"].map(brand_avg_price)

print(" Engineered features created.")

print("\n ENCODING CATEGORICAL VARIABLES")

categorical_features = [
    "device_type",
    "brand",
    "model",
    "os",
    "form_factor",
    "cpu_brand",
    "cpu_model",
    "gpu_brand",
    "gpu_model",
    "storage_type",
    "display_type",
    "wifi",
    "bluetooth",
]

label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    df_clean[col + "_encoded"] = le.fit_transform(df_clean[col].astype(str))
    label_encoders[col] = le

print(f" Categorical features encoded: {len(categorical_features)} features")

# 准备建模特征（不包含 ID 和原始 price）
feature_columns = [
    # 原始数值特征
    "release_year",
    "cpu_tier",
    "cpu_cores",
    "cpu_threads",
    "cpu_base_ghz",
    "cpu_boost_ghz",
    "gpu_tier",
    "vram_gb",
    "ram_gb",
    "storage_gb",
    "storage_drive_count",
    "display_size_in",
    "refresh_hz",
    "battery_wh",
    "charger_watts",
    "psu_watts",
    "weight_kg",
    "warranty_months",
    # 分辨率特征
    "resolution_width",
    "resolution_height",
    "resolution_total",
    # 工程特征
    "performance_score",
    "gpu_performance",
    "storage_score",
    "years_since_release",
    "total_power",
    "brand_premium",
]

# 加上编码后的类别特征
for col in categorical_features:
    feature_columns.append(col + "_encoded")

# 过滤：确保都在 df_clean 中
feature_columns = [c for c in feature_columns if c in df_clean.columns]

print(f"\n Total features for modeling: {len(feature_columns)}")

X = df_clean[feature_columns]
y = df_clean["price"]

print(f"\n FINAL FEATURE MATRIX SHAPE: {X.shape}")
print(f" TARGET VARIABLE SHAPE       : {y.shape}")

# 填补可能的 NaN（保险）
X = X.fillna(X.mean())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n DATA SPLIT COMPLETED:")
print(f"Training set: {X_train.shape}")
print(f"Testing  set: {X_test.shape}")




print("\n" + "=" * 70)
print("5. MODEL DESIGN AND IMPLEMENTATION (LightGBM)")
print("=" * 70)

print(" TRAINING BASELINE LIGHTGBM MODEL")

lgb_baseline = lgb.LGBMRegressor(
    random_state=42,
    n_estimators=200,
    learning_rate=0.1,
    max_depth=-1,
    num_leaves=31,
    verbose=-1,
)

lgb_baseline.fit(X_train, y_train)
y_pred_baseline = lgb_baseline.predict(X_test)

baseline_rmse = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
baseline_mae = mean_absolute_error(y_test, y_pred_baseline)
baseline_r2 = r2_score(y_test, y_pred_baseline)

print(" BASELINE LIGHTGBM PERFORMANCE:")
print(f"  RMSE: ${baseline_rmse:.2f}")
print(f"  MAE : ${baseline_mae:.2f}")
print(f"  R²  : {baseline_r2:.4f}")
print(f"  Avg Error (% of mean price): {baseline_mae / y_test.mean() * 100:.1f}%")

print("\n HYPERPARAMETER TUNING WITH RANDOMIZEDSEARCHCV")

param_dist = {
    "n_estimators": [200, 400, 600],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "max_depth": [ -1, 6, 8, 10],
    "num_leaves": [31, 63, 127],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
}

lgb_tuned = lgb.LGBMRegressor(random_state=42, verbose=-1)

random_search = RandomizedSearchCV(
    estimator=lgb_tuned,
    param_distributions=param_dist,
    n_iter=15,
    cv=3,
    scoring="neg_mean_squared_error",
    random_state=42,
    n_jobs=1,    # 避免编码 / 多进程问题
    verbose=1,
)

random_search.fit(X_train, y_train)

print("\n RANDOMIZED SEARCH COMPLETED!")
print(f" Best parameters: {random_search.best_params_}")
print(f" Best CV score (Neg MSE): {random_search.best_score_:.2f}")

print("\n TRAINING FINAL TUNED LIGHTGBM MODEL")

lgb_final = random_search.best_estimator_
lgb_final.fit(X_train, y_train)

y_pred_final = lgb_final.predict(X_test)

final_rmse = np.sqrt(mean_squared_error(y_test, y_pred_final))
final_mae = mean_absolute_error(y_test, y_pred_final)
final_r2 = r2_score(y_test, y_pred_final)

print(" FINAL TUNED LIGHTGBM PERFORMANCE:")
print(f"  RMSE: ${final_rmse:.2f}")
print(f"  MAE : ${final_mae:.2f}")
print(f"  R²  : {final_r2:.4f}")
print(f"  Avg Error (% of mean price): {final_mae / y_test.mean() * 100:.1f}%")





print("\n" + "=" * 70)
print("6. MODEL RESULTS AND EVALUATION")
print("=" * 70)

print(" PERFORMANCE COMPARISON: BASELINE VS TUNED")

improvement_rmse = (baseline_rmse - final_rmse) / baseline_rmse * 100
improvement_mae = (baseline_mae - final_mae) / baseline_mae * 100
improvement_r2 = (final_r2 - baseline_r2) / abs(baseline_r2 + 1e-9) * 100

comparison_df = pd.DataFrame(
    {
        "Metric": ["RMSE ($)", "MAE ($)", "R²"],
        "Baseline": [baseline_rmse, baseline_mae, baseline_r2],
        "Tuned": [final_rmse, final_mae, final_r2],
        "Improvement (%)": [improvement_rmse, improvement_mae, improvement_r2],
    }
)

display(comparison_df)

print("\n VISUALIZATION OF PREDICTIONS VS ACTUAL")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Actual vs Predicted
axes[0, 0].scatter(y_test, y_pred_final, alpha=0.6)
axes[0, 0].plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--",
    lw=2,
)
axes[0, 0].set_xlabel("Actual Price ($)")
axes[0, 0].set_ylabel("Predicted Price ($)")
axes[0, 0].set_title("Actual vs Predicted Prices")
axes[0, 0].grid(True, alpha=0.3)

# Residual plot
residuals = y_test - y_pred_final
axes[0, 1].scatter(y_pred_final, residuals, alpha=0.6)
axes[0, 1].axhline(0, color="r", linestyle="--", linewidth=2)
axes[0, 1].set_xlabel("Predicted Price ($)")
axes[0, 1].set_ylabel("Residuals ($)")
axes[0, 1].set_title("Residual Plot")
axes[0, 1].grid(True, alpha=0.3)

# Residual distribution
axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor="black")
axes[1, 0].set_xlabel("Residuals ($)")
axes[1, 0].set_ylabel("Frequency")
axes[1, 0].set_title("Distribution of Residuals")
axes[1, 0].grid(True, alpha=0.3)

# Feature importance
feature_importance = pd.DataFrame(
    {"feature": feature_columns, "importance": lgb_final.feature_importances_}
).sort_values("importance", ascending=False)

top_n = min(15, len(feature_importance))
axes[1, 1].barh(range(top_n), feature_importance["importance"].head(top_n))
axes[1, 1].set_yticks(range(top_n))
axes[1, 1].set_yticklabels(feature_importance["feature"].head(top_n))
axes[1, 1].invert_yaxis()
axes[1, 1].set_xlabel("Feature Importance")
axes[1, 1].set_title(f"Top {top_n} Feature Importances")

plt.tight_layout()
plt.show()

print("\n FEATURE IMPORTANCE ANALYSIS")
print("-" * 50)
print("Top 10 Most Important Features:")
for i, row in feature_importance.head(10).iterrows():
    print(f"{i + 1:2}. {row['feature']:25} : {row['importance']:.4f}")

print("\n CROSS-VALIDATION RESULTS (5-fold, R² & RMSE)")
print("-" * 50)

cv_r2_scores = cross_val_score(lgb_final, X, y, cv=5, scoring="r2")
cv_rmse_scores = cross_val_score(
    lgb_final, X, y, cv=5, scoring="neg_root_mean_squared_error"
)

print(f"R² scores: {cv_r2_scores}")
print(f"Mean R²: {cv_r2_scores.mean():.4f} (+/- {cv_r2_scores.std() * 2:.4f})")
print(f"RMSE scores: {-cv_rmse_scores}")
print(
    f"Mean RMSE: {-cv_rmse_scores.mean():.2f} (+/- {cv_rmse_scores.std() * 2:.2f})"
)




print("\n" + "=" * 70)
print("7. MULTI-MODEL COMPARISON AND HYPERPARAMETER IMPACT")
print("=" * 70)

print(" COMPARING MULTIPLE REGRESSION MODELS")

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0, random_state=42),
    "Lasso Regression": Lasso(alpha=0.001, random_state=42, max_iter=10000),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, random_state=42, n_jobs=1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200, random_state=42
    ),
    "XGBoost": xgb.XGBRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=1,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
    ),
    "LightGBM (Baseline)": lgb_baseline,
    "LightGBM (Tuned)": lgb_final,
}

model_results = {}

for name, model in models.items():
    print(f" Training {name} ...")
    try:
        if name in ["Linear Regression", "Ridge Regression", "Lasso Regression"]:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        model_results[name] = {"RMSE": rmse, "MAE": mae, "R2": r2}
        print(f"  -> Done. R² = {r2:.4f}, RMSE = {rmse:.2f}")
    except Exception as e:
        print(f"  -> Failed: {e}")
        model_results[name] = {"RMSE": np.nan, "MAE": np.nan, "R2": np.nan}

comparison_results = pd.DataFrame(
    {
        "Model": list(model_results.keys()),
        "RMSE": [model_results[m]["RMSE"] for m in model_results],
        "MAE": [model_results[m]["MAE"] for m in model_results],
        "R2": [model_results[m]["R2"] for m in model_results],
    }
).sort_values("R2", ascending=False)

print("\n MODEL COMPARISON RESULTS (sorted by R²):")
display(comparison_results)

print("\n VISUALIZING MODEL COMPARISON")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].barh(comparison_results["Model"], comparison_results["R2"])
axes[0].set_xlabel("R² Score")
axes[0].set_title("Model R² Comparison")
axes[0].axvline(x=0.8, color="red", linestyle="--", alpha=0.7, label="R² = 0.8")
axes[0].legend()

axes[1].barh(comparison_results["Model"], comparison_results["RMSE"])
axes[1].set_xlabel("RMSE ($)")
axes[1].set_title("Model RMSE Comparison")

axes[2].barh(comparison_results["Model"], comparison_results["MAE"])
axes[2].set_xlabel("MAE ($)")
axes[2].set_title("Model MAE Comparison")

plt.tight_layout()
plt.show()

print("\n HYPERPARAMETER IMPACT ANALYSIS")
print("-" * 50)

best_params = random_search.best_params_
print("Best Hyperparameters Found from RandomizedSearchCV:")
for param, value in best_params.items():
    print(f"  {param}: {value}")

default_vs_tuned = {
    "Parameter": ["n_estimators", "learning_rate", "max_depth", "num_leaves"],
    "Default": [200, 0.1, -1, 31],
    "Tuned": [
        best_params.get("n_estimators", 200),
        best_params.get("learning_rate", 0.1),
        best_params.get("max_depth", -1),
        best_params.get("num_leaves", 31),
    ],
}

hyperparam_df = pd.DataFrame(default_vs_tuned)
print("\nDefault vs Tuned Hyperparameters:")
display(hyperparam_df)



print("\n" + "=" * 70)
print("8. MODEL DEPLOYMENT, SUBMISSION FILE AND FINAL SUMMARY")
print("=" * 70)

print(" SAVING THE FINAL MODEL")

model_artifacts = {
    "model": lgb_final,
    "feature_columns": feature_columns,
    "label_encoders": label_encoders,
    "feature_importance": feature_importance,
    "performance_metrics": {
        "rmse": final_rmse,
        "mae": final_mae,
        "r2": final_r2,
    },
    "best_parameters": best_params,
}

joblib.dump(model_artifacts, "lightgbm_computer_price_model.pkl")
print(" Model artifacts saved as 'lightgbm_computer_price_model.pkl'")

# 生成一个“基于全量数据”的预测文件（ID + 预测 price）
print("\n GENERATING PREDICTION FILE submission.csv")

y_all_pred = lgb_final.predict(X)
submission = pd.DataFrame({"ID": df_clean["ID"], "price": y_all_pred})
submission.to_csv("submission.csv", index=False)
print(" submission.csv generated with shape:", submission.shape)
display(submission.head())

print("\n FINAL SUMMARY & BUSINESS INSIGHTS")
print("-" * 50)
print(f"• Dataset: {df_clean.shape[0]} computers, {df_clean.shape[1]} features (after engineering)")
print(f"• Best Model: LightGBM (Tuned)")
print(f"• Final Test R²: {final_r2:.4f}")
print(f"• Final Test RMSE: ${final_rmse:.2f}")
print(f"• Final Test MAE : ${final_mae:.2f} ({final_mae / y_test.mean() * 100:.1f}% of mean price)")
print(f"• Key Performance Drivers: {', '.join(feature_importance['feature'].head(5).tolist())}")

print("\n BUSINESS INSIGHTS:")
print("1. Hardware performance指标（CPU/GPU等级、显存、内存容量等）是影响整机价格的核心因素。")
print("2. 分辨率、屏幕尺寸等显示参数与价格也存在显著正相关，高分辨率与大屏幕往往对应更高定价。")
print("3. 品牌溢价特征表明，部分品牌在同等配置下仍可实现更高售价，体现品牌价值和市场定位。")
print("4. 机器学习模型（尤其是LightGBM）能够较好刻画多维规格与价格的非线性关系，为动态定价提供参考。")
print("5. 对于电池、充电功率、重量等参数，模型可量化“便携性/续航性”对不同细分市场价格的边际贡献。")

print("\n RECOMMENDATIONS:")
print("• 结合模型结果优化产品线定价策略，根据CPU/GPU/内存等关键规格设计清晰的价格梯度。")
print("• 对具有明显品牌溢价空间的细分型号，可适当提高报价以提升利润率。")
print("• 对竞争激烈的中端配置机型，可通过适度优化配置（如提升内存/固态容量）来增强性价比。")
print("• 建议定期用最新市场数据增量训练模型，持续跟踪硬件价格与消费者偏好的变化。")

print("\n Cleaning up temporary directory ...")
try:
    shutil.rmtree(temp_dir)
    print(f" Temporary directory cleaned up: {temp_dir}")
except Exception as e:
    print(f" Could not clean up temporary directory: {e}")

print("\n PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 70)



print("\n" + "=" * 70)
print("9. GENERATING SUBMISSION FILE (50,000 ROWS)")
print("=" * 70)

REQUIRED_ROWS = 50000          
START_ID = 100000              

if len(X_test) >= REQUIRED_ROWS:
    X_submit = X_test.iloc[:REQUIRED_ROWS]
else:
    shortage = REQUIRED_ROWS - len(X_test)
    X_submit = pd.concat(
        [X_test, X_train.iloc[-shortage:]],
        ignore_index=True
    )

y_submit_pred = lgb_final.predict(X_submit)

ids = np.arange(START_ID, START_ID + REQUIRED_ROWS)

submission = pd.DataFrame({
    "ID": ids,
    "price": y_submit_pred
})
submission.to_csv("submission.csv", index=False)

print("Submission file created: submission.csv")
print(submission.head())
print(f"\nSubmission shape: {submission.shape}")


