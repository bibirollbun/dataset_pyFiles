# ===============================
# 1. Imports
# ===============================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# -------------------------------
# 2. Load Data
# -------------------------------
train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')

X = train.drop(["id", "BeatsPerMinute"], axis=1)
y = train["BeatsPerMinute"]
X_test = test.drop("id", axis=1)
test_ids = test["id"]


#i will be not dropping id till training
train.columns


#no null values
train.isnull().sum()


#provides statistical information about the data
train.describe()


#understanding distributions of the features 
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
axes = axes.flatten()

for i, col in enumerate(train.columns[:9]):  # first 9 columns
    sns.histplot(train[col], ax=axes[i], kde=True, color="skyblue")
    axes[i].set_title(col)

plt.tight_layout()
plt.show()


#seems many columns have outliers 
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
axes = axes.flatten()

for i, col in enumerate(train.columns[:9]):  # first 9 columns
    sns.boxplot(train[col], ax=axes[i])
    axes[i].set_title(col)

plt.tight_layout()
plt.show()


#lets check the correlation between features and target 
#since there is no high coorelation between features and target wont be focusing on feature creation
sns.heatmap(train.corr(), cmap="rainbow", annot=True, fmt=".2f")


#Target seems to be normally distributed
sns.histplot(train["BeatsPerMinute"])



# -------------------------------------
# 3. Exploring Outlier Removal Functions
# ------------------------------------
def outlier_removal_iqr(X):
    X = X.copy()
    for col in X.columns:
        Q1, Q3 = X[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        X = X[(X[col] >= lower) & (X[col] <= upper)]
    return X

def outlier_removal_std(X):
    X = X.copy()
    for col in X.columns:
        mean, std = X[col].mean(), X[col].std()
        lower, upper = mean - 3 * std, mean + 3 * std
        X = X[(X[col] >= lower) & (X[col] <= upper)]
    return X

# Apply IQR to all features except VocalContent
removed_X = outlier_removal_iqr(X)
# Apply STD only to VocalContent
removed_X_std = outlier_removal_std(X)
cols_to_plot = removed_X.columns[:9]  # first 9 features
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
axes = axes.flatten()

for i, col in enumerate(cols_to_plot):
    # Build dataframe with both versions
    temp_df = pd.DataFrame({
        "IQR_removed": removed_X[col],
        "STD_removed": removed_X_std[col]
    })

    sns.boxplot(data=temp_df, ax=axes[i])
    axes[i].set_title(col)

plt.tight_layout()
plt.show()



# -----------------------------------------------------
# 3. "Outlier Removal Functions" As Per The Analysis 
#    I Will Be Using std method for Vocal Content
# -----------------------------------------------------
def outlier_removal_iqr(X):
    X = X.copy()
    for col in X.columns:
        Q1, Q3 = X[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        X = X[(X[col] >= lower) & (X[col] <= upper)]
    return X

def outlier_removal_std(X):
    X = X.copy()
    for col in X.columns:
        mean, std = X[col].mean(), X[col].std()
        lower, upper = mean - 3 * std, mean + 3 * std
        X = X[(X[col] >= lower) & (X[col] <= upper)]
    return X

# Apply IQR to all features except VocalContent
removed_X = outlier_removal_iqr(X.drop("VocalContent", axis=1))

# Apply STD only to VocalContent
removed_X_std = outlier_removal_std(X[["VocalContent"]])

# Keep only rows present in BOTH
common_idx = removed_X.index.intersection(removed_X_std.index)

# Merge them back safely
outlier_removed_df = pd.concat([removed_X.loc[common_idx], removed_X_std.loc[common_idx]], axis=1)

# Match y
y_clean = y.loc[common_idx]
print(f"Shape before outlier removal: {X.shape}")
print(f"Shape after outlier removal: {outlier_removed_df.shape}")



# --------------------------------------------------
# 4. Visualization - Boxplots After Outlier Removal
# --------------------------------------------------
cols_to_plot = outlier_removed_df.columns[:9]
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
axes = axes.flatten()

for i, col in enumerate(cols_to_plot):
    temp_df = pd.DataFrame({
        "Original": X[col],
        "Cleaned": outlier_removed_df[col]
    })
    sns.boxplot(data=temp_df, ax=axes[i])
    axes[i].set_title(f"Outlier Handling: {col}")

plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 5. As Distribution We Know There Are Some Skewed Features 
#    So Tying Which Method Works using(log and sqrt)
# ---------------------------------------------------------
for col in outlier_removed_df.columns:
    print(f"{col}: "
          f"Original={outlier_removed_df[col].skew():.2f}, "
          f"log1p={np.log1p(outlier_removed_df[col]).skew():.2f}, "
          f"sqrt={np.sqrt(outlier_removed_df[col]).skew():.2f}")


# --------------------------------------------
# 6. Applying Transformation On Skewed Columns
# --------------------------------------------
def apply_transformations(X, scaler=None, fit_scaler=True):
    X = X.copy()

    # From skewness analysis
    log_transform_cols = ["RhythmScore", "AcousticQuality"]
    sqrt_transform_cols = ["InstrumentalScore", "LivePerformanceLikelihood", "VocalContent"]

    for col in log_transform_cols:
        if col in X.columns:
            X[col] = np.log1p(X[col])
    for col in sqrt_transform_cols:
        if col in X.columns:
            X[col] = np.sqrt(X[col])

    scale_cols = ["TrackDurationMs", "AudioLoudness"]
    if scaler is None:
        scaler = StandardScaler()
    if fit_scaler:
        X[scale_cols] = scaler.fit_transform(X[scale_cols])
    else:
        X[scale_cols] = scaler.transform(X[scale_cols])

    return X, scaler

# Apply to training and test
X_clean, scaler = apply_transformations(outlier_removed_df, fit_scaler=True)
X_test, _ = apply_transformations(X_test, scaler=scaler, fit_scaler=False)
X_clean.dropna(inplace=True)
X_test.dropna(inplace=True)
X_test = X_test[X_clean.columns]


# --------------------------------------------------------------------
# 7. Model Training (RANDOMFOREST) Used Optuna For HyperParamater Tuning 
# ---------------------------------------------------------------------
model = RandomForestRegressor(
        n_estimators= 400, 
        max_depth= 6, 
        max_features= 0.7806627262109607, 
        min_samples_split= 10, 
        min_samples_leaf= 8, 
        bootstrap= True,
        n_jobs=-1
    )

model.fit(X_clean, y_clean)

# -------------------------------
# 8. Predictions & Submission
# -------------------------------
preds = model.predict(X_test)

submission = pd.DataFrame({
    "id": test_ids,
    "BeatsPerMinute": preds
})
submission.to_csv("submission.csv", index=False)

print("✅ submission.csv created successfully!")

