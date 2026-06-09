import pandas as pd

train_path = "/kaggle/input/widsdatathon2022/train.csv"
test_path  = "/kaggle/input/widsdatathon2022/test.csv"

train_df = pd.read_csv(train_path)
test_df  = pd.read_csv(test_path)


# Basic dataset inspection 

print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)

print("\nTrain columns:")
print(train_df.columns.tolist())

print("\nTarget statistics (site_eui):")
print(train_df["site_eui"].describe())

print("\nMissing values (Top 15 columns):")
missing_ratio = (train_df.isnull().sum() / len(train_df)) * 100
display(missing_ratio.sort_values(ascending=False).head(15))



import matplotlib.pyplot as plt
import seaborn as sns

# Plot distribution BEFORE outlier removal
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
sns.histplot(train_df["site_eui"], bins=100, kde=True)
plt.title("Site EUI Distribution (Before Outlier Removal)")
plt.xlabel("site_eui")

# Boxplot for extreme values
plt.subplot(1, 2, 2)
sns.boxplot(x=train_df["site_eui"])
plt.title("Site EUI Boxplot (Before Outlier Removal)")

plt.tight_layout()
plt.show()

# Quantiles to guide threshold selection
train_df["site_eui"].quantile([0.90, 0.95, 0.97, 0.99, 0.995])


# --- Cell 3: Outlier removal using 99th percentile ---

# Compute 99th percentile threshold
q99 = train_df["site_eui"].quantile(0.99)
print(f"99th percentile threshold: {q99:.2f}")

# Filter data
train_df_99 = train_df[train_df["site_eui"] <= q99].copy()

print("Shape before:", train_df.shape)
print("Shape after :", train_df_99.shape)

# Plot distribution AFTER outlier removal
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
sns.histplot(train_df_99["site_eui"], bins=100, kde=True)
plt.title("Site EUI Distribution (After 99% Outlier Removal)")
plt.xlabel("site_eui")

plt.subplot(1, 2, 2)
sns.boxplot(x=train_df_99["site_eui"])
plt.title("Site EUI Boxplot (After 99% Outlier Removal)")

plt.tight_layout()
plt.show()



# --- Cell 4: Final outlier removal using 97th percentile ---

# Compute 97th percentile threshold
q97 = train_df["site_eui"].quantile(0.97)
print(f"97th percentile threshold: {q97:.2f}")

# Filter data
train_df_clean = train_df[train_df["site_eui"] <= q97].copy()

print("Shape before:", train_df.shape)
print("Shape after :", train_df_clean.shape)

# Plot distribution AFTER 97% outlier removal
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
sns.histplot(train_df_clean["site_eui"], bins=100, kde=True)
plt.title("Site EUI Distribution (After 97% Outlier Removal)")
plt.xlabel("site_eui")

plt.subplot(1, 2, 2)
sns.boxplot(x=train_df_clean["site_eui"])
plt.title("Site EUI Boxplot (After 97% Outlier Removal)")

plt.tight_layout()
plt.show()



# --- Final Outlier Removal (95th percentile) ---

# Compute 95th percentile threshold
q95 = train_df["site_eui"].quantile(0.95)
print(f"95th percentile threshold: {q95:.2f}")

# Filter data
train_df_clean = train_df[train_df["site_eui"] <= q95].copy()

print("Shape before:", train_df.shape)
print("Shape after :", train_df_clean.shape)

# Plot distribution AFTER 95% outlier removal
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
sns.histplot(train_df_clean["site_eui"], bins=100, kde=True)
plt.title("Site EUI Distribution (After 95% Outlier Removal)")
plt.xlabel("site_eui")

plt.subplot(1, 2, 2)
sns.boxplot(x=train_df_clean["site_eui"])
plt.title("Site EUI Boxplot (After 95% Outlier Removal)")

plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

# Missing ratio before handling
missing_before = (train_df_clean.isnull().sum() / len(train_df_clean)) * 100
missing_before = missing_before[missing_before > 0].sort_values(ascending=False)

plt.figure(figsize=(10, 4))
sns.barplot(x=missing_before.index, y=missing_before.values)
plt.xticks(rotation=90)
plt.ylabel("Missing Percentage (%)")
plt.title("Missing Values Before Handling")
plt.tight_layout()
plt.show()



# --- Missing Values Handling (according to the paper) ---

# Calculate missing value percentage per column
missing_ratio = (train_df_clean.isnull().sum() / len(train_df_clean)) * 100

# Columns to drop (>= 50% missing values)
cols_to_drop = missing_ratio[missing_ratio >= 50].index.tolist()
print("Dropped columns (>=50% missing):")
print(cols_to_drop)

# Drop high-missing columns
train_df_mv = train_df_clean.drop(columns=cols_to_drop)

# Fill remaining missing values with the most frequent value (mode)
for col in train_df_mv.columns:
    if train_df_mv[col].isnull().sum() > 0:
        train_df_mv[col].fillna(train_df_mv[col].mode()[0], inplace=True)

print("\nShape before missing handling:", train_df_clean.shape)
print("Shape after missing handling :", train_df_mv.shape)

# Sanity check
print("\nRemaining missing values:")
print(train_df_mv.isnull().sum().sum())



# Missing ratio after handling
missing_after = (train_df_mv.isnull().sum() / len(train_df_mv)) * 100
missing_after = missing_after[missing_after > 0]

if len(missing_after) == 0:
    print(" No missing values remain after preprocessing.")
else:
    plt.figure(figsize=(6, 3))
    sns.barplot(x=missing_after.index, y=missing_after.values)
    plt.title("Missing Values After Handling")
    plt.ylabel("Missing Percentage (%)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()



# Identify categorical columns
categorical_cols = train_df_mv.select_dtypes(include=["object"]).columns.tolist()
print("Categorical columns:")
print(categorical_cols)



# One-Hot Encoding of categorical features
train_df_encoded = pd.get_dummies(
    train_df_mv,
    columns=categorical_cols,
    drop_first=False
)

print("Shape before encoding:", train_df_mv.shape)
print("Shape after encoding :", train_df_encoded.shape)


# Check if any categorical columns remain
remaining_cat = train_df_encoded.select_dtypes(include=["object"]).columns
print("Remaining categorical columns:", remaining_cat.tolist())


# --- Pearson Correlation with target (site_eui) ---

# Separate target
target = "site_eui"

# Compute correlation of all features with target
pearson_corr = train_df_encoded.corr()[target].drop(target)

# Sort by absolute correlation
pearson_corr_sorted = pearson_corr.abs().sort_values(ascending=False)

pearson_corr_sorted.head(10)


import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
pearson_corr_sorted.head(20).plot(kind="barh")
plt.title("Top 20 Features by Pearson Correlation with site_eui")
plt.xlabel("Absolute Correlation")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# Keep features with meaningful correlation
corr_threshold = 0.02

selected_features_pearson = pearson_corr_sorted[
    pearson_corr_sorted >= corr_threshold
].index.tolist()

print("Number of selected features (Pearson):", len(selected_features_pearson))


train_df_pearson = train_df_encoded[selected_features_pearson + [target]]

train_df_pearson.shape


from sklearn.feature_selection import f_regression
import pandas as pd

# Separate features and target
X = train_df_pearson.drop(columns=["site_eui"])
y = train_df_pearson["site_eui"]

# Apply ANOVA F-test
f_scores, p_values = f_regression(X, y)

# Create ANOVA results dataframe
anova_df = pd.DataFrame({
    "feature": X.columns,
    "f_score": f_scores,
    "p_value": p_values
})

# Sort by importance
anova_df = anova_df.sort_values("f_score", ascending=False)

anova_df.head(10)



# Keep top-k features (example: top 90 like Pearson)
top_k = 90
selected_features_anova = anova_df.head(top_k)["feature"].tolist()

train_df_anova = train_df_pearson[selected_features_anova + ["site_eui"]]

print("Shape after ANOVA selection:", train_df_anova.shape)



import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.barplot(
    x=anova_df.head(20)["f_score"],
    y=anova_df.head(20)["feature"]
)
plt.title("Top 20 Features by ANOVA F-score")
plt.xlabel("F-score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


from sklearn.preprocessing import MinMaxScaler

# Separate features and target
X = train_df_anova.drop(columns=["site_eui"])
y = train_df_anova["site_eui"]

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)

# Convert back to DataFrame
X_scaled_df = pd.DataFrame(
    X_scaled,
    columns=X.columns,
    index=X.index
)

# Recombine with target
train_df_norm = pd.concat([X_scaled_df, y], axis=1)

train_df_norm.shape



import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
sns.histplot(X["energy_star_rating"], bins=50)
plt.title("Before Normalization")

plt.subplot(1,2,2)
sns.histplot(train_df_norm["energy_star_rating"], bins=50)
plt.title("After Min-Max Normalization")

plt.tight_layout()
plt.show()


# Figure 3 (Paper-like): site_eui distribution
# by building_class (Commercial vs Residential)

import matplotlib.pyplot as plt
import seaborn as sns

# Use the preprocessed dataframe (after outlier removal + missing handling)
df_plot = train_df_mv.copy()

# Keep only the two main classes (as in the paper)
df_plot = df_plot[df_plot["building_class"].isin(["Commercial", "Residential"])].copy()

# Optional: ensure consistent ordering in the legend
class_order = ["Commercial", "Residential"]

plt.figure(figsize=(10, 5))

# Histogram comparison (count-based) similar to the paper's distribution plot
sns.histplot(
    data=df_plot,
    x="site_eui",
    hue="building_class",
    hue_order=class_order,
    bins=60,
    element="step",     # gives a cleaner look
    stat="count",
    common_norm=False,  # do not normalize across classes
    kde=False
)

plt.title("Data Distribution with Respect to Energy Utilization and Building Class")
plt.xlabel("Site Energy Use Intensity (site_eui)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()



# Separate features and target
X = train_df_norm.drop(columns=["site_eui"])
y = train_df_norm["site_eui"]

print("X shape:", X.shape)
print("y shape:", y.shape)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training set:", X_train.shape, y_train.shape)
print("Test set    :", X_test.shape, y_test.shape)


from sklearn.model_selection import KFold

# 10-fold cross validation (as used in the paper)
cv = KFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
)


from sklearn.model_selection import GridSearchCV

param_grid_rf = {
    "n_estimators": [635],
    "max_features": [1.0],   # equivalent to "auto"
    "max_depth": [150],
    "min_samples_leaf": [1]
}

grid_rf = GridSearchCV(
    estimator=rf,
    param_grid=param_grid_rf,
    cv=cv,
    scoring="neg_mean_absolute_error",  # paper reports MAE & RMSE
    n_jobs=-1,
    verbose=2
)



grid_rf.fit(X_train, y_train)


grid_rf.best_params_


best_rf = grid_rf.best_estimator_

print("Best RF parameters:")
print(grid_rf.best_params_)



from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Train predictions
y_train_pred = best_rf.predict(X_train)
y_test_pred  = best_rf.predict(X_test)

mae_train = mean_absolute_error(y_train, y_train_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))

mae_test = mean_absolute_error(y_test, y_test_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

print("RF Performance:")
print(f"Train MAE : {mae_train:.2f}")
print(f"Train RMSE: {rmse_train:.2f}")
print(f"Test  MAE : {mae_test:.2f}")
print(f"Test  RMSE: {rmse_test:.2f}")


plt.figure(figsize=(5,5))
plt.scatter(y_test, y_test_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--')
plt.xlabel("True site_eui")
plt.ylabel("Predicted site_eui")
plt.title("Random Forest: True vs Predicted (Test Set)")
plt.tight_layout()
plt.show()



residuals = y_test - y_test_pred

plt.figure(figsize=(6,3))
sns.histplot(residuals, bins=50, kde=True)
plt.xlabel("Prediction Error (Residual)")
plt.title("Random Forest Residual Distribution (Test Set)")
plt.tight_layout()
plt.show()


importances = best_rf.feature_importances_
feature_names = X_train.columns

imp_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values("importance", ascending=False).head(20)

plt.figure(figsize=(6,6))
sns.barplot(x="importance", y="feature", data=imp_df)
plt.title("Top 20 Feature Importances (Random Forest)")
plt.tight_layout()
plt.show()


from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(
    random_state=42
)


from sklearn.model_selection import GridSearchCV

param_grid_dt = {
    "max_depth": [5],
    "max_features": [1],      
    "min_samples_leaf": [2],
    "max_leaf_nodes": [40],
    "min_weight_fraction_leaf": [0.1],
    "splitter": ["random"]
}

grid_dt = GridSearchCV(
    estimator=dt,
    param_grid=param_grid_dt,
    cv=cv,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=2
)


grid_dt.fit(X_train, y_train)


grid_dt.best_params_


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

best_dt = grid_dt.best_estimator_

y_train_pred = best_dt.predict(X_train)
y_test_pred  = best_dt.predict(X_test)

mae_train = mean_absolute_error(y_train, y_train_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))

mae_test = mean_absolute_error(y_test, y_test_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

print("DT Performance:")
print(f"Train MAE : {mae_train:.2f}")
print(f"Train RMSE: {rmse_train:.2f}")
print(f"Test  MAE : {mae_test:.2f}")
print(f"Test  RMSE: {rmse_test:.2f}")


from sklearn.model_selection import cross_val_score

dt = DecisionTreeRegressor(
    max_depth=5,
    max_features=1.0,
    min_samples_leaf=2,
    max_leaf_nodes=40,
    min_weight_fraction_leaf=0.1,
    splitter="random",
    random_state=42
)

cv_mae = -cross_val_score(
    dt, X, y,
    cv=10,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

cv_rmse = np.sqrt(
    -cross_val_score(
        dt, X, y,
        cv=10,
        scoring="neg_mean_squared_error",
        n_jobs=-1
    )
)

print(cv_mae.mean(), cv_rmse.mean())



from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

gbdt = GradientBoostingRegressor(
    random_state=42
)

param_grid_gbdt = {
    "n_estimators": [142],
    "max_depth": [40],
    "max_features": [1.0],
    "min_samples_leaf": [63],
    "subsample": [0.65],
    "learning_rate": [0.05]
}

grid_gbdt = GridSearchCV(
    estimator=gbdt,
    param_grid=param_grid_gbdt,
    cv=10,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=2
)


grid_gbdt.fit(X_train, y_train)


grid_gbdt.best_params_


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

best_gbdt = grid_gbdt.best_estimator_

y_train_pred = best_gbdt.predict(X_train)
y_test_pred  = best_gbdt.predict(X_test)

mae_train = mean_absolute_error(y_train, y_train_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))

mae_test = mean_absolute_error(y_test, y_test_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

print("GBDT Performance:")
print(f"Train MAE : {mae_train:.2f}")
print(f"Train RMSE: {rmse_train:.2f}")
print(f"Test  MAE : {mae_test:.2f}")
print(f"Test  RMSE: {rmse_test:.2f}")


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_svm = scaler.fit_transform(X_train)
X_test_svm  = scaler.transform(X_test)


from sklearn.svm import SVR

svm = SVR(kernel="rbf")


from sklearn.model_selection import GridSearchCV

param_grid_svm = {
    "C": [1.0],
    "gamma": [0.4],
    "epsilon": [0.2],
    "kernel": ["rbf"]
}

grid_svm = GridSearchCV(
    estimator=svm,
    param_grid=param_grid_svm,
    cv=10,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=2
)


grid_svm.fit(X_train_svm, y_train)


grid_svm.best_params_


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

best_svm = grid_svm.best_estimator_

y_train_pred = best_svm.predict(X_train_svm)
y_test_pred  = best_svm.predict(X_test_svm)

mae_train = mean_absolute_error(y_train, y_train_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))

mae_test = mean_absolute_error(y_test, y_test_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

print("SVM Performance:")
print(f"Train MAE : {mae_train:.2f}")
print(f"Train RMSE: {rmse_train:.2f}")
print(f"Test  MAE : {mae_test:.2f}")
print(f"Test  RMSE: {rmse_test:.2f}")



import numpy as np
from sklearn.model_selection import cross_val_score

cv = 10

cv_results = {}

models = {
    "RF": best_rf,
    "DT": best_dt,
    "GBDT": best_gbdt,
    "SVM": best_svm
}

for name, model in models.items():
    mae = -cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )

    rmse = np.sqrt(
        -cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="neg_mean_squared_error",
            n_jobs=-1
        )
    )

    cv_results[name] = {
        "MAE": mae,
        "RMSE": rmse
    }

    print(f"{name} CV done.")


import matplotlib.pyplot as plt

folds = np.arange(1, 11)

plt.figure(figsize=(8, 4))

plt.plot(folds, cv_results["SVM"]["MAE"], 'o--', label="SVM")
plt.plot(folds, cv_results["RF"]["MAE"],  'o-',  label="RF")
plt.plot(folds, cv_results["GBDT"]["MAE"],'o-',  label="GBDT")
plt.plot(folds, cv_results["DT"]["MAE"],  's:',  label="DT")

plt.xlabel("10-Fold CV")
plt.ylabel("MAE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 4))

plt.plot(folds, cv_results["SVM"]["RMSE"], 'o--', label="SVM")
plt.plot(folds, cv_results["RF"]["RMSE"],  'o-',  label="RF")
plt.plot(folds, cv_results["GBDT"]["RMSE"],'o-',  label="GBDT")
plt.plot(folds, cv_results["DT"]["RMSE"],  's:',  label="DT")

plt.xlabel("10-Fold CV")
plt.ylabel("RMSE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

