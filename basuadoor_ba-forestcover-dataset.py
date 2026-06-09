!pip install xgboost

# Imports
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


# Dataset Paths 
TRAIN_CSV = "/kaggle/input/forest-cover-type-prediction/train.csv"
TEST_CSV  = "/kaggle/input/forest-cover-type-prediction/test.csv"
SUBMISSION_CSV = "/kaggle/input/forest-cover-type-prediction/sampleSubmission.csv"


train = pd.read_csv(TRAIN_CSV)
test  = pd.read_csv(TEST_CSV)


train.head()


test.head()


train.dtypes


test.dtypes


train.isnull().sum(axis = 0)


print("\nBasic Info")
train.info()

# Summary stats
print("\nSummary Statistics")
display(train.describe().T)

# Unique values in target
if "Cover_Type" in train.columns:
    print("\nTarget Class Distribution")
    print(train["Cover_Type"].value_counts(normalize=True).sort_index())

train['Cover_Type'].value_counts()


# Missing TRAIN values
missing_train_set = train.isnull().sum()
missing_train_set = missing_train_set[missing_train_set > 0]
if len(missing_train_set) == 0:
    print("No missing train values found.")
else:
    print("Missing train values detected:\n", missing_train_set)

# Check for duplicates
dup_count = train.duplicated().sum()
print(f"\nDuplicate train rows: {dup_count}")


print(train.shape)
print(test.shape)


from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

num_cols = [
    "Elevation", "Aspect", "Slope",
    "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways", "Horizontal_Distance_To_Fire_Points",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"
]

# Z-score method to detect potential outliers
z_scores = np.abs(stats.zscore(train[num_cols]))
outlier_mask = (z_scores > 3).any(axis=1)
print(f"Outliers detected: {outlier_mask.sum()} rows ({100*outlier_mask.mean():.2f}%)")

# Remove outliers
train = train[~outlier_mask].reset_index(drop=True)
print(f"Cleaned dataset shape: {train.shape}")

outlier_percentage = outlier_mask.mean() * 100
print(f"\nApprox. {outlier_percentage:.2f}% of rows have extreme outliers (|z| > 3)")

vc = train["Cover_Type"].value_counts()
vc_percent = (vc / vc.sum() * 100).round(2)
print(pd.DataFrame({"Count": vc, "Percent": vc_percent}))

# Boxplots for a few key features
for c in ["Elevation", "Slope", "Vertical_Distance_To_Hydrology"]:
    sns.boxplot(x=train[c])
    plt.title(f"Boxplot: {c}")
    plt.show()




# Ensure train/test IDs are unique and disjoint
if "Id" in train.columns and "Id" in test.columns:
    overlap = set(train["Id"]) & set(test["Id"])
    print(f"Overlapping IDs between train/test: {len(overlap)}")

# Check for perfectly correlated columns (leakage risk)
corr = train[num_cols].corr().abs()
upper_tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
high_corr = [col for col in upper_tri.columns if any(upper_tri[col] > 0.98)]
if high_corr:
    print("\n Highly correlated columns (possible redundancy):", high_corr)
else:
    print("\n No perfect correlations detected.")


train[num_cols].hist(bins=25, figsize=(14, 10))
plt.suptitle("Distribution of Continuous Features", y=1.02)
plt.show()


# Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(train[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap – Continuous Features")
plt.show()

# Bivariate example: Elevation vs Cover_Type
sns.boxplot(x="Cover_Type", y="Elevation", data=train)
plt.title("Elevation by Cover Type")
plt.show()

# Relationship between Elevation & Hillshade
sns.scatterplot(x="Elevation", y="Hillshade_Noon", hue="Cover_Type", data=train.sample(2000))
plt.title("Elevation vs Hillshade (sampled 2000 points)")
plt.show()


np.round(train[num_cols].corr(), decimals=2)


# Select a manageable subset of continuous features
cont_cols = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Horizontal_Distance_To_Fire_Points",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm"
]

# Sample a smaller subset for readability (e.g., 1000 points)
sample_df = train.sample(n=1000, random_state=42)

# Create a pair plot
sns.set(style="whitegrid", context="notebook")
pair_plot = sns.pairplot(
    sample_df,
    vars=cont_cols[:5],       # choose top 5 for clarity
    hue="Cover_Type",
    palette="tab10",
    corner=True,
    plot_kws=dict(alpha=0.6, s=30, edgecolor="none")
)

pair_plot.fig.suptitle("Pair Plot of Continuous Features (Sample of 1000)", y=1.02)
plt.show()


# Data Cleaning & Preparation

# Dataset is clean; if not, fill or drop
train = train.fillna(train.median())


# Encode Categorical Variables
# Wilderness_Area and Soil_Type are already one-hot encoded (binary columns).
# If not encoded, use pd.get_dummies()
cat_cols = ["Wilderness_Area", "Soil_Type"] if "Wilderness_Area" in train.columns else []
if cat_cols:
    train = pd.get_dummies(train, columns=cat_cols, drop_first=False)


#Feature Scaling for continuous features
cont_cols = [
    "Elevation", "Aspect", "Slope",
    "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways", "Horizontal_Distance_To_Fire_Points",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"
]

scaler = StandardScaler()
train[cont_cols] = scaler.fit_transform(train[cont_cols])
test[cont_cols] = scaler.fit_transform(test[cont_cols])


def add_features(df):
    df = df.copy()
    df["Hillshade_Total"] = df["Hillshade_9am"] + df["Hillshade_Noon"] + df["Hillshade_3pm"]
    df["Hillshade_Mean"] = df[["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]].mean(axis=1)
    df["Hydrology_Abs_V"] = df["Vertical_Distance_To_Hydrology"].abs()
    df["Dist_Road_Fire_sum"] = df["Horizontal_Distance_To_Roadways"] + df["Horizontal_Distance_To_Fire_Points"]
    rad = np.deg2rad(df["Aspect"].replace({360: 0}))
    df["Aspect_Sin"] = np.sin(rad)
    df["Aspect_Cos"] = np.cos(rad)
    return df




train = add_features(train)
test  = add_features(test)


print("\nDerived features created:", [
    "Hillshade_Total", "Hillshade_Mean",
    "Hydrology_Abs_V", "Dist_Road_Fire_sum",
    "Aspect_Sin", "Aspect_Cos"
])


# y is original Cover_Type with values 1..7
y_raw = train["Cover_Type"]
le = LabelEncoder()
y = le.fit_transform(y_raw)   # maps {1,2,3,4,5,6,7} -> {0,1,2,3,4,5,6}

# Define Features & Target
X = train.drop(columns=["Cover_Type", "Id"], errors="ignore")
X_test = test[X.columns]
print(f"\nTrain: {X.shape}, Test: {test.shape}")

X.head(5)


# Define Stratified 5-Fold CV for cross validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ==========================================================
# FOREST COVER TYPE: Baseline Model Selection & Benchmarking
# ==========================================================

# ======================================================
# BASELINE MODELS
# ======================================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, multi_class="ovr", n_jobs=-1),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)    
}

results = []

print("BASELINE MODEL EVALUATION")
for name, model in models.items():
    model.fit(X, y)
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    cv_acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1).mean()
    print(f"{name}:Cross Validation Accuracy = {acc:.4f}, CV Accuracy = {cv_acc:.4f}")
    results.append((name, acc, cv_acc))

# Convert results to DataFrame for comparison
results_df = pd.DataFrame(results, columns=["Model", "Cross Validation_Accuracy", "CV_Accuracy"]).sort_values(by="CV_Accuracy", ascending=False)
print("\nModel Benchmark Results")
print(results_df)

# Plot benchmark results
sns.barplot(x="CV_Accuracy", y="Model", data=results_df, palette="viridis")
plt.title("Model Benchmark Comparison (Cross-Validation Accuracy)")
plt.xlabel("Cross-Validation Accuracy")
plt.ylabel("Model")
plt.show()


# ======================
# HYPERPARAMETER TUNING 
# ======================

# Logistic Regression tuning
lr_param = {
    "C": [0.01, 0.1, 1, 10, 100],
    "penalty": ["l2"],                  # l2 works with lbfgs/saga
    "solver": ["lbfgs", "saga"],        # both support multinomial
    "class_weight": [None, "balanced"],
    "multi_class": ["multinomial"]      # better for multi-class
}

lr_grid = GridSearchCV(
    estimator=LogisticRegression(max_iter=2000, multi_class="ovr", n_jobs=-1),
    param_grid=lr_param,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

lr_grid.fit(X, y)
print("Logistic Regression:")
print("Best params:", lr_grid.best_params_)
print("Best CV accuracy:", lr_grid.best_score_)
print()


#Decision Tree tuning
dt_param = {
    "criterion": ["gini", "entropy", "log_loss"],   # what to optimize
    "max_depth": [None, 5, 10, 15, 20, 25],         # tree depth
    "min_samples_split": [2, 5, 10, 20],            # min samples to split node
    "min_samples_leaf": [1, 2, 4, 8],               # min samples in each leaf
    "max_features": [None, "sqrt", "log2"]          # how many features considered per split
}

dt_grid = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=dt_param,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

dt_grid.fit(X, y)
print("Decision Tree:")
print("Best hyperparameters:", dt_grid.best_params_)
print("Best cross-validation accuracy:", dt_grid.best_score_)
print()


from scipy.stats import randint, uniform

#Gradient Boosting tuning
gb_param_dist = {
    "n_estimators": randint(50, 301),         # 50–300
    "learning_rate": uniform(0.01, 0.19),     # 0.01–0.20
    "max_depth": randint(2, 5),               # 2–4
    "min_samples_split": randint(2, 11),      # 2–10
    "min_samples_leaf": randint(1, 5),        # 1–4
    "subsample": uniform(0.7, 0.3),           # 0.7–1.0
    "max_features": ["sqrt", "log2"]          # fix choices
}


gb_grid = RandomizedSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_distributions=gb_param_dist,
    n_iter=40,            # try 20–50 instead of 1458
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1,
    random_state=42
)
gb_grid.fit(X, y)
print("Gradient Boosting:")
print("Best hyperparameters:", gb_grid.best_params_)
print("Best cross-validation accuracy:", gb_grid.best_score_)
print()


# Random Forest tuning 
rf_params = {
    "n_estimators": [200, 400, 600],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5, 10]
}
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid=rf_params,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)
rf_grid.fit(X, y)
print("Random Forest:")
print("Best hyperparameters:", rf_grid.best_params_)
print("Best CV Accuracy:", rf_grid.best_score_)
print()


# XGBoost tuning 
xgb_params = {
    "n_estimators": [200, 500],
    "learning_rate": [0.05, 0.1],
    "max_depth": [8, 10],
    "subsample": [0.8, 0.9],
    "colsample_bytree": [0.8, 0.9]
}
xgb_grid = GridSearchCV(
    XGBClassifier(
        objective="multi:softprob",
        num_class=7,                 # fixed from best_model
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=42,
        use_label_encoder=False
    ),
    param_grid=xgb_params,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)
xgb_grid.fit(X, y)
print("XGBoost:")
print("Best hyperparameters:", xgb_grid.best_params_)
print("Best CV Accuracy:", xgb_grid.best_score_)


# ======================================================
# FINAL EVALUATION & COMPARISON
# ======================================================
best_lr = lr_grid.best_estimator_
best_dt = dt_grid.best_estimator_
best_rf = rf_grid.best_estimator_
best_gb = gb_grid.best_estimator_
best_xgb = xgb_grid.best_estimator_

final_models = {
    "Logistic Regression (Tuned)":best_lr,
    "Decision Tree (Tuned)": best_dt,   
    "Random Forest (Tuned)": best_rf,
    "Gradient Boosting (Tuned)": best_gb,
    "XGBoost (Tuned)": best_xgb   
}

final_results = []
for name, model in final_models.items():
    model.fit(X, y)
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    final_results.append((name, acc))

final_results_df = pd.DataFrame(final_results, columns=["Model", "Validation_Accuracy"]).sort_values("Validation_Accuracy", ascending=False)
print("\nFinal Tuned Model Results")
print(final_results_df)


#Evaluate generalization performance using Cross-Validation
best_model = best_xgb.fit(X, y)
y_test_pred = best_model.predict(X_test)

scores = cross_val_score(best_model, X, y, cv=cv, scoring='accuracy')
print("CV accuracy mean:", scores.mean())


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ======================================================
# Print CV Summary Metrics
# ======================================================

print("Cross-Validation Accuracy Scores (All Folds):", scores)
print("Mean CV Accuracy:", scores.mean())
print("Std Dev of CV Accuracy:", scores.std())
print("Min Fold Accuracy:", scores.min())
print("Max Fold Accuracy:", scores.max())


# ======================================================
# Cross-Validation Score Distribution
# ======================================================

plt.figure(figsize=(8, 5))
sns.histplot(scores, kde=True, bins=5, color='green')
plt.title("Cross-Validation Accuracy Distribution")
plt.xlabel("Accuracy")
plt.ylabel("Frequency")
plt.show()


# ======================================================
# Boxplot of CV Accuracy
# ======================================================

plt.figure(figsize=(5, 5))
sns.boxplot(x=scores, color='lightblue')
plt.title("Cross-Validation Accuracy Boxplot")
plt.xlabel("Accuracy")
plt.show()


# ======================================================
# Violin Plot (more detailed shape)
# ======================================================

plt.figure(figsize=(5, 5))
sns.violinplot(x=scores, color='lightcoral')
plt.title("Cross-Validation Accuracy Violin Plot")
plt.xlabel("Accuracy")
plt.show()


# ======================================================
# Summary Interpretation (printed results)
# ======================================================

if scores.std() < 0.01:
    print("\nInterpretation: Very stable model — low variance across folds.")
elif scores.std() < 0.03:
    print("\nInterpretation: Moderately stable — some variation but acceptable.")
else:
    print("\nInterpretation: High variance — the model may be sensitive to training data or overfitting.")

print("Overall CV accuracy suggests how well the model generalizes.")


from sklearn.metrics import accuracy_score
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

best_model.fit(X_train, y_train)
y_valid_pred = best_model.predict(X_valid)
print("X:", X.shape)
print("X_train:", X_train.shape)
print("X_valid:", X_valid.shape)


print("Shapes:")
print("X_train:", X_train.shape)
print("X_valid:", X_valid.shape)
print("y_train:", y_train.shape)
print("y_valid:", y_valid.shape)

print("\nAccuracy on validation:", accuracy_score(y_valid, y_valid_pred))
print("All labels match?", np.array_equal(y_valid, y_valid_pred))


# ======================================================
# Basic performance summary
# ======================================================

print("Validation Accuracy:", accuracy_score(y_valid, y_valid_pred))

print("\nClassification Report:")
print(classification_report(y_valid, y_valid_pred, digits=3))


# ======================================================
# Build error dataframe
# ======================================================

errors_df = X_valid.copy()
errors_df["true"] = y_valid
errors_df["pred"] = y_valid_pred

wrong = errors_df[errors_df["true"] != errors_df["pred"]]
right = errors_df[errors_df["true"] == errors_df["pred"]]

print("\nTotal validation samples:", len(errors_df))
print("Correct predictions:", len(right))
print("Misclassifications:", len(wrong))


# ======================================================
# Which TRUE classes have the most errors?
# ======================================================

print("\nMisclassifications per TRUE class:")
print(wrong["true"].value_counts().sort_index())


# ======================================================
# Which predicted classes cause the most mistakes?
# ======================================================

print("\nMisclassifications per PREDICTED class:")
print(wrong["pred"].value_counts().sort_index())


# ======================================================
# True vs Predicted class distribution
# ======================================================

true_counts = pd.Series(y_valid).value_counts().sort_index()
pred_counts = pd.Series(y_valid_pred).value_counts().sort_index()

dist_df = pd.DataFrame({"True Count": true_counts, "Predicted Count": pred_counts})
dist_df.plot(kind="bar", figsize=(10,5))
plt.title("True vs Predicted Class Distribution")
plt.xlabel("Cover Type Label")
plt.ylabel("Samples")
plt.xticks(rotation=0)
plt.show()


# ======================================================
# Show examples of misclassified samples
# ======================================================

print("\nSample misclassified rows:")
print(wrong.head())


# ======================================================
# Normalized Confusion Matrix (hardest classes)
# ======================================================
cm = confusion_matrix(y_valid, y_valid_pred)
cm_norm = cm / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(10, 7))
sns.heatmap(cm_norm, annot=True, cmap="Reds", fmt=".2f")
plt.title("Normalized Confusion Matrix (Error Concentration)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


errors = y_valid != y_valid_pred
print("Number of errors:", errors.sum())
print("Error rate:", errors.mean())


# ----------------------------
# Compute Internal CV Score 
# ----------------------------

y_test_pred = best_model.predict(X_test)

scores = cross_val_score(best_model, X, y, cv=cv, scoring='accuracy')
print("CV accuracy mean:", scores.mean())

# ======================================================
# Print CV Summary Metrics
# ======================================================

print("Cross-Validation Accuracy Scores (All Folds):", scores)
print("Mean CV Accuracy:", scores.mean())
print("Std Dev of CV Accuracy:", scores.std())
print("Min Fold Accuracy:", scores.min())
print("Max Fold Accuracy:", scores.max())


import joblib

joblib.dump(best_model, "best_forest_cover_model.pkl")
print("Saved best_forest_cover_model.pkl")

