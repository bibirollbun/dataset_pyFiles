import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")


def check(df):
    """
    Generates a concise summary of DataFrame columns.
    """
    # Compute values that are constant across columns
    total_rows = len(df)
    duplicates = df.duplicated().sum()

    # Use vectorized operations 
    dtypes = df.dtypes
    instances = df.count()
    unique = df.nunique()
    sum_null = df.isnull().sum()
    #null_percentage = (df.isnull().sum() / total_rows * 100).round(2)

    # Create the summary 
    df_check = pd.DataFrame({
        #'column': df.columns,
        'dtype': dtypes,
        'instances': instances,
        'unique': unique,
        'sum_null': sum_null,
        #'null_percentage': null_percentage,
        'duplicates': duplicates  
    })

    return df_check

print("Train Data")
display(check(train))
display(train.head())

print("Test Data")
display(check(test))
display(test.head())



df = train

# ─────────────────────
# Target preparation
# ─────────────────────
# diagnosed_diabetes is already float, we'll treat 1 = diabetes, 0 = no diabetes
df["has_diabetes"] = df["diagnosed_diabetes"].astype("Int8")  # clean name

# ─────────────────────
# Target distribution
# ─────────────────────
plt.figure(figsize=(5, 4))
sns.countplot(
    data=df,
    x="has_diabetes",
    hue="has_diabetes",
    palette="Set2",

)
plt.title("Diabetes Distribution (700k records)", fontsize=13, pad=10)
plt.xlabel("Has Diabetes (1 = Yes)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

print(df["has_diabetes"].value_counts(normalize=True).round(3) * 100)
print("\nRaw counts:\n", df["has_diabetes"].value_counts())

# ─────────────────────────
# Age vs Diabetes 
# ─────────────────────────
plt.figure(figsize=(10, 5))
sns.histplot(
    data=df,
    x="age",
    hue="has_diabetes",
    multiple="stack",
    bins=40,
    palette="Set1",
    stat="proportion"
)
plt.title("Age Distribution by Diabetes Status", fontsize=13)
plt.xlabel("Age (years)")
plt.tight_layout()
plt.show()

# Smoothed version
plt.figure(figsize=(10, 5))
sns.kdeplot(
    data=df,
    x="age",
    hue="has_diabetes",
    fill=True,
    common_norm=False,
    palette="Set1"
)
plt.title("Age Density by Diabetes Status", fontsize=13)
plt.xlabel("Age (years)")
plt.tight_layout()
plt.show()

# ────────────────────────────────────────────────
# Numerical features vs target – box + violin style
# ────────────────────────────────────────────────
num_cols = [
    "age",
    "bmi",
    "waist_to_hip_ratio",
    "systolic_bp",
    "diastolic_bp",
    "cholesterol_total",
    "hdl_cholesterol",
    "ldl_cholesterol",
    "triglycerides",
    "sleep_hours_per_day",
    "physical_activity_minutes_per_week",
    "diet_score"
]

# plot only the most important ones to avoid too many subplots
important_num = ["age", "bmi", "waist_to_hip_ratio", "systolic_bp", "triglycerides", "hdl_cholesterol"]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flat

for i, col in enumerate(important_num):
    sns.boxplot(
        data=df,
        x="has_diabetes",
        y=col,
        hue="has_diabetes",
        palette="Set2",
        ax=axes[i]
    )
    axes[i].set_title(f"{col.replace('_', ' ').title()} by Diabetes")
    axes[i].set_xlabel("")

# Hide unused subplots if any
for j in range(len(important_num), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()

# ────────────────────────────────────────────────
# Categorical features – diabetes rate within each category
# ────────────────────────────────────────────────
cat_cols = [
    "gender",
    "ethnicity",
    "smoking_status",
    "family_history_diabetes",
    "hypertension_history",
    "cardiovascular_history",
    # "education_level", "income_level", "employment_status" usually weaker signals
]

fig, axes = plt.subplots(3, 2, figsize=(14, 12))
axes = axes.ravel()

for i, col in enumerate(cat_cols):
    if col not in df.columns:
        continue
        
    # Mean diabetes rate per category
    prop = df.groupby(col)["has_diabetes"].mean().sort_values(ascending=False)
    
    sns.barplot(
        x=prop.index.astype(str),
        y=prop.values,
        palette="viridis",
        ax=axes[i]
    )
    axes[i].set_title(f"Diabetes Rate by {col.replace('_', ' ').title()}", fontsize=11)
    axes[i].set_ylabel("Proportion with Diabetes")
    axes[i].set_ylim(0, 1)
    axes[i].tick_params(axis='x', rotation=30 if len(prop) > 5 else 0)

plt.tight_layout()
plt.show()

# ────────────────────────────────────────────────
# Correlation heatmap (numeric features + target)
# ────────────────────────────────────────────────
corr_cols = [
    "age", "bmi", "waist_to_hip_ratio",
    "systolic_bp", "diastolic_bp", "heart_rate",
    "cholesterol_total", "hdl_cholesterol", "ldl_cholesterol", "triglycerides",
    "sleep_hours_per_day", "physical_activity_minutes_per_week", "diet_score",
    "has_diabetes"
]

plt.figure(figsize=(12, 10))
corr = df[corr_cols].corr()
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    annot_kws={"size": 8}
)
plt.title("Correlation Matrix – Key Numeric Features + Diabetes", fontsize=13)
plt.tight_layout()
plt.show()

# ────────────────────────────────────────────────
#pairplot on most important features (sample!)
# ────────────────────────────────────────────────
sample_df = df.sample(20000, random_state=42)

sns.pairplot(
    sample_df,
    vars=["age", "bmi", "waist_to_hip_ratio", "triglycerides", "hdl_cholesterol"],
    hue="has_diabetes",
    palette="Set1",
    corner=True,
    plot_kws={"s": 6, "alpha": 0.5}
)
plt.suptitle("Pairplot – Selected Risk Factors (20k sample)", y=1.02, fontsize=14)
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Target and exclusions
target_col = "has_diabetes"
exclude = ["id", "diagnosed_diabetes", "has_diabetes"]

features = [c for c in df.columns if c not in exclude]

X = train[features].copy()
y = train[target_col].astype(int)

print("Features used:", len(features))
print("Target positive rate:", y.mean())


# ────────────────────────────────────────────────
# Identify categorical columns
# ────────────────────────────────────────────────
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

print("Categorical columns:", cat_cols)

# Convert dtype
for col in cat_cols:
    X[col] = X[col].astype("category")

# ────────────────────────────────────────────────
# train / validation split
# ────────────────────────────────────────────────
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.15,
    stratify=y,
    random_state=42
)

print(f"Train size: {len(X_train):,}")
print(f"Valid size: {len(X_valid):,}")
print("Train positive rate:", y_train.mean())
print("Valid positive rate:", y_valid.mean())


# Prepare Test Set

X_test = test[features].copy()

for col in cat_cols:
    X_test[col] = X_test[col].astype("category")

print("Test shape:", X_test.shape)


import lightgbm as lgb
from sklearn.metrics import roc_auc_score, roc_curve, ConfusionMatrixDisplay
from sklearn.metrics import classification_report, precision_recall_curve


lgb_train = lgb.Dataset(
    X_train,
    label=y_train,
    categorical_feature=cat_cols,
    free_raw_data=False
)

lgb_valid = lgb.Dataset(
    X_valid,
    label=y_valid,
    reference=lgb_train,
    free_raw_data=False
)


# parameters
pos_rate = y_train.mean()
scale_pos_weight = (1 - pos_rate) / pos_rate

params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": 9,
    "min_data_in_leaf": 80,
    "feature_fraction": 0.75,
    "bagging_fraction": 0.85,
    "bagging_freq": 5,
    "scale_pos_weight": scale_pos_weight,
    "random_state": 42,
    "verbosity": -1
}

# Train Model
model = lgb.train(
    params,
    lgb_train,
    num_boost_round=4000,
    valid_sets=[lgb_valid],
    valid_names=["valid"],
    callbacks=[
        lgb.early_stopping(stopping_rounds=80, verbose=True),
        lgb.log_evaluation(period=100)
    ]
)

# Validation Evaluation
y_valid_pred = model.predict(X_valid)

auc = roc_auc_score(y_valid, y_valid_pred)
print(f"\nValidation ROC AUC: {auc:.4f}")


# Optimal Threshold Selection
prec, rec, thresh = precision_recall_curve(y_valid, y_valid_pred)
f1 = 2 * (prec * rec) / (prec + rec + 1e-9)

best_thresh = thresh[f1.argmax()]
print("Best threshold (F1-optimal):", round(best_thresh, 4))

y_valid_bin = (y_valid_pred >= best_thresh).astype(int)

print("\nClassification Report @ optimal threshold:")
print(classification_report(y_valid, y_valid_bin))


# ROC curve
fpr, tpr, _ = roc_curve(y_valid, y_valid_pred)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Validation Set")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# Feature Importance
plt.figure(figsize=(10, 8))
lgb.plot_importance(
    model,
    importance_type="gain",
    max_num_features=20
)
plt.title("Top 20 Features by Gain")
plt.tight_layout()
plt.show()


# Confusion Matrix
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(
    y_valid,
    y_valid_bin,
    normalize="true",
    cmap="Blues",
    ax=ax
)
plt.title("Confusion Matrix (Normalized)")
plt.show()


test_pred = model.predict(X_test)
#test_pred = (test_pred_prob >= best_thresh).astype(int)

submission = pd.DataFrame({
    "id": test["id"],
    "diagnosed_diabetes": test_pred      
})

submission.to_csv("submission.csv", index=False)
submission.head()

