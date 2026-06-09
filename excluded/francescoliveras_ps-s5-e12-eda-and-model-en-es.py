import os
import random
import time
import warnings
from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Preprocessing & Evaluation
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, FunctionTransformer

warnings.filterwarnings("ignore")


SEED = 42
ID_COL = "id"
TARGET_COL = "diagnosed_diabetes"
PAIRPLOT_SAMPLE = 1000

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed()

# Paths - Adjusted for Docs/ folder as requested
DATA_DIR = Path("/kaggle/input/playground-series-s5e12")
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"

# Palette reproduced from Plantilla.ipynb
yellow = "#F7C53E"
cyan_g = "#0CF7AF"
cyan_dark = "#11AB7C"
purple = "#D826F8"
purple_dark = "#9309AB"
purple_light = "#b683d6"
blue = "#0C97FA"
red = "#FA1D19"
orange = "#FA9F19"
green = "#0CFA58"
light_blue = "#01FADC"
soft_blue = "#81c9e6"
dark_blue = "#394be6"

PALETTE_2 = [cyan_g, purple]
PALETTE_7 = [purple_dark, purple_light, purple, blue, light_blue, dark_blue, soft_blue]
PALETTE_7_C = [purple_dark, blue, purple, light_blue, purple_light, soft_blue, dark_blue]

# Plotting styling
sns.set_style("whitegrid")
sns.set_palette(PALETTE_7)  # Set the requested palette
plt.style.use({"figure.facecolor": "#f8fafc"})
pd.set_option("display.float_format", "{:.4f}".format)


if not TRAIN_PATH.exists():
    raise FileNotFoundError(f"File not found: {TRAIN_PATH}. Please check the path.")

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
sample_submission = pd.read_csv(SAMPLE_SUB_PATH)

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

display(train.head())


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    numeric_df = df.select_dtypes(include=["number"])
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, mask=mask, cmap=PALETTE_2, annot=True, fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()

def plot_pairgrid(df: pd.DataFrame, columns: list, sample_size: int = PAIRPLOT_SAMPLE) -> None:
    if len(columns) < 2:
        print("Not enough numeric features for pairplot.")
        return
    sample_df = df[columns].sample(min(len(df), sample_size), random_state=SEED)
    sns.pairplot(sample_df, diag_kind="kde", corner=True, plot_kws={"alpha": 0.5, "s": 20, "color": PALETTE_7[3]})
    plt.suptitle("Pairplot of Numeric Features", y=1.02)
    plt.show()

def plot_numeric_distributions(df: pd.DataFrame, numeric_cols: list, target: str) -> None:
    if not numeric_cols:
        print("No numeric features to profile.")
        return

    n_cols = 2
    n_rows = math.ceil(len(numeric_cols) / n_cols)
    plt.figure(figsize=(16, 5 * n_rows))
    for idx, col in enumerate(numeric_cols, 1):
        ax = plt.subplot(n_rows, n_cols, idx)
        sns.histplot(data=df, x=col, hue=target, kde=True, element="step", palette=PALETTE_2, ax=ax)
        ax.set_title(f"{col} distribution by {target}")
    plt.tight_layout()
    plt.show()

def plot_categorical_summary(df: pd.DataFrame, categorical_cols: list, target: str) -> None:
    if not categorical_cols:
        print("No categorical features to profile.")
        return

    for col in categorical_cols:
        plt.figure(figsize=(12, 5))
        ax1 = plt.subplot(1, 2, 1)
        order = df[col].value_counts().index
        sns.countplot(data=df, x=col, order=order, palette=PALETTE_7, ax=ax1)
        ax1.set_title(f"{col} Counts")
        ax1.tick_params(axis='x', rotation=45)
        
        ax2 = plt.subplot(1, 2, 2)
        sns.barplot(data=df, x=col, y=target, order=order, palette=PALETTE_7_C, errorbar=None, ax=ax2)
        ax2.set_title(f"Mean {target} by {col}")
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()


# Target Balance
plt.figure(figsize=(6, 4))
sns.countplot(data=train, x=TARGET_COL, palette=PALETTE_7)
plt.title("Target Distribution (Diagnosed Diabetes)")
plt.show()

normalize_counts = train[TARGET_COL].value_counts(normalize=True)
print("Target prevalence:\n", normalize_counts)


# Define column groups
CAT_COLS = [
    "gender", "ethnicity", "education_level", "income_level", 
    "smoking_status", "employment_status"
]
BINARY_COLS = ["family_history_diabetes", "hypertension_history", "cardiovascular_history"]

NUM_COLS = [
    c for c in train.columns 
    if c not in CAT_COLS and c not in [ID_COL, TARGET_COL] + BINARY_COLS
]

print("Numeric Columns:", NUM_COLS)
print("Categorical Columns:", CAT_COLS)


# Correlation Heatmap
plot_correlation_heatmap(train.drop(columns=[ID_COL]))


# Numeric Distributions
plot_numeric_distributions(train, NUM_COLS, TARGET_COL)


# Categorical Summaries
plot_categorical_summary(train, CAT_COLS, TARGET_COL)


# Pairplot for a subset of numeric features
plot_pairgrid(train, NUM_COLS[:5])  # Limit to first 5 to avoid clutter


# Splitting X and y
X = train.drop(columns=[ID_COL, TARGET_COL])
y = train[TARGET_COL]
X_test = test.drop(columns=[ID_COL])

# Preprocessing Pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Binary cols - passthrough or simple impute
binary_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, NUM_COLS),
        ('cat', categorical_transformer, CAT_COLS),
        ('bin', binary_transformer, BINARY_COLS)
    ]
)


# Define Models
models = {
    "Logistic Regression": LogisticRegression(random_state=SEED, max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
    "AdaBoost": AdaBoostClassifier(random_state=SEED),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=SEED, n_jobs=-1, use_label_encoder=False),
    "LightGBM": LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1)
}

results = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

print("Starting cross-validation...")
for name, model in models.items():
    start = time.time()
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])
    
    # Evaluamos con ROC AUC
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    
    mean_auc = scores.mean()
    std_auc = scores.std()
    elapsed = time.time() - start
    
    print(f"{name}: mean ROC AUC = {mean_auc:.4f} (+/- {std_auc:.4f})  [Time: {elapsed:.2f}s]")
    results.append({"Model": name, "Mean AUC": mean_auc, "Std AUC": std_auc, "Time": elapsed})

results_df = pd.DataFrame(results).sort_values(by="Mean AUC", ascending=False)
display(results_df)


# Visualizing Performance
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df, x="Mean AUC", y="Model", palette=PALETTE_7)
plt.xlim(0.5, 1.0)
plt.title("Model Comparison (5-Fold Stratified CV)")
plt.show()


best_model_name = results_df.iloc[0]["Model"]
best_model_instance = models[best_model_name]

print(f"Training best model ({best_model_name}) on full dataset...")

final_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', clone(best_model_instance))])

final_pipeline.fit(X, y)

# Predict probabilities for the positive class
test_preds = final_pipeline.predict_proba(X_test)[:, 1]

# Create submission file
submission = pd.DataFrame({
    ID_COL: test[ID_COL],
    TARGET_COL: test_preds
})

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file saved to {submission_path}")

display(submission.head())

