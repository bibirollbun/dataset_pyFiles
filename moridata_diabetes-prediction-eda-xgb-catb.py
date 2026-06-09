# =============================================================================
# ğŸ“¦ STANDARD LIBRARIES
# =============================================================================
import warnings
import numpy as np
import pandas as pd

# =============================================================================
# ğŸ“Š VISUALIZATION
# =============================================================================
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# ğŸ”§ PREPROCESSING & ENCODING
# =============================================================================
from category_encoders import TargetEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# =============================================================================
# ğŸ”§ MODELING & EVALUATION
# =============================================================================
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from catboost import CatBoostClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier


# =============================================================================
# âš™ï¸� SETTINGS / WARNINGS / ENVIRONMENT
# =============================================================================
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)


import pandas as pd

# ğŸ“¥ Load the datasets
train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
original = pd.read_csv("/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv")

# ---------------------------------------------------------
# STEP 1 â€” Add dataset labels
# ---------------------------------------------------------
train['dataset'] = 'train'
test['dataset']  = 'test'
original['dataset'] = 'train'   # treat as training-like

# ---------------------------------------------------------
# STEP 2 â€” Match ORIGINAL columns to TRAIN structure
# ---------------------------------------------------------
# âš ï¸� original dataset has many extra columns (glucose, insulin, A1C...)
# We must keep only the columns that train has.

train_cols = train.columns
original_trimmed = original.reindex(columns=train_cols)

# ---------------------------------------------------------
# STEP 3 â€” Combine train, test, original
# ---------------------------------------------------------
df = pd.concat([train, test, original_trimmed], axis=0, ignore_index=True)

# ---------------------------------------------------------
# STEP 4 â€” Show result
# ---------------------------------------------------------
print("Dataset shape:", df.shape)
df.head()


# # Load Data

# # ğŸ“¥ Load the dataset
# train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
# test  = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

# # original = pd.read_csv("")


# # Add a 'dataset' column to track source
# train['dataset'] = 'train'
# test['dataset'] = 'test'

# # original['dataset'] = 'train'



# # Combine train and test datasets for unified preprocessing
# df = pd.concat([train, test], axis=0).reset_index(drop=True)

# # ğŸ§¾ Display dataset shape
# print("Dataset shape:", df.shape)

# # ğŸ‘�ï¸� Preview the data
# df


original


df.shape


# ğŸ“‹ Check column types and non-null counts
df.info()


# âœ… Separate numerical and categorical columns
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()

print("Numerical Columns:", numerical_cols)
print("Categorical Columns:", categorical_cols)


# ğŸ”� Check for missing values
missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100
missing_df = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percent})
missing_df = missing_df[missing_df['Missing Values'] > 0]
missing_df


# ğŸ“Š Descriptive statistics for numerical columns
df[numerical_cols].describe()


# ğŸ”¢ Unique value counts for categorical columns
for col in categorical_cols:
    print(f"\nUnique values in '{col}':")
    print(df[col].value_counts())


def plot_categorical_target(df, target):
    """
    Plots distribution and statistics for any categorical target column.
    """

    # ---
    # 1. Count Plot
    # ---
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=target, palette='Set2')

    plt.title(f'Distribution of "{target}"', fontsize=14)
    plt.xlabel(target, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # ---
    # 2. Percentage Distribution
    # ---
    category_counts = df[target].value_counts(normalize=True) * 100

    plt.figure(figsize=(6, 4))
    sns.barplot(x=category_counts.index, 
                y=category_counts.values, 
                palette='Set2')

    plt.title(f'"{target}" (% Distribution)', fontsize=14)
    plt.xlabel(target, fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # ---
    # 3. Descriptive Statistics
    # ---
    print(f"\nğŸ“Š Summary for '{target}':")
    print(df[target].value_counts())

    print("\nğŸ”¢ Percentage Distribution:")
    print((df[target].value_counts(normalize=True) * 100)
          .round(2).astype(str) + '%')


# -----------------------------
target = "diagnosed_diabetes"   # 
plot_categorical_target(df, target)


def visualize_numeric_features(df, discrete_threshold=15):
    """
    Automatically visualizes all numerical features in a dataset.
    Detects discrete vs. continuous numeric columns.
    
    discrete_threshold = max number of unique values for a feature to be treated as discrete
    """

    # Detect numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Separate continuous vs discrete
    continuous_features = [
        col for col in numeric_cols 
        if df[col].nunique() > discrete_threshold
    ]

    discrete_features = [
        col for col in numeric_cols 
        if df[col].nunique() <= discrete_threshold
    ]

    print("ğŸ“Œ Detected Continuous Features:", continuous_features)
    print("ğŸ“Œ Detected Discrete Features:", discrete_features)
    print("=====================================================\n")

    # Loop through all numeric columns
    for col in numeric_cols:
        print(f"--- Visualizing: {col} ---")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f'Distribution of {col}', fontsize=16)

        # Continuous features â†’ Histogram + Boxplot
        if col in continuous_features:
            sns.histplot(df[col].dropna(), kde=True, bins=30,
                         ax=axes[0], edgecolor='black')
            axes[0].set_title('Histogram (Density)')
            axes[0].set_xlabel(col)

            sns.boxplot(x=df[col].dropna(), ax=axes[1])
            axes[1].set_title('Box Plot (Outliers)')
            axes[1].set_xlabel(col)

        # Discrete features â†’ Countplot + Boxplot
        else:  
            sns.countplot(x=df[col].dropna(), ax=axes[0], edgecolor='black')
            axes[0].set_title('Count Plot')
            axes[0].set_xlabel(col)

            sns.boxplot(x=df[col].dropna(), ax=axes[1])
            axes[1].set_title('Box Plot')
            axes[1].set_xlabel(col)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

        # Print stats
        print("\nğŸ“Š Descriptive Statistics:")
        print(df[col].describe().round(3))
        print("\n" + "="*60 + "\n")


# -------------------------------
# ğŸ�¯ Run on YOUR dataset
# -------------------------------
visualize_numeric_features(df)


# -------------------------------------------------------
# ğŸ“Š Distribution of Categorical Features
# -------------------------------------------------------

# -------------------------------------------------------
# âœ¨ Categorical Columns from YOUR dataset
# -------------------------------------------------------

cat_cols = [
    'gender',
    'ethnicity',
    'education_level',
    'income_level',
    'smoking_status',
    'employment_status',
    'family_history_diabetes',
    'hypertension_history',
    'cardiovascular_history',
    'diagnosed_diabetes'   # target, still categorical!
]

# -------------------------------------------------------
# âœ¨ Plot each categorical feature (same style as yours)
# -------------------------------------------------------

for col in cat_cols:
    plt.figure(figsize=(8, 4))

    # Countplot ordered by frequency
    sns.countplot(
        data=df,
        x=col,
        order=df[col].value_counts().index,
        palette='Set2',
        edgecolor='black'
    )

    plt.title(f'Distribution of {col}', fontsize=14)
    plt.xlabel(col.replace('_', ' ').title(), fontsize=12)
    plt.ylabel('Count', fontsize=12)

    # Adjust label rotation if categories are long
    plt.xticks(rotation=30, ha='right')

    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Print category proportions
    print(f'\nğŸ“Š Proportion of Each Category in "{col}":\n')
    print(df[col].value_counts(normalize=True).round(3))
    print('-' * 40 + "\n")


# ğŸ�¨ Categorical Feature Distributions by Diabetes Diagnosis Status - Custom Colors

# Select key categorical columns from your dataset
cols_to_plot = [
    'gender',
    'ethnicity',
    'education_level',
    'income_level',
    'smoking_status',
    'employment_status',
    'family_history_diabetes',
    'hypertension_history',
    'cardiovascular_history'
]

# NEW COLORS â€” Purple & Teal
custom_palette = ['#8E44AD', '#16A085']  # 0 = Purple, 1 = Teal

target_col = 'diagnosed_diabetes'

for col in cols_to_plot:
    plt.figure(figsize=(8, 5))
    
    sns.countplot(
        data=df,
        x=col,
        hue=target_col,
        palette=custom_palette,
        edgecolor='black',
        order=df[col].value_counts().index
    )
    
    plt.title(f'{col.replace("_", " ").title()} by Diabetes Diagnosis', fontsize=14)
    plt.xlabel(col.replace('_', ' ').title(), fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    plt.xticks(rotation=25, ha='right')
    
    plt.legend(title='Diabetes Diagnosis', labels=['No (0)', 'Yes (1)'])
    
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()


# ğŸ�¨ Numerical Feature Distributions by Diabetes Diagnosis Status

# Select key numerical columns from your dataset
num_cols = [
    'age',
    'alcohol_consumption_per_week',
    'physical_activity_minutes_per_week',
    'diet_score',
    'sleep_hours_per_day',
    'screen_time_hours_per_day',
    'bmi',
    'waist_to_hip_ratio',
    'systolic_bp',
    'diastolic_bp',
    'heart_rate',
    'cholesterol_total',
    'hdl_cholesterol',
    'ldl_cholesterol',
    'triglycerides'
]

target_col = 'diagnosed_diabetes'  # Binary target variable (0 = No, 1 = Yes)

# Custom colors: Purple = No Diabetes (0), Teal = Yes Diabetes (1)
custom_palette = ['#8E44AD', '#16A085']

for col in num_cols:
    plt.figure(figsize=(8, 4))
    
    sns.boxplot(
        data=df,
        x=target_col,
        y=col,
        palette=custom_palette,
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black"}
    )
    
    plt.title(f'{col.replace("_", " ").title()} by Diabetes Diagnosis', fontsize=14)
    plt.xlabel('Diabetes Diagnosis (0 = No, 1 = Yes)', fontsize=12)
    plt.ylabel(col.replace("_", " ").title(), fontsize=12)
    
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()


# Select numerical features + target
num_cols = [
    'age',
    'alcohol_consumption_per_week',
    'physical_activity_minutes_per_week',
    'diet_score',
    'sleep_hours_per_day',
    'screen_time_hours_per_day',
    'bmi',
    'waist_to_hip_ratio',
    'systolic_bp',
    'diastolic_bp',
    'heart_rate',
    'cholesterol_total',
    'hdl_cholesterol',
    'ldl_cholesterol',
    'triglycerides',
    'diagnosed_diabetes'  # Target
]

# Compute correlation matrix
corr_matrix = df[num_cols].corr()

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
plt.title('Correlation Heatmap (Numerical Features + Diabetes Diagnosis)', fontsize=16)
plt.tight_layout()
plt.show()


# Numerical columns from your diabetes dataset
num_cols = [
    'age',
    'alcohol_consumption_per_week',
    'physical_activity_minutes_per_week',
    'diet_score',
    'sleep_hours_per_day',
    'screen_time_hours_per_day',
    'bmi',
    'waist_to_hip_ratio',
    'systolic_bp',
    'diastolic_bp',
    'heart_rate',
    'cholesterol_total',
    'hdl_cholesterol',
    'ldl_cholesterol',
    'triglycerides'
]

# Determine the number of columns for plotting layout
n_cols = 5
n_rows = (len(num_cols) + n_cols - 1) // n_cols

plt.figure(figsize=(n_cols * 4, n_rows * 4))

for i, col in enumerate(num_cols, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.boxplot(y=df[col], color='lightcoral')
    plt.title(col.replace('_', ' ').title(), fontsize=10)
    plt.tight_layout()

plt.suptitle('ğŸ“¦ Outlier Detection via Boxplots', fontsize=16, y=1.02)
plt.show()


# =============================================================================
# DATA PREPARATION
# =============================================================================

# Split into train/test sets
train_df = df[df["dataset"] == "train"].copy()
test_df = df[df["dataset"] == "test"].copy()

print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")
print(f"Missing target values: {train_df['diagnosed_diabetes'].isna().sum()}")

# Separate features and target
X_train = train_df.drop(["id", "diagnosed_diabetes", "dataset"], axis=1)
y_train = train_df["diagnosed_diabetes"]

X_test = test_df.drop(["id", "diagnosed_diabetes", "dataset"], axis=1)

# =============================================================================
# FEATURE ENGINEERING (OPTIONAL)
# =============================================================================
# You can add new features here (e.g., BMI categories, activity ratios, etc.)

# =============================================================================
# TARGET ENCODING + PREPROCESSING
# =============================================================================

# Numerical columns to scale
num_cols = [
    "age",
    "alcohol_consumption_per_week",
    "physical_activity_minutes_per_week",
    "diet_score",
    "sleep_hours_per_day",
    "screen_time_hours_per_day",
    "bmi",
    "waist_to_hip_ratio",
    "systolic_bp",
    "diastolic_bp",
    "heart_rate",
    "cholesterol_total",
    "hdl_cholesterol",
    "ldl_cholesterol",
    "triglycerides"
]

# Categorical columns
cat_cols = [
    "gender",
    "ethnicity",
    "education_level",
    "income_level",
    "smoking_status",
    "employment_status",
    "family_history_diabetes",
    "hypertension_history",
    "cardiovascular_history"
]

# Columns to encode using Target Encoding (categorical + boolean)
cols_to_encode = cat_cols  # boolean columns included here (0/1)

# ColumnTransformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("target_enc", TargetEncoder(cols=cols_to_encode, smoothing=25.0), cols_to_encode),
        ("scaler", StandardScaler(), num_cols)
    ],
    remainder="drop"  # drop any columns not specified
)


results = []

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Precompute cat feature indices (used in all folds + final CatBoost)
cat_features_idx = [X_train.columns.get_loc(c) for c in cat_cols]

for fold_no, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train), 1):

    print(f"\n--- Fold {fold_no} ---")

    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # ---------- Preprocessing for XGBoost/LightGBM ----------
    preprocessor = ColumnTransformer(
        transformers=[
            ("target_enc", TargetEncoder(cols=cat_cols, smoothing=25.0), cat_cols),
            ("scaler", StandardScaler(), num_cols)
        ],
        remainder="drop"
    )

    X_tr_enc = preprocessor.fit_transform(X_tr, y_tr)
    X_val_enc = preprocessor.transform(X_val)

    # ---------- XGBoost ----------
    xgb_model = xgb.XGBClassifier(
        n_estimators=3000,
        learning_rate=0.013,
        subsample=0.7,
        colsample_bytree=0.6,
        max_depth=5,
        min_child_weight=3,
        reg_alpha=3.5,
        reg_lambda=4.0,
        tree_method='hist',
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )

    xgb_model.fit(
        X_tr_enc, y_tr,
        eval_set=[(X_val_enc, y_val)],
        early_stopping_rounds=20,
        verbose=False
    )

    xgb_auc = roc_auc_score(y_val, xgb_model.predict_proba(X_val_enc)[:, 1])

    # ---------- LightGBM ----------
    lgbm_model = LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.02,
        num_leaves=64,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )

    lgbm_model.fit(
        X_tr_enc, y_tr,
        eval_set=[(X_val_enc, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
    )

    lgbm_auc = roc_auc_score(y_val, lgbm_model.predict_proba(X_val_enc)[:, 1])

    # ---------- CatBoost (raw data) ----------
    cat_model = CatBoostClassifier(
        iterations=3000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=10,
        subsample=0.8,
        loss_function="Logloss",
        eval_metric='AUC',
        random_seed=42,
        verbose=False
    )

    cat_model.fit(
        X_tr,
        y_tr,
        cat_features=cat_features_idx,
        eval_set=(X_val, y_val),
        early_stopping_rounds=200,
        use_best_model=True
    )

    cat_auc = roc_auc_score(y_val, cat_model.predict_proba(X_val)[:, 1])

    results.append({
        "Fold": fold_no,
        "XGBoost_AUC": xgb_auc,
        "LightGBM_AUC": lgbm_auc,
        "CatBoost_AUC": cat_auc
    })

# ---------- CV Results ----------
results_df = pd.DataFrame(results)
print("\nCV Results:")
print(results_df)

avg_results = results_df.drop(columns=["Fold"]).mean().to_frame(name="Average").T
print("\nAverage Results:")
print(avg_results)

# Plot
avg_results_plot = avg_results[['XGBoost_AUC', 'LightGBM_AUC', 'CatBoost_AUC']].T
avg_results_plot.plot(kind="bar", figsize=(9, 4))
plt.title("Average Model Performance (5-fold CV)")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

# Optional: explicit AUCs
avg_auc_xgb = avg_results["XGBoost_AUC"].iloc[0]
avg_auc_lgbm = avg_results["LightGBM_AUC"].iloc[0]
avg_auc_cat = avg_results["CatBoost_AUC"].iloc[0]

print("\nAverage AUC Scores:")
print(f"XGBoost AUC:  {avg_auc_xgb:.4f}")
print(f"LightGBM AUC: {avg_auc_lgbm:.4f}")
print(f"CatBoost AUC: {avg_auc_cat:.4f}")

print("\n" + "=" * 80)

# ---------- Final Model Selection & Training ----------
best_model_name = avg_results[['XGBoost_AUC', 'LightGBM_AUC', 'CatBoost_AUC']].idxmax(axis=1).iloc[0]
best_auc = avg_results[['XGBoost_AUC', 'LightGBM_AUC', 'CatBoost_AUC']].max(axis=1).iloc[0]

print(f"FINAL MODEL SELECTED: {best_model_name}")
print(f"Best CV AUC: {best_auc:.4f}")
print("=" * 80)

# Shared preprocessor for XGB/LGBM final pipelines
final_preprocessor = ColumnTransformer(
    transformers=[
        ("target_enc", TargetEncoder(cols=cat_cols, smoothing=25.0), cat_cols),
        ("scaler", StandardScaler(), num_cols)
    ],
    remainder="drop"
)

if best_model_name == "XGBoost_AUC":
    print("\nTraining final XGBoost model on full data...")

    best_model = xgb.XGBClassifier(
        n_estimators=5000,
        learning_rate=0.007,
        subsample=0.7,
        colsample_bytree=0.6,
        max_depth=4,
        min_child_weight=3,
        reg_alpha=3.5,
        reg_lambda=4.0,
        tree_method='hist',
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )

    final_pipeline = Pipeline([
        ("preprocessor", final_preprocessor),
        ("model", best_model)
    ])

    final_pipeline.fit(X_train, y_train)
    final_model = final_pipeline

elif best_model_name == "LightGBM_AUC":
    print("\nTraining final LightGBM model on full data...")

    best_model = LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.02,
        num_leaves=64,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )

    final_pipeline = Pipeline([
        ("preprocessor", final_preprocessor),
        ("model", best_model)
    ])

    final_pipeline.fit(X_train, y_train)
    final_model = final_pipeline

else:
    print("\nTraining final CatBoost model on full data...")

    best_model = CatBoostClassifier(
        iterations=5000,
        learning_rate=0.007,
        eval_metric='AUC',
        random_seed=42,
        verbose=False
    )

    best_model.fit(
        X_train,
        y_train,
        cat_features=cat_features_idx
    )

    final_model = best_model

print("âœ… Final model training complete.")


print("\n" + "=" * 80)
print("GENERATING TEST PREDICTIONS")
print("=" * 80)

test_predictions = final_model.predict_proba(X_test)[:, 1]


# ============================================================================
# CREATE SUBMISSION FILE
# ============================================================================

submission = pd.DataFrame({
    'id': test_df['id'].values,
    'diagnosed_diabetes': test_predictions
})

submission.to_csv('submission.csv', index=False)
print(f"\nâœ“ Submission file saved: submission.csv")
print(f"  Shape: {submission.shape}")
print(f"\nFirst few predictions:")
print(submission.head(10))

