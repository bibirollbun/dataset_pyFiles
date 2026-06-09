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
# ğŸ¤– MACHINE LEARNING & PREPROCESSING
# =============================================================================
from sklearn.model_selection import KFold, cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier

# =============================================================================
# ğŸ”§ SETTINGS / WARNINGS / ENVIRONMENT
# =============================================================================
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
sns.set(style="whitegrid")


# ğŸ“¥ Load the dataset
train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")

original = pd.read_csv("/kaggle/input/loan-prediction-dataset-2025/loan_dataset_20000.csv")


# Add a 'dataset' column to track source
train['dataset'] = 'train'
test['dataset'] = 'test'

original['dataset'] = 'train'



# Combine train and test datasets for unified preprocessing
df = pd.concat([train, test, original], axis=0).reset_index(drop=True)

# ğŸ§¾ Display dataset shape
print("Dataset shape:", df.shape)

# ğŸ‘�ï¸� Preview the data
df


train


test


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


# ğŸ�¯ Target Variable Distribution (Categorical)
# Target: "loan_paid_back"

import matplotlib.pyplot as plt
import seaborn as sns

# ---
## 1. Count Plot: Showing the Frequency Distribution
# ---

plt.figure(figsize=(6, 4))
# Use 'countplot' for categorical data
sns.countplot(data=df, x='loan_paid_back', palette='Set2')

plt.title('Distribution of Loan Payment Status', fontsize=14)
plt.xlabel('Loan Paid Back', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ---
## 2. Percentage Distribution
# ---

# Calculate the percentage of each category
category_counts = df['loan_paid_back'].value_counts(normalize=True) * 100

# Create a bar plot for percentage distribution
plt.figure(figsize=(6, 4))
sns.barplot(x=category_counts.index, y=category_counts.values, palette='Set2')

plt.title('Loan Payment Status (% Distribution)', fontsize=14)
plt.xlabel('Loan Paid Back', fontsize=12)
plt.ylabel('Percentage (%)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ---
## 3. Descriptive Statistics (Categorical Summary)
# ---

print("\nğŸ“Š Loan Payment Status Summary:")
print(df['loan_paid_back'].value_counts())
print("\nğŸ”¢ Percentage Distribution:")
print((df['loan_paid_back'].value_counts(normalize=True) * 100).round(2).astype(str) + '%')


# -------------------------------------------------------
# ğŸ�¯ Feature Distribution Visualization (Numerical Features)
# Target: loan_paid_back (Categorical)
# -------------------------------------------------------

import matplotlib.pyplot as plt
import seaborn as sns

# Define numerical columns based on your dataset
numerical_cols = [
    'annual_income',
    'debt_to_income_ratio',
    'credit_score',
    'loan_amount',
    'interest_rate'
]

# Separate the columns based on their nature
continuous_features = [
    'annual_income',
    'debt_to_income_ratio',
    'credit_score',
    'loan_amount',
    'interest_rate'
]

# (If you later add any discrete numeric features, like num_of_loans, put them here)
discrete_features = []

# Loop through each numerical column
for col in numerical_cols:
    print(f"--- Visualizing: {col} ---")

    # Set up a figure with two subplots side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'Distribution of {col}', fontsize=16)

    if col in continuous_features:
        # Left: Histogram for density/shape
        sns.histplot(df[col].dropna(), kde=True, bins=30,
                     ax=axes[0], color='skyblue', edgecolor='black')
        axes[0].set_title('Histogram (Shape & Density)')
        axes[0].set_xlabel(col)
        axes[0].set_ylabel('Frequency')

        # Right: Boxplot for quartiles/outliers
        sns.boxplot(x=df[col].dropna(), ax=axes[1], color='lightcoral')
        axes[1].set_title('Box Plot (Outliers & Spread)')
        axes[1].set_xlabel(col)

    elif col in discrete_features:
        # Left: Count Plot for small integer-like features
        sns.countplot(x=df[col].dropna(), ax=axes[0], palette='viridis', edgecolor='black')
        axes[0].set_title('Count Plot (Frequency)')
        axes[0].set_xlabel(col)
        axes[0].set_ylabel('Count')

        # Right: Boxplot (still useful)
        sns.boxplot(x=df[col].dropna(), ax=axes[1], color='lightcoral')
        axes[1].set_title('Box Plot (Summary)')
        axes[1].set_xlabel(col)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # Print descriptive statistics
    print("\nğŸ“Š Descriptive Statistics:")
    print(df[col].describe().round(3))
    print("\n" + "="*50 + "\n")


# ğŸ“Š Distribution of Categorical Features

# Updated list with the categorical columns from your dataset
cat_cols = [
    'gender',
    'marital_status',
    'education_level',
    'employment_status',
    'loan_purpose',
    'grade_subgrade'
]

for col in cat_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(
        data=df,
        x=col,
        order=df[col].value_counts().index,  # order bars by frequency
        palette='Set2',
        edgecolor='black'
    )

    plt.title(f'Distribution of {col}', fontsize=14)
    plt.xlabel(col.replace('_', ' ').title(), fontsize=12)
    plt.ylabel('Count', fontsize=12)

    # Rotate labels if categories are long
    plt.xticks(rotation=30, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # ğŸ§® Print Category Proportions
    print(f'\nğŸ“Š Proportion of Each Category in "{col}":\n')
    print(df[col].value_counts(normalize=True).round(3), '\n' + '-'*40)


# ğŸ�¨ Categorical Feature Distributions by Loan Repayment Status - Custom Colors

# Select key categorical columns to explore their relationship with the target
cols_to_plot = [
    'gender',
    'marital_status',
    'education_level',
    'employment_status',
    'loan_purpose',
    'grade_subgrade'
]

# Custom colors: green for Paid (1), red for Not Paid (0)
custom_palette = ['#E74C3C', '#27AE60']  # Red = Not Paid, Green = Paid

target_col = 'loan_paid_back'  # Binary target variable (0 or 1)

for col in cols_to_plot:
    plt.figure(figsize=(8, 5))
    sns.countplot(
        data=df,
        x=col,
        hue=target_col,
        palette=custom_palette,
        edgecolor='black',
        order=df[col].value_counts().index  # Order bars by frequency
    )
    
    plt.title(f'{col.replace("_", " ").title()} by Loan Repayment Status', fontsize=14)
    plt.xlabel(col.replace('_', ' ').title(), fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=25, ha='right')
    
    # Legend reflecting your target variable meaning
    plt.legend(title='Loan Paid Back', labels=['No (0)', 'Yes (1)'])
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()


# ğŸ�¨ Numerical Feature Distributions by Loan Repayment Status

import matplotlib.pyplot as plt
import seaborn as sns

# Select your key numerical columns
num_cols = [
    'annual_income',
    'debt_to_income_ratio',
    'credit_score',
    'loan_amount',
    'interest_rate'
]

target_col = 'loan_paid_back'  # Binary target variable (0 = Not Paid, 1 = Paid)
custom_palette = ['#E74C3C', '#27AE60']  # Red = Not Paid, Green = Paid

for col in num_cols:
    plt.figure(figsize=(8, 4))
    
    # Left: Boxplot (distribution + outliers)
    sns.boxplot(
        data=df,
        x=target_col,
        y=col,
        palette=custom_palette,
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black"}
    )
    
    plt.title(f'{col.replace("_", " ").title()} by Loan Repayment Status', fontsize=14)
    plt.xlabel('Loan Paid Back (0 = No, 1 = Yes)', fontsize=12)
    plt.ylabel(col.replace("_", " ").title(), fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()


for col in ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose']:
    ctab = pd.crosstab(df[col], df['loan_paid_back'], normalize='index') * 100
    print(f"\nğŸ“Š {col} vs Loan Paid Back (%):\n")
    print(ctab.round(2))


plt.figure(figsize=(8, 6))
sns.heatmap(df[['annual_income', 'debt_to_income_ratio', 'credit_score', 
                'loan_amount', 'interest_rate', 'loan_paid_back']].corr(),
            annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap (Numerical Features + Target)', fontsize=14)
plt.show()


# Categorical columns to analyze
cat_cols = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose']
target_col = 'loan_paid_back'

for col in cat_cols:
    # Calculate % of repayment within each category
    ctab = pd.crosstab(df[col], df[target_col], normalize='index') * 100
    
    print(f"\nğŸ“Š {col} vs Loan Paid Back (%):\n")
    print(ctab.round(2))
    
    # Reset for plotting
    ctab_plot = ctab.reset_index()
    ctab_plot = ctab_plot.melt(id_vars=col, var_name='Loan Paid Back', value_name='Percentage')

    # Plot
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=ctab_plot,
        x=col,
        y='Percentage',
        hue='Loan Paid Back',
        palette=['#E74C3C', '#27AE60'],  # red = not paid, green = paid
        edgecolor='black'
    )
    
    plt.title(f'Loan Repayment Rate by {col.replace("_", " ").title()}', fontsize=14)
    plt.xlabel(col.replace("_", " ").title(), fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.xticks(rotation=25, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.legend(title='Loan Paid Back', labels=['No (0)', 'Yes (1)'])
    plt.tight_layout()
    plt.show()


# Outlier Detection

num_cols = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate']

plt.figure(figsize=(12, 6))
for i, col in enumerate(num_cols, 1):
    plt.subplot(1, len(num_cols), i)
    sns.boxplot(y=df[col], color='lightcoral')
    plt.title(col.replace('_', ' ').title())
    plt.tight_layout()
plt.suptitle('ğŸ“¦ Outlier Detection via Boxplots', fontsize=16, y=1.05)
plt.show()


import numpy as np

def create_advanced_features(df):
    df = df.copy()

    # Core affordability
    df['income_loan_ratio'] = df['annual_income'] / df['loan_amount'].replace(0, np.nan)
    df['loan_to_income'] = df['loan_amount'] / df['annual_income'].replace(0, np.nan)
    
    # Debt metrics
    df['total_debt'] = df['debt_to_income_ratio'] * df['annual_income']
    df['available_income'] = df['annual_income'] * (1 - df['debt_to_income_ratio'])
    df['debt_burden'] = df['debt_to_income_ratio'] * df['loan_amount']
    
    # Payment analysis (simple proxy)
    df['monthly_payment'] = df['loan_amount'] * df['interest_rate'] / 1200
    df['payment_to_income'] = df['monthly_payment'] / (df['annual_income'] / 12).replace(0, np.nan)
    df['affordability'] = df['available_income'] / df['loan_amount'].replace(0, np.nan)
    
    # Risk scoring
    df['default_risk'] = (
        df['debt_to_income_ratio'] * 0.40 + 
        (850 - df['credit_score']) / 850 * 0.35 + 
        df['interest_rate'] / 100 * 0.25
    ).clip(0, 1)
    
    # Credit analysis
    df['credit_utilization'] = df['credit_score'] * (1 - df['debt_to_income_ratio'])
    df['credit_interest_product'] = df['credit_score'] * df['interest_rate'] / 100
    
    # Log transformations
    for col in ['annual_income', 'loan_amount']:
        df[f'{col}_log'] = np.log1p(df[col].clip(lower=0))
    
    # Grade parsing
    df['grade_letter'] = df['grade_subgrade'].astype(str).str[0]
    df['grade_number'] = (
        df['grade_subgrade'].astype(str).str[1:].str.extract(r'(\d+)')[0].astype('float')
    )
    grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    df['grade_rank'] = df['grade_letter'].map(grade_map).astype('float')
    
    return df

NEW_FEATURES = [
    'income_loan_ratio', 'loan_to_income', 'total_debt', 
    'available_income', 'debt_burden', 'monthly_payment',
    'payment_to_income', 'affordability', 'default_risk',
    'credit_utilization', 'credit_interest_product',
    'annual_income_log', 'loan_amount_log', 'grade_letter',
    'grade_number', 'grade_rank'
]

print(f"Created {len(NEW_FEATURES)} new features")

df = create_advanced_features(df)


# =============================================================================
# DATA PREPARATION
# =============================================================================

# Split into train/test sets
train_df = df[df["dataset"] == "train"].copy()
test_df = df[df["dataset"] == "test"].copy()

print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")
print(f"Missing target values: {train_df['loan_paid_back'].isna().sum()}")

# Separate features and target
X_train = train_df.drop(["id", "loan_paid_back", "dataset"], axis=1)
y_train = train_df["loan_paid_back"]

X_test = test_df.drop(["id", "loan_paid_back", "dataset"], axis=1)

# =============================================================================
# FEATURE ENGINEERING (OPTIONAL)
# =============================================================================


# =============================================================================
# TARGET ENCODING + PREPROCESSING
# =============================================================================

# Define feature types
num_cols = [
    "annual_income",
    "debt_to_income_ratio",
    "credit_score",
    "loan_amount",
    "interest_rate"
]

cat_cols = [
    "gender",
    "marital_status",
    "education_level",
    "employment_status",
    "loan_purpose",
    "grade_subgrade"
]

# You can add binary flags (if any exist)
bool_cols = []  

# Columns to encode using Target Encoding (categorical + bools)
cols_to_encode = cat_cols + bool_cols

# ColumnTransformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("target_enc", TargetEncoder(cols=cols_to_encode, smoothing=25.0), cols_to_encode),
        ("scaler", StandardScaler(), num_cols)
    ],
    remainder="drop"
)


# # =============================================================================
# # IMPORTS (add these on top of your notebook/script)
# # =============================================================================
# import cudf
# from cuml.preprocessing import TargetEncoder as cuTargetEncoder
# from cuml.preprocessing import StandardScaler as cuStandardScaler

# # =============================================================================
# # DATA PREPARATION
# # =============================================================================

# # Split into train/test sets
# train_df = df[df["dataset"] == "train"].copy()
# test_df  = df[df["dataset"] == "test"].copy()

# print(f"Training samples: {len(train_df)}")
# print(f"Test samples: {len(test_df)}")
# print(f"Missing target values: {train_df['loan_paid_back'].isna().sum()}")

# # Separate features and target
# X_train = train_df.drop(["id", "loan_paid_back", "dataset"], axis=1)
# y_train = train_df["loan_paid_back"]

# X_test  = test_df.drop(["id", "loan_paid_back", "dataset"], axis=1)

# # =============================================================================
# # FEATURE ENGINEERING (OPTIONAL)
# # =============================================================================
# # ... your feature engineering on X_train / X_test here (if any) ...


# # =============================================================================
# # TARGET ENCODING + PREPROCESSING (cuML on GPU)
# # =============================================================================

# # Define feature types
# num_cols = [
#     "annual_income",
#     "debt_to_income_ratio",
#     "credit_score",
#     "loan_amount",
#     "interest_rate"
# ]

# cat_cols = [
#     "gender",
#     "marital_status",
#     "education_level",
#     "employment_status",
#     "loan_purpose",
#     "grade_subgrade"
# ]

# bool_cols = []   # add any binary flags here if you have them
# cols_to_encode = cat_cols + bool_cols

# # --- send data to GPU (cuDF) ---
# X_train_gpu = cudf.from_pandas(X_train)
# X_test_gpu  = cudf.from_pandas(X_test)
# y_train_gpu = cudf.Series(y_train.values)

# # --- cuML Target Encoding per categorical column (fit on train, transform test) ---
# te_encoders = {}  # keep if you want to reuse encoders later

# for col in cols_to_encode:
#     te = cuTargetEncoder(
#         n_folds=5,
#         smooth=25.0,   # similar idea to your smoothing=25.0
#         seed=42
#     )
#     # fit+encode train
#     X_train_gpu[col] = te.fit_transform(X_train_gpu[col], y_train_gpu)
#     # encode test
#     X_test_gpu[col]  = te.transform(X_test_gpu[col])
#     te_encoders[col] = te

# # --- cuML StandardScaler on numeric columns (fit on train, transform test) ---
# scaler_gpu = cuStandardScaler()
# X_train_gpu[num_cols] = scaler_gpu.fit_transform(X_train_gpu[num_cols])
# X_test_gpu[num_cols]  = scaler_gpu.transform(X_test_gpu[num_cols])

# # =============================================================================
# # FINAL MATRICES FOR MODELS
# # =============================================================================

# # If you use GPU models (cuML, XGBoost GPU, etc.), keep them as cuDF:
# X_train_final_gpu = X_train_gpu
# X_test_final_gpu  = X_test_gpu
# y_train_final_gpu = y_train_gpu

# # # If you want to stay with sklearn/CPU models, convert back to pandas:
# # X_train_final = X_train_gpu.to_pandas()
# # X_test_final  = X_test_gpu.to_pandas()
# # y_train_final = y_train  # already pandas Series



X_train


# =============================================================================
# DEFINE MODELS
# =============================================================================

models = {
    # "LightGBM": LGBMClassifier(
    #     metric='auc',
    #     n_estimators=1000,
    #     learning_rate=0.03,
    #     max_depth=6,
    #     num_leaves=50,
    #     colsample_bytree=0.8,
    #     subsample=0.8,
    #     subsample_freq=1,
    #     min_child_samples=20,
    #     reg_alpha=0.05,
    #     reg_lambda=0.1,
    #     random_state=42,
    #     n_jobs=-1,
    #     device='gpu',
    #     verbose=-1
    # ),
    "CatBoost": CatBoostClassifier(
        iterations=3000,
        learning_rate=0.03,
        depth=8,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=42,
        verbose=0,
        auto_class_weights='Balanced',
        l2_leaf_reg=5
    ),
    "XGBoost": XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        learning_rate=0.01,
        max_depth=6,
        min_child_weight=3,
        colsample_bytree=0.3,
        subsample=0.6,
        reg_alpha=0.5,
        reg_lambda=2.0,
        n_estimators=10000,
        random_state=42,
        n_jobs=-1,
        tree_method='hist',
        device="cuda"
    )
}

# Ensemble model (soft voting)
models["Ensemble_All"] = VotingClassifier(
    estimators=[
        ("CatBoost", models["CatBoost"]),
        ("XGBoost", models["XGBoost"])
    ],
    voting="soft",
    weights=[1, 1]  
)



# =============================================================================
# CROSS-VALIDATION SETUP
# =============================================================================

print("\n" + "=" * 80)
print("CROSS-VALIDATION RESULTS (5-Fold)")
print("=" * 80)

kfold = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)
cv_results = {}

# Classification metrics
scoring = {
    "Accuracy": "accuracy",
    "Precision": "precision",
    "Recall": "recall",
    "F1": "f1",
    "ROC_AUC": "roc_auc"
}


# =============================================================================
# TRAINING + CROSS-VALIDATION
# =============================================================================

for name, model in models.items():
    print(f"\n{name}:")
    print("-" * 40)

    # Create a unified pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # Cross-validation
    cv_scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=kfold,
        scoring=scoring,
        n_jobs=-1
    )

    # Store mean results
    cv_results[name] = {metric: np.mean(scores) for metric, scores in cv_scores.items() if "test_" in metric}

    print(f"Accuracy:  {cv_results[name]['test_Accuracy']:.4f}")
    print(f"Precision: {cv_results[name]['test_Precision']:.4f}")
    print(f"Recall:    {cv_results[name]['test_Recall']:.4f}")
    print(f"F1-score:  {cv_results[name]['test_F1']:.4f}")
    print(f"ROC-AUC:   {cv_results[name]['test_ROC_AUC']:.4f}")


# =============================================================================
# MODEL COMPARISON SUMMARY
# =============================================================================

results_df = pd.DataFrame({
    model: {
        "Accuracy": cv_results[model]["test_Accuracy"],
        "Precision": cv_results[model]["test_Precision"],
        "Recall": cv_results[model]["test_Recall"],
        "F1": cv_results[model]["test_F1"],
        "ROC_AUC": cv_results[model]["test_ROC_AUC"]
    } for model in cv_results.keys()
}).T.round(4)

print("\n" + "=" * 80)
print("MODEL PERFORMANCE SUMMARY (5-Fold CV)")
print("=" * 80)
print(results_df)

# Optional: plot comparison
results_df.plot(kind='bar', figsize=(10,6))
plt.title('Model Performance Comparison (Cross-Validation)')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()


# =============================================================================
# FINAL MODEL TRAINING (CHOOSE BEST MODEL)
# =============================================================================

best_model_name = results_df['ROC_AUC'].idxmax()
best_model = models[best_model_name]

print("\n" + "=" * 80)
print(f"FINAL MODEL: {best_model_name}")
print("=" * 80)

# Create final pipeline
final_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", best_model)
])

print(f"\nTraining final {best_model_name} model on full training data...")
final_pipeline.fit(X_train, y_train)
print("âœ… Final model training complete.")


# ============================================================================
# GENERATE PREDICTIONS
# ============================================================================

print("\n" + "="*80)
print("GENERATING TEST PREDICTIONS")
print("="*80)

test_predictions = final_pipeline.predict_proba(X_test)[:, 1]


# ============================================================================
# CREATE SUBMISSION FILE
# ============================================================================

submission = pd.DataFrame({
    'id': test_df['id'].values,
    'loan_paid_back': test_predictions
})

submission.to_csv('submission.csv', index=False)
print(f"\nâœ“ Submission file saved: submission.csv")
print(f"  Shape: {submission.shape}")
print(f"\nFirst few predictions:")
print(submission.head(10))

