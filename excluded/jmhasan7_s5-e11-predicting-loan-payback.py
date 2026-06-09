import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier
import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')

train_df.head()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", palette="Set2")

# Basic info

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("\nTrain columns:", train_df.columns.tolist())


# Checking Missing Values
print("\nMissing values in train:", train_df.isnull().sum().sum())
print("Missing values in test:", test_df.isnull().sum().sum())


# Checking target distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='loan_paid_back', data=train_df)
plt.title('Target Variable Distribution')
plt.show()

print("Target distribution:")
print(train_df['loan_paid_back'].value_counts(normalize=True))


# Outlier Detection
numerical_features = ['annual_income', 'debt_to_income_ratio', 
                      'credit_score', 'loan_amount', 'interest_rate']

for col in numerical_features:
    plt.figure(figsize=(8,4))
    sns.boxplot(x=train_df[col])
    plt.title(f"Boxplot for {col}")
    plt.show()


# Feature-Target Relationships

for col in numerical_features:
    plt.figure(figsize=(8,5))
    sns.boxplot(x='loan_paid_back', y=col, data=train_df)
    plt.title(f"{col} vs Loan Paid Back")
    plt.show()


# Categorical Feature-Target Relationship

categorical_features = ['gender', 'marital_status', 'education_level', 
                        'employment_status', 'loan_purpose', 'grade_subgrade']

for col in categorical_features:
    plt.figure(figsize=(8,4))
    sns.countplot(data=train_df, x=col, hue='loan_paid_back')
    plt.title(f"{col} vs Loan Paid Back")
    plt.xticks(rotation=45)
    plt.show()


plt.figure(figsize=(10,8))
corr_matrix = train_df[numerical_features + ['loan_paid_back']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix (Numerical Features + Target)')
plt.show()


# Cardinality Check

for col in categorical_features:
    print(f"{col}: {train_df[col].nunique()} unique values")

# Visualize top categories for each
for col in categorical_features:
    plt.figure(figsize=(8,4))
    train_df[col].value_counts(normalize=True).head(10).plot(kind='bar')
    plt.title(f"{col} - Top 10 Category Proportions")
    plt.ylabel("Proportion")
    plt.show()


# Checking for Unseen Categories in Test

for col in categorical_features:
    train_unique = set(train_df[col].dropna().unique())
    test_unique = set(test_df[col].dropna().unique())
    unseen = test_unique - train_unique
    # unseen = train_unique - test_unique
    if unseen:
        print(f"'{col}' has unseen categories in test data: {unseen}")


# Pairwise Feature Relationships
# sns.pairplot(train_df, vars=numerical_features, hue='loan_paid_back', diag_kind='kde', corner=True)
# plt.suptitle("Pairplot of Numerical Features Colored by Target", y=1.02)
# plt.show()


# Descriptive Statistics
display(train_df.describe().T)


print("Mean annual_income :",train_df['annual_income'].mean())
print("Mean debt_to_income_ratio :",train_df['debt_to_income_ratio'].mean())
print("Mean credit_score :",train_df['credit_score'].mean())
print("Mean loan_amount :",train_df['loan_amount'].mean())
print("Mean interest_rate :",train_df['interest_rate'].mean())



import numpy as np
import pandas as pd
import os

# Defining the numerical features and the optimal capping strategy based on box plot analysis
# 'upper-only': Outliers detected only on the high end (e.g., income, loan amount)
# 'lower-only': Outliers detected only on the low end (e.g., credit score)
# 'two-sided': Outliers detected on both ends (e.g., interest rate)
CAPPING_STRATEGIES = {
    'annual_income': 'upper-only',
    'debt_to_income_ratio': 'upper-only',
    'credit_score': 'lower-only',
    'loan_amount': 'upper-only',
    'interest_rate': 'two-sided'
}

def implement_capping(df, column, strategy='two-sided', overwrite=True):
    """
    Applies IQR-based capping to a specified column in the DataFrame.

    If overwrite=True, the original column is replaced (used for final output).
    If overwrite=False, a new '_capped' column is created (used for comparison reporting).

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The name of the column to treat.
        strategy (str): 'two-sided', 'upper-only', or 'lower-only'.
        overwrite (bool): If True, replaces the original column. If False, creates a new one.

    Returns:
        pd.DataFrame: The modified DataFrame.
    """
    if column not in df.columns:
        print(f"Error: Column '{column}' not found in DataFrame.")
        return df

    # 1. Calculate IQR bounds
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_cap = Q1 - 1.5 * IQR
    upper_cap = Q3 + 1.5 * IQR

    # Create a copy of the original data to apply capping
    # We use .copy() here to avoid SettingWithCopyWarning if we are creating a new column for comparison
    processed_data = df[column].copy()

    # 2. Apply Capping based on strategy
    if strategy == 'upper-only':
        processed_data = np.where(processed_data > upper_cap, upper_cap, processed_data)
        processed_data = np.maximum(0, processed_data)

    elif strategy == 'lower-only':
        processed_data = np.where(processed_data < lower_cap, lower_cap, processed_data)

    elif strategy == 'two-sided':
        processed_data = np.where(processed_data > upper_cap, upper_cap, processed_data)
        processed_data = np.where(processed_data < lower_cap, lower_cap, processed_data)

    # 3. Add the processed data back to the DataFrame
    new_col_name = column if overwrite else f'{column}_capped'
    df[new_col_name] = processed_data

    # Print the cap values used
    print(f"Capping applied to '{column}' (Strategy: {strategy}) ::")
    if strategy != 'lower-only':
        print(f"  Upper Cap Value: {upper_cap:,.2f}")
    if strategy != 'upper-only':
        print(f"  Lower Cap Value: {lower_cap:,.2f}")

    return df


def run_outlier_treatment(df, overwrite=True):
    """
    Iterates through all defined features and applies the appropriate capping.
    If overwrite=True, original columns are replaced.
    """
    print(f"\n--- Running Outlier Treatment (Overwrite={overwrite}) ---")
    for col, strategy in CAPPING_STRATEGIES.items():
        df = implement_capping(df, col, strategy, overwrite=overwrite)
    return df


def generate_comparison_report(original_df, treated_df, original_col):
    """
    Generates and prints a statistical comparison between original (untreated) 
    and capped (treated) data, where the treated data replaces the original column.
    """
    original_series = original_df[original_col]
    treated_series = treated_df[original_col] # Now accessing the overwritten column

    # Create a temporary DataFrame for comparison
    comparison_df = pd.DataFrame({
        f'{original_col}_Original': original_series,
        f'{original_col}_Treated': treated_series
    })

    # Calculate descriptive statistics
    comparison_stats = comparison_df.agg(['mean', 'median', 'std', 'min', 'max'])
    
    # Print the comparison table
    print(f"\n{'*'*50}\nStatistical Comparison: {original_col} (Original vs. Treated)")
    # Print the stats, where features are columns and stats are rows
    print(comparison_stats.to_string(float_format="{:,.2f}".format))

    # Calculate the percentage change in the Mean
    original_mean = comparison_stats.loc['mean', f'{original_col}_Original']
    treated_mean = comparison_stats.loc['mean', f'{original_col}_Treated']
    mean_change_percent = ((treated_mean - original_mean) / original_mean) * 100

    print(f"\nSummary of Mean Change:")
    print(f"  Original Mean: {original_mean:,.2f}")
    print(f"  Treated Mean: {treated_mean:,.2f}")
    print(f"  Change in Mean: {mean_change_percent:+.2f}%")
    print(f"\nNote: A change is expected after capping extreme values.")
    print('*'*50)


# 1. Create a copy for treatment so we can compare against the original data later
train_df_treated = train_df.copy()

# 2. Apply Capping to all features, **OVERWRITING** the original columns
train_df_treated = run_outlier_treatment(train_df_treated, overwrite=True)



# 3. Generate a comparison report for each feature to see the effect of replacement
# We compare the ORIGINAL data with the TREATED data (which has overwritten columns)
print("\n" + "*"*70)
print("OUTLIER TREATMENT RESULTS AND STATISTICAL IMPACT (In-Place Replacement)")
print("*"*70)

for col in CAPPING_STRATEGIES.keys():
    # Pass both the original and the treated DataFrame for the comparison report
    generate_comparison_report(train_df, train_df_treated, col)


print(train_df.columns)


print(train_df_treated.columns)


columns_to_update = [
    'annual_income',
    'debt_to_income_ratio',
    'credit_score',
    'loan_amount',
    'interest_rate'
]

treated_subset = train_df_treated[columns_to_update]
# Update train_df in place
train_df.update(treated_subset)

print("Values in train_df for numeric columns have been updated in place.")
print(train_df[columns_to_update].head())



print("Mean annual_income :",train_df['annual_income'].mean())
print("Mean debt_to_income_ratio :",train_df['debt_to_income_ratio'].mean())
print("Mean credit_score :",train_df['credit_score'].mean())
print("Mean loan_amount :",train_df['loan_amount'].mean())
print("Mean interest_rate :",train_df['interest_rate'].mean())


# FEATURE ENGINEERING

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
import numpy as np

# --- Feature Engineering Function ---
def create_features(df):
    df = df.copy()
    
    # Ratio-based financial features
    df['income_to_loan_ratio'] = df['annual_income'] / (df['loan_amount'] + 1)
    df['debt_burden'] = df['annual_income'] * df['debt_to_income_ratio']
    df['affordability_ratio'] = (df['annual_income'] / 12) / (df['loan_amount'] * df['interest_rate'] / 1200 + 1)
    df['credit_income_ratio'] = df['credit_score'] / (df['annual_income'] + 1)
    
    # Weighted Risk Score
    df['risk_score'] = (
        df['debt_to_income_ratio'] * 0.3 + 
        (800 - df['credit_score']) / 800 * 0.3 + 
        df['interest_rate'] / 25 * 0.2 +
        (df['loan_amount'] / (df['annual_income'] + 1)) * 0.2
    )
    # Grade and Subgrade (if available)
    if 'grade_subgrade' in df.columns:
        df['grade'] = df['grade_subgrade'].str[0]
        df['subgrade_num'] = df['grade_subgrade'].str[1].astype(int)
    
    # Employment and Education Encodings
    employment_mapping = {
        'Unemployed': 0, 'Student': 1, 'Self-employed': 2, 
        'Employed': 3, 'Retired': 2
    }
    df['employment_stability'] = df['employment_status'].map(employment_mapping)
    
    education_mapping = {
        'High School': 1, 'Other': 2, 
        'Bachelor\'s': 3, 'Master\'s': 4, 'PhD': 5
    }
    df['education_num'] = df['education_level'].map(education_mapping)
    
    # Log Transform Skewed Features (from EDA)
    skewed_feats = ['annual_income', 'loan_amount', 'debt_burden']
    for feat in skewed_feats:
        df[f'log_{feat}'] = np.log1p(df[feat])
    
    # Interaction Features (based on EDA correlations)
    df['income_x_credit'] = df['annual_income'] * df['credit_score']
    df['loan_x_interest'] = df['loan_amount'] * df['interest_rate']
    
    return df


# --- Applying to Train and Test ---
train_df_eng = create_features(train_df)
test_df_eng = create_features(test_df)

print(f"New feature columns added: {set(train_df_eng.columns) - set(train_df.columns)}")


# Separate numeric and categorical columns
num_cols = train_df_eng.select_dtypes(include=np.number).columns.tolist()
cat_cols = train_df_eng.select_dtypes(exclude=np.number).columns.tolist()

# Drop target column from numeric list if present
if 'loan_paid_back' in num_cols:
    num_cols.remove('loan_paid_back')

# --- Handle Missing Values ---
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

# Fit on train and transform both
train_df_eng[num_cols] = num_imputer.fit_transform(train_df_eng[num_cols])
test_df_eng[num_cols] = num_imputer.transform(test_df_eng[num_cols])

train_df_eng[cat_cols] = cat_imputer.fit_transform(train_df_eng[cat_cols])
test_df_eng[cat_cols] = cat_imputer.transform(test_df_eng[cat_cols])


# --- Encoding Categorical Variables ---
# Label Encoding for Ordinal Columns
ordinal_cols = ['grade'] if 'grade' in train_df_eng.columns else []
le = LabelEncoder()
for col in ordinal_cols:
    train_df_eng[col] = le.fit_transform(train_df_eng[col])
    test_df_eng[col] = le.transform(test_df_eng[col])


# One-Hot Encoding for Nominal Columns
nominal_cols = [col for col in cat_cols if col not in ordinal_cols]
train_df_eng = pd.get_dummies(train_df_eng, columns=nominal_cols, drop_first=True)
test_df_eng = pd.get_dummies(test_df_eng, columns=nominal_cols, drop_first=True)

# Align train and test (important!)
train_df_eng, test_df_eng = train_df_eng.align(test_df_eng, join='left', axis=1, fill_value=0)


# --- Feature Scaling ---
scaler = StandardScaler()
scaled_cols = num_cols + ['subgrade_num', 'education_num', 'employment_stability']

# Validating that all scaling columns exist in both train and test:
missing_cols = [c for c in scaled_cols if c not in train_df_eng.columns or c not in test_df_eng.columns]
print("Missing columns:", missing_cols)

train_df_eng[scaled_cols] = scaler.fit_transform(train_df_eng[scaled_cols])
test_df_eng[scaled_cols] = scaler.transform(test_df_eng[scaled_cols])


# --- Feature Selection (Low Variance Filter) ---
selector = VarianceThreshold(threshold=0.0)
train_selected = selector.fit_transform(train_df_eng)
test_selected = selector.transform(test_df_eng)

train_df_final = pd.DataFrame(train_selected, columns=train_df_eng.columns[selector.get_support()])
test_df_final = pd.DataFrame(test_selected, columns=train_df_eng.columns[selector.get_support()])


print("Feature engineering, encoding, and scaling complete.")
print(f"Final train shape: {train_df_final.shape}")
print(f"Final test shape: {test_df_final.shape}")



# 4: Preprocessing, Encoding & Feature Selection


from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
import pandas as pd

def preprocess_data(train_df_eng, test_df_eng):
    """
    Performs final preprocessing steps:
      - Defines feature sets
      - Scales numerical features
      - Applies low variance feature selection
      - Returns final processed train and test datasets
    """


    # Define Target and Feature Columns

    target_col = 'loan_paid_back'
    
    # Select numeric columns (excluding target)
    num_cols = train_df_eng.select_dtypes(include=np.number).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)
    
    # Optional safety check: ensure key engineered columns exist
    scaling_cols = [col for col in num_cols if col in train_df_eng.columns]

    # Feature Scaling
    
    scaler = StandardScaler()
    train_df_eng[scaling_cols] = scaler.fit_transform(train_df_eng[scaling_cols])
    test_df_eng[scaling_cols] = scaler.transform(test_df_eng[scaling_cols])

    # Feature Selection (Low Variance)

    selector = VarianceThreshold(threshold=0.0)
    train_selected = selector.fit_transform(train_df_eng)
    test_selected = selector.transform(test_df_eng)

    # Keep only selected columns
    selected_features = train_df_eng.columns[selector.get_support()]
    
    # Convert back to DataFrames
    train_final = pd.DataFrame(train_selected, columns=selected_features)
    test_final = pd.DataFrame(test_selected, columns=selected_features)
    
    
    # Separate Features and Target
    
    if target_col in train_final.columns:
        X_train = train_final.drop(columns=[target_col])
        y_train = train_final[target_col]
    else:
        X_train = train_final.copy()
        y_train = train_df_eng[target_col]

    X_test = test_final.copy()
    
   
    # Summary
   
    print("Preprocessing pipeline complete.")
    print(f"Final training shape: {X_train.shape}")
    print(f"Final test shape: {X_test.shape}")
    print(f"Selected feature count: {len(selected_features)}")

    return X_train, y_train, X_test, scaler, selector, selected_features


# Executing the final preprocessing pipeline

X_train, y_train, X_test, scaler, selector, selected_features = preprocess_data(train_df_eng, test_df_eng)

print("\nSample of selected features:")
print(selected_features[:15])




# Stacking Ensemble with LightGBM 


import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
import optuna
import warnings
warnings.filterwarnings("ignore")

TARGET_COL = "loan_paid_back"

# Preparig dataset
X = train_df_final.drop(columns=[TARGET_COL], errors='ignore')
y = train_df_final[TARGET_COL]
X_test = test_df_final.drop(columns=[TARGET_COL], errors='ignore')

print(f"Train shape: {X.shape}, Test shape: {X_test.shape}")

# Defining base models
xgb_model = XGBClassifier(
    n_estimators=400, max_depth=6, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
    eval_metric='logloss', n_jobs=-1
)

cat_model = CatBoostClassifier(
    iterations=400, depth=6, learning_rate=0.05,
    random_state=42, verbose=False, thread_count=-1
)

lgb_model = LGBMClassifier(
    n_estimators=400, learning_rate=0.03, max_depth=-1,
    subsample=0.8, colsample_bytree=0.8, num_leaves=31,
    random_state=42, n_jobs=-1
)

rf_model = RandomForestClassifier(
    n_estimators=300, max_depth=10,
    random_state=42, n_jobs=-1
)

base_models = [
    ("XGB", xgb_model),
    ("CatBoost", cat_model),
    ("LightGBM", lgb_model),
    ("RandomForest", rf_model)
]

# Stacking
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
meta_train = np.zeros((X.shape[0], len(base_models)))
meta_test = np.zeros((X_test.shape[0], len(base_models)))

print("Training base models with 5-Fold Stacking...\n")

for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    print(f"Fold {fold + 1}/5")
    X_tr, X_va = X.iloc[train_idx], X.iloc[valid_idx]
    y_tr, y_va = y.iloc[train_idx], y.iloc[valid_idx]
    
    for j, (name, model) in enumerate(base_models):
        print(f" → Training {name}...", end=" ")
        model.fit(X_tr, y_tr)
        meta_train[valid_idx, j] = model.predict_proba(X_va)[:, 1]
        meta_test[:, j] += model.predict_proba(X_test)[:, 1] / kf.n_splits
        print("✓")

print("\n Meta-features created for full train and test data.")



# Optuna for Meta-Model Optimisation
def objective(trial):
    C = trial.suggest_float('C', 1e-3, 10.0, log=True)
    solver = trial.suggest_categorical('solver', ['lbfgs', 'liblinear'])
    penalty = trial.suggest_categorical('penalty', ['l2'])
    
    model = LogisticRegression(C=C, solver=solver, penalty=penalty, random_state=42, max_iter=500)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    aucs = []
    for train_idx, val_idx in kf.split(meta_train, y):
        X_tr, X_val = meta_train[train_idx], meta_train[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(X_tr, y_tr)
        preds = model.predict_proba(X_val)[:, 1]
        aucs.append(roc_auc_score(y_val, preds))
    
    return np.mean(aucs)

print("\nRunning Optuna optimization for meta-model...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20, show_progress_bar=True)

best_params = study.best_params
print("\n Best Meta-Model Parameters:", best_params)

# Training final meta-model with best params
meta_model = LogisticRegression(
    **best_params, random_state=42, max_iter=500
)
meta_model.fit(meta_train, y)

meta_pred = meta_model.predict_proba(meta_train)[:, 1]
auc_score = roc_auc_score(y, meta_pred)
print(f"\nFinal Meta-Model AUC after Optuna: {auc_score:.4f}")


# Final test predictions
final_test_pred = meta_model.predict_proba(meta_test)[:, 1]

# Create submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'loan_paid_back': final_test_pred
})

submission.to_csv("submission_stacking_optuna.csv", index=False)
print("\nSubmission file saved: submission_stacking_optuna.csv")
print(submission.head())



















