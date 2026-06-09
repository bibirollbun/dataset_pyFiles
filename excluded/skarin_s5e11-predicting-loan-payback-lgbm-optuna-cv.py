import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier, early_stopping
import warnings

warnings.filterwarnings('ignore')


train_df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')

test_ids = test_df['id']


print(train_df.head())
print(train_df.info())
print(train_df.describe())


# Compute counts
counts = train_df['loan_paid_back'].value_counts().sort_index()  # ensures 0 comes before 1

plt.style.use('seaborn-v0_8-deep')
plt.bar(counts.index, counts.values, color=['skyblue', 'salmon'], edgecolor='black')

# Add count labels on top of bars
for i, v in enumerate(counts.values):
    plt.text(counts.index[i], v + 10, str(v), ha='center', fontweight='bold')

# Labels and title
plt.xlabel('Loan Paid Back (0 = No, 1 = Yes)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Loan Repayment Distribution', fontsize=14, fontweight='bold')

# Ensure x-axis shows 0 and 1 clearly
plt.xticks([0, 1])

plt.show()


numeric_cols = ['annual_income', 'loan_amount', 'debt_to_income_ratio', 'credit_score', 'interest_rate']
train_df[numeric_cols].hist(bins=30, figsize=(10,8))


sns.heatmap(train_df.corr(numeric_only=True), annot=True, fmt=".2f")


# --- Feature Engineering (My "Golden" Set) ---
def feature_engineer(df):
    df['grade'] = df['grade_subgrade'].str[0]
    df['subgrade'] = df['grade_subgrade'].str[1]
    total_annual_debt = df['annual_income'] * df['debt_to_income_ratio']
    safe_annual_income = df['annual_income'].replace(0, 1e-6)
    safe_interest_rate = df['interest_rate'].replace(0, 1e-6)
    safe_loan_amount = df['loan_amount'].replace(0, 1e-6)
    
    # 1. Credit-to-Interest Ratio: Is the interest rate "fair" for their score?
    df['credit_to_interest_ratio'] = df['credit_score'] / safe_interest_rate
    
    # 2. Income-to-Debt Ratio: Inverse of DTI, but in absolute terms.
    df['income_to_debt_ratio'] = (safe_annual_income / total_annual_debt + 1e-6)
    
    # 3. Loan-to-Credit-Score: Is this a big loan for their score?
    df['loan_to_credit_score'] = df['loan_amount'] / df['credit_score']
    
    df = df.drop(columns=['grade_subgrade', 'id'])
    return df

print("Applying feature engineering...")
X = feature_engineer(train_df.copy())
y = X.pop('loan_paid_back')
X_test = feature_engineer(test_df.copy())
print("New features created.")


# --- Define Pre-processing Pipelines ---

numeric_features = ['annual_income', 'debt_to_income_ratio', 'credit_score',
                    'loan_amount', 'interest_rate']
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

ordinal_features = ['education_level', 'grade', 'subgrade']
education_categories = ['Other', 'High School', "Bachelor's", "Master's", 'PhD']
grade_categories = ['G', 'F', 'E', 'D', 'C', 'B', 'A']
subgrade_categories = ['5', '4', '3', '2', '1']

ordinal_transformer = Pipeline(steps=[
    ('ordinal', OrdinalEncoder(categories=[
        education_categories,
        grade_categories,
        subgrade_categories
    ]))
])

one_hot_features = ['gender', 'marital_status', 'employment_status', 
                    'loan_purpose']

one_hot_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])


# --- Combine All Tranformers into a Single Prprocessor ---

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('ord', ordinal_transformer, ordinal_features),
        ('onehot', one_hot_transformer, one_hot_features)
    ],
    remainder='passthrough'
)


# --- Apply the Pre-processing ---
print("Applying pre-processing pipeline to all V6 data...")
X_processed = preprocessor.fit_transform(X)
X_test_processed = preprocessor.transform(X_test)
print(f"Data pre-processing complete. New feature shape: {X_processed.shape}\n")


# --- Part 1: K-Fold Cross-Validation Training ---
print("--- Part 1: K-Fold Cross-Validation with V5 FINAL Params ---")

# --- These are your NEW BEST PARAMETERS from V5 ---
best_params = {
    'objective': 'binary',
    'metric': 'auc',
    'n_estimators': 2000,
    'is_unbalance': True,
    'n_jobs': -1,
    'random_state': 42,
    'verbosity': -1,
    # V3A Tuned Values Below:
    'learning_rate': 0.07715287679481146,
    'num_leaves': 28,
    'max_depth': 4,
    'subsample': 0.9956825861346151,
    'colsample_bytree': 0.6863609614892977
}

# --- K-Fold Setup ---
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_scores = []
test_preds = np.zeros(len(X_test_processed))

print(f"Starting {N_FOLDS}-Fold Cross-Validation...")
for fold, (train_index, val_index) in enumerate(skf.split(X_processed, y)):
    print(f"\n--- Fold {fold+1}/{N_FOLDS} ---")
    
    X_train_fold, X_val_fold = X_processed[train_index], X_processed[val_index]
    y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
    
    print(f"Training model for Fold {fold+1}...")
    model = LGBMClassifier(**best_params)
    
    model.fit(
        X_train_fold, 
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        eval_metric='auc',
        callbacks=[early_stopping(100, verbose=False)]
    )
    
    val_preds = model.predict_proba(X_val_fold)[:, 1]
    fold_auc = roc_auc_score(y_val_fold, val_preds)
    oof_scores.append(fold_auc)
    print(f"Fold {fold+1} AUC: {fold_auc:.6f}")
    
    test_preds += model.predict_proba(X_test_processed)[:, 1] / N_FOLDS


# --- Part 2: Final Results & Submission (V7-B) ---
print("\n\n--- K-Fold CV Finished ---")
mean_auc = np.mean(oof_scores)
print(f"All Fold AUCs: {oof_scores}")
print(f"Mean CV AUC Score (V5 FINAL): {mean_auc:.6f}")
print(f"Your V4 (Base Features) CV score was: 0.923141")

# Create the final submission file
print("\n--- Part 2: Generating Submission File (V5) ---")
submission_df = pd.DataFrame({
    'id': test_ids,
    'loan_paid_back': test_preds
})

submission_filename = 'submission.csv'
submission_df.to_csv(submission_filename, index=False)

print(f"\nSubmission file saved successfully as '{submission_filename}'")
print("\n--- Kaggle V5 Script Finished ---")

