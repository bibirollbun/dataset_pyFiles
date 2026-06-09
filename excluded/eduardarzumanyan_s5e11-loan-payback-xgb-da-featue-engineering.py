import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')


def analyze_dataframe(df: pd.DataFrame, corr_method: str = 'pearson'):
    print("="*80)
    print("ğŸ”� DATAFRAME OVERVIEW")
    print("="*80)
    print(f"Shape: {df.shape}")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    print("\nData types:\n", df.dtypes)
    print("\nMemory usage:")
    print(df.memory_usage(deep=True).sum(), "bytes total")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\n" + "="*80)
    print("ğŸ§© MISSING VALUES")
    print("="*80)
    missing = df.isna().sum()
    missing_percent = (missing / len(df)) * 100
    missing_table = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_percent})
    print(missing_table[missing_table['Missing Count'] > 0].sort_values(by='Missing %', ascending=False))
    print("\n" + "="*80)
    print("ğŸ”¢ NUMERIC FEATURES SUMMARY")
    print("="*80)
    numeric_df = df.select_dtypes(include=np.number)
    if not numeric_df.empty:
        print(numeric_df.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T)
        print("\nSkewness:\n", numeric_df.skew())
        print("\nKurtosis:\n", numeric_df.kurt())
    else:
        print("No numeric columns found.")
    print("\n" + "="*80)
    print("ğŸ”¤ CATEGORICAL FEATURES SUMMARY")
    print("="*80)
    categorical_df = df.select_dtypes(exclude=np.number)
    if not categorical_df.empty:
        for col in categorical_df.columns:
            print(f"\nâ–¶ Column: {col}")
            print(f"Unique values: {categorical_df[col].nunique()}")
            print(f"Top 5 most frequent values:\n{categorical_df[col].value_counts().head(5)}\n")
    else:
        print("No categorical columns found.")
    print("\n" + "="*80)
    print("ğŸ“Š CORRELATION MATRIX")
    print("="*80)
    if not numeric_df.empty:
        corr = numeric_df.corr(method=corr_method)
        print(corr)
        print("\nStrong correlations (|corr| > 0.7):")
        strong_corrs = corr[(abs(corr) > 0.7) & (abs(corr) < 1)]
        print(strong_corrs.dropna(how='all').dropna(axis=1, how='all'))
    else:
        print("No numeric columns to compute correlation.")
    print("\n" + "="*80)
    print("ğŸ§¬ DUPLICATES")
    print("="*80)
    dup_count = df.duplicated().sum()
    print(f"Duplicate rows: {dup_count} ({dup_count/len(df)*100:.2f}%)")
    print("\n" + "="*80)
    print("âœ… UNIQUE COUNTS PER COLUMN")
    print("="*80)
    uniques = df.nunique().sort_values(ascending=False)
    print(uniques)
    print("\n" + "="*80)
    print("ğŸ“ˆ VALUE DISTRIBUTIONS (Top Columns)")
    print("="*80)
    for col in df.columns[:5]:
        print(f"\n--- {col} ---")
        if df[col].dtype == 'object' or df[col].dtype == 'category':
            print(df[col].value_counts(normalize=True).head(10))
        else:
            print(df[col].describe())
    print("\nAnalysis complete âœ…")


analyze_dataframe(train)


train = train.drop('id',axis=1)
test = test.drop('id',axis=1)

categorical_cols = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']
le = LabelEncoder()

for col in categorical_cols:
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])


def feature_adding(df):
    df['log_annual_income'] = np.log1p(df['annual_income'])
    df['log_loan_amount'] = np.log1p(df['loan_amount'])
    df['income_to_loan_ratio'] = df['annual_income'] / (df['loan_amount'] + 1)
    df['interest_to_income_ratio'] = df['interest_rate'] / (df['annual_income'] + 1)
    df['income_minus_loan'] = df['annual_income'] - df['loan_amount']
    df['loan_to_income_ratio'] = df['loan_amount'] / (df['annual_income'] + 1)
    df['debt_burden_score'] = df['loan_to_income_ratio'] * df['interest_rate']
    df['estimated_debt'] = df['debt_to_income_ratio'] * df['annual_income']
    df['loan_per_credit_point'] = df['loan_amount'] / (df['credit_score'] + 1)
    df['income_x_credit'] = df['annual_income'] * df['credit_score']
    df['loan_x_interest'] = df['loan_amount'] * df['interest_rate']
    df['credit_x_interest'] = df['credit_score'] * df['interest_rate']
    df['loan_amount_squared'] = df['loan_amount']**2
    df['high_income_low_credit'] = ((df['annual_income'] > 70000) & (df['credit_score'] < 650)).astype(int)
    df['low_income_high_interest'] = ((df['annual_income'] < 30000) & (df['interest_rate'] > 0.15)).astype(int)
    df['high_dti_high_interest'] = ((df['debt_to_income_ratio'] > 0.4) & (df['interest_rate'] > 0.1)).astype(int)
    return df


train = feature_adding(train)


test = feature_adding(test)


X = train.drop('loan_paid_back', axis = 1)
y = train['loan_paid_back']
X_test = test


skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
fold_scores = []
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X,y), 1):
    print(f"-----FOLD {fold}")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = XGBClassifier(
        n_estimators = 100000,
        learning_rate = 0.01,
        max_depth = 5,
        subsample = 0.8,
        colsample_bytree = 0.8,
        eval_metric = 'auc',
        random_state = 42,
        tree_method = 'hist',
        use_label_encoder = False,
        n_jobs = -1,
        early_stopping_rounds = 100
    )

    model.fit(
        X_train, y_train, 
        eval_set = [(X_val, y_val)],
        verbose = 100
    )

    y_val_pred = model.predict_proba(X_val)[:, 1]
    y_test_pred = model.predict_proba(X_test)[:, 1]

    # Evaluate ROC-AUC
    auc = roc_auc_score(y_val, y_val_pred)
    fold_scores.append(auc)
    oof_preds[val_idx] = y_val_pred

    # Average test predictions
    test_preds += y_test_pred / 5

    print(f"Fold {fold} ROC-AUC: {auc:.5f}")


submission['loan_paid_back'] = test_preds
submission.to_csv('submission.csv',index=False)

