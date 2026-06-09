import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (10, 6)

import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')


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
        'column': df.columns,
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


# ---------------------------------------------
# DISTRIBUTION COMPARISON (TRAIN vs TEST)
# ---------------------------------------------
def compare_distributions(train, test, num_features):
    for col in num_features:
        plt.figure()
        sns.kdeplot(train[col], label='Train', fill=True, alpha=0.4)
        sns.kdeplot(test[col], label='Test', fill=True, alpha=0.4)
        plt.title(f"Distribution of {col} (Train vs Test)")
        plt.legend()
        plt.tight_layout()
        plt.show()


def compare_categorical_proportions(train, test, cat_features):
    for col in cat_features:
        plt.figure(figsize=(8,4))
        train_counts = train[col].value_counts(normalize=True)
        test_counts = test[col].value_counts(normalize=True)
        compare_df = pd.concat([train_counts, test_counts], axis=1)
        compare_df.columns = ["Train", "Test"]
        compare_df.plot(kind='bar', width=0.7)
        plt.title(f"{col} Proportion (Train vs Test)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# ---------------------------------------------
# TARGET RELATIONSHIPS
# ---------------------------------------------
def numeric_vs_target(train, num_features, target='loan_paid_back'):
    for col in num_features:
        plt.figure()
        sns.boxplot(x=target, y=col, data=train, palette="coolwarm")
        plt.title(f"{col} vs {target}")
        plt.tight_layout()
        plt.show()


def scatter_with_target(train, x, y, target='loan_paid_back'):
    plt.figure()
    sns.scatterplot(
        data=train,
        x=x,
        y=y,
        hue=target,
        alpha=0.5,
        palette="viridis"
    )
    plt.title(f"{y} vs {x} colored by {target}")
    plt.tight_layout()
    plt.show()

# ---------------------------------------------
# DEMOGRAPHIC ANALYSIS
# ---------------------------------------------
def demographic_target_rate(train, cat_features, target='loan_paid_back'):
    for col in cat_features:
        plt.figure(figsize=(8,4))
        sns.barplot(
            data=train,
            x=col,
            y=target,
            estimator=np.mean,
            order=train[col].value_counts().index
        )
        plt.title(f"Average {target} by {col}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# ---------------------------------------------
# LOAN PURPOSE & GRADE
# ---------------------------------------------
def purpose_and_grade_analysis(train, target='loan_paid_back'):
    # Loan purpose
    plt.figure(figsize=(10,5))
    sns.barplot(
        data=train,
        x='loan_purpose',
        y=target,
        estimator=np.mean,
        order=train.groupby('loan_purpose')[target].mean().sort_values(ascending=False).index
    )
    plt.title("Average Loan Repayment by Loan Purpose")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Grade subgrade
    plt.figure(figsize=(12,5))
    sns.barplot(
        data=train,
        x='grade_subgrade',
        y=target,
        estimator=np.mean,
        order=train.groupby('grade_subgrade')[target].mean().sort_values(ascending=False).index
    )
    plt.title("Average Loan Repayment by Grade/Subgrade")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

# ---------------------------------------------
# FEATURE INTERACTIONS
# ---------------------------------------------
def credit_heatmap(train, credit_col='credit_score', dti_col='debt_to_income_ratio', target='loan_paid_back'):
    df = train.copy()
    df['credit_bin'] = pd.qcut(df[credit_col], q=10, duplicates='drop')
    df['dti_bin'] = pd.qcut(df[dti_col], q=10, duplicates='drop')

    pivot = df.pivot_table(values=target, index='credit_bin', columns='dti_bin', aggfunc='mean')
    plt.figure(figsize=(10,8))
    sns.heatmap(pivot, cmap='YlGnBu', annot=False)
    plt.title(f"{target} rate by Credit Score & Debt-to-Income bins")
    plt.tight_layout()
    plt.show()

# ---------------------------------------------
# CORRELATION MATRIX
# ---------------------------------------------
def correlation_heatmap(train, target='loan_paid_back'):
    numeric_df = train.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, cmap='coolwarm', annot=True, fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------
# MAIN EXECUTION WRAPPER 
# ---------------------------------------------
def run_all_visuals(train, test):
    num_features = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate']
    cat_features = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']

    print("1️⃣ Comparing distributions...")
    compare_distributions(train, test, num_features)
    compare_categorical_proportions(train, test, cat_features)

    print("2️⃣ Exploring target relationships...")
    numeric_vs_target(train, num_features)
    scatter_with_target(train, x='loan_amount', y='interest_rate')

    print("3️⃣ Demographic analysis...")
    demographic_target_rate(train, ['gender', 'marital_status', 'education_level', 'employment_status'])

    print("4️⃣ Loan purpose & grade analysis...")
    purpose_and_grade_analysis(train)

    print("5️⃣ Creditworthiness heatmap...")
    credit_heatmap(train)

    print("6️⃣ Correlation heatmap...")
    correlation_heatmap(train)

# Call Functions
run_all_visuals(train, test)


for df in [train, test]:
    
    # Numeric features
    df['income_to_loan'] = df['annual_income'] / (df['loan_amount'] + 1e-6)
    df['debt_burden'] = df['debt_to_income_ratio'] * df['loan_amount']
    df['total_interest_cost'] = df['interest_rate'] * df['loan_amount']
    df['disposable_income'] = df['annual_income'] * (1 - df['debt_to_income_ratio'])
    
    # Credit score bucket
    df['credit_score_bucket'] = pd.cut(
        df['credit_score'],
        bins=[300,650,700,750,850],
        labels=['poor','fair','good','excellent']
    )
    
    # Interaction features
    df['gender_marital'] = df['gender'] + '_' + df['marital_status']
    df['education_employment'] = df['education_level'] + '_' + df['employment_status']
    
    # Grade/Subgrade split
    df['grade'] = df['grade_subgrade'].str[0]
    df['subgrade'] = df['grade_subgrade'].str[1:]
    
    # Rare categories
    for col in ['loan_purpose','education_level','employment_status']:
        freq = df[col].value_counts(normalize=True)
        rare = freq[freq < 0.01].index
        df[col+'_rare'] = df[col].apply(lambda x: 'other' if x in rare else x)


    #  Interest burden: how much interest adds relative to principal
    df['effective_interest_burden'] = df['total_interest_cost'] / (df['loan_amount'] + 1e-6)

    # Credit grade numeric encoding: ordinal mapping for model interpretability
    grade_order = {g: i+1 for i, g in enumerate(sorted(df['grade'].dropna().unique()))}
    df['credit_grade_numeric'] = df['grade'].map(grade_order)

    # Debt-to-income × interest rate: captures compound stress on repayment ability
    df['dti_interest_interaction'] = df['debt_to_income_ratio'] * df['interest_rate']

    # Monthly payment ratio: approximates repayment strain relative to income
    # Assuming a standard 36month loan term
    assumed_term = 36  
    df['monthly_payment_ratio'] = (
        (df['loan_amount'] * (1 + df['interest_rate'])) / assumed_term
    ) / (df['annual_income'] / 12 + 1e-6)

    # Target-encoded mean loan repayment rate by loan purpose (smoothed using train data)
    # NOTE: Compute this ONLY on train; then map to test
train_means = train.groupby('loan_purpose')['loan_paid_back'].mean()
train['purpose_default_rate'] = train['loan_purpose'].map(train_means)
test['purpose_default_rate'] = test['loan_purpose'].map(train_means)



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score, roc_curve
import xgboost as xgb
import lightgbm as lgb



# Preprocessing Function
def preprocess_data(train, test=None, target_col='loan_paid_back', drop_cols=['id']):
    """
    Preprocess training and test data.
    Automatically encodes categorical columns with OrdinalEncoder and handles unseen categories.
    """
    # Keep test IDs 
    test_ids = test['id'].copy() 

    # Drop ID and target
    X = train.drop(columns=drop_cols + [target_col])
    y = train[target_col]

    # Train/validation split before encoding
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Identify categorical columns
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    # OrdinalEncoder: encodes unseen categories as -1
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=np.int32)

    if cat_cols:
        encoder.fit(X_train[cat_cols])

        # Transform training and validation data
        X_train[cat_cols] = encoder.transform(X_train[cat_cols])
        X_val[cat_cols] = encoder.transform(X_val[cat_cols])

        # Transform holdout test
        if test is not None:
            X_test = test.drop(columns=drop_cols, errors='ignore').copy()
            for col in cat_cols:
                if col in X_test.columns:
                    X_test[col] = X_test[col].astype(str)
            X_test[cat_cols] = encoder.transform(X_test[cat_cols])
        else:
            X_test = None
    else:
        X_test = test.drop(columns=drop_cols, errors='ignore').copy() if test is not None else None

    return X_train, X_val, y_train, y_val, X_test, test_ids, encoder, cat_cols


# Model Training

def train_models(X_train, y_train, X_val, y_val):
    """Train and evaluate XGBoost and LightGBM models."""
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='auc',
        early_stopping_rounds=50,
        verbosity=1
    )

    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

    lgb_model = lgb.LGBMClassifier(
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        metric='auc',
        verbose=-1
    )

    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=100)
        ]
    )

    return xgb_model, lgb_model



# Evaluation
def evaluate_models(xgb_model, lgb_model, X_val, y_val):
    """Compute AUC and plot ROC curves for both models and ensemble."""
    y_pred_xgb = xgb_model.predict_proba(X_val)[:, 1]
    y_pred_lgb = lgb_model.predict_proba(X_val)[:, 1]
    ensemble_pred = 0.6 * y_pred_xgb + 0.4 * y_pred_lgb

    # AUC scores
    auc_xgb = roc_auc_score(y_val, y_pred_xgb)
    auc_lgb = roc_auc_score(y_val, y_pred_lgb)
    auc_ens = roc_auc_score(y_val, ensemble_pred)

    print("\n" + "=" * 50)
    print("MODEL PERFORMANCE (Validation Set)")
    print("=" * 50)
    print(f"XGBoost   - AUC: {auc_xgb:.5f}")
    print(f"LightGBM  - AUC: {auc_lgb:.5f}")
    print(f"Ensemble  - AUC: {auc_ens:.5f}")
    print(f"Best Model: {'XGBoost' if auc_xgb > auc_lgb else 'LightGBM'} "
          f"({max(auc_xgb, auc_lgb):.5f})")
    print("=" * 50)

    # Plot ROC curves
    fpr_xgb, tpr_xgb, _ = roc_curve(y_val, y_pred_xgb)
    fpr_lgb, tpr_lgb, _ = roc_curve(y_val, y_pred_lgb)
    fpr_ens, tpr_ens, _ = roc_curve(y_val, ensemble_pred)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {auc_xgb:.4f})', color='blue', lw=2)
    plt.plot(fpr_lgb, tpr_lgb, label=f'LightGBM (AUC = {auc_lgb:.4f})', color='green', lw=2)
    plt.plot(fpr_ens, tpr_ens, label=f'Ensemble (AUC = {auc_ens:.4f})', color='red', lw=2.5)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5, label='Random Guess')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison', fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()



# Call Pipeline
X_train, X_val, y_train, y_val, X_test, test_ids, encoder, cat_cols = preprocess_data(train, test)
xgb_model, lgb_model = train_models(X_train, y_train, X_val, y_val)
evaluate_models(xgb_model, lgb_model, X_val, y_val)



# Predict on test
ensemble_test_pred = 0.6 * xgb_model.predict_proba(X_test)[:, 1] + 0.4 * lgb_model.predict_proba(X_test)[:, 1]


submission = pd.DataFrame({'id': test_ids, 'loan_paid_back': ensemble_test_pred})
submission.to_csv('submission.csv', index=False)
submission.head(10)

