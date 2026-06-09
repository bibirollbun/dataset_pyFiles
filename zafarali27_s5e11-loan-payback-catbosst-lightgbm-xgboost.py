import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split,cross_val_score,StratifiedKFold,GridSearchCV
from sklearn.preprocessing import LabelEncoder,  StandardScaler,OrdinalEncoder
from sklearn.metrics import accuracy_score,confusion_matrix
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
# from imblearn.under_sampling import RandomUnderSampler


train_df = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
sub_sam = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")


from colorama import Fore, Style

# Print the shape of the dataframe (number of rows and columns)
print(Fore.CYAN + "train_df shape: " + Style.RESET_ALL)
print(f"{train_df.shape}\n")

# Print basic information about the dataframe (column names, data types, non-null values)
print(Fore.GREEN + "train_df info: " + Style.RESET_ALL)
print(f"{train_df.info()}\n")

# Print the count of missing (NaN) values in each column
print(Fore.YELLOW + "train_df isnull sum: " + Style.RESET_ALL)
print(f"{train_df.isnull().sum()}\n")

# Print summary statistics for numerical columns (count, mean, std, min, max, etc.)
print(Fore.MAGENTA + "train_df describe: " + Style.RESET_ALL)
print(f"{train_df.describe()}\n")


# Define the numerical & categorical
numerical_col = train_df.select_dtypes(include = ["int64","float64"]).columns
# Define the categorical
categorical_col = train_df.select_dtypes(include = "object").columns

print(f"We have features: {len(numerical_col)} numerical features {numerical_col}")
print("-"*100)
print(f"We have features: {len(categorical_col)} categorical features {categorical_col}")


# -------------------------------------------------------
# ðŸŽ¯ Feature Distribution Visualization (Numerical Features)
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
        sns.histplot(train_df[col].dropna(), kde=True, bins=30,
                     ax=axes[0], color='skyblue', edgecolor='black')
        axes[0].set_title('Histogram (Shape & Density)')
        axes[0].set_xlabel(col)
        axes[0].set_ylabel('Frequency')

        # Right: Boxplot for quartiles/outliers
        sns.boxplot(x=train_df[col].dropna(), ax=axes[1], color='lightcoral')
        axes[1].set_title('Box Plot (Outliers & Spread)')
        axes[1].set_xlabel(col)

    elif col in discrete_features:
        # Left: Count Plot for small integer-like features
        sns.countplot(x=train_df[col].dropna(), ax=axes[0], palette='viridis', edgecolor='black')
        axes[0].set_title('Count Plot (Frequency)')
        axes[0].set_xlabel(col)
        axes[0].set_ylabel('Count')

        # Right: Boxplot (still useful)
        sns.boxplot(x=train_df[col].dropna(), ax=axes[1], color='lightcoral')
        axes[1].set_title('Box Plot (Summary)')
        axes[1].set_xlabel(col)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # Print descriptive statistics
    print("\nðŸ“Š Descriptive Statistics:")
    print(train_df[col].describe().round(3))
    print("\n" + "="*50 + "\n")


# ðŸ“Š Distribution of Categorical Features

for col in categorical_col:
    counts = train_df[col].value_counts()
    plt.figure(figsize = (20,6))
    plt.subplot(1,2,1)
    sns.countplot(data = train_df, x = col,  palette = "Set2")
    plt.title(f"Count of {col}")
    plt.xticks(rotation = 90)
    plt.ylabel("Count")

    plt.subplot(1,2,2)
    plt.pie(counts,labels = counts.index,autopct = "%1.1f%%",startangle=90)
    plt.title(f"Percentage of {col}")
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


# ðŸŽ¨ Categorical Feature Distributions by Loan Repayment Status - Custom Colors

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
        data=train_df,
        x=col,
        hue=target_col,
        palette=custom_palette,
        edgecolor='black',
        order=train_df[col].value_counts().index  # Order bars by frequency
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


from scipy.stats import chi2_contingency
chi2_test = []
for feature in categorical_col:
    if chi2_contingency(pd.crosstab(train_df['loan_paid_back'], train_df[feature]))[1] < 0.05:
        chi2_test.append('Reject Null Hypothesis')
    else:
        chi2_test.append('Fail to Reject Null Hypothesis')
result = pd.DataFrame(data=[categorical_col, chi2_test]).T # Create a DataFrame to store the chi-squared test results
result.columns = ['Column', 'Hypothesis Result']
result


plt.figure(figsize=(8, 6))
sns.heatmap(train_df[['annual_income', 'debt_to_income_ratio', 'credit_score',
                'loan_amount', 'interest_rate', 'loan_paid_back']].corr(),
            annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap (Numerical Features + Target)', fontsize=14)
plt.show()


num_cols = train_df[["annual_income","debt_to_income_ratio","credit_score","loan_amount","interest_rate"]]

from scipy.stats import skew

skew_values = num_cols.apply(lambda x: skew(x.dropna()))
print(skew_values.sort_values(ascending=False))


skewed_cols = skew_values[abs(skew_values) > 1].index.tolist()
print("Highly skewed columns:", skewed_cols)

for col in skewed_cols:
    train_df[col] = np.log1p(train_df[col])
    test_df[col]  = np.log1p(test_df[col])

from sklearn.preprocessing import PowerTransformer

# Initialize Yeo-Johnson transformer
pt = PowerTransformer(method='yeo-johnson')


# Outliers (IQR)

for col in num_cols:
    Q1 = train_df[col].quantile(0.25)
    Q3 = train_df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    train_df[col] = train_df[col].clip(lower=lower_bound, upper=upper_bound)
    test_df[col] = test_df[col].clip(lower=lower_bound, upper=upper_bound)


train_df["loan_paid_back"].value_counts().plot(kind = "pie", autopct='%1.1f%%',figsize=(6, 6),title = "Loan_paid_back")
plt.show()


def create_features(df):

    df = df.copy()

    df['income_to_loan_ratio'] = df['annual_income'] / df['loan_amount']
    df['affordability_ratio'] = (df['annual_income'] / 12) / (df['loan_amount'] * df['interest_rate'] / 1200)

    df['risk_score'] = (
        df['debt_to_income_ratio'] * 0.3 +
        (800 - df['credit_score']) / 800 * 0.3 +
        df['interest_rate'] / 25 * 0.2 +
        (df['loan_amount'] / df['annual_income']) * 0.2
    )

    if 'grade_subgrade' in df.columns:
        df['grade'] = df['grade_subgrade'].str[0]
        df['subgrade_num'] = df['grade_subgrade'].str[1].astype(int)


    employment_mapping = {
        'Unemployed': 0,
        'Student': 1,
        'Self-employed': 2,
        'Employed': 3,
        'Retired': 2
    }
    df['employment_stability'] = df['employment_status'].map(employment_mapping)

    education_mapping = {
        'High School': 1,
        'Other': 2,
        'Bachelor\'s': 3,
        'Master\'s': 4,
        'PhD': 5
    }
    df['education_num'] = df['education_level'].map(education_mapping)

    return df

# Applying Feature engineering
# Applying Feature engineering
train_df_eng = create_features(train_df)
test_df_eng = create_features(test_df)


def preprocess_data(train_df, test_df):
    # make copies so original dfs are not modified unexpectedly
    train_df = train_df.copy()
    test_df = test_df.copy()

    # Drop columns properly ---
    cols_to_drop = ['education_level', 'employment_status', 'grade_subgrade']
    # only drop columns that actually exist to avoid errors
    cols_to_drop = [c for c in cols_to_drop if c in train_df.columns]
    if cols_to_drop:
        train_df = train_df.drop(columns=cols_to_drop)
        test_df  = test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns])

    # Numerical features
    features = [
        'id', 'annual_income', 'debt_to_income_ratio', 'credit_score',
        'loan_amount', 'interest_rate', 'loan_paid_back',
        'income_to_loan_ratio', 'affordability_ratio', 'risk_score',
        'subgrade_num', 'employment_stability', 'education_num'
    ]
    # categorical features
    categorical_cols = ['gender', 'marital_status', 'loan_purpose', 'grade']

    # Keep only categorical columns that actually exist
    categorical_cols = [c for c in categorical_cols if c in train_df.columns]

    #  Encode categorical columns robustly ---
    if categorical_cols:
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        # fit on train's categorical columns
        enc.fit(train_df[categorical_cols])
        # transform both train and test (returns numpy array; assign back)
        train_df[categorical_cols] = enc.transform(train_df[categorical_cols])
        test_df[categorical_cols]  = enc.transform(test_df[categorical_cols])

    return train_df, test_df


# Applay the function
train_processed, test_processed = preprocess_data(train_df_eng, test_df_eng)
print(train_processed.shape, test_processed.shape)


# Drop target + ID
X = train_processed.drop(columns=["loan_paid_back", "id"])
y = train_processed["loan_paid_back"]

X_test_final = test_processed.drop(columns=["id"])

# Convert to numpy (NO SCALING for tree models)
X = X.values
X_test_final = X_test_final.values




# Models dicts
models = {
    "LightGBM": LGBMClassifier(
        objective='binary',
        metric = "auc",
        boosting_type = "gbdt",
        n_estimators = 1000,
        learning_rate = 0.01,
        colsample_freq = 1,
        min_child_samples = 20,
        reg_alpha = 0.05,
        reg_lambda=0.1,
        random_state = 42,
        n_jobs = -1,
        device = "cpu",
        verbose = -1
    ),
    "CatBoost": CatBoostClassifier(
        iterations = 3000,
        learning_rate = 0.03,
        depth = 8,
        loss_function = "Logloss",
        eval_metric = "AUC",
        random_state = 42,
        # verbosity= 0
        auto_class_weights = "Balanced",
        l2_leaf_reg=5,
        task_type="CPU"
    ),
    "XGBoost" : XGBClassifier(
        objective = "binary:logistic",
        eval_metric = "auc",
        learning_rate = 0.01,
        max_depth = 8,
        min_child_weight = 3,
        colsample_bytree = 0.3,
        subsample = 0.6,
        reg_alpha = 0.5,
        reg_lambda = 2.0,
        n_estimators = 10000,
        random_state = 42,
        n_jobs = -1,
        verbose = -1,
        device = 'cpu',
        tree_method = 'hist'
    )
}


def off_predictions(model,X,y,X_test,n_splits):
  skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

  off_train = np.zeros(len(X))
  off_test = np.zeros(len(X_test))

  for fold, (tr_idx, val_idx) in enumerate(skf.split(X,y)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    model.fit(X_tr, y_tr)

    off_train[val_idx] = model.predict_proba(X_val)[:,1]
    off_test += model.predict_proba(X_test)[:,1] / n_splits

    fold_auc = roc_auc_score(y_val, off_train[val_idx])
    print(f"Fold {fold+1} AUC: {fold_auc:.5f}")

  total_auc = roc_auc_score(y, off_train)
  print(f"\n OOF AUC: {total_auc:.5f}")

  return off_train, off_test



from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np

def oof_predictions(model, X, y, X_test, n_splits=7):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_train = np.zeros(len(X))
    oof_test = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model.fit(X_tr, y_tr)

        oof_train[val_idx] = model.predict_proba(X_val)[:, 1]
        oof_test += model.predict_proba(X_test)[:, 1] / n_splits

        fold_auc = roc_auc_score(y_val, oof_train[val_idx])
        print(f"Fold {fold+1} AUC: {fold_auc:.5f}")

    total_auc = roc_auc_score(y, oof_train)
    print(f"\nOOF AUC: {total_auc:.5f}")

    return oof_train, oof_test



oof_cat, test_cat = oof_predictions(
    models["CatBoost"], X, y, X_test_final
)

oof_lgb, test_lgb = oof_predictions(
    models["LightGBM"], X, y, X_test_final
)

oof_xgb, test_xgb = oof_predictions(
    models["XGBoost"], X, y, X_test_final
)




blend_oof = (
    0.4 * oof_cat +
    0.3 * oof_lgb +
    0.3 * oof_xgb
)

blend_test = (
    0.4 * test_cat +
    0.3 * test_lgb +
    0.3 * test_xgb
)

print("Blended OOF AUC:", roc_auc_score(y, blend_oof))



submission = sub_sam.copy()
submission["loan_paid_back"] = blend_test
submission.to_csv("submission_oof_blend.csv", index=False)

