# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import warnings
import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt
# Default env settings
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (6, 5)
%matplotlib inline


train_df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')

print('--- TRAIN DATA ---')
display(train_df.head())
print('--- TEST DATA ---')
display(test_df.head())


df = train_df.copy()

# General info
print('-' * 15)
display(df.info())
print('-' * 15)
display(f'Size: {df.shape[0]} returns, {df.shape[1]} features.')
print('-' * 15)
if not df.isna().any().any() and not df.duplicated().sum(): # No msn values & duplicates
    print('No missing values')
    print('No duplicates.')
else:
    print(f'Missing values:')
    display(df.isna().sum())
    display(msno.matrix(df))
    print(f'Duplicates: {df.duplicated().sum()}')


# One client info
df.head(1)


df.columns


df = df.drop(columns=['id'])

num_features = df.select_dtypes(include=['int64', 'float64'])
cat_features = pd.concat([df.select_dtypes(exclude=['int64', 'float64']), df['loan_paid_back']], axis=1)
target = 'loan_paid_back'
# cat_features.head()

print(f'Numeric features: {num_features.columns}\n')
print(f'Categorial features: {cat_features.columns}')


for feature in num_features.drop(columns=['loan_paid_back']):
    sns.histplot(
        data=df,
        x=feature,
        kde=True,
        hue=target,
        bins=30
    )
    plt.title(f'Distribution: {feature.replace("_", " ")}')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.legend(title='Is loan paid back?', labels=['Yes', 'No'])
    plt.show()


for feature in cat_features[:-1]:
    sns.countplot(
        data=df,
        x=feature,
        hue=target,
        palette='viridis'
    )
    plt.title(f'Distribution: {feature.replace("_", " ")}')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.legend(title='Is loan paid back?', labels=['No', 'Yes'])
    plt.show()


corr_mtx = df.corr(numeric_only=True)

# Correlation: feature scores dataframe
corr_with_target = corr_mtx['loan_paid_back']
corr_scoring_df = (
    corr_with_target
    .reset_index()
    .rename(columns={'index': 'feature', 'loan_paid_back': 'correlation'})
)
corr_scoring_df['Abs(correlation)'] = np.abs(corr_scoring_df['correlation'])

print('Feature correlation (sorted by absolute correlation)')
display(corr_scoring_df.sort_values(by='Abs(correlation)'))

print('-' * 45)

# Heatmap: feature correlation
sns.heatmap(corr_mtx[['loan_paid_back']].sort_values(by='loan_paid_back',ascending=False),
            annot=True, fmt='.2f', cmap='rocket')
plt.title('Correlation map')
plt.show()


for feature in num_features:
    sns.boxplot(data=df, x=feature)
    plt.title(f'Boxplot (outliers): {feature.replace("_", " ")}')
    plt.show()


from scipy.stats import ttest_ind, chi2_contingency


def ttest(feature):
    loan_paid_df = df[df['loan_paid_back'] == 1][feature]
    loan_default_df = df[df['loan_paid_back'] == 0][feature]

    t_stat, p_value = ttest_ind(loan_paid_df, loan_default_df, equal_var=False)
    print(f'{feature.replace("_", " ").upper()} is {"SIGNIFICANT" if p_value < .05 else "NOT SIGNIFICANT"}.')
    print(f'T-statistics: {t_stat:.2f} | P-value: {p_value:.5f}')


for feature in num_features:
    print('-' * 15)
    ttest(feature)


def chi2_test(feature):
    observed = pd.crosstab(index=df['loan_paid_back'], columns=df[feature])

    chi2, p_value, dof, expeceted = chi2_contingency(observed) 
    
    print(f'{feature.replace("_", " ").upper()} is {"SIGNIFICANT" if p_value < .05 else "NOT SIGNIFICANT"}.')
    print(f'P-value: {p_value:.5f}')


for feature in cat_features:
    print('-' * 15)
    chi2_test(feature)


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score


def add_features(df):
    df = df.copy()

    df["loan_amount"] = np.log1p(df["loan_amount"])
    df["annual_income"] = np.log1p(df["annual_income"])

    df["income_to_debt_ratio"] = df["annual_income"] / (df["annual_income"] * df["debt_to_income_ratio"] + 1e-6)
    df["credit_score_scaled"] = (df["credit_score"] - df["credit_score"].mean()) / df["credit_score"].std()

    df["risk_index"] = df["interest_rate"] * (1 / (df["credit_score"] + 1e-6))

    df["income_credit_interaction"] = df["annual_income"] * df["credit_score"]
    df["loan_interest_interaction"] = df["loan_amount"] * df["interest_rate"]

    df["is_self_employed"] = (df["employment_status"] == "Self-employed").astype(int)
    # df["is_married"] = (df["marital_status"] == "Married").astype(int)
    df["is_debt_consolidation"] = (df["loan_purpose"].str.lower().str.contains("debt")).astype(int)

    edu_map = {
        "High School": 1,
        "Associate": 2,
        "Bachelor's": 3,
        "Master's": 4,
        "Doctorate": 5
    }
    df["education_rank"] = df["education_level"].map(edu_map).fillna(0)

    df["grade_letter"] = df["grade_subgrade"].str[0]
    df["grade_num_only"] = df["grade_subgrade"].str[1].astype(float)
    df["grade_value"] = df["grade_letter"].apply(lambda x: ord(x) - 64) + df["grade_num_only"] / 10
    df["is_high_grade"] = (df["grade_value"] <= 3).astype(int)

    df["debt_to_credit_ratio"] = df["debt_to_income_ratio"] / (df["credit_score"] + 1e-6)
    
    df["solvency_index"] = (
        df["annual_income"] / (df["loan_amount"] * (1 + df["interest_rate"] / 100))
    ) * df["credit_score"]

    df["interest_vs_grade_median"] = (
        df["interest_rate"] / df.groupby("grade_letter")["interest_rate"].transform("median")
    )

    return df


df = add_features(df)
df.head()


loan_not_paid_count, loan_paid_count = df['loan_paid_back'].value_counts()
scale_pos_weight = loan_paid_count / loan_not_paid_count

scale_pos_weight


X = df.drop(columns=[target, 'marital_status'])
y = df[target]


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X_train.select_dtypes(exclude=['int64', 'float64']).columns


numeric_transformer = MinMaxScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")


preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ]
)


pipe_lr = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(C=1, penalty='l2', solver='saga', max_iter=10000, class_weight='balanced'))
])


pipe_lr.fit(X_train, y_train)


# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
# print("Mean AUC:", scores.mean(), "+-", scores.std())


y_pred = pipe_lr.predict(X_val)

print("Accuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:\n", classification_report(y_val, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_pred))
print("--> ROC-AUC Score:", roc_auc_score(y_val, y_pred))


# param_grid = {
#     'classifier__max_iter': [1000, 5000, 10000],
#     'classifier__C': [0.1, 1, 10, 100],
#     'classifier__penalty': ['l1', 'l2'],
#     "classifier__solver": ['saga']
# }

# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# grid_search_logreg = GridSearchCV(
#     model,
#     param_grid=param_grid,
#     cv=cv,
#     scoring="roc_auc",
#     n_jobs=-1,
#     verbose=2
# )

# grid_search_logreg.fit(X, y)

# print('-' * 45)
# print("Best parameters:", grid_search_logreg.best_params_)
# print("Best ROC-AUC:", grid_search_logreg.best_score_)
# print('-' * 45)


X_test_processed = add_features(test_df).copy()
test_predictions = pipe_lr.predict(X_test_processed)
test_probabilities = pipe_lr.predict_proba(X_test_processed)

# test_predictions[:10]
test_probabilities[0]


encoder = pipe_lr.named_steps['preprocessor'].named_transformers_['cat']
ohe_features = encoder.get_feature_names_out(cat_cols)

# Features
features = np.concatenate([num_cols, ohe_features])
# Coefficients
coefs = pipe_lr.named_steps['classifier'].coef_.flatten()

feature_importance = pd.DataFrame({
    'Feature': features,
    'Coefficient': coefs,
    'Abs_importance': np.abs(coefs)
}).sort_values('Abs_importance', ascending=False)

print('10 most significant features')

feature_importance.head(20)


from sklearn.ensemble import RandomForestClassifier, StackingClassifier
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score


# LightGBM pipeline
pipe_lgb = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", lgb.LGBMClassifier(
        boosting_type='gbdt',
        objective='binary',
        metric='auc',
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=8,
        lambda_l1=0.5,
        lambda_l2=1.0,
        min_child_samples=50,
        scale_pos_weight=scale_pos_weight,
        is_unbalance=False,
        random_state=42,
        verbose=-1
    ))
])

# XGBoost pipeline
pipe_xgb = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=500, 
        learning_rate=.05, 
        max_depth=6,
        tree_method='hist',
        random_state=42
    ))
])


lgb_model = pipe_lgb.fit(X_train, y_train)
display(lgb_model)

y_pred = lgb_model.predict(X_val)
y_pred_proba = lgb_model.predict_proba(X_val)[:, 1]

print("Accuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:\n", classification_report(y_val, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_pred))
print("--> ROC-AUC Score:", roc_auc_score(y_val, y_pred_proba))


# xgb_model = pipe_xgb.fit(X_train, y_train)
# display(xgb_model)

# y_pred = xgb_model.predict(X_val)
# y_pred_proba = xgb_model.predict_proba(X_val)[:, 1]

# print("Accuracy:", accuracy_score(y_val, y_pred))
# print("\nClassification Report:\n", classification_report(y_val, y_pred))
# print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_pred))
# print("--> ROC-AUC Score:", roc_auc_score(y_val, y_pred_proba))


test_df = add_features(test_df)

test_df.head()


# LightGBM predictions fixation  
test_probs = lgb_model.predict_proba(test_df)[:, 1]
test_preds = lgb_model.predict(test_df)

# Pasting in test df
test_df["loan_paid_back_prob"] = test_probs
test_df["loan_paid_back_pred"] = test_preds

test_df.head(10)


submission = test_df[['id']].copy()
submission['loan_paid_back'] = test_probs

submission.to_csv('submission.csv', index=False)


submission.head()

