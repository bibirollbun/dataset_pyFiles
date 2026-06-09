import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split

from lightgbm import LGBMClassifier
import xgboost as xgb
import catboost as cb

from sklearn.metrics import roc_curve, auc, roc_auc_score


df_train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')


print("The train set's shape is",df_train.shape[0], "rows and", df_train.shape[1], "columns.")
print("The test set's shape is",df_test.shape[0], "rows and", df_test.shape[1], "columns.")
df_train.head()


df_train.isnull().sum()


msno.matrix(df_train)


df_test.isnull().sum()


msno.matrix(df_test)


print(f"There are {df_train.duplicated().sum()} duplicates in train set.")


df_train['is_duplicate'] = df_train.duplicated(keep=False).astype(int)
sns.countplot(x='is_duplicate', data = df_train, palette=['lightblue','salmon'])
plt.xticks([0,1], ['Unique', 'Duplicate'])
plt.title('Number of duplicates vs unique rows in train set')
plt.show()


print(f"There are {df_test.duplicated().sum()} duplicates in test set.")


df_test['is_duplicate'] = df_test.duplicated(keep=False).astype(int)
sns.countplot(x='is_duplicate', data = df_test, palette=['lightblue','salmon'])
plt.xticks([0,1], ['Unique', 'Duplicate'])
plt.title('Number of duplicates vs unique rows in test set')
plt.show()


df_train = df_train.drop(columns=['is_duplicate'])
df_test = df_test.drop(columns=['is_duplicate'])


df_train['loan_paid_back'] = df_train['loan_paid_back'].astype(int)


ax = sns.countplot(data=df_train, x="loan_paid_back", palette=['red', 'lightgreen'])

total = len(df_train)
for p in ax.patches:
    percentage = 100 * p.get_height() / total
    ax.annotate(f'{percentage:.1f}%', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=10)

ax.set_ylabel('Count')
plt.title('Distribution of Loan Paid Back - with Percent Labels')
plt.xlabel('Loan Paid Back (No = 0, Yes = 1)')
plt.tight_layout()
plt.show()


# Define numerical features
numerical_features = ["annual_income", "debt_to_income_ratio", "credit_score", "loan_amount", "interest_rate"]


df_train[numerical_features].describe()


for column in numerical_features:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    sns.histplot(data=df_train, x=column, ax=axes[0])
    mean_value = df_train[column].mean()
    median_value = df_train[column].median()
    axes[0].axvline(mean_value, color='orange', linestyle='--', linewidth=2, label=f'Mean: {mean_value:.2f}')
    axes[0].axvline(median_value, color='yellow', linestyle='-.', linewidth=2, label=f'Median: {median_value:.2f}')
    axes[0].set_title(f'Histogram of {column}')
    axes[0].legend()

    sns.boxplot(data=df_train, x=column, ax=axes[1])
    axes[1].set_title(f'Boxplot of {column}')

    plt.tight_layout()
plt.show()


# Define categorical features
categorical_features = ["gender", "marital_status", "education_level", "employment_status", "loan_purpose", "grade_subgrade"]


for column in categorical_features:
    order = df_train[column].value_counts().index
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df_train, x=column, order=order)
    plt.title(f'Distribution of {column}')
plt.show()


for col in numerical_features:
    Q1 = df_train[col].quantile(0.25)
    Q3 = df_train[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    print(f"{col}: Lower bound = {lower_bound:.2f}, Upper bound = {upper_bound:.2f}")
    outliers = df_train[(df_train[col] < lower_bound) | (df_train[col] > upper_bound)]
    print(f"Number of outliers in {col}: {outliers.shape[0]}\n")


num_features = len(numerical_features)
fig, axes = plt.subplots(num_features, 1, figsize=(12, 4*num_features))

for ax, column in zip(axes, numerical_features):
    sns.kdeplot(df_train[column], ax=ax, label='Train', fill=True, alpha=0.5)
    sns.kdeplot(df_test[column], ax=ax, label='Test', fill=True, alpha=0.3)
    ax.set_title(f'Distribution comparison (KDE) of {column}')
    ax.legend()
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
sns.heatmap(df_train.corr(numeric_only=True), annot=True,fmt='.2f', cmap='coolwarm')
plt.title("Feature Linear Correlation Matrix")
plt.show()


for column in categorical_features:
    order = df_train[column].value_counts().index
    sns.catplot(
        data=df_train,
        x=column,
        hue='loan_paid_back',
        kind='count',
        order=order,
        palette=['red','lightgreen'],
        height=5,
        aspect=2
    )
    plt.title(f'Distribution of {column} by Loan Status')
    plt.xticks(rotation=45)
    plt.show()


for df in [df_train, df_test]:
    df['credit_rank'] = df['grade_subgrade'].str[0].map({'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7}) + df['grade_subgrade'].str[1].astype(int)/10
    df['loan_to_income'] = df['loan_amount'] / (df['annual_income'] + 1)
    df['interest_burden'] = df['loan_amount'] * df['interest_rate'] / (df['annual_income'] + 1)


X = df_train.drop(columns=['id', 'loan_paid_back'])
y = df_train['loan_paid_back']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)


X_train_lgb = X_train.copy()
X_val_lgb = X_val.copy()
X_test_lgb = df_test.drop(columns=['id']).copy()


cat_cols = ['gender','marital_status','education_level','employment_status','loan_purpose']

for col in cat_cols:
    X_train_lgb[col] = X_train_lgb[col].astype('category')
    X_val_lgb[col] = X_val_lgb[col].astype('category')
    X_test_lgb[col] = X_test_lgb[col].astype('category')


lgb_features = ['annual_income','debt_to_income_ratio','credit_score','loan_amount','interest_rate',
                'gender','marital_status','education_level','employment_status','loan_purpose',
                'credit_rank','loan_to_income','interest_burden']

X_train_lgb = X_train_lgb[lgb_features]
X_val_lgb = X_val_lgb[lgb_features]
X_test_lgb = X_test_lgb[lgb_features]


lgb_model = LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=90,
                           colsample_bytree=0.8, subsample=0.8, reg_alpha=0.1, reg_lambda=0.1,
                           random_state=42, n_jobs=-1, verbose=-1)


lgb_model.fit(X_train_lgb, y_train, eval_set=[(X_val_lgb, y_val)])


lgb_val_pred = lgb_model.predict_proba(X_val_lgb)[:, 1]


fpr, tpr, thresholds = roc_curve(y_val, lgb_val_pred)

auc_score = auc(fpr, tpr)
print("Validation AUC score is:", auc_score)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.4f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random guess')
plt.plot([0, 0, 1], [0, 1, 1], color='lightgreen', linestyle='--', linewidth=2, label='Perfect model')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for LGBM model")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


X_train_xgb = X_train.copy()
X_val_xgb = X_val.copy()
X_test_xgb = df_test.drop(columns=['id']).copy()


le = LabelEncoder()
le_dict = {}
for col in cat_cols:
    combined = pd.concat([X_train_xgb[col], X_val_xgb[col]], axis=0)
    le.fit(combined.astype(str))
    X_train_xgb[f'{col}_le'] = le.transform(X_train_xgb[col].astype(str))
    X_val_xgb[f'{col}_le'] = le.transform(X_val_xgb[col].astype(str))
    X_test_xgb[f'{col}_le'] = le.transform(X_test_xgb[col].astype(str))
    le_dict[col] = le


xgb_features = ['annual_income','debt_to_income_ratio','credit_score','loan_amount','interest_rate',
                'gender_le','marital_status_le','education_level_le','employment_status_le','loan_purpose_le',
                'credit_rank','loan_to_income','interest_burden']


X_train_xgb = X_train_xgb[xgb_features]
X_val_xgb = X_val_xgb[xgb_features]
X_test_xgb = X_test_xgb[xgb_features]


xgb_model = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.03, max_depth=7,
                              min_child_weight=10, subsample=0.85, colsample_bytree=0.7,
                              reg_alpha=0.1, reg_lambda=1.0, n_jobs=-1,
                              random_state=42, tree_method='hist', verbosity=0)


xgb_model.fit(X_train_xgb, y_train, eval_set=[(X_val_xgb, y_val)], verbose=False)


xgb_val_pred = xgb_model.predict_proba(X_val_xgb)[:, 1]


fpr, tpr, thresholds = roc_curve(y_val, xgb_val_pred)

auc_score = auc(fpr, tpr)
print("Validation AUC score is:", auc_score)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.4f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random guess')
plt.plot([0, 0, 1], [0, 1, 1], color='lightgreen', linestyle='--', linewidth=2, label='Perfect model')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for XGBoost model")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


cat_model = cb.CatBoostClassifier(
    iterations=2000,
    learning_rate=0.05,
    depth=8,
    cat_features=cat_cols,
    random_seed=42,
    verbose=0
)


cat_model.fit(X_train_lgb, y_train, eval_set=(X_val_lgb, y_val), verbose=False)


cat_val_pred = cat_model.predict_proba(X_val_lgb)[:, 1]


fpr, tpr, thresholds = roc_curve(y_val, cat_val_pred)

auc_score = auc(fpr, tpr)
print("Validation AUC score is:", auc_score)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.4f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random guess')
plt.plot([0, 0, 1], [0, 1, 1], color='lightgreen', linestyle='--', linewidth=2, label='Perfect model')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for XGBoost model")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


best_auc = 0
best_w1, best_w2 = 0, 0

for w1 in np.arange(0.0, 1.01, 0.05):
    for w2 in np.arange(0.0, 1.01 - w1, 0.05):
        blend = w1 * xgb_val_pred + w2 * lgb_val_pred + (1 - w1 - w2) * cat_val_pred
        auc_i = roc_auc_score(y_val, blend)
        if auc_i > best_auc:
            best_auc = auc_i
            best_w1, best_w2 = w1, w2


print(f"Best: XGB {best_w1:.2f} + LGBM {best_w2:.2f} + Cat {1-best_w1-best_w2:.2f} â†’ {best_auc:.6f}")


lgb_test_pred = lgb_model.predict_proba(X_test_lgb)[:, 1]
xgb_test_pred = xgb_model.predict_proba(X_test_xgb)[:, 1]
cat_test_pred = cat_model.predict_proba(X_test_lgb)[:, 1]


final_pred = best_w1 * xgb_test_pred + best_w2 * lgb_test_pred + (1 - best_w1 - best_w2) * cat_test_pred


submission = pd.DataFrame({
    'id': df_test.id,
    'loan_paid_back': final_pred
})
submission.to_csv('submission.csv', index=False)

