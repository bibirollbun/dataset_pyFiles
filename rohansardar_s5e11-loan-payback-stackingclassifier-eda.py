import numpy as np
import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42


train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv", index_col="id")
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv", index_col="id")


train.head()


train.isna().sum()


test.isna().sum()


categorical_cols = test.select_dtypes(include=['object']).columns
numerical_cols = test.select_dtypes(include=['int64', 'float64']).columns

print(f"The categorical value columns are: {categorical_cols.values}")
print(f"The numerical value columns are: {numerical_cols.values}")


sns.set_style('whitegrid')
sns.countplot(data=train, x='loan_paid_back', palette='Set2')
plt.title('Distribution of loan_paid_back')
plt.xlabel('loan_paid_back')
plt.ylabel('Count')
plt.show()


for col in categorical_cols:
    plt.figure(figsize=(7, 4))
    sns.countplot(x=col, hue='loan_paid_back', data=train, palette='Set2')
    plt.title(f"{col} vs loan_paid_back count")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


for col in numerical_cols:
    plt.figure(figsize=(4, 6))
    sns.boxplot(x='loan_paid_back', y=col, data=train, palette='Set2')
    plt.title(f"Distribution of {col} by loan_paid_back")
    plt.tight_layout()
    plt.show()


plt.figure(figsize=(7, 6))
corr = test[numerical_cols].corr()

sns.heatmap(corr, annot=True, cmap='viridis_r', fmt=".2f")
plt.title("Correlation of Numeric columns", fontsize=16)
plt.tight_layout()
plt.show()


def feature_engineering(df):

    df = df.copy()
    
    df['loan_to_income_ratio'] = df['loan_amount'] / (df['annual_income'] + 1)
    df['monthly_income'] = df['annual_income'] / 12
    df['monthly_payment_estimate'] = (df['loan_amount'] * df['interest_rate']) / 1200
    df['payment_to_income_ratio'] = df['monthly_payment_estimate'] / (df['monthly_income'] + 1)
    df['current_debt_amount'] = df['debt_to_income_ratio'] * df['annual_income']
    df['total_debt_with_loan'] = df['current_debt_amount'] + df['loan_amount']
    df['new_debt_to_income'] = df['total_debt_with_loan'] / (df['annual_income'] + 1)
    df['debt_increase_ratio'] = df['new_debt_to_income'] / (df['debt_to_income_ratio'] + 0.01)
    df['disposable_income'] = df['annual_income'] - df['current_debt_amount']
    df['disposable_income_ratio'] = df['disposable_income'] / (df['annual_income'] + 1)
    df['loan_to_disposable_income'] = df['loan_amount'] / (df['disposable_income'] + 1)
    df['monthly_disposable_income'] = df['disposable_income'] / 12
    df['payment_to_disposable_ratio'] = df['monthly_payment_estimate'] / (df['monthly_disposable_income'] + 1)
    df['annual_payment_burden'] = df['monthly_payment_estimate'] * 12
    df['payment_burden_ratio'] = df['annual_payment_burden'] / (df['annual_income'] + 1)
    
    df['debt_coverage_ratio'] = df['monthly_income'] / (df['current_debt_amount'] / 12 + 1)
    df['interest_to_income_ratio'] = (df['loan_amount'] * df['interest_rate'] / 100) / (df['annual_income'] + 1)
    return df

train_fe = feature_engineering(train)
test_fe = feature_engineering(test)


encoder = OrdinalEncoder()
train_fe[categorical_cols] = encoder.fit_transform(train[categorical_cols])
test_fe[categorical_cols] = encoder.transform(test[categorical_cols])


X = train_fe.drop('loan_paid_back', axis=1)
y = train_fe['loan_paid_back']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22, stratify=y)


rf = RandomForestClassifier(random_state=RANDOM_STATE)

rf_params = {
    'n_estimators': randint(100, 600),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 5),
    'max_features': ['sqrt', 'log2', None]
}

rf_search = RandomizedSearchCV(
    rf, rf_params, n_iter=10, scoring='roc_auc', cv=3,
    n_jobs=-1, random_state=RANDOM_STATE, verbose=1
)

rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_


xgb = XGBClassifier(
    use_label_encoder=False,
    eval_metric='auc',
    verbosity=0,
    random_state=RANDOM_STATE
)

xgb_params = {
    'n_estimators': randint(200, 800),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.2),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'gamma': uniform(0, 0.5)
}

xgb_search = RandomizedSearchCV(
    xgb, xgb_params, n_iter=40, scoring='roc_auc', cv=3,
    n_jobs=-1, random_state=RANDOM_STATE, verbose=1
)

xgb_search.fit(X_train, y_train)
best_xgb = xgb_search.best_estimator_


lgbm = LGBMClassifier(
    verbose=-1,
    allow_writing_files=False,
    random_state=RANDOM_STATE
)

lgbm_params = {
    'n_estimators': randint(200, 800),
    'num_leaves': randint(20, 150),
    'max_depth': randint(-1, 15),
    'learning_rate': uniform(0.01, 0.2),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4)
}

lgbm_search = RandomizedSearchCV(
    lgbm, lgbm_params, n_iter=40, scoring='roc_auc', cv=3,
    n_jobs=-1, random_state=RANDOM_STATE, verbose=1
)

lgbm_search.fit(X_train, y_train)
best_lgbm = lgbm_search.best_estimator_


catb = CatBoostClassifier(verbose=0, random_state=RANDOM_STATE)

catb_params = {
    'depth': randint(4, 10),
    'learning_rate': uniform(0.01, 0.2),
    'iterations': randint(300, 800),
    'l2_leaf_reg': uniform(1, 10)
}

catb_search = RandomizedSearchCV(
    catb, catb_params, n_iter=10, scoring='roc_auc',cv=3, 
    n_jobs=-1, random_state=RANDOM_STATE, verbose=1
)

catb_search.fit(X_train, y_train)
best_catb = catb_search.best_estimator_


model = StackingClassifier(
    estimators=[
        ('xgb', best_xgb),
        ('lgbm', best_lgbm),
        ('catb', best_catb),
        ('rf', best_rf)
    ],
    final_estimator=LogisticRegression(),
    n_jobs=-1,
    stack_method='predict_proba'
)
model.fit(X_train,  y_train)


y_pred = model.predict(X_test)
print(f"VotingClassifier Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred)}")
print(f"Classification Report: \n{classification_report(y_test, y_pred)}")


sub = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')
test_preds = model.predict_proba(test_fe)
test_preds_proba = test_preds[:, 1]
submission = pd.DataFrame({
    'id': sub['id'],
    'loan_paid_back': test_preds_proba
})

submission.to_csv('submission.csv', index=False)

