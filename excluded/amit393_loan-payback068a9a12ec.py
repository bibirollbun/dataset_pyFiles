

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






df_train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
df_submission = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")
print(df_train.head())
print("Shape:", df_train.shape)




df_train.describe()


df_train.duplicated().sum()


for col in df_train.select_dtypes(include=['object', 'category']):
    print(f"{col}: {df_train[col].unique()}")


df_train['loan_to_income'] = df_train['loan_amount'] / df_train['annual_income']
df_test['loan_to_income'] = df_test['loan_amount'] / df_test['annual_income']


df_train['interest_burden'] = df_train['interest_rate'] * df_train['loan_amount'] / df_train['annual_income']
df_test['interest_burden'] = df_test['interest_rate'] * df_test['loan_amount'] / df_test['annual_income']


# Cell 3: Additional feature engineering (KEEP AS IS - but use BEFORE log transform)
df_train['income_credit_interaction'] = df_train['annual_income'] * df_train['credit_score']
df_test['income_credit_interaction'] = df_test['annual_income'] * df_test['credit_score']

df_train['monthly_payment'] = df_train['loan_amount'] * df_train['interest_rate'] / 12
df_test['monthly_payment'] = df_test['loan_amount'] * df_test['interest_rate'] / 12

df_train['payment_to_income'] = df_train['monthly_payment'] / (df_train['annual_income'] / 12)
df_test['payment_to_income'] = df_test['monthly_payment'] / (df_test['annual_income'] / 12)


# Cell 4: Credit buckets (KEEP AS IS)
df_train['credit_bucket'] = pd.cut(
    df_train['credit_score'],
    bins=[0, 580, 670, 740, 800, 900],
    labels=['Poor','Fair','Good','VeryGood','Excellent']
)

df_test['credit_bucket'] = pd.cut(
    df_test['credit_score'],
    bins=[0, 580, 670, 740, 800, 900],
    labels=['Poor','Fair','Good','VeryGood','Excellent']
)


for col in df_train.select_dtypes(include='number'):
    print(f"{col}  skew: {df_train[col].skew()}")


cols = ['annual_income', 'debt_to_income_ratio', 'loan_to_income', 'interest_burden']

for col in cols:
    df_train[col] = np.log1p(df_train[col])
    df_test[col] = np.log1p(df_test[col])


for col in df_train.select_dtypes(include='number'):
    print(f"{col}  skew: {df_train[col].skew()}")


import matplotlib.pyplot as plt
import seaborn as sns
corr = df_train.select_dtypes(include='number').corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, fmt=".2f")
plt.show()


for col in df_train.select_dtypes(include=['object', 'category']):
    plt.figure(figsize=(10,6))
    sns.countplot(x=col, data=df_train)
    plt.title(f"{col} Distribution")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.show()


for col in df_train.select_dtypes(include=['object', 'category']).columns:
    df_train[col] = df_train[col].astype('category')
    df_test[col] = df_test[col].astype('category')
df_train.drop('id',axis=1)
df_test.drop("id",axis=1)


from sklearn.model_selection import train_test_split
X = df_train.drop('loan_paid_back', axis=1)
y = df_train['loan_paid_back']

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


cat_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
num_features = X_train.select_dtypes(include=['int', 'float']).columns.tolist()

# Convert categorical columns to category dtype
for col in cat_features:
    X_train[col] = X_train[col].astype("category")
    X_valid[col] = X_valid[col].astype("category")


from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

# Properly drop 'id' column
X_train = X_train.drop('id', axis=1, errors='ignore')
X_valid = X_valid.drop('id', axis=1, errors='ignore')
df_test_features = df_test.drop('id', axis=1, errors='ignore')

# Identify features
cat_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
num_features = X_train.select_dtypes(include=['int', 'float']).columns.tolist()

# Use RobustScaler instead of StandardScaler (better for outliers)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', RobustScaler(), num_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_features)
    ]
)

# STRONGER base models with optimized hyperparameters
estimators = [
    ('lgb', LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=50,
        max_depth=12,
        min_child_samples=15,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=0.2,
        random_state=42,
        verbose=-1,
        class_weight='balanced'
    )),
    ('xgb', XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=8,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        gamma=0.1,
        reg_alpha=0.2,
        reg_lambda=0.2,
        random_state=42,
        eval_metric='logloss',
        scale_pos_weight=1
    )),
    ('catboost', CatBoostClassifier(
        iterations=300,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=False,
        auto_class_weights='Balanced'
    )),
    ('rf', RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=8,
        min_samples_leaf=4,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    ))
]

# MORE POWERFUL final estimator
final_estimator = CatBoostClassifier(
    iterations=500,
    learning_rate=0.02,
    depth=10,
    l2_leaf_reg=5,
    random_seed=42,
    verbose=False,
    auto_class_weights='Balanced'
)

# Stacking with MORE cross-validation folds
stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=final_estimator,
    cv=10,  # Increased from 5 to 10
    passthrough=True,
    n_jobs=-1
)

# Pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('stacking', stacking)
])

# Fit
print("Training optimized model (this will take longer)...")
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_valid_pred = model.predict(X_valid)
y_valid_proba = model.predict_proba(X_valid)[:, 1]

# Evaluate
train_acc = accuracy_score(y_train, y_train_pred)
valid_acc = accuracy_score(y_valid, y_valid_pred)
roc_auc = roc_auc_score(y_valid, y_valid_proba)

print(f"\n{'='*50}")
print(f"Training accuracy: {train_acc:.4f}")
print(f"Validation accuracy: {valid_acc:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")
print(f"{'='*50}")
print(f"\nClassification Report:\n{classification_report(y_valid, y_valid_pred)}")

# Test predictions
y_test_pred = model.predict(df_test_features)



y_test_proba = model.predict_proba(df_test_features)[:, 1]

# Create submission dataframe
submission = pd.DataFrame({
    'id': df_test['id'],
    'loan_paid_back': y_test_proba  
})

# Save to CSV
submission.to_csv('submission.csv', index=False)


