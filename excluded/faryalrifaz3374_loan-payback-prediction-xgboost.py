# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import os
print('ready')


# paths, adjust if needed
TRAIN_PATH = '/kaggle/input/playground-series-s5e11/train.csv'
TEST_PATH = '/kaggle/input/playground-series-s5e11/test.csv'
SAMPLE_SUB_PATH = '/kaggle/input/playground-series-s5e11/sample_submission.csv'

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
print('train shape', train.shape, 'test shape', test.shape)


# quick peek
display(train.head(10))
display(train.info())
display(train.describe(include='all'))


# missing values and basic checks
print('missing in train\n', train.isnull().sum())
print('\nmissing in test\n', test.isnull().sum())
print('\nunique counts for categoricals:')
for c in train.select_dtypes(include=['object']).columns:
    print(c, train[c].nunique())


# target checks
target = 'loan_paid_back'
print('target unique values and counts:\n', train[target].value_counts(dropna=False))


# target distribution plot
vals = train[target].value_counts().sort_index()
plt.figure(figsize=(6,4))
plt.bar(vals.index.astype(str), vals.values)
plt.title('target distribution')
plt.xlabel('loan_paid_back')
plt.ylabel('count')
plt.show()


# numeric histograms
num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c not in ['id', target]]
for c in num_cols:
    plt.figure(figsize=(6,3))
    plt.hist(train[c].dropna(), bins=40)
    plt.title(c)
    plt.xlabel(c)
    plt.ylabel('count')
    plt.show()


# categorical counts 
cat_cols = train.select_dtypes(include=['object']).columns.tolist()
for c in cat_cols:
    plt.figure(figsize=(6,3))
    vc = train[c].value_counts().head(20)
    plt.bar(vc.index.astype(str), vc.values)
    plt.title(c)
    plt.xticks(rotation=45, ha='right')
    plt.show()


# correlation matrix for numeric features
nums = train.select_dtypes(include=[np.number]).columns.tolist()
nums = [c for c in nums if c not in ['id']]
corr = train[nums].corr()
plt.figure(figsize=(8,6))
plt.imshow(corr, interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(nums)), nums, rotation=45, ha='right')
plt.yticks(range(len(nums)), nums)
plt.title('numeric feature correlations')
plt.show()


# ratio-based features
train['loan_to_income'] = train['loan_amount'] / (train['annual_income'] + 1e-9)
test['loan_to_income'] = test['loan_amount'] / (test['annual_income'] + 1e-9)

train['income_per_interest'] = train['annual_income'] / (train['interest_rate'] + 1e-9)
test['income_per_interest'] = test['annual_income'] / (test['interest_rate'] + 1e-9)

# credit score binning
train['credit_score_bin'] = pd.qcut(train['credit_score'], 10, duplicates='drop').astype(str)
test['credit_score_bin'] = pd.qcut(test['credit_score'], 10, duplicates='drop').astype(str)


# frequency encoding for categorical features
def freq_encode(df, col):
    vc = df[col].value_counts(normalize=True)
    return df[col].map(vc).fillna(0)

enc_cols = [
    'gender', 'marital_status', 'education_level', 
    'employment_status', 'loan_purpose', 
    'grade_subgrade', 'credit_score_bin'
]

for c in enc_cols:
    if c in train.columns:
        train[c + '_freq'] = freq_encode(train, c)
        mapping = train[c].value_counts(normalize=True)
        if c in test.columns:
            test[c + '_freq'] = test[c].map(mapping).fillna(0)


# prepare final datasets
drop_cols = ['id', 'loan_paid_back']
features = [c for c in train.columns if c not in drop_cols]
test_features = [c for c in features if c in test.columns]

X = train[features].copy()
y = train['loan_paid_back'].copy()
X_test = test[test_features].copy()


# handle missing values and categorical encoding
for c in X.columns:
    if X[c].dtype == 'object':
        X[c] = X[c].astype('category').cat.codes
        if c in X_test.columns:
            X_test[c] = X_test[c].astype('category').cat.codes
    else:
        med = X[c].median()
        X[c].fillna(med, inplace=True)
        if c in X_test.columns:
            X_test[c].fillna(med, inplace=True)

print('Feature Engineering Complete')
print('Train shape:', X.shape)
print('Test shape:', X_test.shape)
print('Any nulls in X:', X.isnull().any().any())
print('Any nulls in X_test:', X_test.isnull().any().any())
print('Sample features:', features[:30])



# train validation split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(X_train.shape, X_valid.shape)


# XGBoost Hyperparameter Tuning with RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
import numpy as np

# base model
xgb_base = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False,
    random_state=42
)

# parameter grid for tuning
param_dist = {
    'n_estimators': [300, 500, 800, 1000],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'max_depth': [4, 6, 8, 10],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2, 0.3],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.01, 0.1, 1],
    'reg_lambda': [0.5, 1, 2]
}

# randomized search setup
random_search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_dist,
    scoring='roc_auc',
    n_iter=25,                 # increase if you want more exploration
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# fit search
random_search.fit(X_train, y_train)

# best parameters and score
print("Best Parameters:", random_search.best_params_)
print("Best CV AUC:", random_search.best_score_)

# train final model with best params
best_model = random_search.best_estimator_



# validation predictions and metrics
y_val_proba = best_model.predict_proba(X_valid)[:,1]
y_val_pred = (y_val_proba >= 0.5).astype(int)
print('roc_auc', roc_auc_score(y_valid, y_val_proba))
print('accuracy', accuracy_score(y_valid, y_val_pred))
print(classification_report(y_valid, y_val_pred))

fpr, tpr, _ = roc_curve(y_valid, y_val_proba)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr)
plt.plot([0,1],[0,1], linestyle='--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()


# feature importance plot 
fi = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=True)
plt.figure(figsize=(6,8))
fi.tail(25).plot(kind='barh')
plt.title('feature importance (top 25)')
plt.show()


# Generate predictions on test set
X_test = X_test.reindex(columns=X.columns, fill_value=0)
sub_probs = best_model.predict_proba(X_test)[:, 1]

# Load sample submission and create new submission dataframe
sample_sub = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')
submission = sample_sub.copy()
submission['loan_paid_back'] = sub_probs

# Save submission file 
submission.to_csv('submission_xgb.csv', index=False)

# Display message
print("File 'submission_xgb.csv' saved successfully!")

# Show first few rows
submission.head()


