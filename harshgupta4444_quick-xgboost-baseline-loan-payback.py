import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')


train.shape


test.shape


train['loan_paid_back'].value_counts(normalize=True)


X = train.drop(['id', 'loan_paid_back'], axis=1)
y = train['loan_paid_back']
X_test = test.drop('id', axis=1)


categorical_cols = ['gender', 'marital_status', 'education_level','employment_status', 'loan_purpose', 'grade_subgrade']


label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    # Fit on combined train and test to handle all categories
    combined = pd.concat([X[col], X_test[col]])
    le.fit(combined)
    X[col] = le.transform(X[col])
    X_test[col] = le.transform(X_test[col])
    label_encoders[col] = le


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


X_train.shape


X_val.shape


model = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='auc',
    n_jobs=-1
)


model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=100,
    verbose=100
)


val_pred = model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, val_pred)
print(f"\nValidation AUC: {val_auc:.4f}")


model_full = xgb.XGBClassifier(
    n_estimators=model.best_iteration,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)


model_full.fit(X, y)


test_pred = model_full.predict_proba(X_test)[:, 1]


submission = pd.DataFrame({
    'id': test['id'],
    'loan_paid_back': test_pred
})


submission.to_csv('submission.csv', index=False)


submission.shape


submission.head()




