import numpy as np
import pandas as pd 
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv', index_col='id')


train.head()


train.info()


train.describe()


test.info()


categorical_cols = test.select_dtypes(include=['object']).columns
numerical_cols = test.select_dtypes(include=['int64', 'float64']).columns

print(f"The categorical value columns are: {categorical_cols.values}")
print(f"The numerical value columns are: {numerical_cols.values}")


encoder = OrdinalEncoder()
train[categorical_cols] = encoder.fit_transform(train[categorical_cols])
test[categorical_cols] = encoder.transform(test[categorical_cols])


X = train.drop('diagnosed_diabetes', axis=1)
y = train['diagnosed_diabetes']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


xgb = XGBClassifier(
    use_label_encoder=False,
    eval_metric='auc',
    verbosity=0,
    random_state=42
)

xgb_params = {
    'n_estimators': randint(300, 1500),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.005, 0.2),
    'subsample': uniform(0.6, 0.9),
    'colsample_bytree': uniform(0.6, 0.4),
    'gamma': uniform(0, 0.5)
}

xgb_search = RandomizedSearchCV(
    xgb, xgb_params, n_iter=50, scoring='roc_auc', cv=3,
    n_jobs=-1, random_state=42, verbose=1
)

xgb_search.fit(X_train, y_train)
best_xgb = xgb_search.best_estimator_


lgbm = LGBMClassifier(
    verbose=-1,
    allow_writing_files=False,
    random_state=42
)

lgbm_params = {
    'n_estimators': randint(300, 1500),
    'num_leaves': randint(5, 150),
    'max_depth': randint(-1, 15),
    'learning_rate': uniform(0.005, 0.2),
    'subsample': uniform(0.6, 0.9),
    'colsample_bytree': uniform(0.6, 0.4)
}

lgbm_search = RandomizedSearchCV(
    lgbm, lgbm_params, n_iter=50, scoring='roc_auc', cv=3,
    n_jobs=-1, random_state=42, verbose=1
)

lgbm_search.fit(X_train, y_train)
best_lgbm = lgbm_search.best_estimator_


model = StackingClassifier(
    estimators=[
        ('xgb', best_xgb),
        ('lgbm', best_lgbm)
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


sub = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')
test_preds = model.predict_proba(test)
test_preds_proba = test_preds[:, 1]
submission = pd.DataFrame({
    'id': sub['id'],
    'diagnosed_diabetes': test_preds_proba
})

submission.to_csv('submission.csv', index=False)

