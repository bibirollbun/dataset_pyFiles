import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn import metrics


train = pd.read_csv('/kaggle/input/multiclassificationtask/train.csv', index_col=0)
test = pd.read_csv('/kaggle/input/multiclassificationtask/test.csv', index_col=0)
sub = pd.read_csv('/kaggle/input/multiclassificationtask/sample_submission.csv', index_col=0)


X = train.drop('Status', axis=1)
y = train['Status']


le = LabelEncoder()
y_encoded = le.fit_transform(y)


class ForwardFillImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.fillna(method='ffill')


X['Age'] = X['Age'] // 365
test['Age'] = test['Age'] // 365


ffill_cols = ['Ascites', 'Edema']
most_freq_cols = ['Drug', 'Sex', 'Hepatomegaly', 'Spiders']
num_cols = X.select_dtypes(include=['number']).columns.tolist()




num_cols = [col for col in num_cols if col not in ffill_cols + most_freq_cols]


ffill_pipeline = Pipeline([
    ('ffill', ForwardFillImputer()),
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
])

most_freq_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
])

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Column transformer
preprocessor = ColumnTransformer([
    ('ffill_cat', ffill_pipeline, ffill_cols),
    ('freq_cat', most_freq_pipeline, most_freq_cols),
    ('num', num_pipeline, num_cols)
])



X_prepared = preprocessor.fit_transform(X)
test_prepared = preprocessor.transform(test)


X_train, X_test, y_train, y_test = train_test_split(X_prepared, y_encoded, test_size=0.2, random_state=42)


xgb_model = XGBClassifier(
    use_label_encoder=False,
    learning_rate=0.1,
    max_depth=5,
    n_estimators=180,
    eval_metric='mlogloss',
    objective='multi:softprob',
    random_state=42,
    subsample=0.7,
    colsample_bytree=0.2,
)

xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
print(metrics.classification_report(y_test, y_pred))
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

y_pred_proba = xgb_model.predict_proba(X_test)
print("Log Loss:", metrics.log_loss(y_test, y_pred_proba, labels=[0, 1, 2]))


y_proba = xgb_model.predict_proba(test_prepared)
y_proba_clip = np.clip(y_proba, 1e-15, 1 - 1e-15)
sub[['Status_C', 'Status_CL', 'Status_D']] = y_proba_clip
sub.to_csv('sub.csv')




