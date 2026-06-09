import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

from sklearn.model_selection import train_test_split

import numpy as np

from lightgbm import LGBMClassifier
import lightgbm as lgb

from sklearn.metrics import roc_auc_score


df = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
df.head()


df['diagnosed_diabetes'].value_counts(normalize=True)


X = df.drop(columns=['id', 'diagnosed_diabetes'])
y = df['diagnosed_diabetes']



df.dtypes


cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
num_cols = df.select_dtypes(include=["number"]).columns.tolist()


for col in cat_cols:
    X[col] = X[col].astype("category")


fi_model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    class_weight="balanced",
    random_state=42
)

fi_model.fit(
    X, y,
    categorical_feature=cat_cols
)


fi = pd.DataFrame({
    "feature": X.columns,
    "importance": fi_model.feature_importances_
}).sort_values(by="importance", ascending=False)

fi


features_to_drop = ['alcohol_consumption_per_week','screen_time_hours_per_day',
       'gender', 'ethnicity', 'education_level',
       'income_level','employment_status','hypertension_history',
       'cardiovascular_history']

X = X.drop(columns=features_to_drop)



X.columns


cat_features = ['smoking_status']


for col in cat_features:
    X[col] = X[col].astype("category")


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)



lgb_model = LGBMClassifier(
    n_estimators=700,
    learning_rate=0.07,
    num_leaves=31,
    
    class_weight="balanced",
    random_state=42
)

lgb_model.fit(
    X_train, y_train,
    categorical_feature=cat_features,
    eval_metric="auc"
    
)



y_test_pred = lgb_model.predict_proba(X_test)[:, 1]
roc_auc_score(y_test, y_test_pred)


test_df = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
test_df.head()


X_test_final = test_df.drop(columns = 'id')
X_test_final = X_test_final.drop(columns = features_to_drop)


for col in cat_features:
    X_test_final[col] = X_test_final[col].astype("category")


  y_test_pred_final = lgb_model.predict_proba(X_test_final)[:, 1]


submission_df = pd.DataFrame({'id': test_df['id'], 'diagnosed_diabetes':y_test_pred_final })
submission_df.to_csv('submission.csv', index=False)

