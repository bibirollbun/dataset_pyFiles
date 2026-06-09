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


train_dataset=pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test_dataset = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
submit=pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")


import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, GridSearchCV,train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier




train_dataset.head()


train_dataset.info()


target="loan_paid_back"
x=train_dataset.drop(columns=[target])
y=train_dataset[target]


categorical_features = x.select_dtypes(include=['object','category']).columns.to_list()
numerical_features=x.select_dtypes(exclude=['object','category']).columns.to_list()

print("Categorical:", categorical_features)
print("Numerical:", numerical_features)


def features(dataset):
    dataset['interest_to_credit'] = dataset['interest_rate'] / dataset['credit_score']
    dataset['interest_to_credit'] = (
        dataset['interest_to_credit']
        .replace([float('inf'), -float('inf')], np.nan)
        .fillna(dataset['interest_to_credit'].median())
    )

    dataset['loan_to_income'] = dataset['loan_amount'] / dataset['annual_income']
    dataset['loan_to_income'] = (
        dataset['loan_to_income']
        .replace([float('inf'), -float('inf')], np.nan)
        .fillna(dataset['loan_to_income'].median())
    )
    dataset['interest_burden'] = dataset['interest_rate'] * dataset['loan_amount'] / (dataset['annual_income'] + 1)
    dataset['income_x_credit'] = dataset['annual_income'] * dataset['credit_score']
    dataset['loan_x_interest'] = dataset['loan_amount'] * dataset['interest_rate']
    return dataset



x_added_features=features(x)



for col in categorical_features:
    le=LabelEncoder()
    x_added_features[col]=le.fit_transform(x_added_features[col])
scaler=StandardScaler()
x_added_features[numerical_features]=scaler.fit_transform(x_added_features[numerical_features])
   



x_train,x_test,y_train,y_test= train_test_split(x_added_features,y,test_size=0.2,random_state=42)



feature_model = lgb.LGBMClassifier()
feature_model.fit(x_train, y_train.squeeze())
importance = pd.DataFrame({
    'feature': x_train.columns,
    'importance': feature_model.feature_importances_
}).sort_values('importance', ascending=False)

print(importance.head(10))



xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_params = {
    'n_estimators': [300, 600],
    'max_depth': [4, 6],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0]
}


lgb_model = lgb.LGBMClassifier(verbose=0,random_state=42)
lgb_params = {
    'n_estimators': [900, 600],
    'num_leaves': [31, 50],
    'learning_rate':[0.05, 0.1]
}

cat_model = CatBoostClassifier(verbose=0, random_seed=42)
cat_params = {
    'depth': [4, 6],
    'iterations': [200, 300],
    'learning_rate':  [0.05, 0.1]
}

log_reg = LogisticRegression(max_iter=500)



skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

def grid_search_model(model, params, x_added_features, y, preprocess=None):
    if preprocess:
        pipe = Pipeline(steps=[('preprocess', preprocess), ('model', model)])
    else:
        pipe = model
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=params,
        scoring='roc_auc',
        cv=skf,
        n_jobs=-1,
        verbose=1
    )
    grid.fit(x_added_features, y)
    print(f"Best AUC: {grid.best_score_:.4f}")
    print("Best Params:", grid.best_params_)
    return grid.best_estimator_

best_xgb = grid_search_model(xgb_model, xgb_params, x_added_features, y)
best_lgb = grid_search_model(lgb_model, lgb_params, x_added_features, y)  
best_cat = grid_search_model(cat_model, cat_params, x_added_features, y)  


test_transformed=features(test_dataset)
for col in categorical_features:
    le=LabelEncoder()
    test_transformed[col]=le.fit_transform(test_transformed[col])
scaler=StandardScaler()
test_transformed[numerical_features]=scaler.fit_transform(test_transformed[numerical_features])



meta_train = np.zeros((x_added_features.shape[0],3))
meta_test=np.zeros((test_dataset.shape[0],3))
for folds,(train_idx,val_idx) in enumerate(skf.split(x_added_features,y)):
    print(f"fold: {folds+1}")
    x_train,x_val = x_added_features.iloc[train_idx],x_added_features.iloc[val_idx]
    y_train,y_val=y.iloc[train_idx],y.iloc[val_idx]
    best_cat.fit(x_train, y_train)
    best_lgb.fit(x_train, y_train)
    best_xgb.fit(x_train, y_train)
    meta_train[val_idx] = np.column_stack([
        best_xgb.predict_proba(x_val)[:, 1],
        best_lgb.predict_proba(x_val)[:, 1],
        best_cat.predict_proba(x_val)[:, 1]
    ])
    meta_test += np.column_stack([
        best_xgb.predict_proba(test_transformed)[:, 1],
        best_lgb.predict_proba(test_transformed)[:, 1],
        best_cat.predict_proba(test_transformed)[:, 1]
    ]) / skf.n_splits



meta_model = LogisticRegression(max_iter=1000)
meta_model.fit(meta_train, y)
final_preds = meta_model.predict_proba(meta_test)[:, 1]



submission = pd.DataFrame({
    "id": test_dataset["id"],
    "loan_paid_back": (final_preds > 0.5).astype(int)
})
submission.to_csv("submission.csv", index=False)
print("Submission saved as submission.csv")



submission.head()







