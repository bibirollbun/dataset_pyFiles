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


X_train_full = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
X_test_full = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
X_train_full.head()


X_train_full.isna().sum()


y = X_train_full['y']
X_train_full.drop(['y'], axis = 1, inplace = True)


X_train_full.info()


# # drop useless columns
# # id --> doesn't affect prediction
# # campaign + duration --> current data ; possibly cause data leakage
# # day --> mostlikely irrelevant?
# X_train_full.drop(['id','campaign','duration','day'], axis = 1, inplace = True)



# X_train_full['age_group'] = pd.cut(
#     X_train_full['age'],
#     bins=[18, 39, 59, 100],
#     labels=['adult', 'middle_aged', 'senior']
# )



# print(X_train_full.balance[X_train_full['balance'] < 0].sum())
# print(X_train_full.balance[X_train_full['balance'] >= 0].sum())



# X_train_full.balance.max()


num_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtypes in ['float64','int64']]
cat_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtypes in ['object','category' ]]
print(f'num_cols : \n{num_cols}\n\ncat_cols : \n{cat_cols}')


y.value_counts(normalize=True)


from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder

X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y, test_size = 0.2, random_state = 0)


preprocessor = ColumnTransformer(transformers = [
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown= 'ignore'), cat_cols)
])




pipeline = make_pipeline(
    preprocessor,
    XGBClassifier(scale_pos_weight=9.0,random_state = 0, use_label_encoder = False, eval_metric = 'auc')
)

param_grid = {
    'xgbclassifier__n_estimators' : [100,200,500,100],
    'xgbclassifier__max_depth': [3, 5, 7],
    'xgbclassifier__learning_rate': [0.01, 0.1, 0.3],
    'xgbclassifier__subsample': [0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=3,
    scoring='roc_auc', 
    n_jobs=-1,
    verbose=2
)
grid_search.fit(X_train, y_train)

print("Best params:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

y_pred = grid_search.predict(X_valid)


X_test_pred = grid_search.predict(X_test_full)
submission = pd.DataFrame({'id': X_test_full.index,
                          'y': X_test_pred})
submission.to_csv('./output.csv', index= False)


from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.metrics import roc_auc_score


best_model  = make_pipeline(
    preprocessor,
    XGBClassifier(
        scale_pos_weight=9.0,
        random_state = 0, 
        use_label_encoder = False, 
        eval_metric = 'auc',
        learning_rate = 0.1,
        max_depth = 5,
        n_estimators = 500,
        subsample = 0.8
    )
)

best_model.fit(X_train, y_train)
X_valid_y_pred = best_model.predict(X_valid)
print("ROC-AUC:", roc_auc_score(y_valid, best_model.predict_proba(X_valid)[:, 1]))


X_test_y_pred = best_model.predict(X_test_full)
submission = pd.DataFrame({'id': X_test_full.id,
                          'y': X_test_y_pred})
submission.to_csv('./submission.csv', index= False)


tmp = pd.read_csv('./submission.csv')
tmp




