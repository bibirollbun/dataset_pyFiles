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


import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer # <--- CẢI TIẾN: Import SimpleImputer
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
import os

warnings.filterwarnings("ignore", category=UserWarning, module='xgboost.core')
warnings.filterwarnings("ignore", message="1 warning generated.")


train_df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")

df_orig = (
    pd.read_csv('/kaggle/input/extrovert-vs-introvert-behavior-data-backup/personality_datasert.csv')
    .rename(columns={'Personality': 'match_p'})
    .drop_duplicates(['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
                      'Going_outside', 'Drained_after_socializing', 'Friends_circle_size',
                      'Post_frequency'])
)

train_df = train_df.merge(df_orig, how='left')
test_df = test_df.merge(df_orig, how='left')


TARGET_COLUMN = 'Personality'
mapping = {'Introvert': 0, 'Extrovert': 1}
y = train_df[TARGET_COLUMN].map(mapping)

X = train_df.drop(["id", TARGET_COLUMN], axis=1)
X_test = test_df.drop("id", axis=1)

X['match_p_is_null'] = X['match_p'].isna().astype(int)
X_test['match_p_is_null'] = X_test['match_p'].isna().astype(int)

X_test = X_test[X.columns]

categorical_features = X.select_dtypes(include=['object', 'category']).columns
numerical_features = X.select_dtypes(include=np.number).columns 

X[categorical_features] = X[categorical_features].fillna('Unknown')
X_test[categorical_features] = X_test[categorical_features].fillna('Unknown')

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')) # Điền giá trị thiếu bằng trung bình
])

categorical_transformer = Pipeline(steps=[
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)


xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, device='cuda')
lgbm_clf = LGBMClassifier(random_state=42, device='gpu')

param_grid_xgb = {
    'classifier__n_estimators': [200, 300, 400],
    'classifier__learning_rate': [0.01, 0.05, 0.1],
    'classifier__max_depth': [5, 7, 10],
    'classifier__subsample': [0.7, 0.8],
    'classifier__colsample_bytree': [0.7, 0.8],
}
param_grid_lgbm = {
    'classifier__n_estimators': [200, 300, 400],
    'classifier__learning_rate': [0.01, 0.05, 0.1],
    'classifier__num_leaves': [20, 31, 40],
    'classifier__subsample': [0.7, 0.8],
    'classifier__colsample_bytree': [0.7, 0.8],
}

xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', xgb_clf)
])
lgbm_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', lgbm_clf)
])


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("Optimizing XGBoost...")
grid_search_xgb = GridSearchCV(xgb_pipeline, param_grid_xgb, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search_xgb.fit(X, y)

print("\nOptimizing LightGBM...")
grid_search_lgbm = GridSearchCV(lgbm_pipeline, param_grid_lgbm, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search_lgbm.fit(X, y)

print("\nBest XGBoost params:", grid_search_xgb.best_params_)
print("Best XGBoost score:", grid_search_xgb.best_score_)
print("\nBest LightGBM params:", grid_search_lgbm.best_params_)
print("Best LightGBM score:", grid_search_lgbm.best_score_)


best_xgb = grid_search_xgb.best_estimator_
best_lgbm = grid_search_lgbm.best_estimator_
xgb_weight = grid_search_xgb.best_score_
lgbm_weight = grid_search_lgbm.best_score_

ensemble = VotingClassifier(
    estimators=[
        ('xgb', best_xgb),
        ('lgbm', best_lgbm)
    ],
    voting='soft', 
    weights=[xgb_weight, lgbm_weight],  
    n_jobs=-1
)

print("\nTraining ensemble model...")
ensemble.fit(X, y)


predictions = ensemble.predict(X_test)
reverse_mapping = {1: 'Extrovert', 0: 'Introvert'}
final_predictions_text = pd.Series(predictions).map(reverse_mapping)

submission_df = pd.DataFrame({'id': test_df['id'], 'Personality': final_predictions_text})
submission_df.to_csv('submission.csv', index=False)

print("\nSubmission file 'submission.csv' created successfully.")
print("Submission file head:")
print(submission_df.head())

