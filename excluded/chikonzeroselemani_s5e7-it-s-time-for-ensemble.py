# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import warnings 
warnings.filterwarnings("ignore")


def preprocess(df):
    df = df.copy()
    df['Stage_fear'] = df['Stage_fear'].fillna('Unknown').map({'Yes': 1, 'No': 0, 'Unknown': -1})
    df['Drained_after_socializing'] = df['Drained_after_socializing'].fillna('Unknown').map({'Yes': 1, 'No': 0, 'Unknown': -1})
    num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
               'Friends_circle_size', 'Post_frequency']
    df[num_cols] = IterativeImputer(random_state=42).fit_transform(df[num_cols])
    return df

train, test = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv'), pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
X, y = preprocess(train.drop(['id', 'Personality'], axis=1)), train['Personality'].map({'Introvert': 0, 'Extrovert': 1})
X_test = preprocess(test.drop('id', axis=1))



# Model definitions
models = {
    'xgb': XGBClassifier(
        n_estimators=2940, max_depth=8, learning_rate=0.058513724672961474, subsample=0.966903548205041,
        colsample_bytree=0.6389163126034817, gamma=0.511952128846015, reg_alpha=0.5813029236097625,
        reg_lambda=0.06307134961195304, random_state=42, eval_metric='logloss', tree_method='hist',
        enable_categorical=True),
    'cat': CatBoostClassifier(
        iterations=1924, depth=9, learning_rate=0.1516720383883906, l2_leaf_reg=2.0561906552795515,
        random_strength=0.5612770485084783, bagging_temperature=0.5521031704243804, random_state=42,
        verbose=0, cat_features=[X.columns.get_loc(c) for c in ['Stage_fear', 'Drained_after_socializing']]),
    'rf': RandomForestClassifier(
        n_estimators=4227, max_depth=12, min_samples_split=3, min_samples_leaf=1, max_features='log2',
        max_samples=0.8265062803917577, random_state=42, n_jobs=-1),
    'lgb': LGBMClassifier(
        n_estimators=4923, max_depth=9, learning_rate=0.07247628224517862, num_leaves=99, min_child_samples=11,
        reg_alpha=0.5534605185815967, reg_lambda=0.4542620702300151, feature_fraction=0.9481042324480506,verbose=-1,
        random_state=42, n_jobs=-1, categorical_feature=[X.columns.get_loc(c) for c in ['Stage_fear', 'Drained_after_socializing']]),
    'lr': make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.010505770800081918, penalty='elasticnet', solver='saga', l1_ratio=0.2679428960515307,
                         max_iter=1000, random_state=42))
}


# Train and predict
preds = {}
for name, model in models.items():
    model.fit(X, y)
    preds[name] = model.predict_proba(X_test)[:, 1]

# Ensemble
final_probs = (0.25*preds['xgb'] + 0.25*preds['cat'] + 0.20*preds['rf'] + 0.25*preds['lgb'] + 0.05*preds['lr'])
pd.DataFrame({
    'id': test['id'],
    'Personality': ['Extrovert' if p > 0.5 else 'Introvert' for p in final_probs]
}).to_csv('submission.csv', index=False)


submission = pd.read_csv("submission.csv")
submission.head(5)

