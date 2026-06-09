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


train_all = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')

random_state = 57


import kagglehub

ADD_ORIGINAL_DATASET = False

if ADD_ORIGINAL_DATASET:
    # Download latest version
    path = kagglehub.dataset_download("sushant097/bank-marketing-dataset-full")
    
    print("Path to dataset files:", path)
    
    orig_df = pd.read_csv('/kaggle/input/bank-marketing-dataset-full/bank-full.csv', sep=';')
    orig_df['y'] = orig_df['y'].apply(lambda x: 0 if x == 'no' else 1)
    train_all = pd.concat([train_all.drop('id', axis=1), orig_df])


x_columns = "age job marital education default balance housing loan contact day month duration campaign pdays previous poutcome".split()
y_column = ["y"]


from sklearn.model_selection import train_test_split

def get_train_test_split(df: pd.DataFrame, test_split=0.2):
    X, y = df[x_columns], df[y_column]
    train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=random_state)
    return (train_X, train_y), (val_X, val_y)
    

(train_X, train_y), (val_X, val_y) = get_train_test_split(train_all)


from sklearn.preprocessing import LabelEncoder


class Encoder:
    def __init__(
            self,
            columns: list[str]=[
                'job',
                'marital',
                'education',
                'default',
                'housing',
                'loan',
                'contact',
                'month',
                'poutcome',
            ]
    ):
        self.encoders = {
            col: LabelEncoder()
            for col in columns
        }

    def fit(self, df):
        for col in self.encoders:
            self.encoders[col].fit(df[col])

    def transform(self, df):
        for col in self.encoders:
            df[f'{col}_'] = self.encoders[col].transform(df[col])



categorical = [
    'job', 'marital', 'education', 'default', 'housing',
    'loan', 'contact', 'month', 'poutcome',
]

train_X_ = train_X.drop(columns=categorical)
train_X_.reset_index().head()


corr = train_X_.copy()
corr['y'] = train_y
corr.corr()


train_y.value_counts()


# import matplotlib.pyplot as plt

_columns = "balance duration pdays previous housing_ contact_".split()
# fig, ax = plt.subplots(2, 3, figsize=(20, 15))
# for ind, col in enumerate(_columns):
#     i, j = ind // 3, ind % 3
#     ax[i, j].plot(train_X_[col].values, train_y.values, 'bo', alpha=0.01)
#     ax[i, j].set_xlabel(col)
#     ax[i, j].set_ylabel('y')
# fig.show()


import functools

combined_features = [mask for mask in range(1, 1 << len(_columns))]

def augument_df(df: pd.DataFrame, cols: list[str] = _columns):
    for feature in combined_features:
        selected = [_columns[i] for i, _ in enumerate(_columns) if (1 << i ) & feature]
        if len(selected) > 2:
            continue
        name = "_&_".join(selected)
        df[name] = df[selected].apply(lambda x: functools.reduce(lambda a, y: a*y, x), axis=1)
    return df

# train_X_copy = train_X_.copy()
# augment_df(train_X_copy)


from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier


# we'll do a grid search on its parameters
# mlp_classifier = MLPClassifier(hidden_layer_sizes=(100, 10))

# logistic = LogisticRegressionCV()
# ensemble = VotingClassifier([('xgb', xgb_model), ('mlp', mlp_classifier), ('logistic', logistic)])



# parameters = {'max_depth': list(range(3, 7)), 'max_leaves': list(range(2, 5))}
# grid_search = GridSearchCV(xgb_model, parameters)
# grid_search.fit(train_X_copy, train_y)
# grid_search.cv_results_


parameters = {'max_leaves': list(range(4, 12))}
xgb = XGBClassifier()
encoder_all = Encoder()

# grid_search = GridSearchCV(xgb_model, parameters)

class PrepareDF:

    def __init__(self):
         self.encoder = Encoder()
    
    def __call__(self, *args, **kwargs):
        return self.prepare_df(*args, **kwargs)

    def prepare_df(self, _train: pd.DataFrame, augument: bool, is_training: bool) -> pd.DataFrame:
        _train_X_all = _train[x_columns]
        if is_training:
            self.encoder.fit(_train_X_all)
        self.encoder.transform(_train_X_all)
        _train_X_all = _train_X_all.drop(columns=categorical)
        if augument:
            augument_df(_train_X_all)
        if is_training:
            return _train_X_all, _train[y_column]
        else:
            return _train_X_all


def make_prediction(
    model,
    _X: pd.DataFrame,
    _Y: pd.DataFrame,
    is_training: bool = True,
):
    if is_training:
        model.fit(_train_X_all, _train_y_all)
    return model.transform(_train_X_all)

# grid_search.cv_results_

mlp = MLPClassifier(hidden_layer_sizes=(512, 16))
prepare_df_no_aug = PrepareDF()
x_mlp, y_mlp = prepare_df_no_aug(train_all, augument=False, is_training=True)
mlp.fit(x_mlp, y_mlp)

prepare_df_aug = PrepareDF()
x_xgb, y_xgb = prepare_df_aug(train_all, augument=True, is_training=True)
xgb_model.fit(x_xgb, y_xgb)
print(f'{xgb_model.score(x_xgb, y_xgb)=}')
print(f'{mlp.score(x_mlp, y_mlp)=}')

y_pred_mlp = mlp.predict(x_mlp)
y_pred_xgb = xgb.predict(x_xgb)

print(f"& count: {((y_pred_mlp & y_pred_xgb) == y_mlp).sum() / len(y_mlp)}")
print(f"| count: {((y_pred_mlp | y_pred_xgb) == y_mlp).sum() / len(y_mlp)}")


test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


test_X_mlp = prepare_df_no_aug(test, augument=False, is_training=False)
test_X_xgb = prepare_df_aug(test, augument=True, is_training=False)


# test_X_copy = augment_df(test_X)
y_pred_xgb = xgb.predict(test_X_xgb)
y_pred_mlp = mlp.predict(test_X_mlp)


submission = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')

print(f'{len(y_pred)=}, {len(test)=}')

submission_test = test[['id']]

submission_test['y'] = y_pred_xgb | y_pred_mlp



submission_test.to_csv('submission.csv', index=False)

