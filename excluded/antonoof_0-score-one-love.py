!pip install xgboost > /dev/null


import os
import warnings
import numpy as np
import pandas as pd
import polars as pl # read train -> pd
import xgboost as xgb

from sklearn.metrics import ndcg_score
from ydata_profiling import ProfileReport # Exploratory Data Analysis(EDA)
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")


pd.read_parquet('/kaggle/input/aeroclub-recsys-2025/sample_submission.parquet').head()


train = pl.read_parquet('/kaggle/input/aeroclub-recsys-2025/train.parquet')


train.head()


train = train.select(['ranker_id', 'taxes', 'totalPrice', 'selected'])
train = train.to_pandas()


ProfileReport(train, title="EDA")


test = pd.read_parquet('/kaggle/input/aeroclub-recsys-2025/test.parquet')


test.head()


train['selected'].unique()


# set(train['Id']) & set(test['Id']) # return set() =(


train['ranker_id'] = train['ranker_id'].astype('category')

X = train[['ranker_id', 'taxes', 'totalPrice']]
y = train['selected']

train_categories = X['ranker_id'].cat.categories


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train['ranker_id'] = X_train['ranker_id'].cat.codes
X_test['ranker_id'] = X_test['ranker_id'].cat.set_categories(train_categories, ordered=True)
X_test['ranker_id'] = X_test['ranker_id'].cat.codes

train_group_sizes = X_train['ranker_id'].value_counts().sort_index().tolist()


model = xgb.XGBRanker(
    objective='rank:pairwise',
    n_estimators=333,  # Number of trees
    tree_method='gpu_hist',
    predictor='gpu_predictor',
    gpu_id=0,
    max_depth=8,       # Max depth
    learning_rate=0.01,
    reg_alpha=0.3,     # L1 regularization
    reg_lambda=0.2,    # L2 regularization
)

model.fit(X_train, y_train, group=train_group_sizes)


y_scores = model.predict(X_test)
y_ranks = y_scores.argsort().argsort() + 1

y_ranks[:5]


true_ranks = y_test.values.reshape(1, -1)
predicted_ranks = y_ranks.reshape(1, -1)

ndcg = ndcg_score(true_ranks, predicted_ranks)
print(f"NDCG Score: {ndcg:.2f}")


# del train


submission = pd.DataFrame({
    'Id': test['Id'],
    'ranker_id': test['ranker_id']
})

test['ranker_id'] = test['ranker_id'].astype('category')
test['ranker_id'] = test['ranker_id'].cat.set_categories(train_categories, ordered=True)
test['ranker_id'] = test['ranker_id'].cat.codes

test_scores = model.predict(test[['ranker_id', 'taxes', 'totalPrice']])
test_ranks = test_scores.argsort().argsort() + 1


submission['selected'] = test_ranks
submission.to_parquet('submission.parquet', index=False)
submission.head()


sum(submission['selected'] < 10)

