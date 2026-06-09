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


X_train, X_test, y_train, y_test = train_test_split(
    X.copy(), y.copy(), test_size=0.2, random_state=42
)


# Ğ�Ğ±Ñ�Ğ·Ğ°Ñ‚ĞµĞ»ÑŒĞ½Ğ¾ Ñ�Ğ¾Ñ…Ñ€Ğ°Ğ½Ñ�ĞµĞ¼ Ñ�Ñ‚Ñ€Ğ¾ĞºĞ¾Ğ²Ñ‹Ğ¹ ranker_id Ğ´Ğ»Ñ� Ğ³Ñ€ÑƒĞ¿Ğ¿Ğ¸Ñ€Ğ¾Ğ²ĞºĞ¸
X_train['ranker_id_str'] = X_train['ranker_id'].astype(str)
X_test['ranker_id_str'] = X_test['ranker_id'].astype(str)


#3. ĞšĞ¾Ğ´Ğ¸Ñ€ÑƒĞµĞ¼ ranker_id Ğ´Ğ»Ñ� Ğ¿Ğ¾Ğ´Ğ°Ñ‡Ğ¸ Ğ² Ğ¼Ğ¾Ğ´ĞµĞ»ÑŒ:
# ĞŸĞ¾Ğ»ÑƒÑ‡Ğ°ĞµĞ¼ ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ğ¸
train_categories = X_train['ranker_id'].astype('category').cat.categories

# ĞŸÑ€Ğ¸Ğ¼ĞµĞ½Ñ�ĞµĞ¼ Ğº train
X_train['ranker_id'] = X_train['ranker_id'].astype('category')
X_train['ranker_id'] = X_train['ranker_id'].cat.set_categories(train_categories)
X_train['ranker_id_code'] = X_train['ranker_id'].cat.codes

# ĞŸÑ€Ğ¸Ğ¼ĞµĞ½Ñ�ĞµĞ¼ Ñ‚Ğµ Ğ¶Ğµ ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ğ¸ Ğº test
X_test['ranker_id'] = X_test['ranker_id'].astype('category')
X_test['ranker_id'] = X_test['ranker_id'].cat.set_categories(train_categories)
X_test['ranker_id_code'] = X_test['ranker_id'].cat.codes



# 4. Ğ’Ñ‹Ñ‡Ğ¸Ñ�Ğ»Ñ�ĞµĞ¼ group Ğ´Ğ»Ñ� Ğ¼Ğ¾Ğ´ĞµĞ»Ğ¸:
train_group_sizes = X_train.groupby('ranker_id_str').size().tolist()



#ĞŸÑ€Ğ¸Ğ¼ĞµÑ€ Ğ±Ğ°Ğ·Ğ¾Ğ²Ñ‹Ñ… params Ğ´Ğ»Ñ� Ğ·Ğ°Ğ´Ğ°Ñ‡Ğ¸ Ñ€Ğ°Ğ½Ğ¶Ğ¸Ñ€Ğ¾Ğ²Ğ°Ğ½Ğ¸Ñ� (Learning to Rank):
params = {
    "objective": "lambdarank",     # Ğ¸Ğ»Ğ¸ "rank_xendcg"
    "metric": "ndcg",              # Ğ¸Ğ»Ğ¸ "map", "ndcg@10"
    "ndcg_eval_at": [1, 3, 5, 10], # Ğ½Ğ° ĞºĞ°ĞºĞ¸Ñ… Ğ¿Ğ¾Ğ·Ğ¸Ñ†Ğ¸Ñ�Ñ… Ğ¾Ñ†ĞµĞ½Ğ¸Ğ²Ğ°Ñ‚ÑŒ
    "learning_rate": 0.1,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "verbose": -1
}


#ğŸ“¦ Ğ¢ĞµĞ¿ĞµÑ€ÑŒ Ğ´Ğ»Ñ� Ğ¾Ğ±ÑƒÑ‡ĞµĞ½Ğ¸Ñ� Ğ¼Ğ¾Ğ´ĞµĞ»Ğ¸:
import lightgbm as lgb

train_data = lgb.Dataset(
    X_train[['ranker_id_code', 'taxes', 'totalPrice']],
    label=y_train,
    group=train_group_sizes
)

# Ğ�Ğ±ÑƒÑ‡ĞµĞ½Ğ¸Ğµ
model = lgb.train(params, train_data)



X_test['score'] = model.predict(X_test[['ranker_id_code', 'taxes', 'totalPrice']])

X_test['rank'] = (
    X_test.sort_values(['ranker_id_str', 'score'], ascending=[True, False])
          .groupby('ranker_id_str')
          .cumcount() + 1
)



from sklearn.metrics import ndcg_score

# ĞŸÑ€ĞµĞ´Ñ�ĞºĞ°Ğ·Ğ°Ğ½Ğ½Ñ‹Ğµ Ñ�ĞºĞ¾Ñ€Ñ‹
X_test['score'] = model.predict(X_test[['ranker_id_code', 'taxes', 'totalPrice']])
X_test['true_label'] = y_test.values



print(y_test.shape)  # Ğ”Ğ¾Ğ»Ğ¶Ğ½Ğ¾ Ğ±Ñ‹Ñ‚ÑŒ (1, N)
print(type(y_test))  # Ğ”Ğ¾Ğ»Ğ¶ĞµĞ½ Ğ±Ñ‹Ñ‚ÑŒ numpy.ndarray


from sklearn.metrics import ndcg_score
import numpy as np

ndcgs = []

for _, group in X_test.groupby('ranker_id_str'):
    if len(group) < 2:
        continue  # âš ï¸� ĞŸÑ€Ğ¾Ğ¿ÑƒÑ�ĞºĞ°ĞµĞ¼ Ñ�Ğ»Ğ¸ÑˆĞºĞ¾Ğ¼ Ğ¼Ğ°Ğ»ĞµĞ½ÑŒĞºĞ¸Ğµ Ğ³Ñ€ÑƒĞ¿Ğ¿Ñ‹

    true_labels = group['true_label'].values
    scores = group['score'].values

    if true_labels.sum() == 0:
        continue  # âš ï¸� ĞŸÑ€Ğ¾Ğ¿ÑƒÑ�ĞºĞ°ĞµĞ¼ Ğ³Ñ€ÑƒĞ¿Ğ¿Ñ‹ Ğ±ĞµĞ· Ğ²Ñ‹Ğ±Ñ€Ğ°Ğ½Ğ½Ğ¾Ğ³Ğ¾ Ñ€ĞµĞ¹Ñ�Ğ°

    y_true = np.array([true_labels])
    y_score = np.array([scores])

    ndcg = ndcg_score(y_true, y_score)
    ndcgs.append(ndcg)

print(f"Mean NDCG: {np.mean(ndcgs):.4f}")




# 1. ĞšĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ğ¸ Ğ¸Ğ· train
test['ranker_id_cat'] = test['ranker_id'].astype('category')
test['ranker_id_cat'] = test['ranker_id_cat'].cat.set_categories(train_categories)
test['ranker_id_code'] = test['ranker_id_cat'].cat.codes
test['ranker_id_str'] = test['ranker_id'].astype(str)

# 2. ĞŸÑ€ĞµĞ´Ñ�ĞºĞ°Ğ·Ğ°Ğ½Ğ¸Ğµ
test['score'] = model.predict(test[['ranker_id_code', 'taxes', 'totalPrice']])

# 3. Ğ Ğ°Ğ½Ğ¶Ğ¸Ñ€Ğ¾Ğ²Ğ°Ğ½Ğ¸Ğµ Ğ²Ğ½ÑƒÑ‚Ñ€Ğ¸ ĞºĞ°Ğ¶Ğ´Ğ¾Ğ³Ğ¾ ranker_id
test['selected'] = (
    test.sort_values(['ranker_id_str', 'score'], ascending=[True, False])
        .groupby('ranker_id_str')
        .cumcount() + 1
)




# 4. Ğ¡Ğ¾Ñ…Ñ€Ğ°Ğ½Ñ�ĞµĞ¼ submission
submission = test[['Id', 'ranker_id', 'selected']]
submission.to_csv('OneLove56.csv', index=False)



print(submission.head(10))
print(submission['selected'].min(), submission['selected'].max())
print(submission.groupby('ranker_id').size().head())

