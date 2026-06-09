!pip install xgboost > /dev/null


import numpy as np
import pandas as pd
import polars as pl # read train -> pd
import xgboost as xgb
from sklearn.model_selection import train_test_split


features = ['bySelf', 'companyID', 'frequentFlyer', 'nationality', 'isAccess3D', 'isVip',
            'legs0_segments0_baggageAllowance_quantity', 'legs0_segments0_baggageAllowance_weightMeasurementType', 'legs0_segments0_cabinClass',
            'legs0_segments0_flightNumber', 'legs0_segments0_seatsAvailable', 'profileId', 'pricingInfo_isAccessTP', 'pricingInfo_passengerCount',
            'legs0_segments0_arrivalTo_airport_iata', 'legs0_segments0_marketingCarrier_code',
            'ranker_id', 'taxes', 'totalPrice'
]
features_train = features + ['selected']


train = pl.read_parquet('/kaggle/input/aeroclub-recsys-2025/train.parquet')
train = train.select(features_train)
train = train.to_pandas()
test = pd.read_parquet('/kaggle/input/aeroclub-recsys-2025/test.parquet')


categorical = ['frequentFlyer', 'legs0_segments0_flightNumber', 'legs0_segments0_arrivalTo_airport_iata', 'legs0_segments0_marketingCarrier_code']
for col in categorical:
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')

for col in categorical:
    train[col] = train[col].cat.codes
    test[col] = test[col].cat.codes


train['ranker_id'] = train['ranker_id'].astype('category')
test['ranker_id'] = test['ranker_id'].astype('category')

X = train[features]
y = train['selected']


train_categories = X['ranker_id'].cat.categories

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train['ranker_id'] = X_train['ranker_id'].cat.codes
X_test['ranker_id'] = X_test['ranker_id'].cat.set_categories(train_categories, ordered=True).cat.codes

train_group_sizes = X_train['ranker_id'].value_counts().sort_index().tolist()


model = xgb.XGBRanker(
    objective='rank:pairwise',
    n_estimators=3000,
    max_depth=5,
    learning_rate=0.002,
    eval_metric='ndcg@3',
    reg_alpha=0.6,
    reg_lambda=0.8
)

model.fit(X_train, y_train, group=train_group_sizes)


def take_rank(data, group_col, score_col, rank_col):
    data[rank_col] = data.groupby(group_col, observed=False)[score_col].rank(method='first', ascending=False).astype(int)
    return data


def HitRate3(data, rank_col, true_col):
    positive = 0
    Q = data['ranker_id'].nunique()

    for _, group in data.groupby('ranker_id'):
        preds_top_3 = group.nsmallest(3, rank_col) # top 3
        if any(preds_top_3[true_col] == 1):
            positive += 1

    hit_rate = positive / Q
    return hit_rate

X_test['y_scores'] = model.predict(X_test[features])
X_test = take_rank(X_test, 'ranker_id', 'y_scores', 'y_ranks')
X_test['selected'] = y_test.values

hitrate_score = HitRate3(X_test, 'y_ranks', 'selected')
print(f"HitRate@3: {hitrate_score:.2f}")


test['y_scores'] = model.predict(test[features])

test = take_rank(test, 'ranker_id', 'y_scores', 'selected')

submission = test[['Id', 'ranker_id', 'selected']]
submission.to_parquet('submission.parquet', index=False)
submission.head()




