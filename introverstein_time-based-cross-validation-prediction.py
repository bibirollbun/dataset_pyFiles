import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_percentage_error
import lightgbm as lgb

warnings.filterwarnings('ignore')


class CFG:
    train_path = '/kaggle/input/playground-series-s5e1/train.csv'
    test_path = '/kaggle/input/playground-series-s5e1/test.csv'
    sample_sub_path = '/kaggle/input/playground-series-s5e1/sample_submission.csv'
    seed = 4321


train = pd.read_csv(CFG.train_path)
train['date'] = pd.to_datetime(train['date'])
train = train.sort_values('date')

test = pd.read_csv(CFG.test_path)
test['date'] = pd.to_datetime(test['date'])
test = test.sort_values('date')

cat_features = ['country', 'store', 'product']


print(f"Shape before dropping NaN categories: {len(train)}")

nan_percentages = (
    train.groupby(cat_features)['num_sold']
    .apply(lambda x: (x.isna().sum() / len(x)) * 100)
    .reset_index(name='nan_percentage')
)

categories_to_drop = nan_percentages[nan_percentages['nan_percentage'] > 0]
train = train.merge(categories_to_drop[cat_features], on=cat_features, how='left', indicator=True)
train = train[train['_merge'] == 'left_only'].drop(columns='_merge').reset_index(drop=True)

print(f"Shape after dropping NaN categories: {len(train)}")


def create_date_features(data):
    data['quarter'] = data['date'].dt.quarter
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    data['day_of_week'] = data['date'].dt.dayofweek
    data['day_of_year'] = data['date'].dt.dayofyear
    data['week_of_month'] = (data['date'].dt.day - 1) // 7 + 1
    data['week_of_year'] = data['date'].dt.isocalendar().week.astype(int)
    data['is_weekend'] = data['date'].dt.dayofweek.isin([5, 6]).astype(int)
    data['is_month_end'] = data['date'].dt.is_month_end.astype(int)
    data['year'] = data['date'].dt.year

    date_feats = [
        'quarter', 'month', 'day', 'day_of_week', 'day_of_year',
        'week_of_month', 'week_of_year', 'is_weekend', 'is_month_end', 'year'
    ]
    return data, date_feats


def encode_categorical_features(data, cat_features):
    encoders = {}
    for feature in cat_features:
        encoder = LabelEncoder()
        data[feature + '_encoded'] = encoder.fit_transform(data[feature])
        encoders[feature] = encoder
    return data, encoders

train, encoders = encode_categorical_features(train, cat_features)
test, _ = encode_categorical_features(test, cat_features)

encoded_cat_features = [feature + '_encoded' for feature in cat_features]


train['num_sold_log'] = np.log1p(train['num_sold'])

all_mapes = []
oof_predictions = np.zeros(len(train))
test_predictions = np.zeros(len(test))

n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)

for fold, (train_index, val_index) in enumerate(tscv.split(train)):
    print(f"\nFold {fold + 1}:\n")

    train_data, val_data = train.iloc[train_index], train.iloc[val_index]
    print(f"Train date range: {train_data['date'].min().date()} to {train_data['date'].max().date()}")
    print(f"Val date range: {val_data['date'].min().date()} to {val_data['date'].max().date()}")
    print(f"Train shape: {train_data.shape}")
    print(f"Val shape: {val_data.shape}\n")

    train_data, date_feats = create_date_features(train_data)
    val_data, _ = create_date_features(val_data)

    feature_cols = encoded_cat_features + date_feats
    X_train, y_train = train_data[feature_cols], train_data['num_sold_log']
    X_val, y_val = val_data[feature_cols], val_data['num_sold_log']

    model = lgb.LGBMRegressor(
        n_estimators=1000, learning_rate=0.1, boosting_type='gbdt',
        categorical_feature=encoded_cat_features, random_state=42
    )
    model.fit(X_train, y_train, categorical_feature=encoded_cat_features)

    val_preds_log = model.predict(X_val)
    oof_predictions[val_index] = val_preds_log

    val_preds = np.expm1(val_preds_log)
    mape = mean_absolute_percentage_error(np.expm1(y_val), val_preds)
    all_mapes.append(mape)
    print(f"Fold {fold + 1} MAPE: {mape:.4f}\n")

    test_data, _ = create_date_features(test)
    X_test = test_data[feature_cols]
    test_preds_log = model.predict(X_test)
    test_preds = np.expm1(test_preds_log)
    test_predictions += test_preds / n_splits


valid_mapes = [m for m in all_mapes if not np.isnan(m)]
if valid_mapes:
    print(f"\nAverage MAPE across all folds: {np.mean(valid_mapes):.4f}")
else:
    print("No valid MAPE scores calculated")


submission = test[['id']].copy()
submission['num_sold'] = test_predictions
submission.to_csv('submission.csv', index=False)

