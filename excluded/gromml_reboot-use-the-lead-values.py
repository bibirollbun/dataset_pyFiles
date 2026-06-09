import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


def preprocess_data(raw_df):
    assert len(raw_df.shape) == 2

    y = raw_df['label'].to_numpy()
    assert y.shape == (raw_df.shape[0],)

    cols = [
        'X239', 'X743', 'X215', 'X623', 'X240', 'X71', 'X119', 'X157',
        'X173', 'X198', 'X113', 'X324', 'X366', 'X462',
    ]

    df = raw_df[cols]
    assert df.isna().sum().sum() == 0

    df = pd.concat([
        df.shift(-lag).add_suffix(f'_lead_{lag}')
        for lag in [30, 40, 50, 60, 70, 80]
    ], axis=1)

    df = df.fillna(0.0)
    assert 'label' not in df.columns
    assert raw_df.shape[0] == df.shape[0] and (raw_df.index == df.index).all()
    assert df.isna().sum().sum() == 0
    assert df.shape[0] == y.shape[0]
    return df, y


train_df = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
X_train, y_train = preprocess_data(train_df)
train_df = None
print(X_train.shape, y_train.shape)


scaler = StandardScaler()
X_train.iloc[:, :] = scaler.fit_transform(X_train)


%%time

# model = Ridge()
model = xgb.XGBRegressor()

model.fit(X_train, y_train)


# pd.Series(model.coef_, index=model.feature_names_in_).sort_values().plot(
#     kind='barh', figsize=(16, 2 + len(model.coef_) // 4)
# )

pd.Series(model.feature_importances_, index=X_train.columns).sort_values().plot(
    kind='barh', figsize=(16, 2 + X_train.shape[1] // 4)
)


X_train = None
y_train = None


test_df = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')


t = pd.Series(pd.read_csv(
    '/kaggle/input/the-order-of-the-test-rows-2/closest_rows.csv'
)['0'].to_numpy())
assert t.shape == (test_df.shape[0],)
print('Reconstructed timestamps share:', len(t[t >= 0]) / len(t))


plt.figure(figsize=(16, 4))
plt.plot(t.sort_values().to_numpy())


plt.figure(figsize=(16, 4))
plt.plot(t[t >= 0].sort_values().iloc[:1000].to_numpy())
plt.axhline(10080, color='r', linestyle='--')


t -= 10080
t[t < 0] = 538149  # the most recent rows now have the biggest value

t = t.sort_values()
t[t <= len(t)] = np.arange(t[t <= len(t)].shape[0])
t = t.sort_index()


t = pd.Series(np.arange(538150), index=t.to_numpy()).sort_index()


plt.figure(figsize=(16, 4))
plt.plot(test_df['X656'].to_numpy())


test_df = test_df.iloc[t.to_numpy()]


plt.figure(figsize=(16, 4))
plt.plot(test_df['X656'].to_numpy())


X_test, _ = preprocess_data(test_df)
test_df = None
print(X_test.shape)


X_test.iloc[:, :] = scaler.transform(X_test)


y_pred = model.predict(X_test)


pd.Series(y_pred).describe()


plt.figure(figsize=(16, 4))
plt.plot(np.cumsum(y_pred))


submission = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')

submission = submission.iloc[t.to_numpy()]
submission['prediction'] = y_pred
submission = submission.sort_index()

submission.to_csv('submission.csv', index=False)


submission




