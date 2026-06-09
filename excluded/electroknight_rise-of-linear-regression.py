import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import gc
from scipy.stats import pearsonr


train = pd.read_parquet("/kaggle/input/drw-crypto-market-prediction/train.parquet",
                       engine = 'pyarrow')
train.shape


def drop_inf_columns(df):
    """Drop columns from DataFrame if any value is inf or -inf in that column."""
    return df.loc[:, ~df.isin([float('inf'), float('-inf')]).any()]


train = drop_inf_columns(train)
train.shape


X = train.drop(columns = ['label'])
y = train['label']


X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)


del X,y,train
gc.collect()


n_features = X_train.shape[1]
models = []
for i in range(n_features):
    model = LinearRegression()
    Xi = X_train.iloc[:, i].values.reshape(-1, 1)
    model.fit(Xi, y_train)
    models.append(model)

preds = np.zeros((X_test.shape[0], n_features))
for i, model in enumerate(models):
    Xi_test = X_test.iloc[:, i].values.reshape(-1, 1)
    preds[:, i] = model.predict(Xi_test)
y_pred = np.median(preds, axis=1)


corr, _ = pearsonr(y_pred, y_test)
print(corr)


test = pd.read_parquet("/kaggle/input/drw-crypto-market-prediction/test.parquet",
                      engine = 'pyarrow')
test = test.drop(columns = ['label'])
test = drop_inf_columns(test)
test.shape


preds = np.zeros((test.shape[0], n_features))
for i, model in enumerate(models):
    i_test = test.iloc[:, i].values.reshape(-1, 1)
    preds[:, i] = model.predict(i_test)
y_pred = np.median(preds, axis=1)


submission = pd.DataFrame({
    "ID": np.arange(1, len(y_pred) + 1),
    "prediction": y_pred
})
submission.head()


submission.to_csv("submission.csv", index=False)

