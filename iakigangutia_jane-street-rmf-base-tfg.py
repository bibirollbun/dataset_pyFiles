import pandas as pd
import polars as pl
import numpy as np
from scipy.stats import jarque_bera
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Evaluation
import os
import kaggle_evaluation.jane_street_inference_server as JS_eval


features_csv = pd.read_csv('/kaggle/input/jane-street-real-time-market-data-forecasting/features.csv') # metadata pertaining to the anonymized features
responders_csv = pd.read_csv('/kaggle/input/jane-street-real-time-market-data-forecasting/responders.csv') # metadata pertaining to the anonymized responders
sample_submission_csv = pd.read_csv('/kaggle/input/jane-street-real-time-market-data-forecasting/sample_submission.csv') # format of the predictions your model should make (responder_6)


# The submission must contain the prediction for each symbol_id, represented as row_id
sample_submission_csv.head()


responders_csv.loc[[6]]


# Extract objective features
cols = ['feature', 'tag_2']
features_sel = features_csv[cols]

features_sel = features_sel.query('tag_2 == True')
features_obj = features_sel['feature'].unique()
features_obj


# Objective columns
ts_columns = ['date_id', 'time_id', 'symbol_id', 'weight', 'responder_6'] + list(features_obj)

X_columns = ['date_id', 'time_id', 'symbol_id', 'weight'] + list(features_obj)
y_columns = ['date_id', 'time_id', 'symbol_id', 'responder_6']

ts_test_columns = ['row_id'] + X_columns
ts_lags_columns = ['date_id', 'time_id', 'symbol_id', 'responder_6_lag_1']


# Train: 9 partitions
train_partitions = [
    pd.read_parquet(f"/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet/partition_id={p}/part-0.parquet", 
                    columns=ts_columns)
    for p in range(10)
]
ts_train = pd.concat(train_partitions, axis=0, ignore_index=True)

'''
train_partitions = [
    pl.read_parquet(f"/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet/partition_id={p}/part-0.parquet",
                    columns=ts_columns)
    for p in range(10)
]
ts_train = pl.concat(train_partitions)
'''

# Test parquet
X_ts_test = pl.read_parquet('/kaggle/input/jane-street-real-time-market-data-forecasting/test.parquet/date_id=0/part-0.parquet',
                            columns=ts_test_columns)

# Lags parquet
y_ts_lags = pl.read_parquet('/kaggle/input/jane-street-real-time-market-data-forecasting/lags.parquet/date_id=0/part-0.parquet',
                            columns=ts_lags_columns)


X_ts_test


# Select the desired columns for X and y
#ts_train = ts_train.drop_nulls()
ts_train = ts_train.dropna()

X = ts_train[X_columns]
y = ts_train[y_columns]

ts_train


#del ts_train, features_csv, responders_csv, sample_submission_csv


print(f"Shape X:{np.shape(X)} \nShape y:{np.shape(y)} \n\nNull X: {X.isnull().sum().sum()} \nNull y: {y.isnull().sum().sum()}")
#print(f"Shape X:{np.shape(X)} \nShape y:{np.shape(y)} \n\nNull X: {sum(X.null_count().sum().row(0))} \nNull y: {sum(y.null_count().sum().row(0))}")


print(f"Number of symbol_id X: {len(X['symbol_id'].unique())} \nNumber of symbol_id y: {len(y['symbol_id'].unique())}")


# Arreglar outliers de salto repentino

for n in range(39):
    df_plot = y.query('symbol_id == @n')
    #df_plot = y.filter(pl.col('symbol_id') == n)
    trajectory = np.cumsum(df_plot['responder_6'])
    plt.plot(df_plot['date_id'], trajectory, linewidth=0.3)

plt.xlim(0, y['date_id'].max())
plt.xlabel('Date ID')
plt.ylabel('Cumulative Sum')
plt.title('Symbols Trajectories')
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
plt.show()


def getWeights_FFD(d, thres):
    w, k = [1.], 1
    
    while True:
        w_ = -w[-1] / k * (d-k+1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k+=1
        
    return np.array(w[::-1]).reshape(-1, 1)



def fracDiff_FFD(series, d, thres=1e-3):
    '''
    series: Pandas DataFrame format (X)
    d: can be any positive fractional, not necessarily bounded [0,1]
    thresh: determines the cut-off weight for the window
    '''
    w = getWeights_FFD(d, thres)
    width = len(w) - 1
    df = {}
    
    for name in series.columns:
        seriesF = series[[name]].ffill().dropna()
        df_ = pd.Series(dtype=float)
        for iloc1 in range(width, seriesF.shape[0]):
            loc0 = seriesF.index[iloc1-width]
            loc1 = seriesF.index[iloc1]
            if not np.isfinite(series.loc[loc1, name]):
                continue
            df_[loc1] = np.dot(w.T, seriesF.loc[loc0:loc1])[0, 0]
        df[name] = df_.copy(deep=True)
    df = pd.concat(df, axis=1)
    
    return df


X_diff = X.query('symbol_id == 0').iloc[:, 4:]
X_diff = X_diff.cumsum()
X_diff = X_diff.iloc[:10000, :]


X_diff


frac = fracDiff_FFD(X_diff, 0.5)
frac


feature = ['feature_01']
plt.plot(X_diff[feature])
plt.plot(frac[feature])
plt.show()


frac = frac.reset_index(drop=True)


np.round(adfuller(X_diff['feature_01'])[1], 2), np.round(adfuller(frac['feature_01'])[1], 2)





# Train Test with TS
    # Filtrar y matchear index en y_
X_LR = X.filter(pl.col('symbol_id') == 0)[:, 4:]
y_LR = y.filter(pl.col('symbol_id') == 0)[:, 3]

train_size = int(len(X_LR) * 0.8)
X_train, X_test = X_LR[:train_size], X_LR[train_size:]
y_train, y_test = y_LR[:train_size], y_LR[train_size:]


# Simplest model LR
LM = LinearRegression()
LM.fit(X_train, y_train)

y_pred = LM.predict(X_test)


# Resultados del modelo lineal (relaciones no lineales)
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f} \nR²: {r2_score(y_test, y_pred):.2f}")

# Con datos en diferencias; MSE: 0.47 R^2: -0.00
# Con diferenciación parcial: 

















'''
lags_ : pl.DataFrame | None = None


# Replace this function with your inference code.
# You can return either a Pandas or Polars dataframe, though Polars is recommended.
# Each batch of predictions (except the very first) must be returned within 1 minute of the batch features being provided.
def predict(test: pl.DataFrame, lags: pl.DataFrame | None) -> pl.DataFrame | pd.DataFrame:
    """Make a prediction."""
    # All the responders from the previous day are passed in at time_id == 0. We save them in a global variable for access at every time_id.
    # Use them as extra features, if you like.
    global lags_
    if lags is not None:
        lags_ = lags

    # Replace this section with your own predictions
    predictions = test.select(
        'row_id',
        pl.lit(0.0).alias('responder_6'),
    )

    if isinstance(predictions, pl.DataFrame):
        assert predictions.columns == ['row_id', 'responder_6']
    elif isinstance(predictions, pd.DataFrame):
        assert (predictions.columns == ['row_id', 'responder_6']).all()
    else:
        raise TypeError('The predict function must return a DataFrame')
    # Confirm has as many rows as the test data.
    assert len(predictions) == len(test)

    return predictions
'''


'''
inference_server = kaggle_evaluation.jane_street_inference_server.JSInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        (
            '/kaggle/input/jane-street-real-time-market-data-forecasting/test.parquet',
            '/kaggle/input/jane-street-real-time-market-data-forecasting/lags.parquet',
        )
    )
'''

