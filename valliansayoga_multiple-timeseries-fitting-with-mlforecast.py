!pip install mlforecast window-ops -q


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import warnings


warnings.filterwarnings("ignore")
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv("/kaggle/input/china-real-estate-demand-prediction/train/new_house_transactions.csv")
train.head()


train.sector = train.sector.str.replace("sector ", "").astype(np.int8)


train["date"] = pd.to_datetime(train.month, format="%Y-%b")


train.date.describe()


import matplotlib.pyplot as plt
import seaborn as sns


plt.rcParams["figure.figsize"] = 15, 5
data_mean = train.groupby("date").amount_new_house_transactions.mean()
data_median = train.groupby("date").amount_new_house_transactions.median()
sns.lineplot(data_mean, label="data_mean")
sns.lineplot(data_median, label="data_median")
plt.legend()


train_reform = pd.pivot_table(train, values="amount_new_house_transactions", index="date", columns="sector")
train_reform.head()


train_reform.insert(94, 95, 0)
train_reform.head()


train_reform = train_reform.interpolate("time").fillna(0)
train_reform


train_reform = pd.melt(train_reform.reset_index(), id_vars="date", value_name="new_house_transaction_amount")
train_reform.head()


train_reform.shape


from mlforecast import MLForecast
from mlforecast.target_transforms import Differences, LocalStandardScaler
from mlforecast.lag_transforms import ExpandingMean, RollingMean, ExponentiallyWeightedMean, RollingStd, Lag, RollingMin, RollingMax, \
    RollingQuantile
from utilsforecast.plotting import plot_series
from mlforecast.auto import AutoXGBoost


plot_series(train_reform, id_col="sector", time_col="date", target_col="new_house_transaction_amount")


lag_transforms = [
    ExpandingMean(), 
    RollingMean(3),
    # RollingMean(6),
    # RollingMean(9),
    ExponentiallyWeightedMean(0.1),
    RollingStd(3),
    # RollingStd(6),
    # RollingStd(9),
    Lag(3),
    # Lag(6),
    # Lag(9),
    RollingMin(3),
    # RollingMin(6),
    # RollingMin(9),
    RollingMax(3),
    # RollingMax(6),
    # RollingMax(9),
    RollingQuantile(0.25, 3),
    # RollingQuantile(0.25, 6),
    # RollingQuantile(0.25, 9),
    RollingQuantile(0.75, 3),
    # RollingQuantile(0.75, 6),
    # RollingQuantile(0.75, 9),
]


from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor


params = dict(
    learning_rate=0.007,
    subsample=0.8,
    reg_lambda=0.25,
    reg_alpha=0.1,
    n_estimators=5000,
    objective="reg:pseudohubererror",
    colsample_bytree=0.9,
    colsample_bylevel=0.9,
    colsample_bynode=0.9,
)

models = [
    XGBRegressor(**params, random_state=42, n_jobs=-1)
    # AutoXGBoost(config=my_searchspace)
]

fcst = MLForecast(
    models=models,
    freq="M",
    target_transforms=[
        Differences([3]),
        LocalStandardScaler()
    ],    
    lag_transforms={
        1: lag_transforms,
        # 2: lag_transforms,
        # 3: lag_transforms
    },
    date_features=["month", "quarter", "year"]
)


train_reform.date = train_reform.date.dt.to_period("M").dt.to_timestamp("M")


train_reform.head()


fcst.fit(
    train_reform,
    id_col="sector",
    time_col="date",
    target_col="new_house_transaction_amount",
)


preds = fcst.predict(12)
preds.date.max()


preds.shape


plot_series(
    train_reform, 
    preds,
    id_col="sector",
    time_col="date",
    target_col="new_house_transaction_amount"
)


preds["id"] = preds.date.dt.strftime("%Y %b") + "_sector " + preds.sector.astype(str)
preds


sub = pd.read_csv("/kaggle/input/china-real-estate-demand-prediction/test.csv")
sub


sub = pd.merge(sub, preds, on="id", how="left")
sub


sub.drop("new_house_transaction_amount", axis=1, inplace=True)
sub.rename({"XGBRegressor": "new_house_transaction_amount"}, axis=1, inplace=True)
sub


sub.new_house_transaction_amount = sub.new_house_transaction_amount.clip(0)


sub[["id", "new_house_transaction_amount"]].to_csv('submission.csv', index=False)
!head submission.csv

