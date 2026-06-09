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


df = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv", parse_dates=["date"])
df.head()


df.info()


df.isna().sum()


df.duplicated().sum()


df["num_sold"] = df["num_sold"].bfill()


df.head()


df.isna().sum()


%pip install darts


from darts import TimeSeries


series = TimeSeries.from_group_dataframe(
    df,
    group_cols = ["country", "store", "product"],
    time_col = "date",
    value_cols = "num_sold"    
)


len(series)


from darts.models import ExponentialSmoothing
from darts.utils.utils import ModelMode
from darts.utils.utils import SeasonalityMode


from tqdm import tqdm


models = {}
for i in tqdm(range(len(series))):
    model =  ExponentialSmoothing(trend=ModelMode.ADDITIVE, seasonal=SeasonalityMode.ADDITIVE)
    model.fit(series[i])
    group = tuple(series[i].static_covariates.values.flatten())
    models[group] = model


xnew = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv", parse_dates=["date"])
xnew.head()


groups = xnew.groupby(by=["country", "store", "product"])


details = {}
for name, group in groups:
    details[name] = len(group)


details


predictions = {}
for key, model in models.items():
    p = model.predict(n=1095)
    predictions[key] = p


predictions[('Canada',
  'Discount Stickers',
  'Holographic Goose')].values()


preds = []
for name, group in groups:
    group["num_sold"] = predictions[name].values()
    preds.append(group)


res = pd.concat(preds, axis=0)
res


res = res[["id", "num_sold"]]


res = res.astype(int)


res


res.to_csv("submission.csv", index=False)

