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


from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import calendar
import seaborn as sns
import warnings

# 關掉所有紅字
warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv")
test = pd.read_csv("/kaggle/input/bike-sharing-demand/test.csv")
submit = pd.read_csv("/kaggle/input/bike-sharing-demand/sampleSubmission.csv")


# 確認是否有缺失值
train.info()
test.info()


# 確認資料分布
train.describe()


test.describe()


# 因 count 有 outlier，透過 
train = train[np.abs(train["count"] - train["count"].mean()) <= (3 * train["count"].std())]
print(train.shape)


data = pd.concat([train,test],ignore_index=True)
print(data.shape)

datetime = pd.to_datetime(data['datetime'], format='%Y-%m-%d %H:%M:%S')
data['hour'] = datetime.dt.hour
data['year'] = datetime.dt.year
data["weekday"] = datetime.dt.weekday
data['month'] = datetime.dt.month




# fig 代表整個畫布， axes 代表子圖
# histplot 代表直方圖， x 軸會根據傳入的值，進行區間分布，然後 y 軸預設是 count ，可透過 stat 修改
fig, axes = plt.subplots(nrows=2, ncols=2)
fig.set_size_inches(24,10)
sns.histplot(data["temp"],ax=axes[0][0], kde=True)
sns.histplot(data["atemp"],ax=axes[0][1], kde=True)
sns.histplot(data["humidity"],ax=axes[1][0], kde=True)
sns.histplot(data["windspeed"],ax=axes[1][1], kde=True)

axes[0][0].set(xlabel="temp",title="distribution of temp")
axes[0][1].set(xlabel="atemp",title="distribution of atemp")
axes[1][0].set(xlabel="humidity",title="distribution of humidity")
axes[1][1].set(xlabel="windspeed",title="distribution of windspeed")


# 從上面看起來， windspeed 看起來有些不符合邏輯的，位於 0~ 10 的中間的風速非常少，而且 0 非常的多
data_wind_0 = data[data["windspeed"] == 0]
print("這邊是 data_wind_0", len(data_wind_0))
data_wind_not_0 = data[data["windspeed"] != 0]
rf = RandomForestRegressor(n_estimators=1000, random_state=42)
wind_columns = ["season","weather","humidity","month","temp","year","atemp"]
rf.fit(data_wind_not_0[wind_columns],data_wind_not_0["windspeed"])
data_wind_0["windspeed"] = rf.predict(data_wind_0[wind_columns])
data = pd.concat([data_wind_0,data_wind_not_0])

print(data.head())


# 可看到有些 0 ~10 的風力分布較為正常了
sns.histplot(data["windspeed"],kde=True)


# 分割訓練跟測試集，並進行排序
train_data = data[data["count"].notnull()].sort_values(by="datetime")
test_data = data[data["count"].isnull()].sort_values(by="datetime")
datetime_col = test_data["datetime"]
train_y = train_data["count"]
# 因為現在右偏太嚴重，所以可以用 log 協助讓分布更對稱
train_y_log = np.log(train_y) 



sns.histplot(train_y,kde=True)


sns.histplot(train_y_log,kde=True)


drop_columns = ["casual","count","datetime","registered"]
train_data = train_data.drop(drop_columns,axis=1)
test_data = test_data.drop(drop_columns,axis=1)


rf = RandomForestRegressor(n_estimators=1000, random_state=42)
rf.fit(train_data, train_y_log)
preds = rf.predict(train_data)



preds_test = rf.predict(test_data)
# 因為先前用 log 更改分布後，需要透過 exp 還原值
submission = pd.DataFrame({
    "datetime": datetime_col,
    "count": [max(0, x) for x in np.exp(preds_test)]
})

submission.to_csv("bike_predictions_RF.csv", index=False)


submission

