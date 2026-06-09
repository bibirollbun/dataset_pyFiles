# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_set = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_set = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


train_set.head()


test_set.head()


train_set.describe().T


test_set.describe().T


train_set.isnull().sum()


test_set.isnull().sum()


train_set.columns


plt.figure(figsize=(14,7))
sns.boxplot(train_set)
plt.show()


corr = train_set.corr()
plt.figure(figsize =(12,8))
sns.heatmap(corr,cmap ="coolwarm")
plt.show()


def get_season(day):
    if 80 <= day <= 171:
        return "spring"
    elif 172 <= day <= 263:
        return "summer"
    elif 264 <= day <= 354:
        return "fall"
    else:
        return "winter"


train_set["season"]=train_set["day"].apply(get_season)
test_set["season"]=test_set["day"].apply(get_season)

train_set["temp_range"] = train_set["maxtemp"] - train_set["mintemp"]
test_set["temp_range"] = test_set["maxtemp"] - test_set["mintemp"]

train_set["dew_humidity_ratio"] = train_set["dewpoint"] / (train_set["humidity"] + 1e-5)
test_set["dew_humidity_ratio"] = test_set["dewpoint"] / (test_set["humidity"] + 1e-5)


train_set["temp_dew_diff"] = train_set["temparature"] - train_set["dewpoint"]
test_set["temp_dew_diff"] = test_set["temparature"] - test_set["dewpoint"]

train_set["low_sun"] = (train_set["sunshine"] < 1).astype(int)
test_set["low_sun"] = (test_set["sunshine"] < 1).astype(int)

train_set["cloud_sun_ratio"] = train_set["cloud"] / (train_set["sunshine"] + 1e-5)
test_set["cloud_sun_ratio"] = test_set["cloud"] / (test_set["sunshine"] + 1e-5)

train_set["cloud_humidity"] = train_set["humidity"] * train_set["cloud"]
test_set["cloud_humidity"] = test_set["humidity"] * test_set["cloud"]

train_set["temp_humidity"] = train_set["humidity"] * train_set["temp_dew_diff"]
test_set["temp_humidity"] = test_set["humidity"] * test_set["temp_dew_diff"]

season_map = {"winter": 0, "spring": 1, "summer": 2, "fall": 3}

train_set["season_num"] = train_set["season"].map(season_map)
test_set["season_num"] = test_set["season"].map(season_map)


train_set["cloud_sun_season"] = train_set["cloud_sun_ratio"] * train_set["season_num"]
test_set["cloud_sun_season"] = test_set["cloud_sun_ratio"] * test_set["season_num"]

train_set["cloud_sun_intersect"] = train_set["cloud"] * train_set["sunshine"]
test_set["cloud_sun_intersect"] = test_set["cloud"] * test_set["sunshine"]

train_set["cloud_humidity_intersect"] = train_set["cloud"] * train_set["humidity"]
test_set["cloud_humidity_intersect"] = test_set["cloud"] * test_set["humidity"]

train_set["cloud_sun_intersect"] = train_set["cloud"] / (train_set["sunshine"] + 1e-3)
test_set["cloud_sun_intersect"] = test_set["cloud"] / (test_set["sunshine"] + 1e-3)

train_set["humidity_dewpoint_intersect"] = train_set["humidity"] * train_set["dewpoint"]
test_set["humidity_dewpoint_intersect"] = test_set["humidity"] * test_set["dewpoint"]

train_set["sun_wind_intersect"] = train_set["sunshine"] / (train_set["windspeed"] + 1e-3)
test_set["sun_wind_intersect"] = test_set["sunshine"] / (test_set["windspeed"] + 1e-3)

train_set["cloud_low_sun_intersect"] = train_set["cloud"] * train_set["low_sun"]
test_set["cloud_low_sun_intersect"] = test_set["cloud"] * test_set["low_sun"]

# Convert boolean columns to integers
bool_cols = train_set.select_dtypes(include='bool').columns

for col in bool_cols:
    train_set[col] = train_set[col].astype(int)
    test_set[col] = test_set[col].astype(int)

# Drop the 'season' column from both DataFrames
train_set = train_set.drop(["season"], axis=1)
test_set = test_set.drop(["season"], axis=1)


test_set.isnull().sum()


test_set["winddirection"] = test_set["winddirection"].fillna(test_set["winddirection"].mean())


X = train_set.drop(["id", "rainfall"], axis=1)
y = train_set["rainfall"]

X_test = test_set.drop(["id"],axis = 1)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)


svc_model = SVC(C=1,gamma=.001,kernel="rbf",probability=True)
scores=cross_val_score(svc_model,X_scaled,y,cv=5,scoring="roc_auc")
print(f"Mean AUC: {scores.mean():.4f}")
svc_model.fit(X_scaled, y)


y_test_pred = svc_model.predict_proba(X_test_scaled)[:, 1]

submission_lr = pd.DataFrame({
    "id": test_set["id"],
    "rainfall": y_test_pred
})
submission_lr.to_csv("submission_svc_final.csv", index=False)

