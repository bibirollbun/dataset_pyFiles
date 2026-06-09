# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import plotly.express as px

#from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Insert Data code, add addition code/markdown blocks as needed


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

#test_data = pd.read_csv("test_V2.csv")
#train_data = pd.read_csv("train_V2.csv")

test_data = pd.read_csv("/kaggle/input/pubg-finish-placement-prediction/test_V2.csv")
train_data = pd.read_csv("/kaggle/input/pubg-finish-placement-prediction/train_V2.csv")


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



train_data.head()

train_data.isna().sum()


train_data["travel_distance"] = train_data["walkDistance"] + train_data["rideDistance"] + train_data["swimDistance"]

train_data["heals_boosts"] = train_data["heals"] + train_data["boosts"]

train_data["knocks"] = (train_data["DBNOs"] + train_data["assists"] + train_data["kills"] + train_data["teamKills"])



test_data["travel_distance"] = test_data["walkDistance"] + test_data["rideDistance"] + test_data["swimDistance"]

test_data["heals_boosts"] = test_data["heals"] + test_data["boosts"]

test_data["knocks"] = (test_data["DBNOs"] + test_data["assists"] + test_data["kills"] + test_data["teamKills"])


train_data.head()

train_data.info()
train_data=train_data.dropna()


fig = px.scatter(train_data, x="winPlacePerc", y="travel_distance")

fig.show()

fig2 = px.scatter(train_data, x = "knocks", y= "heals_boosts",
    color = "winPlacePerc"
)

fig2.show()

X = train_data[["travel_distance","heals_boosts", "weaponsAcquired", "damageDealt", "knocks", "killPlace"]]
y = train_data["winPlacePerc"]

X_test_data = test_data[["travel_distance","heals_boosts", "weaponsAcquired", "damageDealt", "knocks", "killPlace"]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42
)
y.info()
gbr = GradientBoostingRegressor(loss='absolute_error',
    learning_rate=0.16,
    n_estimators=350,
    max_depth=2,
    random_state=42)



gbr.fit(X_train, y_train)

pred_y = gbr.predict(X_test)

mae = mean_absolute_error(y_test, pred_y)

print('mean absolute error:', mae)


predictions = gbr.predict(X_test_data)


result = test_data[['Id']].copy()
result['predictions'] = predictions

result.head()
result.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")



