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


# ãƒ‡ãƒ¼ã‚¿èª­ã�¿è¾¼ã�¿
df = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", parse_dates=["pickup_datetime"], nrows=1000000)
display(df)


# 2009å¹´ä»¥é™�ã�«çµ�ã‚‹
df = df[df['pickup_datetime'] >= '2009-01-01']
display(df)


# å¹´ã�”ã�¨ã�«å¹³å�‡é�‹è³ƒã‚’é›†è¨ˆ
df['year'] = df['pickup_datetime'].dt.year
yearly_avg_fare = df.groupby('year')['fare_amount'].mean()
display(yearly_avg_fare)


#å¹´ã�”ã�¨ã�®å¹³å�‡é�‹è³ƒãƒ»å�ˆè¨ˆé‡‘é¡�ãƒ»ä»¶æ•°
yearly_stats = df.groupby('year')['fare_amount'].agg(['mean', 'sum', 'count'])
display(yearly_stats)


# å¤–ã‚Œå€¤ã‚’æ¶ˆã�™
df = df[(df['fare_amount'] > 0) & (df['fare_amount'] < 200)&(df['passenger_count'] > 0) & (df['passenger_count'] <= 6)]
# df = df[(df['passenger_count'] > 0) & (df['passenger_count'] <= 6)]  # ä¸€èˆ¬çš„ã�ªã‚¿ã‚¯ã‚·ãƒ¼ã�®å®šå“¡ç¯„å›²
yearly_stats = df.groupby('year')['fare_amount'].agg(['mean', 'sum', 'count'])
display(yearly_stats)


# å¹´ã�”ã�¨ã�®å¹³å�‡é�‹è³ƒã‚°ãƒ©ãƒ•
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(4,3))

yearly_avg_fare.plot(kind='bar', ax=ax, color='b')
ax.set_title('Yearly Average Taxi Fare (from 2009 to 2015)')
ax.set_xlabel('Year')
ax.set_ylabel('Average Fare ($)')
ax.grid(axis='y')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

plt.tight_layout()
plt.show()



# ä¹—è»Šäººæ•°ã�”ã�¨ã�®å¹³å�‡å€¤ãƒ»ä¸­å¤®å€¤ãƒ»æ¨™æº–å��å·®ãƒ»ä»¶æ•°
fare_stats = df.groupby('passenger_count')['fare_amount'].agg(['mean', 'median', 'std', 'count'])
display(fare_stats)


# ä¹—è»Šäººæ•°ã�”ã�¨ã�®å¹³å�‡é�‹è³ƒ
fare_by_passenger = df.groupby('passenger_count')['fare_amount'].mean()
display(fare_by_passenger)


# ä¹—å®¢äººæ•°ã�”ã�¨ã�®å¹³å�‡é�‹è³ƒ
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(4,3))

fare_by_passenger.plot(kind='bar', ax=ax, color='g')
ax.set_title('Average Fare by Passenger Count')
ax.set_xlabel('Passenger Count')
ax.set_ylabel('Average Fare ($)')
ax.grid(axis='y')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

plt.tight_layout()
plt.show()


# æ—¥æ™‚ã‚’datetimeå�‹ã�«å¤‰æ�›
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')

# æ›œæ—¥ã‚’æŠ½å‡ºï¼ˆ0=æœˆæ›œæ—¥, 6=æ—¥æ›œæ—¥ï¼‰
df['weekday'] = df['pickup_datetime'].dt.dayofweek

# æ›œæ—¥ã�”ã�¨ã�®å¹³å�‡æ–™é‡‘ã‚’è¨ˆç®—
avg_fare_by_weekday = df.groupby('weekday')['fare_amount'].mean()

# æ›œæ—¥ã‚’æ–‡å­—ãƒ©ãƒ™ãƒ«ã�«å¤‰æ�›ï¼ˆã‚ªãƒ—ã‚·ãƒ§ãƒ³ï¼‰
weekday_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
avg_fare_by_weekday.index = [weekday_labels[i] for i in avg_fare_by_weekday.index]

# çµ�æ�œè¡¨ç¤º
display(avg_fare_by_weekday)


# æ›œæ—¥ã�”ã�¨ã�®å¹³å�‡é�‹è³ƒ
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(4,3))

avg_fare_by_weekday.plot(kind='bar', ax=ax, color='r')

ax.set_title('Average Taxi Fare by Weekday')
ax.set_xlabel('Weekday')
ax.set_ylabel('Average Fare ($)')
ax.grid(axis='y')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

plt.tight_layout()
plt.show()


# ãƒ‡ãƒ¼ã‚¿èª­ã�¿è¾¼ã�¿ï¼ˆé�©å®œãƒ•ã‚¡ã‚¤ãƒ«å��ã�¨è¡Œæ•°ã‚’èª¿æ•´ï¼‰
df = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", parse_dates=["pickup_datetime"], nrows=1000000)
df = df.dropna(subset=["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"])

# 1. ãƒ�ãƒ�ãƒ¼ã‚µã‚¤ãƒ³é–¢æ•°ï¼ˆç·¯åº¦ãƒ»çµŒåº¦ã�‹ã‚‰è·�é›¢è¨ˆç®—ï¼‰
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # åœ°ç�ƒã�®å�Šå¾„ï¼ˆkmï¼‰
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
    a = np.nan_to_num(a, nan=0.0)  # NaNã‚’0ã�«å¤‰æ�›
    a = np.clip(a, 0, 1)
    return 2 * R * np.arcsin(np.sqrt(a))

# 2. è·�é›¢è¨ˆç®—
df["distance_km"] = haversine(
    df["pickup_latitude"], df["pickup_longitude"],
    df["dropoff_latitude"], df["dropoff_longitude"]
)



df = df[
    (df["distance_km"] > 0) & (df["distance_km"] <= 100) &
    (df["fare_amount"] > 0) & (df["fare_amount"] <= 200)
]

# 3. è·�é›¢ã‚’1kmå�˜ä½�ã�«ä¸¸ã‚�ã�¦ã‚°ãƒ«ãƒ¼ãƒ—åŒ–
df["distance_bin"] = df["distance_km"].round().astype(int)

# display(df)

# 4. å�„è·�é›¢ã�”ã�¨ã�®å¹³å�‡æ–™é‡‘ã‚’è¨ˆç®—
fare_by_distance = df.groupby("distance_bin")["fare_amount"].mean().reset_index()

# 5. çµ�æ�œè¡¨ç¤º
display(fare_by_distance)


import matplotlib.pyplot as plt
df_plot = df[(df["distance_km"] <= 2) & (df["fare_amount"] < 100)|
             (df["distance_km"] > 2) &(df["distance_km"] <= 40) & (df["fare_amount"] > 0)|
             (df["distance_km"] > 40) &(df["distance_km"] <= 110) & (df["fare_amount"] > 55)]

df_plot = df_plot[(df_plot["distance_km"] >= 2) & (df_plot["fare_amount"] > 0)& (df_plot["fare_amount"] < 200)]


fig, ax = plt.subplots(figsize=(4, 3))
# ax.plot(fare_by_distance["distance_bin"], fare_by_distance["fare_amount"], marker='o',linestyle='None')
# ax.plot(df["distance_km"], df["fare_amount"], marker='o',linestyle='None')
ax.plot(df_plot["distance_km"], df_plot["fare_amount"], marker='o',linestyle='None')
ax.set_title("Average Fare by Distance (0-100 km)")
ax.set_xlabel("Distance (km)")
ax.set_ylabel("Average Fare ($)")
ax.grid(True)
ax.set_xlim(0, 80)


plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(4, 3))

ax.plot(fare_by_distance["distance_bin"], fare_by_distance["fare_amount"], marker='o',linestyle='None')
ax.set_title("Average Fare by Distance (0-27 km)")
ax.set_xlabel("Distance (km)")
ax.set_ylabel("Average Fare ($)")
ax.grid(True)
ax.set_xlim(0, 27)
ax.set_ylim(0, 80)

plt.tight_layout()
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
print('ğŸ‘Œ')


train_df = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", nrows=1000000)  # 1Mè¡Œã� ã�‘ä½¿ç”¨ï¼ˆãƒ¡ãƒ¢ãƒªç¯€ç´„ï¼‰
test_df = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import mean_squared_error


# æ—¥æ™‚å‡¦ç�†
train_df["pickup_datetime"] = pd.to_datetime(train_df["pickup_datetime"])
train_df["hour"] = train_df["pickup_datetime"].dt.hour
train_df["day"] = train_df["pickup_datetime"].dt.dayofweek
train_df["year"]=train_df["pickup_datetime"].dt.year

#è·�é›¢ã‚’è¿½åŠ 
train_df["distance_km"] = haversine(
    train_df["pickup_latitude"], train_df["pickup_longitude"],
    train_df["dropoff_latitude"], train_df["dropoff_longitude"]
)

# æ¬ æ��å‰Šé™¤ãƒ»ç•°å¸¸é™¤å¤–
# train_df = train_df.dropna()
train_df = train_df[(train_df["distance_km"] <= 2) & (train_df["fare_amount"] < 100)|
             (train_df["distance_km"] > 2) &(train_df["distance_km"] <= 40) & (train_df["fare_amount"] > 0)|
             (train_df["distance_km"] > 40) &(train_df["distance_km"] <= 110) & (train_df["fare_amount"] > 55)]

train_df = train_df[(train_df["distance_km"] >= 2) & (train_df["fare_amount"] > 0)& (train_df["fare_amount"] < 200)]

# display(train_df)

# ç‰¹å¾´é‡�é�¸æŠ�
# features = ["distance_km", "pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]
features = ["distance_km","pickup_latitude", "pickup_longitude","dropoff_latitude", "dropoff_longitude",'day','hour','year']
X = train_df[features]
y = train_df["fare_amount"]


#å­¦ç¿’
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=17)
# RandomForestã�«ã‚ˆã‚‹ãƒ¢ãƒ‡ãƒ«
model = RandomForestRegressor(n_estimators=100, random_state=17)

model.fit(X_train, y_train)

# æ¤œè¨¼ã‚¹ã‚³ã‚¢
val_preds = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_preds))
print(f"Validation RMSE: {rmse:.2f}") 


# ç‰¹å¾´é‡�é‡�è¦�åº¦ã�®è¡¨ç¤º
import matplotlib.pyplot as plt

importances = model.feature_importances_
feature_importance_df = pd.DataFrame({"Feature": features, "Importance": importances})
feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)
fig, ax = plt.subplots(figsize=(4, 3))
ax.barh(feature_importance_df["Feature"], feature_importance_df["Importance"])
ax.set_title("Feature Importance in Random Forest")
plt.gca().invert_yaxis()
ax.set_xlabel("Feature Importance")

plt.tight_layout()
plt.show()


# ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿ã�«å�Œæ§˜ã�®ç‰¹å¾´è¿½åŠ 
test_df["pickup_datetime"] = pd.to_datetime(test_df["pickup_datetime"])
test_df["hour"] = test_df["pickup_datetime"].dt.hour
test_df["day"] = test_df["pickup_datetime"].dt.dayofweek
test_df["year"]=test_df["pickup_datetime"].dt.year

#è·�é›¢ã�®å‡¦ç�†
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # km
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))
test_df["distance_km"] = haversine_distance(
    test_df["pickup_latitude"],
    test_df["pickup_longitude"],
    test_df["dropoff_latitude"],
    test_df["dropoff_longitude"]
)

X_test = test_df[features]

# äºˆæ¸¬ã�¨ä¿�å­˜
test_df["fare_amount"] = model.predict(X_test)
submission = test_df[["key", "fare_amount"]]
submission.to_csv("submission.csv", index=False)




