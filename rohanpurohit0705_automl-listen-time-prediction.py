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


train_df = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


train_df.head()


test_df.head()


import seaborn as sns
import matplotlib.pyplot as plt


train_df["Episode_Length_minutes"].fillna(train_df["Episode_Length_minutes"].median(), inplace=True)
train_df["Guest_Popularity_percentage"].fillna(train_df["Guest_Popularity_percentage"].median(), inplace=True)
train_df["Number_of_Ads"].fillna(train_df["Number_of_Ads"].median(), inplace=True)


categorical_cols = ["Podcast_Name", "Episode_Title", "Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment"]
for col in categorical_cols:
    train_df[col] = train_df[col].astype("category").cat.codes
    test_df[col] = test_df[col].astype("category").cat.codes



fig, axes = plt.subplots(2, 3, figsize=(15, 10))
num_features = ["Episode_Length_minutes", "Host_Popularity_percentage", "Guest_Popularity_percentage",
                "Number_of_Ads", "Listening_Time_minutes"]

for i, col in enumerate(num_features):
    sns.histplot(train_df[col], bins=50, kde=True, ax=axes[i // 3, i % 3])
    axes[i // 3, i % 3].set_title(col)

plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
sns.heatmap(train_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()


plt.figure(figsize=(8, 5))
sns.scatterplot(data=train_df, x="Episode_Length_minutes", y="Listening_Time_minutes", alpha=0.5)
plt.title("Episode Length vs Listening Time")
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


test_df["Episode_Length_minutes"].fillna(test_df["Episode_Length_minutes"].median(), inplace=True)
test_df["Guest_Popularity_percentage"].fillna(test_df["Guest_Popularity_percentage"].median(), inplace=True)
test_df["Number_of_Ads"].fillna(test_df["Number_of_Ads"].median(), inplace=True)


num_cols = ["Episode_Length_minutes", "Host_Popularity_percentage", "Guest_Popularity_percentage", "Number_of_Ads"]
scaler = StandardScaler()
train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
test_df[num_cols] = scaler.transform(test_df[num_cols])


!apt-get install default-jre
!pip install h2o


import h2o
from h2o.automl import H2OAutoML
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


h2o.init()
train_df = train_df.drop(columns=["id"], errors="ignore")
test_df = test_df.drop(columns=["id"], errors="ignore")
h2o_train = h2o.H2OFrame(train_df)
h2o_test = h2o.H2OFrame(test_df)
target = "Listening_Time_minutes"
h2o_train[target] = h2o_train[target].asnumeric()


aml = H2OAutoML(max_models=20, seed=42)
aml.train(y=target, training_frame=h2o_train)

lb = aml.leaderboard
print(lb.head())

best_model = aml.leader

train_pred = best_model.predict(h2o_train).as_data_frame().values.flatten()

y_true = train_df[target].values
mae = mean_absolute_error(y_true, train_pred)
rmse = mean_squared_error(y_true, train_pred, squared=False)
r2 = r2_score(y_true, train_pred)

print(f"✅ MAE: {mae:.4f}")
print(f"✅ RMSE: {rmse:.4f}")
print(f"✅ R² Score: {r2:.4f}")




