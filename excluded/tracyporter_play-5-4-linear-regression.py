import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pylab 
import scipy.stats as stats

import matplotlib.pyplot as plt
import seaborn as sns


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")


train


train.info()


train.isnull().sum()


test


test.isnull().sum()


submission


train.drop(columns=['Episode_Length_minutes','Guest_Popularity_percentage'], inplace=True)
test.drop(columns=['Episode_Length_minutes','Guest_Popularity_percentage'], inplace=True)

train.isnull().sum().sum(), test.isnull().sum().sum()


train['Number_of_Ads'] = train['Number_of_Ads'].fillna(train['Number_of_Ads'].mean())

train.isna().sum().sum(), test.isna().sum().sum()


# Plotting a basic histogram
plt.hist(train['Listening_Time_minutes'], bins=50, color='skyblue', edgecolor='black')
 
# Adding labels and title
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Listening time Histogram')
 
# Display the plot
plt.show()


# Plotting a basic histogram
plt.hist(train['Host_Popularity_percentage'], bins=50, color='skyblue', edgecolor='black')
 
# Adding labels and title
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Host popularty percentage Histogram')
 
# Display the plot
plt.show()


# Plotting a basic histogram
plt.hist(train['Genre'], bins=50, color='skyblue', edgecolor='black')
 
# Adding labels and title
plt.xlabel('Values')
plt.xticks(rotation=45)
plt.ylabel('Frequency')
plt.title('Genre Histogram')
 
# Display the plot
plt.show()


# Plotting a basic histogram
plt.hist(train['Publication_Day'], bins=50, color='skyblue', edgecolor='black')
 
# Adding labels and title
plt.xlabel('Values')
plt.xticks(rotation=45)
plt.ylabel('Frequency')
plt.title('Publication day Histogram')
 
# Display the plot
plt.show()


# Plotting a basic histogram
plt.hist(train['Publication_Time'], bins=50, color='skyblue', edgecolor='black')
 
# Adding labels and title
plt.xlabel('Values')
plt.xticks(rotation=45)
plt.ylabel('Frequency')
plt.title('Publication time Histogram')
 
# Display the plot
plt.show()


# Plotting a basic histogram
plt.hist(train['Episode_Sentiment'], bins=50, color='skyblue', edgecolor='black')
 
# Adding labels and title
plt.xlabel('Values')
plt.xticks(rotation=45)
plt.ylabel('Frequency')
plt.title('Episode sentiment Histogram')
 
# Display the plot
plt.show()


enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

for col in train:
    if train[col].dtype == 'object':
        train[col] = enc.fit_transform(train[col].values.reshape(-1,1))
        test[col] = enc.transform(test[col].values.reshape(-1,1))


corr = train.corr()
sns.heatmap(corr)


y = train.pop('Listening_Time_minutes')
X = train
X_test = test


X_train, X_val, y_train, y_val = train_test_split( X, y, test_size=0.1, shuffle=True, random_state=42)
X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape


model = LinearRegression().fit(X_train, y_train)
model.score(X_train, y_train)


y_pred = model.predict(X_val)
y_pred


mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
rmse


# Plotting a basic histogram
plt.hist(y_pred, bins=50, color='skyblue', edgecolor='black')
 
# Adding labels and title
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('y_pred Histogram')
 
# Display the plot
plt.show()


df = pd.DataFrame({'y_val':y_val, 'y_pred':y_pred})
df


stats.probplot(y_pred, dist="norm", plot=pylab)
pylab.show()


fig, ax = plt.subplots()
ax.scatter(y_val, y_pred, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()


pred = model.predict(X_test)
pred


submission['Listening_Time_minutes'] = pred
submission.to_csv('submission.csv', index=False)
submission = pd.read_csv('submission.csv')
submission

