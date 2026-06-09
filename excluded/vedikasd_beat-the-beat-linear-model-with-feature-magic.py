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


#load data
train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")


print("train:" ,train.shape)
print("test: ",test.shape)


train.head()


train.columns


train.value_counts('BeatsPerMinute')


# Compute correlations with target
corr = train.corr(numeric_only=True)['BeatsPerMinute'].sort_values(ascending=False)
print(corr)


# Heatmap of all correlations

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
sns.heatmap(train.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()


# Add feature: log of TrackDurationMs
train["log_duration"] = np.log1p(train["TrackDurationMs"])  
test["log_duration"] = np.log1p(test["TrackDurationMs"])

# Add interaction features
train["energy_rhythm"] = train["Energy"] * train["RhythmScore"]
test["energy_rhythm"] = test["Energy"] * test["RhythmScore"]

train["mood_vocal"] = train["MoodScore"] * train["VocalContent"]
test["mood_vocal"] = test["MoodScore"] * test["VocalContent"]

train["loudness_sq"] = train["AudioLoudness"] ** 2
test["loudness_sq"] = test["AudioLoudness"] ** 2


train.columns


from sklearn.model_selection import train_test_split
# Features and target
X = train.drop(columns=['id', 'BeatsPerMinute'])
y = train['BeatsPerMinute']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

# Train
pipeline.fit(X_train, y_train)


pipeline.fit(X, y)


X_test = test.drop(columns=['id'])


preds = pipeline.predict(X_test)


submission = pd.DataFrame({
    'id': test['id'],
    'BeatsPerMinute': preds
})
submission.to_csv("submission.csv", index=False)
print("Submission file saved!")

