# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')


df.shape


df.head()


df.sample(5)


df.info()


df.isnull().sum()


df.describe()


df.duplicated().sum()


df.corr()


df.corr()['BeatsPerMinute']


sns.histplot(x='BeatsPerMinute', data=df, kde=True)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df['BeatsPerMinute'] = scaler.fit_transform(df[['BeatsPerMinute']])


sns.histplot(x='TrackDurationMs', data=df, kde=True, color='red', label='Energy')
plt.show()


df.std()


df.hist(figsize=(12,10))
plt.show()



df['TrackDurationMs'] = scaler.fit_transform(df[['TrackDurationMs']])


df.std()


df.drop(columns='id')


sns.boxplot(x='RhythmScore', data=df)


from scipy.stats.mstats import winsorize

df["RhythmScore"] = winsorize(df["RhythmScore"], limits=[0.01, 0.01])



sns.boxplot(x='RhythmScore', data=df)


df['BeatsPerMinute']


corr = df.corr()
corr_target = corr["BeatsPerMinute"].sort_values(ascending=False)
print(corr_target)


import matplotlib.pyplot as plt

features = ["MoodScore", "TrackDurationMs", "RhythmScore", "VocalContent", 
            "LivePerformanceLikelihood", "InstrumentalScore", 
            "AcousticQuality", "AudioLoudness", "Energy"]

for col in features:
    plt.figure(figsize=(6,4))
    plt.scatter(df[col], df["BeatsPerMinute"], alpha=0.3)
    plt.xlabel(col)
    plt.ylabel("BeatsPerMinute")
    plt.title(f"{col} vs BeatsPerMinute")
    plt.show()



df["BPM_bin"] = pd.qcut(df["BeatsPerMinute"], 3, labels=["Low", "Medium", "High"])

for col in ["MoodScore", "TrackDurationMs", "RhythmScore", "VocalContent", 
            "LivePerformanceLikelihood", "InstrumentalScore", 
            "AcousticQuality", "AudioLoudness", "Energy"]:
    plt.figure(figsize=(6,4))
    for bpm_level in df["BPM_bin"].unique():
        sns.kdeplot(df[df["BPM_bin"] == bpm_level][col], label=bpm_level, fill=True)
    plt.title(f"{col} distribution across BPM groups")
    plt.legend()
    plt.show()



import seaborn as sns
sns.pairplot(df[["BeatsPerMinute","MoodScore", "TrackDurationMs", "RhythmScore", "VocalContent", 
            "LivePerformanceLikelihood", "InstrumentalScore", 
            "AcousticQuality", "AudioLoudness", "Energy"]])
plt.show()



from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state=42)
x = df.drop(columns=["BeatsPerMinute", "id","BPM_bin"])
y = df["BeatsPerMinute"]
model.fit(x,y)


importances = pd.Series(model.feature_importances_, index=x.columns)
print(importances.sort_values(ascending=False))


test.head()


x_test = test.drop(columns=['id'])


test_pred = model.predict(x_test)


submission = pd.DataFrame({
    "id":test['id'],
    "BeatsPerMinute": test_pred
})
submission.to_csv("submission.csv",index=False)




