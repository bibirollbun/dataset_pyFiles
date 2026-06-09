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


train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')



from IPython.display import display

display(train.head().style.background_gradient(cmap='coolwarm'))



print("Train data shape:", train.shape)
print("Test data shape:", test.shape)


stats_df = train.describe()
display(stats_df.style.background_gradient(cmap='coolwarm').format('{:.4f}'))



import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.figure(figsize=(10,5))
sns.histplot(train["BeatsPerMinute"], bins=50, kde=True, color="blue")
plt.title("Distribution of BeatsPerMinute")
plt.show()

print(train["BeatsPerMinute"].describe())
print("Skewness:", train["BeatsPerMinute"].skew())
print("Kurtosis:", train["BeatsPerMinute"].kurt())



num_features = ["RhythmScore", "AudioLoudness", "VocalContent", "AcousticQuality", 
                "InstrumentalScore", "LivePerformanceLikelihood", "MoodScore", 
                "TrackDurationMs", "Energy"]

for col in num_features:
    plt.figure(figsize=(14,4))
    
    plt.subplot(1,2,1)
    sns.histplot(train[col], kde=True, bins=40)
    plt.title(f"Distribution of {col}")
    
    plt.subplot(1,2,2)
    sns.scatterplot(x=train[col], y=train["BeatsPerMinute"], alpha=0.3)
    plt.title(f"{col} vs BeatsPerMinute")
    
    plt.show()




corr = train.drop("id", axis=1).corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")
plt.show()

corr_target = corr["BeatsPerMinute"].drop("BeatsPerMinute").sort_values(ascending=False)
print("Correlation with BPM:")
print(corr_target)




top_features = corr_target.abs().sort_values(ascending=False).head(4).index.tolist()

sns.pairplot(train[top_features + ["BeatsPerMinute"]], diag_kind="kde")
plt.show()




num_features = [col for col in train.columns if col not in ["id", "BeatsPerMinute"]]

fig, axes = plt.subplots(len(num_features)//5 + 1, 5, figsize=(20, 3*len(num_features)//5))
axes = axes.flatten()

for i, col in enumerate(num_features):
    sns.boxplot(x=train[col], ax=axes[i])
    axes[i].set_title(col)

# remove empty plots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()




num_features = [col for col in train.columns if col not in ["id", "BeatsPerMinute"]]

rows = len(num_features) // 4 + 1   
fig, axes = plt.subplots(rows, 4, figsize=(20, 4*rows))
axes = axes.flatten()

for i, col in enumerate(num_features):
    sns.kdeplot(train[col], label="Train", shade=True, ax=axes[i])
    sns.kdeplot(test[col], label="Test", shade=True, ax=axes[i])
    axes[i].set_title(f"{col} (Train vs Test)")
    axes[i].legend()

# remove empty subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.figure(figsize=(14,6))

# Raw TrackDurationMs vs BPM
plt.subplot(1,2,1)
sns.scatterplot(x=train["TrackDurationMs"], y=train["BeatsPerMinute"], alpha=0.3)
plt.title("Raw TrackDurationMs vs BeatsPerMinute")

# Log-transformed TrackDurationMs vs BPM
plt.subplot(1,2,2)
sns.scatterplot(x=np.log1p(train["TrackDurationMs"]), y=train["BeatsPerMinute"], alpha=0.3, color="green")
plt.title("Log(TrackDurationMs) vs BeatsPerMinute")

plt.tight_layout()
plt.show()




import numpy as np

corr_raw = train["TrackDurationMs"].corr(train["BeatsPerMinute"])
corr_log = np.log1p(train["TrackDurationMs"]).corr(train["BeatsPerMinute"])

print("Correlation (Raw):", corr_raw)
print("Correlation (Log):", corr_log)






