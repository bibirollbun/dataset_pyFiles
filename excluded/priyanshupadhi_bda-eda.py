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


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Sample submission shape:", sample_submission.shape)


def reduce_mem_usage(df):
    for col in df.columns:
        if df[col].dtype != object:
            col_type = df[col].dtype
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                else:
                    df[col] = df[col].astype(np.int32)
            else:
                df[col] = df[col].astype(np.float32)
    return df

train = reduce_mem_usage(train)
test = reduce_mem_usage(test)


sns.countplot(x='rainfall', data=train, palette='Set2')
plt.title("Rainfall Class Distribution")
plt.xticks([0, 1], ['No Rain', 'Rain'])
plt.show()

print(train['rainfall'].value_counts(normalize=True))


grouped_stats = train.groupby('rainfall').agg(['mean', 'std', 'min', 'max'])
display(grouped_stats.T)


features = [col for col in train.columns if col not in ['id', 'rainfall']]
sampled_train = train.sample(frac=0.1, random_state=42)

for feature in features:
    plt.figure()
    sns.kdeplot(data=sampled_train, x=feature, hue='rainfall', fill=True)
    plt.title(f"Distribution of {feature} by Rainfall")
    plt.show()


corr = train[features + ['rainfall']].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


skew_kurtosis = train[features].agg(['skew', 'kurtosis']).T
display(skew_kurtosis)


# ðŸ“¦ Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")

test_ids = test['id']
train.drop(columns=['id'], inplace=True)
test.drop(columns=['id'], inplace=True)

X = train.drop(columns=['rainfall'])
y = train['rainfall']

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
test_imputed = imputer.transform(test)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
test_scaled = scaler.transform(test_imputed)

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    early_stopping_rounds=20
)

model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

test_preds = model.predict(test_scaled)

submission = pd.DataFrame({'id': test_ids, 'rainfall': test_preds})
submission.to_csv("submission.csv", index=False)




