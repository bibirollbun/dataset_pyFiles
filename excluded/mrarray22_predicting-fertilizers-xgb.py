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

train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
train.head()


print(train.info())
print(train.shape)


train.describe().T


print(test.shape)
print(test.info())


test.describe().T


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(14,8))
sns.countplot(data=train, y='Fertilizer Name', order=train['Fertilizer Name'].value_counts().index)
plt.title('Distribution of Fertilizer Name')
plt.ylabel('Name')
plt.xlabel('Count')
plt.tight_layout()
plt.show()


import numpy as np
numeric_cols = train.select_dtypes(include=np.number).columns.drop('id')


train[numeric_cols].hist(figsize=(12,8),bins=30)
plt.suptitle('Distrbution of Features')
plt.tight_layout()
plt.show()


grouped_stats = train.groupby('Fertilizer Name')[numeric_cols].mean().T
grouped_stats.plot(kind='bar', figsize=(16, 8))
plt.title("Mean of Features Values by Fertilizer")
plt.ylabel("Mean Value")
plt.xlabel("Feature")
plt.tight_layout()
plt.show()


numerical_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']


for feature in numerical_features:
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=train, x='Fertilizer Name', y=feature)
    plt.title(f'{feature} by Fertilizer')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


plt.figure(figsize=(10, 5))
sns.countplot(data=train, x='Soil Type', hue='Fertilizer Name')
plt.title("Fertilizer Distribution by Soil Type")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


plt.figure(figsize=(14, 6))
sns.countplot(data=train, x='Crop Type', hue='Fertilizer Name')
plt.title("Fertilizer Distribution by Crop Type")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA



df = train.copy()
# Encoding the soil and crop type
df['Soil Type'] = LabelEncoder().fit_transform(df['Soil Type'])
df['Crop Type'] = LabelEncoder().fit_transform(df['Crop Type'])


X = df[numerical_features + ['Soil Type', 'Crop Type']]
y = df['Fertilizer Name'] # target variable


# Standaralization
X_scaled = StandardScaler().fit_transform(X)


X_scaled


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


X_pca


pca_df = pd.DataFrame(X_pca, columns=['Pc1', 'Pc2'])
pca_df['Fertilizer Name'] = y

plt.figure(figsize=(10, 6))
sns.scatterplot(data=pca_df, x='Pc1', y='Pc2', hue='Fertilizer Name', palette='tab10')
plt.title("PCA")
plt.tight_layout()
plt.show()


pivot = train.groupby(['Crop Type', 'Soil Type'])['Fertilizer Name'] \
             .agg(lambda x: x.value_counts().index[0]) \
             .unstack()
dummy = pivot.notnull().astype(int)
plt.figure(figsize=(12, 8))
ax = sns.heatmap(dummy, annot=pivot, fmt='', cmap='Blues', cbar=False, linewidths=0.5, linecolor='gray')
plt.title('Most Used Fertilizer by Crop and Soil Type')
plt.ylabel('Crop Type')
plt.xlabel('Soil Type')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


from math import pi
from sklearn.preprocessing import MinMaxScaler


features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
fea_data = train.groupby('Fertilizer Name')[features].mean().reset_index()
# categories = features
N = len(features)


scaler = MinMaxScaler()
fea_scaled = fea_data.copy()
fea_scaled[features] = scaler.fit_transform(fea_scaled[features])


for i in range(len(fea_scaled)):
    values = fea_scaled.iloc[i, 1:].tolist()
    values += values[:1]
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    plt.xticks(angles[:-1], features)
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=fea_scaled['Fertilizer Name'][i])
    ax.fill(angles, values, alpha=0.25)
    plt.title(f"Average Profile (Normalized): {fea_scaled['Fertilizer Name'][i]}")
    plt.tight_layout()
    plt.show()


from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


le = LabelEncoder()
train['Fertilizer Encoded'] = le.fit_transform(train['Fertilizer Name'])


X = pd.get_dummies(train[features])
y = train['Fertilizer Encoded']


model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X, y)


X_test = pd.get_dummies(test[features])
X_test = X_test.reindex(columns=X.columns, fill_value=0)


probs = model.predict_proba(X_test)
top_3 = np.argsort(probs, axis=1)[:, -3:][:, ::-1]


preds = [' '.join(le.inverse_transform(row)) for row in top_3]


submission = pd.DataFrame({
    'id': test['id'],
    'Fertilizer Name': preds
})
submission.to_csv('submission.csv', index=False)




