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
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder


train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")


train["Fertilizer Name"].value_counts()


train['Crop Type'].value_counts()


train['Soil Type'].value_counts()



fertilizer_encoder = LabelEncoder()
train['Fertilizer Name Enc'] = fertilizer_encoder.fit_transform(train['Fertilizer Name'])


train.head()


train.describe()


plt.figure(figsize=(10, 6))
sns.countplot(data=train, x='Fertilizer Name', palette="viridis")
plt.title("Distribution of Fertilizer Classes", fontsize=14)
plt.xlabel("Fertilizer Class", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))
sns.countplot(data=train, x='Soil Type', hue='Fertilizer Name')
plt.title("Fertilizer by Soil Type", fontsize=14)
plt.xlabel("Soil Type", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.legend(title='Fertilizer')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))
sns.countplot(data=train, x='Crop Type', hue='Fertilizer Name')
plt.title("Fertilizer by Crop Type", fontsize=14)
plt.xlabel("Crop Type", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.legend(title='Fertilizer')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


numeric_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for i, col in enumerate(numeric_cols):
    sns.histplot(train[col], ax=axes[i//3, i%3], kde=True, bins=30)
plt.suptitle("Distribution of Numeric Features", fontsize=16)
plt.tight_layout()
plt.show()


fert_list = train['Fertilizer Name'].unique()

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

sns.boxplot(data=train, x='Fertilizer Name', y='Nitrogen', ax=axes[0])
sns.boxplot(data=train, x='Fertilizer Name', y='Phosphorous', ax=axes[1])
sns.boxplot(data=train, x='Fertilizer Name', y='Potassium', ax=axes[2])

plt.suptitle("N-P-K Distribution by Fertilizer", fontsize=16)
plt.tight_layout()
plt.show()


numeric_df = train.select_dtypes(include=[np.number]).dropna()

corr = numeric_df.corr(method='pearson')

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Matrix", fontsize=14)
plt.tight_layout()
plt.show()


mean_npk = train.groupby('Fertilizer Name')[['Nitrogen', 'Phosphorous', 'Potassium']].mean().reset_index()

mean_npk_norm = mean_npk.set_index('Fertilizer Name')
mean_npk_norm = (mean_npk_norm - mean_npk_norm.min()) / (mean_npk_norm.max() - mean_npk_norm.min())
mean_npk_norm.reset_index(inplace=True)

categories = ['Nitrogen', 'Phosphorous', 'Potassium']
N = len(categories)

angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

for i, row in mean_npk_norm.iterrows():
    values = mean_npk_norm.iloc[i].loc[categories].tolist()
    values += values[:1]  
    ax.plot(angles, values, label=row['Fertilizer Name'])
    ax.fill(angles, values, alpha=0.25)

ax.set_xticks(angles[:-1]) 
ax.set_xticklabels(categories)
ax.set_yticklabels([])

plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.title("Normalized N-P-K Profile by Fertilizer", size=16, color="navy", y=1.1)
plt.show()


columns = [
    
    "Temparature",
    "Humidity",
    "Moisture",
    "Nitrogen",
    "Potassium",
    "Phosphorous",
    "Fertilizer Name Enc"
]

train_df = train[columns]
correlation = train_df.corr(method='pearson')


categorical_cols = ["Soil Type", "Crop Type"]
for col in categorical_cols:
    train[col] = train[col].astype("category")
    test[col] = test[col].astype("category")


X = train.drop(columns=["id", "Fertilizer Name", "Fertilizer Name Enc"])
y = train["Fertilizer Name Enc"]

X_test = test.drop(columns=["id"])


model = XGBClassifier(
    enable_categorical=True,
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric="mlogloss"
)

model.fit(X, y)

probs = model.predict_proba(X_test)


top3_indices = np.argsort(probs, axis=1)[:, ::-1][:, :3]

top3_labels = [
    fertilizer_encoder.inverse_transform(row)
    for row in top3_indices
]

submission = pd.DataFrame({
    "id": test["id"],
    "Fertilizer Name": [" ".join(row) for row in top3_labels]
})


submission.to_csv("submission.csv", index=False)

