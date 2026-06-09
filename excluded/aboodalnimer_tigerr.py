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


train_data = pd.read_csv(r'/kaggle/input/cat-in-the-dat-ii/train.csv')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk


train_data.head()


train_data.describe()



sns.countplot(x='target', data=train_data, palette='pastel')
plt.title('Target Class Distribution', fontsize=14)
plt.xlabel('Target')
plt.ylabel('Count')


train_data.drop('id', axis=1, inplace=True)



train_data.head()


for col in train_data.columns:
    print(f"\nColumn: {col}")
    print(train_data[col].unique())


binary_cols = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4']

plt.figure(figsize=(12, 6))

for i, col in enumerate(binary_cols, 1):
    plt.subplot(1, len(binary_cols), i)
    train_data[col].value_counts(dropna=False).plot.pie(
        autopct='%1.1f%%',
        startangle=90,
        colors=['#66b3ff', '#99ff99', '#ffcc99'],  # Added color for NaN
        textprops={'color': 'black', 'fontsize': 10}
    )
    plt.title(col, color='black', fontsize=12, weight='bold')
    plt.ylabel('')

plt.suptitle('Pie Chart of Binary Features (0 vs 1 vs NaN)', color='white', fontsize=16, weight='bold')
plt.tight_layout()
plt.show()



train_data.fillna("missing", inplace=True)


import category_encoders as ce

encoder = ce.BinaryEncoder()
binary_encoded_data = encoder.fit_transform(train_data)

print(binary_encoded_data.head())


%pip install catboost



from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split

X = train_data.drop('target', axis=1)
y = train_data['target']

cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

X[cat_features] = X[cat_features].astype(str)
X[cat_features] = X[cat_features].fillna('missing')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_pool = Pool(X_train, label=y_train, cat_features=cat_features)
test_pool = Pool(X_test, label=y_test, cat_features=cat_features)

model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=42,
    verbose=100
)

model.fit(train_pool, eval_set=test_pool)


test = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/test.csv")

test[cat_features] = test[cat_features].astype(str)
test[cat_features] = test[cat_features].fillna('missing')

test_pool = Pool(test, cat_features=cat_features)

# --------------------------
# 5. Predict probabilities
# --------------------------
test_pred = model.predict_proba(test_pool)[:, 1]

# --------------------------
# 6. Save submission file
# --------------------------
submission = pd.DataFrame({
    "id": test["id"],
    "target": test_pred
})

submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)

print(f"✅ Submission file saved successfully at: {submission_path}")

# --------------------------
# 7. Submit to Kaggle
# --------------------------
#!kaggle competitions submit -c cat-in-the-dat-ii -f /kaggle/working/submission.csv -m "CatBoost baseline submission"




