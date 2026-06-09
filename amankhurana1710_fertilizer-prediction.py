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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df_train=pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
df_train.head()


fertypes=df_train['Fertilizer Name'].value_counts()
plt.bar(fertypes.index,fertypes.values)
plt.xlabel('Fertilizer Type')
plt.ylabel('Total Count')
plt.show()


cropcount=df_train['Crop Type'].value_counts()
plt.bar(cropcount.index,cropcount.values)
plt.xlabel('Crop Type')
plt.ylabel('Total Count')
plt.xticks(rotation=45)
plt.show()


df_train.describe()


df_train.drop('id',inplace=True,axis=1)


df_train.info()


df_train.isnull().sum()


df_train.shape


soilcount=df_train['Soil Type'].value_counts()
plt.bar(soilcount.index,soilcount.values)
plt.xlabel('Soil Type')
plt.ylabel('Total Count')
plt.xticks(rotation=45)
plt.show()


df_train.corr(numeric_only=True).style.background_gradient(cmap='coolwarm')


df_test=pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
df_test.drop('id',inplace=True,axis=1)
df_test.head()


df_train['Soil Type'].value_counts()


df_test['Soil Type'].value_counts()


cat_cols=['Soil Type','Crop Type']
encoders={}


from sklearn.preprocessing import LabelEncoder
target_le = LabelEncoder()
df_train['Fertilizer Name'] = target_le.fit_transform(df_train['Fertilizer Name'])

# Store class names for later decoding
class_names = target_le.classes_
class_names


for col in cat_cols:
    le=LabelEncoder()
    df_train[col]=le.fit_transform(df_train[col])
    df_test[col] = le.transform(df_test[col])
    encoders[col] = le


df_test.shape


df_train.shape


from sklearn.model_selection import train_test_split


from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb


X = df_train.drop('Fertilizer Name', axis=1)
y = df_train['Fertilizer Name']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')  # for multiclass
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


xgb.plot_importance(model)
plt.show()


model.predict(X_test.iloc[[1]])[0]


label = target_le.inverse_transform(y_pred)


label


y_pred


def mapk(actual, predicted, k=3):
    """
    Computes the mean average precision at k.
    
    Parameters:
        actual (list): List of actual target values.
        predicted (list of lists): Each sublist is a list of predicted values (top k predictions).
        k (int): The maximum number of predicted elements.

    Returns:
        score (float): The MAP@k score.
    """
    def apk(a, p, k):
        if len(p) > k:
            p = p[:k]

        score = 0.0
        num_hits = 0.0

        for i, pred in enumerate(p):
            if pred == a and pred not in p[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
                break  # since we only care about the first correct prediction

        return score

    return sum(apk(a, p, k) for a, p in zip(actual, predicted)) / len(actual)



y_proba = model.predict_proba(X_test)

# Get indices of top-3 predicted classes
top_3_preds_idx = np.argsort(y_proba, axis=1)[:, -3:][:, ::-1]

# Convert indices back to labels
top_3_preds_labels = [
    target_le.inverse_transform(row).tolist() for row in top_3_preds_idx
]


actual_labels = target_le.inverse_transform(y_test)


score = mapk(actual=actual_labels, predicted=top_3_preds_labels, k=3)
print("MAP@3 Score:", score)




