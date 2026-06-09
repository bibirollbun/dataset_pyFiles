# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
from scipy import stats

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df_train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
df_sample = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")


df_train.head()


df_train.dtypes


df_test.head(1)


df_test.dtypes


df_sample.head(1)


df_train = df_train.drop(columns=['id'])


label_encoders = {}
categorical_cols = ['Soil Type', 'Crop Type', 'Fertilizer Name']

for col in categorical_cols:
    le = LabelEncoder()
    df_train[col] = le.fit_transform(df_train[col])
    label_encoders[col] = le


z_scores = np.abs(stats.zscore(df_train['Fertilizer Name']))
print(z_scores)
df_train = df_train[z_scores < 3]


X = df_train.drop(columns=['Fertilizer Name'])
y = df_train['Fertilizer Name']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = XGBClassifier(
    objective='multi:softprob',  # çok sınıflı sınıflandırma
    num_class=7,                # 7 farklı gübre türü olduğunu söyledin
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42
)


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


with open('xgb_fertilizer_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)


ids = df_test['id']


with open('xgb_fertilizer_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)


for col in ['Soil Type', 'Crop Type']:
    df_test[col] = label_encoders[col].transform(df_test[col])



X_test = df_test.drop(columns=['id'])



probas = model.predict_proba(X_test)


top_3_indices = np.argsort(probas, axis=1)[:, -3:][:, ::-1]


top_3_labels = [
    label_encoders['Fertilizer Name'].inverse_transform(row)
    for row in top_3_indices
]


top_3_strs = [" ".join(preds) for preds in top_3_labels]



# Submission DataFrame
submission_df = pd.DataFrame({
    'id': ids,
    'Fertilizer Name': top_3_strs
})

submission_df.to_csv("submission.csv", index=False)

