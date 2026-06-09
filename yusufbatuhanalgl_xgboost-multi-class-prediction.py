# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df_train = pd.read_csv("/kaggle/input/playground-series-s4e2/train.csv") 
df_test = pd.read_csv("/kaggle/input/playground-series-s4e2/test.csv")
df_sample = pd.read_csv("/kaggle/input/playground-series-s4e2/sample_submission.csv")


df_train.head()


df_test.head()


df_sample.head()


# Hedef sütunu ayır
y = df_train["NObeyesdad"]

# Sadece özellikleri al (NObeyesdad hariç)
df_train_features = df_train.drop(columns=["NObeyesdad"])
df_test_features = df_test.copy()

# Train/test flag ekle
df_train_features["train_test_flag"] = "train"
df_test_features["train_test_flag"] = "test"

# Birleştir
combined = pd.concat([df_train_features, df_test_features], axis=0)

# get_dummies uygula
combined_dummies = pd.get_dummies(combined, drop_first=False)

# Ayır: train_test_flag sütunu oluşma biçimine göre kontrol et
if "train_test_flag_train" in combined_dummies.columns:
    df_train_processed = combined_dummies[combined_dummies["train_test_flag_train"] == 1].drop(columns=["train_test_flag_train", "train_test_flag_test"])
    df_test_processed = combined_dummies[combined_dummies["train_test_flag_test"] == 1].drop(columns=["train_test_flag_train", "train_test_flag_test"])
else:
    # Sadece 'train_test_flag_test' oluşmuşsa:
    df_test_processed = combined_dummies[combined_dummies["train_test_flag_test"] == 1].drop(columns=["train_test_flag_test"])
    df_train_processed = combined_dummies[combined_dummies["train_test_flag_test"] == 0].drop(columns=["train_test_flag_test"])




df_train_processed["NObeyesdad"] = y.values


x = df_train_processed.drop(columns=["NObeyesdad"])
y = df_train_processed["NObeyesdad"]


le = LabelEncoder()
y_encoded = le.fit_transform(y)


y_encoded.shape


X_train,X_test,y_train,y_test = train_test_split(x,y_encoded,train_size = 0.8,random_state = 58)


num_class = len(np.unique(y))
print("NObeyesdad:", num_class)


model = xgb.XGBClassifier(
    n_estimators = 100,
    objective='multi:softprob',
    num_class=7,                
    eval_metric='mlogloss'
)


model.fit(X_train,y_train)


y_pred_encoded = model.predict(df_test_processed)


y_pred_encoded


y_pred = le.inverse_transform(y_pred_encoded)
y_true = le.inverse_transform(y_test)


model.score(X_test,y_test)


submission = pd.DataFrame({
    'id': df_test['id'],
    'NObeyesdad': y_pred
})


submission.head()


submission.to_csv("submission.csv", index=False)




