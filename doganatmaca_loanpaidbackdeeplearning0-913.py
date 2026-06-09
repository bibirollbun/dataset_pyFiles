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


data_train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
data_test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')


data = pd.concat([data_train,data_test])
data = data.reset_index(drop=True)


data.tail()


data.shape


data.info()


data.isnull().sum()


data.education_level.value_counts()


education_dict={"Other":0,"High School":1,"Bachelor's":2,"Master's":3,"PhD":4}
data['education_level']=data['education_level'].map(education_dict)


data.annual_income.describe()


data['annual_income_category'] = pd.cut(data['annual_income'],bins=5,labels=False,include_lowest=True)


data.credit_score.describe()


data['credit_score_category'] = pd.cut(data['credit_score'],bins=3,labels=False,include_lowest=True)


data.loan_amount.describe()


data['loan_amount_category'] = pd.cut(data['loan_amount'],bins=5,labels=False,include_lowest=True)


data.grade_subgrade.value_counts()


def grade_category(text):
    if 'A' in text:
        return 5
    if 'B' in text:
        return 4
    if 'C' in text:
        return 3
    if 'D' in text:
        return 2
    if 'E' in text:
        return 1
    if 'F' in text:
        return 0
data['grade_category'] = data['grade_subgrade'].apply(grade_category)


data.head()


data.loan_purpose.value_counts()


data.employment_status.value_counts()


data["income_to_loan_ratio"] = data["annual_income"] / (data["loan_amount"] + 1)
data["credit_to_income_ratio"] = data["credit_score"] / (data["annual_income"] + 1)
data["interest_income_ratio"] = data["interest_rate"] / (data["annual_income"] + 1)
data['credit_to_loan_ratio'] = data['credit_score'] / (data['loan_amount'] + 1)



encode_data = data.drop('grade_subgrade',axis=1)
encode_data = pd.get_dummies(encode_data,drop_first=True)


full_data = encode_data[encode_data['loan_paid_back'].notna()]
empty_data = encode_data[encode_data['loan_paid_back'].isna()]


full_data.info()


!pip install protobuf==3.20 --quiet


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# X ve y ayır
X = full_data.drop(columns=["loan_paid_back"])
y = full_data["loan_paid_back"]

# Train-test böl
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, random_state=25, stratify=y
)

# Sadece float ve int kolonları scale edilecek
num_cols = X.select_dtypes(include=["float64", "int64"]).columns

scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])



from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))



from keras.callbacks import ReduceLROnPlateau

lr_reduce = ReduceLROnPlateau(
    monitor='val_auc',  # hangi metriğe bakacak
    factor=0.5,         # learning rate'i bu oranla düşürür
    patience=3,         # 3 epoch plateau olursa düşür
    mode='max',         # AUC maksimum olmalı
    min_lr=1e-6,        # minimum learning rate
    verbose=1
)


model = keras.Sequential([
    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Dense(64, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.25),

    layers.Dense(32, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.2),

    layers.Dense(1, activation="sigmoid")  # binary output
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.AUC(name="auc")]
)

# early stopping
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_auc",
    patience=5,
    mode="max",
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=256,
    validation_split=0.2,
    callbacks=[early_stop,lr_reduce],
    class_weight=class_weight_dict
)



y_proba = model.predict(X_test)        # 0-1 arası olasılık


from sklearn.metrics import roc_auc_score

auc = roc_auc_score(y_test, y_proba)
print("ROC–AUC:", auc)



empty_data_test3 = empty_data.drop('loan_paid_back',axis=1)
num_cols = empty_data_test3.select_dtypes(include=["float64", "int64"]).columns
empty_data_test3[num_cols] = scaler.fit_transform(empty_data_test3[num_cols])



y_proba_test = model.predict(empty_data_test3)        # 0-1 arası olasılık


result = pd.DataFrame()
result['id']=data_test['id']
result['loan_paid_back']=y_proba_test


result = result.set_index("id")


result.head()


result.to_csv('loan_paid_back_end.csv')

