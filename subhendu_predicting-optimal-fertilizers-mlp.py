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


import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, QuantileTransformer, KBinsDiscretizer


train_df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
print(f"{train_df.shape=} {test_df.shape=}")


train_df.head()


test_df.head()


train_cols = list(train_df.columns)
test_cols = list(test_df.columns)

assert len(train_cols) == len(test_cols) + 1
print(f"{train_cols=}\n{test_cols=}")


TARGET_COLUMN = "Fertilizer Name"
NUMERIC_COLS = ["Temparature", "Humidity", "Moisture", "Nitrogen","Potassium", "Phosphorous"]
ID_COL = "id"
CATEGORICAL_COLS = ["Soil Type","Crop Type"]


# Check values
for col in CATEGORICAL_COLS:
    train_counts = train_df[col].value_counts()
    test_counts = test_df[col].value_counts()

    train_labels = sorted(train_counts.index)
    test_labels = sorted(test_counts.index)

    print(f"{col=} {len(train_labels)=}")

    assert train_labels == test_labels


train_df[TARGET_COLUMN].hist(bins = 20)


train_df[NUMERIC_COLS].describe()


# Q_COLS = [c+"_Q" for c in NUMERIC_COLS]
# print(f"{Q_COLS=}")


# QT = QuantileTransformer(n_quantiles=10, random_state=0)
# QT.fit(train_df[NUMERIC_COLS])
# print(f"{QT.n_quantiles_=}\n{QT.n_features_in_=}\n{QT.feature_names_in_=}")
# train_df[Q_COLS] = QT.transform(train_df[NUMERIC_COLS])
# train_df[Q_COLS].describe()


NEW_RATIO_FEATURES = [
    ("N/K", ("Nitrogen","Potassium")),
    ("N/P", ("Nitrogen","Phosphorous")),
    ("K/P", ("Potassium", "Phosphorous")) 
]

NEW_RATIO_COLS = [name for name,_ in NEW_RATIO_FEATURES]
print(f"{NEW_RATIO_COLS=}")

def add_new_cols(df):
    for name,(a,b) in NEW_RATIO_FEATURES:
        df[name] = (df[a]+1.0) / (df[b]+1.0)
    return df

train_df = add_new_cols(train_df)
test_df = add_new_cols(test_df)



train_df.head()


# transformation_parms = df[NUMERIC_COLS].describe().T
# for col in NUMERIC_COLS:
#     std = transformation_parms.loc[col]["std"]
#     _mean = transformation_parms.loc[col]["mean"]
#     df[col] = (df[col] - _mean)/std

# Train Encoders
OHE = OneHotEncoder(sparse_output=False)
OHE.fit(train_df[CATEGORICAL_COLS])
print(f"{OHE.n_features_in_=}\n{OHE.feature_names_in_=}\n{OHE.categories_=}")

KBD = KBinsDiscretizer(
    n_bins=3, encode='onehot-dense', strategy='quantile'
)
KBD.fit(train_df[NUMERIC_COLS + NEW_RATIO_COLS])
print(f"{KBD.n_features_in_=}\n{KBD.n_bins_=}\n{KBD.bin_edges_=}")
# QT = QuantileTransformer(n_quantiles=100, random_state=0)
# QT.fit(train_df[NUMERIC_COLS])
# print(f"{QT.n_quantiles_=}\n{QT.n_features_in_=}\n{QT.feature_names_in_=}")


def df_to_vecs(df):
    x_hot = OHE.transform(df[CATEGORICAL_COLS])
    print(f"{x_hot.shape=}")
    num_hot = KBD.transform(df[NUMERIC_COLS + NEW_RATIO_COLS])
    x_mat = np.hstack((x_hot,num_hot))
    return x_mat

X = df_to_vecs(train_df)
X_test = df_to_vecs(test_df)

print(f"{X.shape=} {X_test.shape=}")


X[0]


LE = LabelEncoder()
y = LE.fit_transform(train_df[TARGET_COLUMN])
classes_ = LE.classes_
num_classes = len(classes_)
print(f"{num_classes=} \n{classes_=}, \n{ y[:10]=}")

# LE = OneHotEncoder(sparse_output=False)
# y = LE.fit_transform(train_df[TARGET_COLUMN].values.reshape(-1, 1))
# classes_ = list(LE.categories_[0])
# num_classes = len(classes_)
# print(f"{y.shape=}")
# print(f"{num_classes=}\n{classes_=}")


x_train, x_val, y_train, y_val = train_test_split(X,y, test_size=0.1, random_state=25)
print(f"{x_train.shape=}, {x_val.shape=}, {y_train.shape=}, {y_val.shape=}")


def create_model(input_dim=0, output_dim=0):
    keras.backend.clear_session()
    model = keras.models.Sequential()
    model.add(keras.layers.Input((input_dim,)))
    model.add(keras.layers.Dense(256, activation="relu"))
    model.add(keras.layers.Dense(256, activation="relu"))
    model.add(keras.layers.Dense(256, activation="relu"))
    model.add(keras.layers.Dense(256, activation="relu"))
    model.add(keras.layers.Dense(256, activation="relu"))
    model.add(keras.layers.Dense(256, activation="relu"))
    model.add(keras.layers.Dense(output_dim, activation="softmax"))
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  optimizer=tf.keras.optimizers.SGD(learning_rate=0.03), 
                  metrics=["accuracy"]
    )
    model.summary()
    return model

model = create_model(input_dim=x_train.shape[1], output_dim=num_classes)



hist = model.fit(x_train, y_train, 
                 epochs=20, 
                 batch_size=128, 
                 validation_data=(x_val,y_val))


hdf = pd.DataFrame(hist.history)
hdf[["accuracy","val_accuracy"]].plot()


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score


y_val_pred = np.argmax(model.predict(x_val), axis=1)
print(y_val_pred.shape)


acc = accuracy_score(y_val, y_val_pred)
print(f"{acc=}")
cm = confusion_matrix(y_val, y_val_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=classes_)
disp.plot()
plt.show()


final_pred = np.argsort(model.predict(X_test), axis=1)
print(final_pred.shape)


output_classes = [" ".join(classes_[i] for i in p[-3:][::-1]) for p in final_pred]
print(output_classes[:10])


test_df[TARGET_COLUMN] = output_classes
out_df = test_df[ [ID_COL, TARGET_COLUMN]]
out_df.head()


# from datetime import datetime, timezone, timedelta


# # Define the UTC+5:30 timezone
# tz = timezone(timedelta(hours=5, minutes=30))
# now_local = datetime.now(tz)
# # Convert the datetime object to a string
# now_str = now_local.strftime("%Y-%m-%d_%H-%M")
# print(now_str)
# out_file_name = f"/kaggle/working/output_{now_str}.csv"
# print(f"{out_file_name=}")


out_df.to_csv("submission.csv", index=False)
print("done")




