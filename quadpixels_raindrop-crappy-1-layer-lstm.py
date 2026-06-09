# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

MODEL_FN = "/kaggle/working/model.keras"


def Train():
    df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
    feats = df.columns[2:12]
    df_X = df[feats]
    df_Y = df[df.columns[12]]

    train_X, test_X, train_Y, test_Y = train_test_split(df_X, df_Y, test_size=0.1, random_state=42)
    print(train_X.head())

    # To make the input acceptable to the LSTM
    train_X = np.expand_dims(train_X, axis=1)
    test_X = np.expand_dims(test_X, axis=1)

    model = Sequential(layers=[
        LSTM(100, input_shape=(None,10)),
        Dense(1),
    ])
    model.compile(loss="binary_crossentropy", optimizer="adam")
    model.fit(train_X, train_Y, epochs=50, validation_data=(test_X, test_Y))
    model.save(MODEL_FN)


def Predict():
    if not os.path.exists(MODEL_FN):
        print("Please train first.")
        return
    df = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
    model = keras.models.load_model(MODEL_FN)
    feats = df.columns[2:12]
    df_X = df[feats]
    df_X = np.expand_dims(df_X, axis=1)
    pred_Y = model.predict(df_X).flatten()
    pred_Y[np.isnan(pred_Y)] = 0.5
    out = pd.DataFrame({
        "id": df["id"],
        "rainfall": pred_Y
    })
    out.to_csv("/kaggle/working/submission.csv", index=False)


Train()


Predict()

