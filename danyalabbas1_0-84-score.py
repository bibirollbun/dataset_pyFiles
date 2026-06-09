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


import warnings
warnings.filterwarnings("ignore")
_ = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")
_


train_df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv", index_col = "id")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv", index_col="id")
train_df


X = train_df.copy()
y = X.pop("rainfall")
X


X.isnull().sum()



test_df["winddirection"].fillna(test_df["winddirection"].mean(), inplace=True)
test_df.isnull().sum()


X


import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(4, 3, figsize=(15, 12))  
columns = train_df.columns
length = len(columns)  

for i, col in enumerate(columns):
    row, col_index = divmod(i, 3)  
    sns.kdeplot(x=train_df[col], y=train_df["rainfall"], data=train_df, ax=ax[row, col_index])
    ax[row, col_index].set_title(f"{col} vs rainfall")

plt.tight_layout()  # Adjust spacing
plt.show()






import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(4, 3, figsize=(15, 12))  
columns = train_df.columns
length = len(columns)  

for i, col in enumerate(columns):
    row, col_index = divmod(i, 3)  
    sns.boxplot(y=train_df[col], x=train_df["rainfall"], data=train_df, ax=ax[row, col_index])
    ax[row, col_index].set_title(f"{col} vs rainfall")

plt.tight_layout()  # Adjust spacing
plt.show()


sns.lineplot(X, linewidth=2.5)


sns.pairplot(train_df[["maxtemp","sunshine", "winddirection", "windspeed"]])


y.sum()


y[y==0].count()


y


import tensorflow as tf
from tensorflow.keras.activations import sigmoid, linear, relu
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import L2

model = Sequential(
    [
        Dense(300, activation="relu", kernel_regularizer=L2(0.01)),
        Dense(500, activation="relu", kernel_regularizer=L2(0.01)),
        Dense(500, activation="relu", kernel_regularizer=L2(0.01)),
        Dense(2, activation="linear")
    ]
)


model.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
)


model.fit(X,y, epochs=100)


preds = model.predict(test_df)
preds


pj = tf.nn.softmax(preds)
d = pj[:,1]


output = pd.DataFrame({"id":test_df.index, "rainfall":d})
output.to_csv("submission.csv", index=False)




