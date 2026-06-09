# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


from packaging import version
import sklearn
from sklearn.model_selection import train_test_split

assert version.parse(sklearn.__version__) >= version.parse("1.0.1")


import pandas as pd
import numpy as np


import tensorflow as tf

assert version.parse(tf.__version__) >= version.parse("2.8.0")


import matplotlib.pyplot as plt

plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)


train = pd.read_csv("/kaggle/input/rossmann-store-sales/train.csv")
test = pd.read_csv("/kaggle/input/rossmann-store-sales/test.csv")


train.info()


train.head()


test.head()


train.hist(bins=50, figsize=(15, 10))


train[['Year','Month','Day']] = train['Date'].astype(str).str.split('-', expand=True).iloc[:, :3]
train[['Year','Month','Day']] = train[['Year','Month','Day']].apply(pd.to_numeric, errors='coerce').astype('int64')


engineered_test = test
engineered_test[['Year','Month','Day']] = engineered_test['Date'].astype(str).str.split('-', expand=True).iloc[:, :3]
engineered_test[['Year','Month','Day']] = engineered_test[['Year','Month','Day']].apply(pd.to_numeric, errors='coerce').astype('int64')


engineered_test.drop(columns=["Date"])


print(train['StateHoliday'].unique())


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

train['StateHoliday'] = train['StateHoliday'].astype(str)
le.fit(train['StateHoliday'])

train['StateHoliday'] = le.transform(train['StateHoliday'].astype(str))
engineered_test['StateHoliday'] = le.transform(engineered_test['StateHoliday'].astype(str))



print(engineered_test['StateHoliday'].unique())


train.info()


Y_train = train["Sales"]
X_train = train.drop(columns=["Sales", "Date"])


X_train.head()


X_training, X_validation, Y_training, Y_validation = train_test_split(
    X_train, Y_train,
    test_size=0.2,
    random_state=42,
)


X_training.info()


X_validation.info()


def create_huber(threshold=1.0):
    def huber_fn(y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < threshold
        squared_loss = tf.square(error) / 2
        linear_loss  = threshold * tf.abs(error) - threshold ** 2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)
    return huber_fn


lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=20_000,
    decay_rate=0.1,
    staircase=False
)


tf.keras.utils.set_random_seed(42)  
l2_reg = tf.keras.regularizers.l2(0.05)
model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(10,)),
    tf.keras.layers.Dense( 30, activation="relu", kernel_initializer="he_normal",
                          kernel_regularizer=l2_reg),
    tf.keras.layers.Dense( 30, activation="relu", kernel_initializer="he_normal",
                          kernel_regularizer=l2_reg),
    tf.keras.layers.Dense(1, kernel_regularizer=l2_reg)
])


def random_batch(X, y, batch_size=32):
    idx = np.random.randint(len(X), size=batch_size)
    return X[idx], y[idx]


def print_status_bar(step, total, loss, metrics=None):
    metrics = " - ".join([f"{m.name}: {m.result():.4f}"
                          for m in [loss] + (metrics or [])])
    end = "" if step < total else "\n"
    print(f"\r{step}/{total} - " + metrics, end=end)


n_epochs = 5
batch_size = 32
n_steps = len(X_training) // batch_size
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
loss_fn = create_huber()
mean_loss = tf.keras.metrics.Mean(name='train_loss') 
metrics = [tf.keras.metrics.MeanAbsoluteError(name='mae')]


X_training_np = X_training.to_numpy() 
Y_training_np = Y_training.to_numpy() 


for epoch in range(1, n_epochs + 1):
    print("Epoch {}/{}".format(epoch, n_epochs))
    for step in range(1, n_steps + 1):
        X_batch, y_batch = random_batch(X_training_np, Y_training_np)
        with tf.GradientTape() as tape:
            y_pred = model(X_batch, training=True)
            main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
            loss = tf.add_n([main_loss] + model.losses)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        mean_loss(loss)
        for metric in metrics:
            metric(y_batch, y_pred)

        print_status_bar(step, n_steps, mean_loss, metrics)

        for metric in [mean_loss] + metrics:
            metric.reset_states()


engineered_test= engineered_test.drop(columns=["Date"])


y_predict = model.predict(engineered_test)
y_pred = np.argmax(y_predict, axis=1)


submission = pd.DataFrame({
    "ImageId": np.arange(1, len(y_pred)+1),
    "Label": y_pred
})
submission.to_csv("submission.csv", index=False)

