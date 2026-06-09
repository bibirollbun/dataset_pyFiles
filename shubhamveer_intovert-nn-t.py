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


!pip install imputepy
!pip install autokeras


from tensorflow.keras import layers, models, metrics
from sklearn.model_selection import train_test_split
import tensorflow as tf
tf.random.set_seed(1100)


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


train.isnull().sum()


from imputepy import LGBMimputer

# Initialize the imputer with your train dataframe
train = LGBMimputer(train, filter=False, unique_count_limit=15)
test = LGBMimputer(test, filter=False, unique_count_limit=15)


ids = test['id']


def process(df, train=True):
    df.drop('id', axis=1, inplace=True)
    df['Stage_fear'] = df['Stage_fear'].map({'No':0, 'Yes':1})
    df['Drained_after_socializing'] = df['Drained_after_socializing'].map({'No':0, 'Yes':1})
    if train:
        df['Personality'] = df['Personality'].map({'Extrovert':0, 'Introvert':1}) 
    return df


train = process(train)
test = process(test, train=False)


true_labels_train = train['Personality']
train.drop(['Personality'], axis=1, inplace=True)


xtrain,xtest,ytrain,ytest = train_test_split(train, true_labels_train.values, test_size=0.2, random_state=102, stratify=true_labels_train.values)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

xttrain = scaler.fit_transform(xtrain)
xttest = scaler.transform(xtest)

test_scaled = scaler.transform(test)



tensorX = tf.convert_to_tensor(xtrain)
tensorXt = tf.convert_to_tensor(xtest)
tensorY = tf.convert_to_tensor(ytrain)
tensorYt = tf.convert_to_tensor(ytest)


opt = tf.keras.optimizers.Adam(
    learning_rate=0.001,  # default is 0.001
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07
)

model = models.Sequential([
    tf.keras.Input(shape=(7,)),
    layers.Dense(32, activation='swish'),
    layers.Dense(32, activation='swish'),
    layers.Dense(1, activation='sigmoid')
])

loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
model.compile(
    optimizer=opt,
    loss=loss,
    metrics=[
        metrics.BinaryAccuracy(),
        metrics.AUC()])

model.fit(tensorX,tensorY, epochs=100, batch_size=128,
    validation_data=(tensorXt, tensorYt))


import autokeras as ak

input_node = ak.Input()
x = ak.DenseBlock(num_layers=5, num_units=128, use_batchnorm=True, dropout=0.3)(input_node)
output_node = ak.ClassificationHead()(x)

clf = ak.AutoModel(
    inputs=input_node,
    outputs=output_node,
    max_trials=100,
    overwrite=True
)

clf.fit(xtrain.values, ytrain, validation_data=(xtest.values, ytest), validation_split=0.2, epochs=500)


 predicted_test = clf.predict(test_scaled, batch_size=32, verbose=1)


predicted_test


# binary_preds = np.round(predicted_test).astype(int)
labels = np.where(predicted_test == 1, "Introvert", "Extrovert")

submission = pd.DataFrame({
    'id': ids, 
    'Personality': labels.flatten()
})



submission.to_csv("submission.csv", index=False)


submission




