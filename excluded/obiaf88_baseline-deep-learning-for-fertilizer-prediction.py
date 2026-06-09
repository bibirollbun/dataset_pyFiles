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
import seaborn as sns
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')


train.head(2)


train['Fertilizer Name'].unique()


test.head()


train.isnull().sum(), test.isnull().sum()


fig,ax = plt.subplots(4,2, figsize = (10,15))
plt.suptitle("Descriptive statistics of features for train dataset" , y=1.02)
axs = ax.ravel()
for i, c in enumerate([col for col in train.columns if col not in ('id','Fertilizer Name')]):
    grouped = train.groupby(['Fertilizer Name',f'{c}'])['Fertilizer Name'].count()
    grouped.unstack(0).describe().plot(kind = 'bar',ax = axs[i])
    axs[i].set_title(f'{c}')
    axs[i].spines['right'].set_visible(False)
    axs[i].spines['top'].set_visible(False)
    axs[i].legend(loc = 'lower center', fontsize = 'small')
plt.tight_layout()


fig,ax = plt.subplots(4,2, figsize = (10,15))
plt.suptitle("Descriptive statistics of features for test dataset", y=1.02)
axs = ax.ravel()
for i, c in enumerate([col for col in test.columns if col not in ('id')]):
    grouped = train.groupby(['Fertilizer Name',f'{c}'])['Fertilizer Name'].count()
    grouped.unstack(0).describe().plot(kind = 'bar',ax = axs[i])
    axs[i].set_title(f'{c}')
    axs[i].spines['right'].set_visible(False)
    axs[i].spines['top'].set_visible(False)
    axs[i].legend(loc = 'lower center', fontsize = 'small')
plt.tight_layout()


numerical_columns = [col for col in train.select_dtypes('number').columns if col not in 'id']
categorical_columns = [col for col in train.select_dtypes('object').columns if col not in 'Fertilizer Name']


num_pipe = Pipeline([('poly_feat',FeatureUnion([('poly', PolynomialFeatures(degree = 3 , interaction_only = True, include_bias = False))])),('scaler', StandardScaler())])


num_pipe


preprocessing = ColumnTransformer(
    remainder = 'passthrough',
    transformers = [
        ('num_preproc', num_pipe,numerical_columns ),
        ('cat_preproc', OneHotEncoder(), categorical_columns)
    ]
)


preprocessing


pd.DataFrame(preprocessing.fit_transform(train[[col for col in train.columns if col not in ('id','Fertilizer Name')]]), columns = preprocessing.get_feature_names_out()).head()


le = LabelEncoder()
encoded_y = le.fit_transform(train['Fertilizer Name'])
y = keras.utils.to_categorical(encoded_y, num_classes=7)



np.unique(encoded_y)


le.inverse_transform(np.unique(encoded_y))


y


X_train, X_valid,y_train, y_valid = train_test_split(preprocessing.fit_transform(train[[col for col in train.columns if col not in ('id','Fertilizer Name')]]), y,test_size = 0.3 ,random_state = 42 )


model = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=[X_train.shape[1]]),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(7,activation = 'softmax')
])


model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=[keras.metrics.CategoricalAccuracy()]
)


early_stopping = keras.callbacks.EarlyStopping(
    patience=20,
    min_delta=0.001,
    restore_best_weights=True,
)


history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=1000,
    epochs=50000,
    callbacks=[early_stopping],
    verbose=0
)


test_transformed = preprocessing.fit_transform(test[[col for col in test.columns if col not in 'id']])


predictions = model.predict(test_transformed)


predictions[0]


np.argsort(-predictions[0])


' '.join(le.inverse_transform(np.argsort(-predictions[0]))[:3])


np.argsort(-predictions)


final_submission = pd.DataFrame(columns = ['id','Fertilizer Name'])


assert test.shape[0] == predictions.shape[0]


predictions_sorted = np.argsort(-predictions)
for  i in range(predictions_sorted.shape[0]):
    temp_df = pd.DataFrame({'id': [test.loc[i,'id']], 'Fertilizer Name': ' '.join(le.inverse_transform(np.argsort(-predictions[i]))[:3])})
    final_submission = pd.concat([final_submission, temp_df], axis = 0)


final_submission.head()


final_submission.to_csv('submission.csv', index=False)
print("Submission created")

