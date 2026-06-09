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
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
from sklearn.preprocessing import OneHotEncoder ,MinMaxScaler,StandardScaler,OrdinalEncoder


#Importing data
train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv', index_col=0)
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv', index_col=0)


test_pred = test.copy()


train.head()


np.round(train.isnull().sum() / train.shape[0],4)*100


median_listening_time = (train.groupby('Podcast_Name',as_index = False)['Episode_Length_minutes'].median())
median_Guest_Popularity_percentage = train.groupby('Podcast_Name', as_index = False)['Guest_Popularity_percentage'].median()


median_listening_time.head(2)


median_Guest_Popularity_percentage.head(2)


train = train.merge(median_listening_time, how = 'left', left_on = 'Podcast_Name', right_on  = 'Podcast_Name',suffixes = ('_left', None))
train = train.merge(median_Guest_Popularity_percentage, how = 'left', left_on = 'Podcast_Name', right_on  = 'Podcast_Name',suffixes = ('_left', None))


train.drop(columns = ['Episode_Length_minutes_left','Guest_Popularity_percentage_left'], axis = 1 , inplace = True)


train['Number_of_Ads'].fillna(0, inplace = True)


train.isnull().sum()


np.round(test.isnull().sum() / test.shape[0],4)*100


median_listening_time_test = (test.groupby('Podcast_Name',as_index = False)['Episode_Length_minutes'].median())
median_Guest_Popularity_percentage_test = test.groupby('Podcast_Name', as_index = False)['Guest_Popularity_percentage'].median()


test = test.merge(median_listening_time_test, how = 'left', left_on = 'Podcast_Name', right_on  = 'Podcast_Name',suffixes = ('_left', None))
test = test.merge(median_Guest_Popularity_percentage_test, how = 'left', left_on = 'Podcast_Name', right_on  = 'Podcast_Name',suffixes = ('_left', None))


test.drop(columns = ['Episode_Length_minutes_left','Guest_Popularity_percentage_left'], axis = 1 , inplace = True)


test.isnull().sum()


categorical_columns = train.select_dtypes(include = ['object']).columns
numerical_columns = [col for col in train.select_dtypes(include = ['number']).columns if col not in 'Listening_Time_minutes']


categorical_columns


numerical_columns


cat_transformer = Pipeline(steps = [('one_hot', OneHotEncoder(sparse_output = False,handle_unknown = "ignore"))])
cat_preprocessor = ColumnTransformer(transformers = [('cat', cat_transformer, categorical_columns)],remainder = 'drop')


num_transformer = Pipeline(steps = [('min_max_scaler',MinMaxScaler())])
num_preprocessor = ColumnTransformer(transformers = [('num', num_transformer, numerical_columns)],remainder = 'drop')


preprocessing = ColumnTransformer([
    ('cat',cat_preprocessor,categorical_columns),
    ('num',num_preprocessor, numerical_columns)
])


preprocessing


X = train[[col for col in train.columns if col not in 'Listening_Time_minutes']]
y = train['Listening_Time_minutes']


train_transformed = preprocessing.fit_transform(X)


assert train.shape[0] == train_transformed.shape[0]


X_train, X_valid, y_train, y_valid = train_test_split(train_transformed, y, test_size=0.30, random_state=42)


model = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=[X_train.shape[1]]),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1,activation = 'relu')
])


model.compile(
    optimizer='adam',
    loss='mse',
    metrics=[keras.metrics.RootMeanSquaredError()]
)


early_stopping = keras.callbacks.EarlyStopping(
    patience=20,
    min_delta=0.001,
    restore_best_weights=True,
)


history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=100000,
    callbacks=[early_stopping],
    verbose=0, # hide the output because we have so many epochs
)


test_transformed = preprocessing.fit_transform(test)


assert test.shape[0] == test_transformed.shape[0]


history_df = pd.DataFrame(history.history)


history_df.head(2)


submission = pd.DataFrame({
    'id': test_pred.index,         
    'rainfall': model.predict(test_transformed).ravel()
})


assert submission.shape[0] == test_pred.shape[0]


# Save the DataFrame to a CSV file
submission.to_csv('submission.csv', index=False)
print("Submission created")



