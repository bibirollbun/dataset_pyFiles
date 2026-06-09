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


train=pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


train


train.columns


train.corr


train.isna().mean()


test.isna().mean()


from sklearn.model_selection import train_test_split
from sklearn.compose        import ColumnTransformer
from sklearn.preprocessing  import StandardScaler, OneHotEncoder
from sklearn.pipeline       import Pipeline


import tensorflow
from tensorflow import keras
from keras import Sequential
from keras.utils import *
from keras.layers import *


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

features = ['id', 'age', 'job', 'marital', 'education', 'default', 'balance',
           'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign',
           'pdays', 'previous', 'poutcome']
target_col = 'y'

X = train[features]
y = train[target_col]
X1 = test[features] 

numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

numeric_pipe = Pipeline([('scaler', StandardScaler())])
categorical_pipe = Pipeline([
    ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipe, numeric_cols),
    ('cat', categorical_pipe, cat_cols),
], remainder='drop')

X_processed = preprocessor.fit_transform(X)
X1_processed = preprocessor.transform(X1) 

if y.dtype == 'object' or y.dtype == 'category':
    y_processed = np.where(y == 'yes', 1, 0)
else:
    y_processed = y.values

X_train, X_val, y_train, y_val = train_test_split(
    X_processed,
    y_processed,
    test_size=0.2,
    random_state=42,
    stratify=y_processed
)


x_train = X_train.astype('float32')
x_test = X_val.astype('float32')
x1_processed = X1_processed.astype('float32')  
y_train = y_train.astype('float32')
y_test = y_val.astype('float32')


print(x_train.shape)


 def mish(x):
        return x * tf.nn.tanh(tf.nn.softplus(x))


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential([
    Dense(512, activation='mish', input_shape=(52,)),
    Dropout(0.3),
    
    Dense(256, activation='mish'),
    Dropout(0.3),
    
    Dense(128, activation='mish'),
    Dropout(0.2),
    
    Dense(1, activation='sigmoid')
])
model.summary()


import tensorflow as tf
model.compile(
    loss='binary_crossentropy',
   optimizer = 'adam',
    metrics=['accuracy']
)


history = model.fit(
    x_train, y_train,
       epochs=184,
    batch_size=200,
    validation_data=(x_test,y_test),
   
)



test_ids = test['id'].values


predicted_test = model.predict(x1_processed, batch_size=1024).flatten()

submission = pd.DataFrame({
    'id': test_ids,
    'y': predicted_test
})

submission.to_csv("/kaggle/working/submission.csv", index=False)
print("Submission saved to /kaggle/working/submission.csv")

