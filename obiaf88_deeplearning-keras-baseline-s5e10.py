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


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
pd.set_option('display.max_columns', 500)
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from sklearn.utils import class_weight


train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")


train.head()


test.head(2)


train['accident_risk'].value_counts().sort_values()


train.shape, test.shape


train.isnull().sum()


test.isnull().sum()


num_columns = [col for col in train.select_dtypes('number').columns if col not in ['id','accident_risk']]
cat_columns = train.select_dtypes('object').columns
bool_columns = train.select_dtypes('bool').columns


num_columns


cat_columns


bool_columns


col_transformer = ColumnTransformer(
    [
        ('cat', OneHotEncoder(), cat_columns),
        ('bool', OneHotEncoder(), bool_columns),
        ('num',MinMaxScaler(), num_columns)
    ]
)


train_transformed = pd.DataFrame(col_transformer.fit_transform(train[[col for col in train.columns if col not in 'accident_risk']]), columns = col_transformer.get_feature_names_out())


train_transformed.head(2)


test_transformed = pd.DataFrame(col_transformer.transform(test), columns = col_transformer.get_feature_names_out())


test_transformed.head(2)


assert all(train_transformed.columns == test_transformed.columns)


train_transformed.shape


weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train['accident_risk']),
    y=train['accident_risk']
)


class_weights = dict(enumerate(weights))


X_train, X_valid,y_train, y_valid = train_test_split(train_transformed, train['accident_risk'],test_size = 0.3 ,random_state = 42 )


model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=[X_train.shape[1]], kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])



model.compile(
        optimizer='adam',
        loss='mean_squared_error',  # <-- regression-style loss,
        metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')]
    )


early_stopping = keras.callbacks.EarlyStopping(
    patience=10,
    min_delta=0.001,
    restore_best_weights=True,
)


lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)


history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=5000,
    epochs=50,
    callbacks=[early_stopping,lr_scheduler],
    class_weight=class_weights,
    verbose=0
)


loss, rmse = model.evaluate(X_valid, y_valid, verbose=0)
print(f"Test RMSE: {rmse:.4f}")


history_df = pd.DataFrame(history.history)


history_df.head(2)


history_df.loc[:, ['loss', 'val_loss']].plot()
history_df.loc[:, ['rmse', 'val_rmse']].plot()


predictions = model.predict(test_transformed)


submission = pd.DataFrame({'id': test['id'], 'accident_risk': predictions[:,0]})


submission.to_csv('submission.csv', index=False)
print("Submission created")

