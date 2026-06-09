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


TRAIN_PATH = r"/kaggle/input/recruitment-task-for-gdsc-ml/MiNDAT.csv"
TEST_PATH  = r"/kaggle/input/recruitment-task-for-gdsc-ml/MiNDAT_UNK.csv"


!pip install keras-tuner -q

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import keras_tuner as kt

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)







train_df.dropna(subset=['CORRUCYSTIC_DENSITY'], inplace=True)
y = train_df['CORRUCYSTIC_DENSITY']
train_features = train_df.drop('CORRUCYSTIC_DENSITY', axis=1)
test_features = test_df.copy()

test_ids = test_features['LOCAL_IDENTIFIER']
train_features = train_features.drop('LOCAL_IDENTIFIER', axis=1)
test_features = test_features.drop('LOCAL_IDENTIFIER', axis=1)


categorical_cols = train_features.select_dtypes(include=['object']).columns
numerical_cols = train_features.select_dtypes(include=np.number).columns

num_imputer = SimpleImputer(strategy='median')
train_features[numerical_cols] = num_imputer.fit_transform(train_features[numerical_cols])
test_features[numerical_cols] = num_imputer.transform(test_features[numerical_cols])

cat_imputer = SimpleImputer(strategy='most_frequent')
train_features[categorical_cols] = cat_imputer.fit_transform(train_features[categorical_cols])
test_features[categorical_cols] = cat_imputer.transform(test_features[categorical_cols])

train_features = pd.get_dummies(train_features, columns=categorical_cols, drop_first=True)
test_features = pd.get_dummies(test_features, columns=categorical_cols, drop_first=True)

train_features, test_features = train_features.align(test_features, join='left', axis=1, fill_value=0)

scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)
test_features_scaled = scaler.transform(test_features)



X_train, X_val, y_train, y_val = train_test_split(train_features_scaled, y, test_size=0.2, random_state=42)


def build_model(hp):
    model = Sequential()
    model.add(Dense(
        units=hp.Int('units_1', min_value=64, max_value=256, step=32),
        activation='relu',
        input_shape=(X_train.shape[1],)
    ))
    model.add(Dropout(rate=hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(Dense(
        units=hp.Int('units_2', min_value=32, max_value=128, step=32),
        activation='relu'
    ))
    model.add(Dropout(rate=hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(Dense(1))

    model.compile(
        optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 5e-4])),
        loss='mean_squared_error'
    )
    return model


tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=20,  
    executions_per_trial=1,
    directory='tuner_results',
    project_name='corrucystic_density'
)



tuner.search(X_train, y_train, epochs=20, validation_data=(X_val, y_val), callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=5)])
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
Best hyperparameters found:
- Units in Layer 1: {best_hps.get('units_1')}
- Dropout in Layer 1: {best_hps.get('dropout_1'):.2f}
- Units in Layer 2: {best_hps.get('units_2')}
- Dropout in Layer 2: {best_hps.get('dropout_2'):.2f}
- Learning Rate: {best_hps.get('learning_rate')}
""")


best_model = tuner.hypermodel.build(best_hps)

history = best_model.fit(
    train_features_scaled, y,
    epochs=100,
    batch_size=32,
    validation_split=0.15,
    callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=10)],
    verbose=0
)


tuned_predictions = best_model.predict(test_features_scaled).flatten()
submission_df_tuned = pd.DataFrame({'LOCAL_IDENTIFIER': test_ids, 'CORRUCYSTIC_DENSITY': tuned_predictions})
submission_df_tuned.to_csv('submission.csv', index=False)
submission_df_tuned.head()

