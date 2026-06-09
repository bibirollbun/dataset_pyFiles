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


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


train_df=pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
train_df.head()


test_df=pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
test_df.head()


# Define the target and identifier columns
TARGET = 'accident_risk'
ID_COL = 'id'

# Identify feature types
numerical_features = train_df.select_dtypes(include=np.number).columns.tolist()
numerical_features.remove(ID_COL)
numerical_features.remove(TARGET)

categorical_features = train_df.select_dtypes(include=['object', 'category']).columns.tolist()
print(f"Numerical features: {numerical_features}")
print(f"Categorical features: {categorical_features}")
# scale the features to have zero mean and unit variance.
numerical_transformer = StandardScaler()

# one hot encoding
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)


# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)


def build_model(input_shape):
    """Builds a Keras sequential model."""
    model = Sequential([
        # Input layer
        Dense(128, activation='relu', input_shape=[input_shape]),
        BatchNormalization(),
        Dropout(0.3),

        # Hidden layer 1
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        # Hidden layer 2
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        # Output layer
        # Sigmoid activation constrains the output to be between 0 and 1
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mean_squared_error', # Minimizing MSE is equivalent to minimizing RMSE
                  optimizer=optimizer,
                  metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])
    return model


# Separate features (X) and target (y)
X = train_df.drop([ID_COL, TARGET], axis=1)
y = train_df[TARGET]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)

input_shape = X_train_processed.shape[1]

model = build_model(input_shape)
model.summary()

early_stopping = EarlyStopping(monitor='val_rmse', patience=15, restore_best_weights=True, mode='min')

# Train the model
history = model.fit(
    X_train_processed, y_train,
    validation_data=(X_val_processed, y_val),
    epochs=100, 
    batch_size=256,
    callbacks=[early_stopping],
    verbose=1
)


print("Making predictions on the test set...")

submission_df = pd.DataFrame({'id': test_df['id']})

X_test = test_df.drop(ID_COL, axis=1)
X_test_processed = preprocessor.transform(X_test)

test_predictions = model.predict(X_test_processed).flatten()

submission_df[TARGET] = test_predictions

submission_filename = 'submission.csv'
submission_df.to_csv(submission_filename, index=False)

print(f"\nSubmission file '{submission_filename}' created successfully!")
print("First 5 rows of the submission file:")
print(submission_df.head())

