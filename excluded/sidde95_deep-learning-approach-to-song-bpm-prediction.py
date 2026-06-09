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


train_df = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
sample_df = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')


train_df.sample(5)


test_df.sample(3)


sample_df.sample(2)


train_df.shape


train_df.info()


train_df.isnull().sum()


import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


def boxplot(data):
    num_cols = train_df.select_dtypes(exclude = 'O').columns
    for i, j in enumerate(num_cols):
        plt.subplot(len(num_cols)//3+1, 3, i+1 )
        sns.boxplot(data = data, x = j)
        plt.title(f"{j} plot")
    plt.tight_layout()

plt.figure(figsize = (15, 10))
boxplot(train_df)


def kdeplot(data):
    num_cols = train_df.select_dtypes(exclude = 'O').columns
    for i, j in enumerate(num_cols):
        plt.subplot(len(num_cols)//3+1, 3, i+1 )
        sns.kdeplot(data = data, x = j)
        plt.title(f"{j} plot")
    plt.tight_layout()

plt.figure(figsize = (15, 10))
kdeplot(train_df)


plt.figure(figsize = (15, 8))
sns.heatmap(train_df.corr(), annot = True)
plt.show()


train_df = train_df.drop('id', axis = 1)


!pip install scikit-learn==1.6.3 --quiet --force-reinstall
!pip install --upgrade scikeras --quiet



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = train_df.drop('BeatsPerMinute', axis = 1)
y = train_df.BeatsPerMinute

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.85, random_state = 42)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


X_train_scaled


import tensorflow
from tensorflow import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential([
    Dense(128, input_shape = (X_train_scaled.shape[1],), activation = 'relu'),
    Dense(64, activation = 'relu'),
    Dense(32, activation = 'relu'),
    Dense(1)
])

model.compile(optimizer = 'adam', loss = 'mse', metrics = [keras.metrics.RootMeanSquaredError(name="rmse")])
early_stopping = EarlyStopping(monitor = "val_loss", restore_best_weights = True, patience = 35)

history = model.fit(X_train_scaled, y_train, 
                   validation_data = (X_test_scaled, y_test),
                   callbacks = [early_stopping],
                   verbose = 1,
                   epochs = 100)


# !pip install scikeras


from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold
from tensorflow.keras.optimizers import Adam, RMSprop

def build_model(n_hidden1 = 128, n_hidden2 = 64, n_hidden3 = 32, activation = 'relu', learning_rate = 0.001, optimizer = 'adam'):
    model = Sequential([
        Dense(n_hidden1, input_shape = (X_train_scaled.shape[1],), activation = activation),
        Dense(n_hidden2, activation = activation),
        Dense(n_hidden3, activation = activation),
        Dense(1)
    ])

    if optimizer == 'adam':
        opt = Adam(learning_rate = learning_rate)
    elif optimizer == 'rmsprop':
        opt = RMSprop(learning_rate = learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")
        
    model.compile(
        optimizer = opt,
        loss = 'mse',
        metrics = [keras.metrics.RootMeanSquaredError(name = 'rmse')]
    )
    return model


# Wrap with KerasRegressor
regressor = KerasRegressor(model = build_model, verbose = 0)

# defining parameter grid
param_grid = {
    "model__n_hidden1": [64, 128],
    "model__n_hidden2": [32, 64],
    "model__n_hidden3": [16],
    "model__activation": ["relu", "elu"], 
    "model__learning_rate": [0.001],
    "batch_size": [64],
    "epochs": [20, 30]
}

random_cv = RandomizedSearchCV(estimator = regressor, param_distributions = param_grid, cv = 2, scoring = 'neg_root_mean_squared_error', verbose = 1, n_jobs = -1, random_state = 42)
early_stopping = EarlyStopping(monitor = "val_loss", restore_best_weights = True, patience = 35)

random_cv.fit(X_train_scaled, y_train, 
            validation_data = (X_test_scaled, y_test), 
            callbacks = [early_stopping])


print("Best Parameters:", random_cv.best_params_)
print("Best Score:", -random_cv.best_score_)  # convert negative RMSE to positive
best_model = random_cv.best_estimator_.model_


test_df.sample(5)


# Data Preprocessing
test_df_scaled = scaler.transform(test_df.drop('id', axis = 1))

test_df_scaled[0]


y_pred = best_model.predict(test_df_scaled)
y_pred


test_df.id


y_pred.flatten()


submission = pd.DataFrame({"id": test_df.id,
                           "BeatsPerMinute": y_pred.flatten()})

submission



submission.to_csv('submission.csv', index = False)
print("Sucessfully saved!")




