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


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam


train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


test.isnull().sum()


test['winddirection'].fillna(test['winddirection'].median(), inplace=True)


for col in train.columns:
    plt.figure(figsize=(4, 2))
    sns.histplot(data=train, hue = 'rainfall', x = col, kde = True)


sns.pairplot(train)


#temp difference
train['temp_diff']= train.maxtemp - train.mintemp
test['temp_diff']= test.maxtemp - test.mintemp


#temp dew ratio in Kelvin
train['temp_dew_ratio']= (train.dewpoint+273.0)/(train.mintemp+273.0)
test['temp_dew_ratio']= (test.dewpoint+273.0)/(test.mintemp+273.0)


#th ratio ratio
train['th_ratio']= (train.maxtemp - train.mintemp)/(train.humidity)
test['th_ratio']= (test.maxtemp - test.mintemp)/(test.humidity)



train['p_diff'] = train['pressure'].diff().fillna(0)
test['p_diff'] = test['pressure'].diff().fillna(0)


train['T_diff'] = train['temparature'].diff().fillna(0)
test['T_diff'] = test['temparature'].diff().fillna(0)


X = train.loc[:,~train.columns.isin(['id','rainfall','min_temp','max_temp'])]
y = train.rainfall


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


#resampling
from imblearn.over_sampling import SMOTE
smote=SMOTE(sampling_strategy='minority')
X_scaled_resampled, y_resampled = smote.fit_resample(X_scaled, y)


param_grid = {
    'n_estimators': np.arange(100, 1500, 400),
    'max_depth': [None] + list(np.arange(10, 50, 20)),
    'min_samples_split': np.arange(2, 10),
    'min_samples_leaf': np.arange(1, 5)}


#Random Forest
# Initialize the Random Forest Classifier
rf = RandomForestClassifier(random_state=42)

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=10,
    cv=3,  
    random_state=42,
    n_jobs=-1 
)

# Fit RandomizedSearchCV to the training data
random_search.fit(X_scaled, y)


# Print the best hyperparameters found
print("Best hyperparameters:", random_search.best_params_)

# Print the best score obtained
print("Best score:", random_search.best_score_)

# Get the best model
best_rf = random_search.best_estimator_


feature_importance = best_rf.feature_importances_
importance_df = pd.DataFrame({
    "Feature": X.columns,  
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)
plt.figure(figsize=(10, 5))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("RF Feature Importance")
plt.gca().invert_yaxis()  
plt.show()


from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)


model = Sequential([
    Dense(128, activation='relu', input_shape=(X_scaled.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])


optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(X_scaled, y, epochs=200, batch_size=32, validation_split=0.2, 
                    callbacks=[early_stopping], verbose=1)


#plot history
fig, ax = plt.subplots()
ax.plot(history.history["loss"],'r', marker='.', label="Train Loss")
ax.plot(history.history["val_loss"],'b', marker='.', label="Validation Loss")
ax.legend()


#!pip install scikeras


from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier, KerasRegressor


# Define Keras model function
def create_model(optimizer='adam', activation='relu'):
    model = Sequential([
    Dense(128, activation=activation, input_shape=(X_scaled.shape[1],)),
    Dropout(0.3),
    Dense(64, activation=activation),
    Dropout(0.3),
    Dense(32, activation=activation),
    Dropout(0.3),
    Dense(16, activation=activation),
    Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Define hyperparameter grid
param_grid = {
    'optimizer': ['adam', 'rmsprop'],
    'activation': [ 'relu','sigmoid', 'tanh']
}

# Create KerasClassifier
model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=64, 
                        activation='relu',verbose=0)

# Perform GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_search_result = grid_search.fit(X_scaled, y)


print("Best: %f using %s" % (grid_search_result.best_score_, grid_search_result.best_params_))


activation = grid_search_result.best_params_['activation']
optimizer = grid_search_result.best_params_['optimizer']


model = Sequential([
    Dense(128, activation=activation, input_shape=(X_scaled.shape[1],)),
    Dropout(0.3),
    Dense(64, activation=activation),
    Dropout(0.3),
    Dense(32, activation=activation),
    Dropout(0.3),
    Dense(16, activation=activation),
    Dense(1, activation='sigmoid')  # Binary classification
])


model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(X_scaled, y, epochs=200, batch_size=64, validation_split=0.2, 
                    callbacks=[early_stopping], verbose=1)


#plot history
fig, ax = plt.subplots()
ax.plot(history.history["loss"],'r', marker='.', label="Train Loss")
ax.plot(history.history["val_loss"],'b', marker='.', label="Validation Loss")
ax.legend()


plt.hist(model.predict(X_scaled))


X_test = test.loc[:,~test.columns.isin(['id','min_temp','max_temp'])]


X_test_scaled = scaler.transform(X_test)


y_test_pred = model.predict(X_test_scaled)


#check the distribution
#plt.hist(y_test_pred.flatten())




#submission
submission = pd.DataFrame()
submission['id'] = test.id
submission['rainfall'] = y_test_pred


submission.to_csv('submission.csv', index = False)

