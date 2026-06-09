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


train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


train.head()


test.head()


train.isnull().any()


test.isnull().any()


test.isnull().sum()


test["winddirection"].dtype


test['winddirection'] = test['winddirection'].fillna(test['winddirection'].mean())


train.shape , test.shape


## dropping id and day column
train.drop(columns =['id','day'],inplace = True)
test.drop(columns=['id','day'],inplace = True)
print(train.head())
print(test.head())


train.head()


test.head()


class_counts = train['rainfall'].value_counts()
print(class_counts)


train.describe()


import matplotlib.pyplot as plt
class_counts.plot(kind='bar', color=['red', 'blue'])
plt.title('Rainfall Distribution')
plt.xlabel('Rainfall (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()


!pip install imblearn


## handling data imbalance SMOTE

from imblearn.over_sampling import SMOTE


X = train.drop(['rainfall'], axis=1)
y = train['rainfall']


train.columns


oversample = SMOTE()
X,y = oversample.fit_resample(train[['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity',
       'cloud', 'sunshine', 'winddirection', 'windspeed']],train['rainfall'])


X.shape, y.shape


y.value_counts()


import seaborn as sns
X_df = pd.DataFrame(X, columns=['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 
                                'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed'])

data = X_df.copy()
data['rainfall'] = y

correlation_matrix = data.corr()

### Create the heatmap
plt.figure(figsize=(7, 7))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, 
            fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Features and Rainfall (Post-SMOTE)')
plt.show()


X_df = pd.DataFrame(X, columns=['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 
                                'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed'])

X_reduced = X_df.drop(['maxtemp', 'mintemp', 'dewpoint'], axis=1)


X_reduced


X_test_reduced = test.drop(['maxtemp', 'mintemp', 'dewpoint'], axis=1)


X_test_reduced


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(test)


X_train_scaled 


X_test_scaled


# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.metrics import AUC
from sklearn.model_selection import train_test_split


X_train_split, X_val, y_train_split, y_val = train_test_split(X_train_scaled, y, test_size=0.2, random_state=42)


# (X_train_split.shape[1],)


# model = Sequential([
#     Dense(64, activation='relu', input_shape=(X_train_split.shape[1],), kernel_regularizer='l2'),
#     Dropout(0.2),
#     Dense(32, activation='relu', kernel_regularizer='l2'),
#     Dropout(0.2),
#     Dense(1, activation='sigmoid')
# ])


# model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=[AUC(name='roc_auc')])


# history = model.fit(X_train_split, y_train_split, validation_data=(X_val, y_val), 
#                     epochs=100, batch_size=32, verbose=1)


# val_loss, val_auc = model.evaluate(X_val, y_val)
# print(f"Validation ROC AUC: {val_auc}")


subm = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")


# y_pred = model.predict(X_test_scaled).flatten()


# subm['rainfall'] = y_pred
# subm.to_csv('submition.csv', index=False)
# subm.head()


# from tensorflow.keras.callbacks import EarlyStopping
# early_stopping = EarlyStopping(monitor='val_roc_auc', patience=10, mode='max', restore_best_weights=True)
# history = model.fit(X_train_split, y_train_split, validation_data=(X_val, y_val), 
#                     epochs=100, batch_size=32, callbacks=[early_stopping], verbose=1)


# y_pred = model.predict(X_test_scaled,verbose=1).flatten()


# subm['rainfall'] = y_pred
# subm.to_csv('submition02.csv', index=False)
# subm.head()


# def create_model(neurons=32,layers=1):
#     model = Sequential()
#     model.add(Dense(neurons,activation='relu',input_shape=(X_train_split.shape[1],)))

#     for _ in range(layers-1):
#         model.add(Dense(neurons,activation='relu'))
    
#     model.add(Dense(1,activation='sigmoid'))
#     model.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics=[AUC(name='roc_auc')])

#     return model


# !pip install scikeras


# ## create a keras classifier
# from scikeras.wrappers import KerasClassifier
# model = KerasClassifier(build_fn=create_model, layers=1, neurons=32, verbose=1)


# param_grid = {
#     'neurons': [16,32,64,128],
#     'layers': [1,2,3],
#     'epochs': [50,100]
# }


# ## GridSearchCV
# from sklearn.model_selection import GridSearchCV
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3)
# grid_result = grid.fit(X_train_split, y_train_split)

# print("Best: %f using %s " % (grid_result.best_score_, grid_result.best_params_))


from sklearn.ensemble import RandomForestClassifier


model = RandomForestClassifier(
    n_estimators=100,
    criterion='gini',
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    n_jobs=-1,
    random_state=42,
    class_weight='balanced',
    ccp_alpha=0.0
)


y.value_counts()


from sklearn.model_selection import cross_val_score
rf_scores = cross_val_score(model, X_train_split, y_train_split, cv=5, scoring='roc_auc')
print("Initial Random Forest Cross-Validation ROC AUC Scores:", rf_scores)
print("Mean ROC AUC:", rf_scores.mean())


model.fit(X_train_split, y_train_split)


# test_predictions = model.predict_proba(X_test_scaled)[:, 1]

y_val_pred = model.predict(X_val)


y_val_pred


y_pred = model.predict(X_test_scaled).flatten()


y_pred


subm['rainfall'] = y_pred
subm.to_csv('submition03.csv', index=False)
subm.head()


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy}")


print(classification_report(y_val, y_val_pred))




