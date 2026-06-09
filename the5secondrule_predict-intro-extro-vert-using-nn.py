import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set_style('whitegrid')

import warnings
warnings.filterwarnings('ignore')


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')


df.head()


df.info()


df.describe()


df.isnull().sum()


sns.pairplot(df, hue = 'Personality')
print('From the visualization, all features other than id seem to significantly contribute to personality')


df.corr(numeric_only = True)


df2 = df.dropna(axis = 0, how = 'any')
len(df2) / len(df)
print('Dropping all NA will reduce the number of data to only {:.1f}% of the original size'.format(100*len(df2) / len(df)))
print('/n')
print('To improve the model, can try to impute missing values')


df2.columns


X = df2.drop(['id','Personality'], axis = 1)


y = df2['Personality']


#This is for competition submission
comp_test_data = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


comp_test_data.head()


X_test_comp = comp_test_data.drop(['id'], axis = 1)


X = pd.get_dummies(data = X, 
               columns = ['Stage_fear','Drained_after_socializing'], 
               drop_first = True)


X_test_comp = pd.get_dummies(data = X_test_comp, 
               columns = ['Stage_fear','Drained_after_socializing'], 
               drop_first = True)


# Split X and y into train and test set
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler()


scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)


# Build a NN with early stopping and dropout layer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


model = Sequential()
model.add(Dense(30, activation = 'relu'))
model.add(Dropout(rate = 0.5, seed = 101))
model.add(Dense(15, activation = 'relu'))
model.add(Dropout(rate = 0.5, seed = 101))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy')


# Early Stopping
from tensorflow.keras.callbacks import EarlyStopping


early_stop = EarlyStopping(monitor='val_loss',
                           mode = 'min',
                           verbose = 0,
                           patience = 25)


type(scaled_X_train)


type(y_train.values)


y_test.values


# Map Extrovert and Introvert to 1 and 0


y_train = y_train.map({'Extrovert':1, 'Introvert':0})


y_train.head()


y_test  = y_test.map({'Extrovert':1, 'Introvert':0})


model.fit(scaled_X_train, y_train.values, epochs = 1000,
         validation_data = (scaled_X_test, y_test.values),
         callbacks = [early_stop],verbose = 0)


pd.DataFrame(model.history.history).plot()


predicted_prob = model.predict(scaled_X_test)
predicted_class = (predicted_prob > 0.5).astype('int')



from sklearn.metrics import classification_report, confusion_matrix


print(classification_report(y_test.values, predicted_class))


print(confusion_matrix(y_test.values, predicted_class))


from tensorflow.keras.models import load_model


model.save('model_1.h5')


# Apply model_1 to predict the test data


model = load_model('model_1.h5')


y_test_comp_pred = model.predict(X_test_comp)


y_test_comp_pred


# Imputate missing value with the average values
X_test_comp['Time_spent_Alone'] = X_test_comp['Time_spent_Alone'].apply(lambda x: X_test_comp['Time_spent_Alone'].mean() if pd.isnull(x) else x)
X_test_comp['Social_event_attendance'] = X_test_comp['Social_event_attendance'].apply(lambda x: X_test_comp['Social_event_attendance'].mean() if pd.isnull(x) else x)
X_test_comp['Going_outside'] = X_test_comp['Going_outside'].apply(lambda x: X_test_comp['Going_outside'].mean() if pd.isnull(x) else x)
X_test_comp['Friends_circle_size'] = X_test_comp['Friends_circle_size'].apply(lambda x: X_test_comp['Friends_circle_size'].mean() if pd.isnull(x) else x)
X_test_comp['Post_frequency'] = X_test_comp['Post_frequency'].apply(lambda x: X_test_comp['Post_frequency'].mean() if pd.isnull(x) else x)


X_test_comp.isnull().sum()


scaled_X_test_comp = scaler.transform(X_test_comp)
y_test_comp_pred = model.predict(scaled_X_test_comp)


y_test_comp_predicted_class = (y_test_comp_pred > 0.5).astype('int')


y_test_comp_predicted_class


submission = pd.DataFrame({'id': comp_test_data['id'], 'Personality':y_test_comp_predicted_class.reshape(-1)})


submission['Personality'] = submission['Personality'].map({1:'Extrovert',0:'Introvert'})


submission.head()


submission.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")

