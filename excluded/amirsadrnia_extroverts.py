import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline


train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


ids = test_df['id']


train_df.head()


train_df.info()


train_df.describe()


train_df.isnull().sum()


train_df.isnull().sum()/len(train_df)


train_df.duplicated().sum()


plt.figure(figsize=(12,5))
sns.pairplot(train_df)


sns.heatmap(train_df.corr(numeric_only=True),cmap='viridis',annot=True)


train_df['Time_spent_Alone'].hist(bins=10)


sns.set_style('whitegrid')
sns.countplot(train_df,x='Time_spent_Alone',hue='Personality')


sns.countplot(train_df,x='Social_event_attendance',hue='Personality')


sns.countplot(train_df,x='Stage_fear',hue='Personality')


sns.countplot(train_df,x='Going_outside',hue='Personality')


sns.countplot(train_df,x='Drained_after_socializing',hue='Personality')


sns.countplot(train_df,x='Friends_circle_size',hue='Personality')


sns.countplot(train_df,x='Post_frequency',hue='Personality')


sns.heatmap(train_df.isnull(),yticklabels=False,cbar=False,annot=False,cmap='viridis')


train_df['Personality'] = train_df['Personality'].map({'Introvert':0, 'Extrovert':1})


introvert_thresh = {
    'Time_spent_Alone': 4.0,
    'Social_event_attendance': 3.0,
    'Going_outside': 2.0,
    'Friends_circle_size': 4.0,
    'Post_frequency': 2.0
}

threshold_cols = list(introvert_thresh.keys())

def should_impute_lower(row, current_col):
    other_cols = [col for col in threshold_cols if col != current_col]
    for col in other_cols:
        if pd.notna(row[col]) and row[col] > introvert_thresh[col]:
            return False  
    return True 

for col in threshold_cols:
    threshold = introvert_thresh[col]
    for idx in train_df[train_df[col].isna()].index: 
        row = train_df.loc[idx]
        if should_impute_lower(row, col):
            train_df.at[idx, col] = threshold - 1 
        else:
            train_df.at[idx, col] = threshold  

for col in threshold_cols:
    threshold = introvert_thresh[col]
    for idx in test_df[test_df[col].isna()].index: 
        row = test_df.loc[idx]
        if should_impute_lower(row, col):
            test_df.at[idx, col] = threshold - 1 
        else:
            test_df.at[idx, col] = threshold 

categorical_cols = ['Stage_fear', 'Drained_after_socializing']
for col in categorical_cols:
    mode_val = train_df[col].mode()[0]  
    train_df[col].fillna(mode_val, inplace=True)
    test_df[col].fillna(mode_val, inplace=True)  


sns.heatmap(train_df.isnull(),yticklabels=False,cbar=False,annot=False,cmap='viridis')


train_df.info()


train_df['Drained_after_socializing'].value_counts()


train_df['Drained_after_socializing'] = train_df['Drained_after_socializing'].map({'No':0, 'Yes':1})
test_df['Drained_after_socializing'] = test_df['Drained_after_socializing'].map({'No':0, 'Yes':1})


train_df['Stage_fear'].value_counts()


train_df['Stage_fear'] = train_df['Stage_fear'].map({'No':0, 'Yes':1})
test_df['Stage_fear'] = test_df['Stage_fear'].map({'No':0, 'Yes':1})


train_df.drop('id',axis=1,inplace=True)
test_df.drop('id',axis=1,inplace=True)


X = train_df.drop('Personality',axis=1)
y = train_df['Personality']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


lr_model = LogisticRegression()

lr_model.fit(X_train,y_train)

lr_predictions = lr_model.predict(X_test)

print(confusion_matrix(y_test,lr_predictions))
print('\n')
print(classification_report(y_test,lr_predictions))


param_grid_svm = {'C' : [0.1, 1, 10, 100, 1000], 'gamma' : [1, 0.1, 0.01, 0.001, 0.0001], 'kernel' : ['rbf']}

grid_svm = GridSearchCV(SVC(),param_grid_svm,verbose=1)

grid_svm.fit(X_train,y_train)

svm_predictions = grid_svm.predict(X_test)

print(confusion_matrix(y_test,svm_predictions))
print('\n')
print(classification_report(y_test,svm_predictions))


dt_model = DecisionTreeClassifier()

dt_model.fit(X_train,y_train)

dt_predictions = dt_model.predict(X_test)

print(confusion_matrix(y_test,dt_predictions))
print('\n')
print(classification_report(y_test,dt_predictions))


rf_model = RandomForestClassifier(n_estimators=100)

rf_model.fit(X_train,y_train)

rf_predictions = rf_model.predict(X_test)

print(confusion_matrix(y_test,rf_predictions))
print('\n')
print(classification_report(y_test,rf_predictions))


X_train.shape


model = Sequential()

# input layer of nn
model.add(Dense(64,activation='relu',input_shape=(7,)))
model.add(BatchNormalization())
model.add(Dropout(0.1))

# first hidden layer
model.add(Dense(32,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))

# output layer
model.add(Dense(units=1,activation='sigmoid'))

optimizer = Adam(learning_rate=0.0001)

model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])


early_stop = EarlyStopping(monitor='val_loss',mode='min',patience=25,verbose=1,restore_best_weights=True)
model.fit(x=X_train, y=y_train, batch_size=128, epochs=600, validation_data=[X_test,y_test], callbacks=[early_stop])


loss = pd.DataFrame(model.history.history)
loss[['loss','val_loss']].plot()


X_test_kaggle = test_df.copy()


classes = {0:'Introvert',1:'Extrovert'}
label_mapping = np.vectorize(classes.get)

# logistic regression 
lr_predictions_kaggle = lr_model.predict(X_test_kaggle)
lr_predicted_labels = label_mapping(lr_predictions_kaggle)
submission_lr = pd.DataFrame({'id': ids,'Personality': lr_predicted_labels.reshape(-1)})
submission_lr.to_csv('submission_lr.csv', index=False)


# SVM 
svm_predictions_kaggle = grid_svm.predict(X_test_kaggle)
svm_predicted_labels = label_mapping(svm_predictions_kaggle)
submission_svm = pd.DataFrame({'id': ids,'Personality': svm_predicted_labels.reshape(-1)})
submission_svm.to_csv('submission_svm.csv', index=False)

# decision tree
dt_predictions_kaggle = dt_model.predict(X_test_kaggle)
dt_predicted_labels = label_mapping(dt_predictions_kaggle)
submission_dt = pd.DataFrame({'id': ids,'Personality' : dt_predicted_labels.reshape(-1)})
submission_dt.to_csv('submission_dt.csv', index=False)

# random forest
rf_predictions_kaggle = rf_model.predict(X_test_kaggle)
rf_predicted_labels = label_mapping(rf_predictions_kaggle)
submission_rf = pd.DataFrame({'id': ids,'Personality' : rf_predicted_labels.reshape(-1)})
submission_rf.to_csv('submission_rf.csv', index=False)

# nn 
nn_predictions_kaggle = model.predict(X_test_kaggle)
nn_predictions_kaggle = (nn_predictions_kaggle > 0.5).astype(int)
nn_predicted_labels = label_mapping(nn_predictions_kaggle)
submission_nn = pd.DataFrame({'id': ids,'Personality' : nn_predicted_labels.reshape(-1)})
submission_nn.to_csv('submission_nn.csv', index=False)




