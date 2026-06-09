# Import
import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression


# Data load
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


train.info()


train.isnull().sum()


test.isnull().sum()


# Stage_fear, Drained_after_socializing : categorize to number(0 or 1)
Stage_fear_name = ['Stage_fear', 'No', 'Yes']
Drained_after_socializing_name = ['Drained_after_socializing', 'No', 'Yes']

def change_01(list_name):
    for i in range(2):
        train.loc[train[list_name[0]]==list_name[i+1], list_name[0]] = i
        test.loc[test[list_name[0]]==list_name[i+1], list_name[0]] = i

change_01(Stage_fear_name)
change_01(Drained_after_socializing_name)


# Stage_fear, Drained_after_socializing null : fill mode 
train['Stage_fear'] = train['Stage_fear'].infer_objects(copy=False).fillna(train['Stage_fear'].mode()[0])
train['Drained_after_socializing'] = train['Drained_after_socializing'].infer_objects(copy=False).fillna(train['Drained_after_socializing'].mode()[0])

test['Stage_fear'] = test['Stage_fear'].infer_objects(copy=False).fillna(test['Stage_fear'].mode()[0])
test['Drained_after_socializing'] = test['Drained_after_socializing'].infer_objects(copy=False).fillna(test['Drained_after_socializing'].mode()[0])


# null value => median 
numerical_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']

for i in range(len(numerical_cols)):
    train[numerical_cols[i]] = train[numerical_cols[i]].infer_objects(copy=False).fillna(train[numerical_cols[i]].median())  
    test[numerical_cols[i]] = test[numerical_cols[i]].infer_objects(copy=False).fillna(test[numerical_cols[i]].median())


# box plot 
plt.figure(figsize=(15, 8)) 

for i, col in enumerate(numerical_cols):
    plt.subplot(2, 3, i + 1) 
    sns.boxplot(y=train[col]) 
    plt.title(f'Box Plot of {col}')
    plt.ylabel(col)

plt.tight_layout()
plt.show()


# histogram
plt.figure(figsize=(15, 8))

for i, col in enumerate(numerical_cols):
    plt.subplot(2, 3, i + 1)
    sns.histplot(train[col], kde=True) 
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    
plt.tight_layout()
plt.show()


# capping outlier : 99% quantile 
upper_bound_tsa = train['Time_spent_Alone'].quantile(0.99)

train['Time_spent_Alone'] = np.where(train['Time_spent_Alone'] > upper_bound_tsa, upper_bound_tsa, train['Time_spent_Alone'])
test['Time_spent_Alone'] = np.where(test['Time_spent_Alone'] > upper_bound_tsa, upper_bound_tsa, test['Time_spent_Alone'])


# Log Transformation
train['Time_spent_Alone'] = np.log1p(train['Time_spent_Alone'])
test['Time_spent_Alone'] = np.log1p(test['Time_spent_Alone'])


# Personality : only train data 
train['Personality'] = train['Personality'].map({'Introvert': 0, 'Extrovert': 1})


X_train = train.drop('Personality', axis=1)
y_train = train['Personality']
y_train = y_train.astype(int)

X_test = test 


# normalization train, test data
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])


param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  
    'penalty': ['l1', 'l2']      
}

model = LogisticRegression(random_state=42, solver='liblinear', max_iter=1000)
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)


grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_


predictions = best_model.predict(X_test)

reverse_personality_map = {0: 'Introvert', 1: 'Extrovert'}
predictions_mapped = np.vectorize(reverse_personality_map.get)(predictions)

submission = pd.DataFrame({
     'id': test['id'],
     'Personality': predictions_mapped
 })

submission.to_csv('submission.csv', index=False)




