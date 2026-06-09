# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# This code block is not related to the project. It is to ignore the future warnings by seaborn and NaN values.

import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')

train.head()


# Information
print("Information : \n")
print(train.info())

# Missing values
print("\nMissing Values : \n")
print(train.isnull().sum())

# Duplicate Entries
print("\nDuplicate Entries : \n")
print(train.duplicated().sum())


plt.figure(figsize = (12,10))
n = 0 

numeric_columns = ['Time_spent_Alone','Social_event_attendance','Going_outside','Friends_circle_size','Post_frequency']

for col in numeric_columns:
    n = n+1
    plt.subplot(2,3,n)
    sns.histplot(data = train, x = train[col], kde = True)
    plt.title(f"{col} distribution")

plt.tight_layout()
plt.show()



# Filling Missing values

train[numeric_columns] = train[numeric_columns].fillna(train[numeric_columns].median())



# Train imputation 
"""
Filling the Stage_fear data with the help of social_attendance. If the social attendance of a person greater than the
overall mean , the person may have "No" stage fear and less than mean may results in "Yes".
"""
mean_social_attendance_1 = train['Social_event_attendance'].mean()
train['Stage_fear'] = train['Stage_fear'].fillna(
    train['Social_event_attendance'].apply(lambda x: "Yes" if x > mean_social_attendance_1 else "No")
)


"""
If the Friends size is greater than the overall mean of the friends size then, the person may not darined with socializing
else the "Yes".
"""
friends_size_mean_1 = train['Friends_circle_size'].mean()
train['Drained_after_socializing'] = train['Drained_after_socializing'].fillna(
    train['Friends_circle_size'].apply(lambda x : "Yes" if x>friends_size_mean_1 else "No")
)


# Categorical columns to numeric
map_values = {"Yes" : 1,
             "No" : 0}

train['Stage_fear'] = train['Stage_fear'].map(map_values)
train['Drained_after_socializing'] = train['Drained_after_socializing'].map(map_values)


personality_map = {"Introvert" : 0,
                  "Extrovert" : 1}

train['Personality'] = train['Personality'].map(personality_map)


train.head()


# Scaling
minmaxscaler = MinMaxScaler()
train[numeric_columns] = minmaxscaler.fit_transform(train[numeric_columns])



train.head()


# removing 'id' feature in train dataset and storing 'id' in test.

train = train.drop('id', axis = 1)


# splitting the train dataset to training and testing
x = train.drop('Personality', axis = 1)
y = train['Personality']

x_train,x_test,y_train,y_test = train_test_split(
    x,y,
    test_size = 0.2,
    stratify = y,
    random_state = 30
)



from sklearn.model_selection import GridSearchCV


# Define the model
loc_reg = LogisticRegression(random_state=2, max_iter=1000)

# Define the hyperparameter grid to search

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

grid_search = GridSearchCV(estimator=loc_reg, param_grid=param_grid, cv=5, scoring='accuracy')

# Perform the grid search on your training data

grid_search.fit(x_train, y_train)

# Get the best parameters and best score
print("Best hyperparameters found:", grid_search.best_params_)
print("Best cross-validation accuracy:", grid_search.best_score_)



# Use the best model to make predictions on the test set
best_model = grid_search.best_estimator_
loc_reg_pred = best_model.predict(x_test)


# Evaluate the final model on the test data
print("Final Logistic Regression Accuracy", accuracy_score(y_test, loc_reg_pred))

