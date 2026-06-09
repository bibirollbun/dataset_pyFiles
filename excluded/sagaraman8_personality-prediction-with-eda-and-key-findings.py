# For data cleaning and visualization.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# For Data preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# For Model Selection and Metric calculation
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Various ML algorithms
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


test_df = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
train_df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
print("Test Data Shape: ", test_df.shape)
print("Train Data Shape: ", train_df.shape)


train_df.head()


test_df.head()


print("Train Data info: ")
print( train_df.info())
print("\nTest Data info: ")
print(test_df.info())


def get_null_values():
  # Creating a DataFrame with percentage of missing data from the columns of Train Data.
  train_null_cols = train_df.columns[train_df.isnull().any()].tolist()
  train_null_cols_num = [train_df[col].isnull().sum() for col in train_null_cols]
  train_null_cols_percentage = [train_df[col].isnull().sum()/train_df.shape[0]*100 for col in train_null_cols]
  train_null_df = pd.DataFrame({'Column Name': train_null_cols,
                                'Missing Number': train_null_cols_num,
                                'Missing Percentage': train_null_cols_percentage})

  # Creating a DataFrame with percentage of missing data from the columns of Test data.
  test_null_cols = test_df.columns[test_df.isnull().any()].tolist()
  test_null_cols_num = [test_df[col].isnull().sum() for col in test_null_cols]
  test_null_cols_percentage = [test_df[col].isnull().sum()/test_df.shape[0]*100 for col in test_null_cols]
  test_null_df = pd.DataFrame({'Column Name': test_null_cols,
                                'Missing Number': test_null_cols_num,
                                'Missing Percentage': test_null_cols_percentage})

  print("Train Data Missing Data: ")
  print(train_null_df)
  print("\nTest Data Missing Data: ")
  print(test_null_df)

get_null_values()


print(train_df['Stage_fear'].value_counts())
print("\nPercentage of values in Stage_fear:")
print(train_df['Stage_fear'].value_counts(normalize=True) * 100)


# Impute missing 'Stage_fear' values with 'U' in both dataframes
train_df['Stage_fear'].fillna('U', inplace=True)
test_df['Stage_fear'].fillna('U', inplace=True)

print("Null values in stage_fear column in train data: ")
display(train_df['Stage_fear'].isnull().sum())
print("Null values in stage_fear column in train data: ")
display(test_df['Stage_fear'].isnull().sum())


sns.histplot(x='Social_event_attendance', data=train_df, kde=True)
plt.show()


imputer_SET = SimpleImputer(strategy='median')
train_df['Social_event_attendance'] = imputer_SET.fit_transform(train_df[['Social_event_attendance']])
test_df['Social_event_attendance'] = imputer_SET.transform(test_df[['Social_event_attendance']])

print("Null values in stage_fear column in train data: ")
display(train_df['Social_event_attendance'].isnull().sum())
print("Null values in stage_fear column in train data: ")
display(test_df['Social_event_attendance'].isnull().sum())


get_null_values()


sns.histplot(x='Going_outside', data=train_df, kde=True)
plt.show()


imputer_GO = SimpleImputer(strategy='median')
train_df['Going_outside'] = imputer_GO.fit_transform(train_df[['Going_outside']])
test_df['Going_outside'] = imputer_GO.transform(test_df[['Going_outside']])

print("Null values in stage_fear column in train data: ")
display(train_df['Going_outside'].isnull().sum())
print("Null values in stage_fear column in train data: ")
display(test_df['Going_outside'].isnull().sum())


sns.histplot(x='Time_spent_Alone', data=train_df, kde=True)
plt.show()


imputer_TSA = SimpleImputer(strategy='median')
train_df['Time_spent_Alone'] = imputer_TSA.fit_transform(train_df[['Time_spent_Alone']])
test_df['Time_spent_Alone'] = imputer_TSA.transform(test_df[['Time_spent_Alone']])

print("Null values in stage_fear column in train data: ")
display(train_df['Time_spent_Alone'].isnull().sum())
print("Null values in stage_fear column in train data: ")
display(test_df['Time_spent_Alone'].isnull().sum())

get_null_values()


train_df['Friends_circle_size'] = train_df.groupby(['Time_spent_Alone',
                                                    'Going_outside',
                                                    'Social_event_attendance']
                                                   )['Friends_circle_size'].transform(lambda x: x.fillna(x.median()))
test_df['Friends_circle_size'] = test_df.groupby(['Time_spent_Alone',
                                                  'Going_outside',
                                                  'Social_event_attendance']
                                                 )['Friends_circle_size'].transform(lambda x: x.fillna(x.median()))

get_null_values()


sns.histplot(x='Friends_circle_size', data=train_df, kde=True)
plt.show()


train_df['Friends_circle_size'] = train_df['Friends_circle_size'].fillna(train_df['Friends_circle_size'].median())
test_df['Friends_circle_size'] = test_df['Friends_circle_size'].fillna(test_df['Friends_circle_size'].median())

get_null_values()


sns.histplot(x='Post_frequency', data=train_df, kde=True)
plt.show()


train_df['Post_frequency'] = train_df['Post_frequency'].fillna(train_df['Post_frequency'].median())
test_df['Post_frequency'] = test_df['Post_frequency'].fillna(test_df['Post_frequency'].median())

get_null_values()


train_df['Drained_after_socializing'].fillna('U', inplace=True)
test_df['Drained_after_socializing'].fillna('U', inplace=True)

print("Total number of null values in train Data: ", train_df.isnull().sum().sum())
print("Total number of null values in test Data: ", test_df.isnull().sum().sum())


def compare_plot(col):
  fig, axes = plt.subplots(1, 2, figsize=(10,5))
  ax1 = sns.countplot(x=col, data=train_df,hue='Personality',ax=axes[0])
  ax1.tick_params(axis='x', rotation=0)
  ax1.set_title(f'{col} vs Personality')
  ax1.legend(title='Personality')

  cross_tab = pd.crosstab(train_df[col], train_df['Personality'])
  ax2 = cross_tab.plot(kind='bar', stacked=True, ax=axes[1])
  ax2.tick_params(axis='x', rotation=0)
  ax2.set_title(f"Bar plot of Crosstab between {col} and Personality")
  plt.show()

  print(f'\nPercentage of Personality by {col}:\n{cross_tab.round(2)}\n')


compare_plot('Stage_fear')


compare_plot('Drained_after_socializing')


def distribution_plot(column):
  plt.figure(figsize=(10, 6))
  sns.histplot(data=train_df, x=column, hue='Personality', kde=True, multiple="stack")
  plt.title(f'Distribution of {column} by Personality')
  plt.xlabel(column)
  plt.ylabel('Frequency')
  plt.show()


distribution_plot('Time_spent_Alone')


distribution_plot('Social_event_attendance')


distribution_plot('Going_outside')


distribution_plot('Friends_circle_size')


distribution_plot('Post_frequency')


train_df = pd.get_dummies(train_df, columns=['Stage_fear','Drained_after_socializing'], dummy_na=False)
test_df = pd.get_dummies(test_df, columns=['Stage_fear','Drained_after_socializing'], dummy_na=False)

print("Shape of train data after creating dummies: ", train_df.shape)
print("Shape of test data after creating dummies: ", test_df.shape)


print(train_df.info())
print(test_df.info())


train_df = train_df.drop(['id'], axis=1)
pre_test_df = test_df.drop(['id'], axis=1)


numerical_cols_train = train_df.select_dtypes(include=np.number).columns.tolist()
numerical_cols_test = pre_test_df.select_dtypes(include=np.number).columns.tolist()

print(numerical_cols_train)
print(numerical_cols_test)


scaler = StandardScaler()
for col in numerical_cols_train:
  train_df[col] = scaler.fit_transform(train_df[[col]])
  pre_test_df[col] = scaler.transform(pre_test_df[[col]])

train_df.head()


train_df['Personality'] = train_df['Personality'].replace({'Extrovert':0, 'Introvert':1})


X = train_df.drop('Personality', axis=1)
y = train_df['Personality']


models = [('Random Forest', RandomForestClassifier()),
          ('Logistic Regression', LogisticRegression()),
          ('SVM', SVC()),
          ('KNeighborsClassifier', KNeighborsClassifier()),
          ('Decision Tree', DecisionTreeClassifier()),
          ('Gaussian NB', GaussianNB()),
          ('Gradient Boosting', GradientBoostingClassifier()),
          ('XG Boosting', XGBClassifier())]

for name, model in models:
  scores = cross_val_score(model, X, y, cv=5)
  print(f'{name} Accuracy: {scores.mean()}')


param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, y)

print("Best Parameters: ", grid_search.best_params_)
print("Best Cross-Validation Accuracy: ", grid_search.best_score_)

best_lr_model = grid_search.best_estimator_


prediction = best_lr_model.predict(pre_test_df)
predictions_final = pd.Series(prediction).map({0:'Extrovert', 1:'Introvert'})
predictions_final


submission = pd.DataFrame({'id': test_df['id'], 'Personality': predictions_final})
submission.to_csv('submission.csv', index=False)
print("Submission file created successfully.")

