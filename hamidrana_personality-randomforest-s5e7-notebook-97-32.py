import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

# for data splitting and evaluation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# for modeling
from sklearn.ensemble import RandomForestClassifier 


#it will show every column in data frame
pd.set_option('display.max_columns',None)


# loading data
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_sample_submission = pd.read_csv('sample_submission.csv')


df_train.head()


df_train.dtypes


df_train.shape


df_train.info()


df_train.duplicated().sum()  # checking for duplicates


df_train.isnull().sum()  # checking for nan/null values


# checking the skewness
df_train[['Time_spent_Alone', 'Social_event_attendance', 
          'Going_outside', 'Friends_circle_size', 'Post_frequency']].skew()

## Skewness Value	Shape
# ≈ 0	Symmetric (Normal)
# > 0	Right Skew (Positive)
# < 0	Left Skew (Negative)


# we will use median for Time_spent_alone and mean for others


# filling the null/nan values of columns with int data types
df_train['Time_spent_Alone'].fillna(np.median(df_train['Time_spent_Alone'].dropna()), inplace= True)
df_train['Social_event_attendance'].fillna(np.mean(df_train['Social_event_attendance']),inplace=True)
df_train['Going_outside'].fillna(np.mean(df_train['Going_outside']),inplace=True)
df_train['Friends_circle_size'].fillna(np.mean(df_train['Friends_circle_size']),inplace=True)
df_train['Post_frequency'].fillna(np.mean(df_train['Post_frequency']),inplace=True)

# filling Categorical data 
df_train['Stage_fear'].fillna((df_train['Stage_fear']).mode()[0],inplace=True)
df_train['Drained_after_socializing'].fillna((df_train['Drained_after_socializing']).mode()[0],inplace=True)



df_train.isnull().sum()  # recheacking for nan/null values


# encoding categorical features


# encoding Stage_fear 
pd.get_dummies(df_train['Stage_fear'])
df_train['Stage_fear'] = df_train['Stage_fear'].map({'No':0,'Yes': 1})


# encoding Drained_after_socializing
pd.get_dummies(df_train['Drained_after_socializing'])
df_train['Drained_after_socializing'] = df_train['Drained_after_socializing'].map({'No':0,'Yes': 1})


# encoding Personality
pd.get_dummies(df_train['Personality'])
df_train['Personality'] = df_train['Personality'].map({'Introvert':0,'Extrovert': 1})


df_train.drop('id',axis=1 ,inplace=True)  # removing id its useless


df_train.head()


df_test.head()


df_test.dtypes


df_test.shape


df_test.info()


df_test.duplicated().sum() # checking for duplicates


df_test.isnull().sum () # checking for null/nan values


# checking the skewness
df_test[['Time_spent_Alone', 'Social_event_attendance', 
          'Going_outside', 'Friends_circle_size', 'Post_frequency']].skew()

## Skewness Value	Shape
# ≈ 0	Symmetric (Normal)
# > 0	Right Skew (Positive)
# < 0	Left Skew (Negative)


# we will use median for Time_spent_alone and mean for others


# filling the null/nan values of columns with int data types
df_test['Time_spent_Alone'].fillna(np.median(df_test['Time_spent_Alone'].dropna()), inplace= True)
df_test['Social_event_attendance'].fillna(np.mean(df_test['Social_event_attendance']),inplace=True)
df_test['Going_outside'].fillna(np.mean(df_test['Going_outside']),inplace=True)
df_test['Friends_circle_size'].fillna(np.mean(df_test['Friends_circle_size']),inplace=True)
df_test['Post_frequency'].fillna(np.mean(df_test['Post_frequency']),inplace=True)

# filling Categorical data 
df_test['Stage_fear'].fillna((df_test['Stage_fear']).mode()[0],inplace=True)
df_test['Drained_after_socializing'].fillna((df_test['Drained_after_socializing']).mode()[0],inplace=True)



df_test.isnull().sum()   # rechecking for nan/null values


# encoding of Categorical data


# encoding Stage_fear 
pd.get_dummies(df_test['Stage_fear'])
df_test['Stage_fear'] = df_test['Stage_fear'].map({'No':0,'Yes': 1})


# encoding Drained_after_socializing
pd.get_dummies(df_test['Drained_after_socializing'])
df_test['Drained_after_socializing'] = df_test['Drained_after_socializing'].map({'No':0,'Yes': 1})


df_test.head()


# splitting data into features and target
X = df_train.drop('Personality', axis=1)
y = df_train['Personality']

# splitting data into train and test 
X_train , X_val , y_train , y_val = train_test_split(X, y , test_size=0.2,stratify=y, random_state=42)


# instantiating model 
clf = RandomForestClassifier(random_state=42)
model = clf.fit(X_train,y_train)

# making prediction
y_pred = clf.predict(X_val)

# Evaluate Accuracy 
print(f"Vaidate Accuracy: {accuracy_score(y_val,y_pred):.6f}")


importances = clf.feature_importances_
features = X.columns
sns.barplot(x=importances, y=features)
plt.title('Feature Importance (RandomForest)')
plt.ylabel('Features')
plt.xlabel('Importance level')
plt.show()


param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=1,
    verbose=2
)

grid_search.fit(X_train,y_train)


print("Best Parameters:", grid_search.best_params_)


# Evaluate best model on validation set
best_rf = grid_search.best_estimator_
y_pred_best = best_rf.predict(X_val)


accuracy_best = accuracy_score(y_val, y_pred_best)
print(f'Tuned Validation Accuracy: {accuracy_best:.6f}')


print(f"Vaidate Accuracy: {accuracy_score(y_val,y_pred):.6f}")


df_train.head()


# Prepare full data (exclude 'id' and 'Personality')
X_full = df_train.drop('Personality',axis=1)
y_full = df_train['Personality']

# Create model with best parameters
final_rf = RandomForestClassifier(
    max_depth=5,
    max_features='sqrt',
    min_samples_leaf=2,
    min_samples_split=2,
    n_estimators=300,
    random_state=42
)

# Train on full dataset
final_rf.fit(X_full, y_full)



y_pred_train = final_rf.predict(X_full)
train_accuracy = accuracy_score(y_full, y_pred_train)
print(f'Accuracy on Full Training Data: {train_accuracy:.6f}')


df_test.head()


# Assuming your test dataset is df_test
X_test = df_test.drop(columns=['id'])
test_predictions = final_rf.predict(X_test)


submission = pd.DataFrame({
    'id': df_test['id'],
    'Personality': test_predictions
})

# Save to CSV
submission.to_csv('results.csv', index=False)


result = pd.read_csv('submission.csv')


result.head(10)


# encoding personality 
result['Personality'] = result['Personality'].map({1: 'Extrovert', 0 : 'Introvert'})


result.head(10)


result.to_csv('submission.csv', index=False)

