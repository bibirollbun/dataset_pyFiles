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


import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
from sklearn.metrics import accuracy_score
#from catboost import CatBoostClassifier
from scipy.stats import mode




from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report




train=pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test=pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


train.head()


numerical_features = train.select_dtypes(include=['number']).columns
categorical_cols = train.select_dtypes(exclude=['number']).columns

train[numerical_features] = train[numerical_features].fillna(train[numerical_features].mean())

for col in categorical_cols:
    if train[col].isnull().any():
        train[col] = train[col].fillna(train[col].mode()[0])


numerical_features = test.select_dtypes(include=['number']).columns
categorical_cols = test.select_dtypes(exclude=['number']).columns

test[numerical_features] = test[numerical_features].fillna(test[numerical_features].mean())

for col in categorical_cols:
    if test[col].isnull().any():
        test[col] = test[col].fillna(test[col].mode()[0])


train.info()


train.head()


for feature in ['Stage_fear','Drained_after_socializing']:
    train[feature]=le.fit_transform(train[feature])
    test[feature]=le.fit_transform(test[feature])


train.head()


train['Time_spent_Alone']=train['Time_spent_Alone'].astype(int)
test['Time_spent_Alone']=test['Time_spent_Alone'].astype(int)


# Social Engagement Score (interaction term)
train['Social_score'] = (train['Social_event_attendance'] + train['Going_outside'] + train['Friends_circle_size'])
# Introvert-Tendency Proxy
train['Introvert_score'] = (train['Time_spent_Alone'] - train['Social_score'])
train['Inp']=train['Introvert_score']-train['Post_frequency']
train['set']=train['Social_event_attendance']-train['Time_spent_Alone']
train['In_ex']=train['Stage_fear']+train['Drained_after_socializing']

# Social Engagement Score (interaction term)
test['Social_score'] = (test['Social_event_attendance'] + test['Going_outside'] + test['Friends_circle_size'])
# Introvert-Tendency Proxy
test['Introvert_score'] = (test['Time_spent_Alone'] - test['Social_score'])
test['Inp']=test['Introvert_score']-test['Post_frequency']
test['set']=test['Social_event_attendance']-test['Time_spent_Alone']
test['In_ex']=test['Stage_fear']+test['Drained_after_socializing']


train.head(10)


train.tail(20)



def team(a):
    if a<-1:
        return 0
    else:
        return 1
train['team']=train['set'].apply(team)
test['team']=test['set'].apply(team)
        


train.head()


# Group by target and calculate the mean of the feature
grouped = train.groupby('Personality')['team']

# Plotting
grouped.plot(kind='bar', color='skyblue')
plt.title('Average Feature Value per Target Class')
plt.xlabel('Target')
plt.ylabel('Average Feature')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.show()


'''#ct = pd.crosstab(train['Personality'], train['set'])
sns.heatmap(train.corr(), annot=True, fmt='d', cmap='coolwarm')
plt.show()'''


'''ct.plot(kind='bar')
plt.xlabel('Feature')
plt.ylabel('Count')
plt.title('Target Distribution by Feature')
plt.show()
plt.show()'''


# Group by target and calculate the mean of the feature
grouped = train.groupby('Personality')['Post_frequency'].mean()

# Plotting
grouped.plot(kind='hist', color='skyblue')
plt.title('Average Feature Value per Target Class')
plt.xlabel('Target')
plt.ylabel('Average Feature')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.show()


plt.scatter(train['Personality'],train['Post_frequency'],color='blue',alpha=0.8)
sns.boxplot()
plt.xlabel("person")
plt.ylabel("post")
plt.grid(True)
plt.tight_layout()
plt.title("diff")
plt.show()


train.describe()


'''categorical_cols=['Stage_fear','Drained_after_socializing']
# One-hot encode and update the original DataFrame
train = pd.get_dummies(train, columns=categorical_cols)
train
test = pd.get_dummies(test, columns=categorical_cols)
test'''


train.head(100)


X=train.drop(['Personality'],axis=1)

y_t=train['Personality']


y_t


y=le.fit_transform(y_t)
y


accuracy_score(y,train['team'])


scaler = StandardScaler()
X_scale = scaler.fit_transform(X)

x_scale = scaler.transform(test)


import numpy as np
import xgboost as xgb
import lightgbm as lgb
#import catboost as cb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression


# Initialize models
'''xgb_model = xgb.XGBClassifier(
    objective= "binary:logistic",
    eval_metric="logloss",
    max_depth=4,
    eta= 0.1,
    subsample= 0.8,
    colsample_bytree = 0.8,
    random_state=42)'''
lgb_model = lgb.LGBMClassifier(random_state=42)
#catboost_model = cb.CatBoostClassifier(learning_rate=0.1, iterations=500, depth=6, random_state=42, verbose=0)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
#log_model=LogisticRegression()

# Initialize the VotingClassifier (Majority Voting)
voting_model = VotingClassifier(
    estimators=[#('xgb', xgb_model),
                ('lgb',lgb_model),
                #('cb',catboost_model),
                #('LogisticRegression',log_model),
               ('RandomForestClassifier', rf_model)
               ],
    voting='hard'  # 'hard' for majority voting, 'soft' for probability averaging
)

# Initialize Stratified K-Fold
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# List to store the accuracies for each fold
accuracies = []

# Stratified K-Fold Cross-Validation
for train_index, test_index in skf.split(X_scale, y):
    # Split the dataset into training and testing sets for each fold
    X_train, X_test = X_scale[train_index], X_scale[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Train the Voting Classifier model
    voting_model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = voting_model.predict(X_test)
    
    # Evaluate the accuracy for this fold
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# Output the average accuracy across all folds
print(f"Average Accuracy across {n_splits} folds: {np.mean(accuracies):.2f}")

# Final Classification Report (using the whole dataset, but typically, you may compute on the final model)
voting_model.fit(X_scale, y)  # Fit on the whole data for final model
final_predictions = voting_model.predict(X_scale)
print("Final Classification Report on Entire Dataset:\n", classification_report(y, final_predictions))



accuracy_score(y,final_predictions)


final_predictions = voting_model.predict(x_scale)
final_predictions


pred=le.inverse_transform(final_predictions)
pred


output = pd.DataFrame({'id': test.id, 'Personalities': pred})
output.to_csv('Submission.csv', index=False)
print("Your submission was successfully saved!")


sub=pd.read_csv("Submission.csv")
sub

