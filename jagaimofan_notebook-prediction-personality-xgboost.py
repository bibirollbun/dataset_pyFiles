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


test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")



train.head()
train['flag'] = 'Train'
test['flag'] = 'Test'
test['Personality'] = None


print(f"Number of rows of train: {len(train)}")
print(f"Number of rows of test: {len(test)}")


null_counts = train.isnull().sum()
null_counts



train_df= train.dropna()
print(f"Number of rows of train: {len(train_df)}")



test_df= test.dropna()
print(f"Number of rows of train: {len(test_df)}")



df = pd.concat([train, test])
print(df.describe())


df.head()



df2 = pd.get_dummies(df, 
                     columns=['Stage_fear', 'Drained_after_socializing', 'Personality'],
                     dtype=int,
                    dummy_na=True)
df2.head()


df2[df2['Stage_fear_nan'] == 1]



df3 = df2.drop(columns=['Stage_fear_nan', 'Drained_after_socializing_nan','Personality_nan','Personality_Introvert'])
df3.describe()



null_counts2 = df3.isnull().sum()
null_counts2



df3[['Time_spent_Alone', 'Social_event_attendance','Going_outside','Friends_circle_size','Post_frequency']] = df3[['Time_spent_Alone', 'Social_event_attendance','Going_outside','Friends_circle_size','Post_frequency']].fillna(df3[['Time_spent_Alone', 'Social_event_attendance','Going_outside','Friends_circle_size','Post_frequency']].mean())

null_counts3 = df3.isnull().sum()
null_counts3


train_df3 = df3[df3['flag']=='Train']
test_df3 = df3[df3['flag']=='Test']



train_df3.head(10)
train_df3 = train_df3.drop(columns=['flag'])


test_df3 = test_df3.drop(columns=['flag'])






from sklearn.model_selection import train_test_split
X = train_df3.drop('Personality_Extrovert', axis = 1)
y = train_df3['Personality_Extrovert']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)


import time


import xgboost as xgb
from sklearn.metrics import accuracy_score
start = time.time()

model_xg = xgb.XGBClassifier(objective='multi:softmax', num_class=3, use_label_encoder=False, eval_metric='mlogloss')
model_xg.fit(X_train, y_train)

# 5. Make predictions on the test set
y_pred = model_xg.predict(X_test)

# 6. Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print('elapsed_time:{}'.format(time.time()-start))


import lightgbm as lgb
start2 = time.time()

# Initialize the LightGBM Classifier
# Parameters like num_leaves, max_depth, learning_rate can be tuned
lgbm_clf = lgb.LGBMClassifier(objective='binary',  # For binary classification
                              metric='binary_logloss', # Evaluation metric
                              n_estimators=100,      # Number of boosting rounds
                              learning_rate=0.05,    # Step size shrinkage
                              num_leaves=31,         # Max number of leaves in one tree
                              random_state=42)       # For reproducibility

# Train the model
lgbm_clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lgbm_clf.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print('elapsed_time:{}'.format(time.time()-start))



from catboost import CatBoostClassifier, Pool
start3 = time.time()

train_pool = Pool(X_train, y_train)
test_pool = Pool(X_test, y_test)

# 2. Initialize and Train the CatBoost Model
# Define CatBoostClassifier with desired parameters
model_cat = CatBoostClassifier(
    iterations=100,  # Number of boosting iterations
    learning_rate=0.1, # Step size shrinkage
    depth=6,          # Depth of the trees
    loss_function='Logloss', # Loss function for binary classification
    verbose=0,        # Suppress training output
    random_seed=42    # For reproducibility
)

# Train the model
model_cat.fit(train_pool)

# 3. Make Predictions and Evaluate
# Predict classes on the test set
y_pred = model_cat.predict(test_pool)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

print('elapsed_time:{}'.format(time.time()-start))



test_df4 = test_df3.drop(columns=['Personality_Extrovert'])
test_pred = model_xg.predict(test_df4)



s = pd.Series(test_pred, name='Personality')
df_test_pred = pd.concat([test_df4['id'], s],axis =1)


print(df_test_pred)


df

df_test_pred['Personality'] = np.where(df_test_pred['Personality'] == 1, 'Extrovert', 'Introvert')



print(df_test_pred)



print('Generating submission.csv file...')

df_test_pred.to_csv('submission_personality.csv', index=False)

print("Submission file created.")

