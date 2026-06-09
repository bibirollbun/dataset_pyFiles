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


df_sample = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
df_train= pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

print(df_train.head())
print(df_test.head())
print(df_sample.head())


from sklearn.preprocessing import MinMaxScaler

def preprocess_xgboost (df):

    new_df_1 = pd.DataFrame(df)
    new_df_2 = pd.DataFrame(df)
    
    try:
        df['Personality'] = df['Personality'].map({'Extrovert': 1, 'Introvert': 0})
        new_df_1 = df.dropna()
        new_df_2 = new_df_1.drop(columns = ["id"])
    except:
        pass

    new_df_2 = pd.get_dummies(new_df_2, columns = ["Stage_fear","Drained_after_socializing"], drop_first = True)

   
    numeric_cols = new_df_2.select_dtypes(include='number').columns
    
    min_max_scaler = MinMaxScaler()
    
    new_df_2[numeric_cols] = min_max_scaler.fit_transform(new_df_2[numeric_cols])
    
    return new_df_2


prep_train = preprocess_xgboost(df_train)

X_train = prep_train.drop(columns = ['Personality'])

y_train = prep_train[['Personality']]

X_t = preprocess_xgboost(df_test)

X_test = X_t.drop(columns = ['id'])



from xgboost import XGBClassifier

bst = XGBClassifier(n_estimators=40, max_depth=2, learning_rate=0.35, objective='binary:logistic',subsample = 0.7)

bst.fit(X_train, y_train)

preds_xgboost = bst.predict(X_test)


"""   

from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

# 1. Define the model
xgb = XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss')

# 2. Define the grid of parameters to search
param_grid = {
    'max_depth': [2,3,4,5,6, 7,8,9,10],
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
    'n_estimators': [10,20,30,40,50,60,70,80,90,100,125,150,175, 200,250, 300],
    'subsample': [0.5, 0.6, 0.7, 0.75, 0.8, 1.0]
}

# 3. Set up the grid search
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1)

# 4. Run the grid search on your data
grid_search.fit(X_train, y_train)

# 5. Get the best parameters
print("Best parameters found: ", grid_search.best_params_)

"""


""" 

submission = pd.DataFrame()

submission['id'] = df_test['id']

Re-assigning the labels different values can lead to different results
#submission['personality_type'] = np.where(preds > 0.01, 'Extrovert', 'Introvert')

print(submission.head())

submission.to_csv('submission.csv',index = False)

"""


df_train_cat = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
df_test_cat = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


for col in ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 'Post_frequency']:
    df_train_cat[col].fillna(df_train_cat[col].mode()[0], inplace=True)
    df_test_cat[col].fillna(df_test_cat[col].mode()[0], inplace=True)


X_cat = df_train_cat.drop(['id', 'Personality'], axis=1)
y_cat = df_train_cat['Personality']
X_test_cat = df_test_cat.drop('id', axis=1)


categorical_features_indices = np.where(X_cat.dtypes != np.float64)[0]


from catboost import CatBoostClassifier
#from sklearn.model_selection import GridSearchCV

cat_model = CatBoostClassifier(iterations=300,
                               learning_rate=0.1,
                               depth=6,
                               l2_leaf_reg=3,
                               loss_function='Logloss',
                               verbose=0,
                               subsample=0.8)

# Train the model
cat_model.fit(X_cat, y_cat, cat_features=categorical_features_indices)

# Get predictions
cat_preds_labels = cat_model.predict(X_test_cat)


print("\n--- Creating Soft Voting Ensemble ---")


xgb_probs = bst.predict_proba(X_test)[:, 1]


cat_probs = cat_model.predict_proba(X_test_cat)[:, 0] 

# You can add weights if you trust one model more than the other
xgb_weight = 0.5
cat_weight = 0.5
ensemble_probs = (xgb_probs * xgb_weight) + (cat_probs * cat_weight)

# Convert the final averaged probabilities to class labels
ensemble_preds = np.where(ensemble_probs > 0.7, 'Extrovert', 'Introvert')

# Create the final submission DataFrame
submission_ensemble = pd.DataFrame()
submission_ensemble['id'] = df_test['id']
submission_ensemble['Personality'] = ensemble_preds

# Save the ensemble predictions to a CSV file
submission_ensemble.to_csv('submission.csv', index=False)

print("\nSoft-voting ensemble submission file created successfully!")
print(submission_ensemble.head())



