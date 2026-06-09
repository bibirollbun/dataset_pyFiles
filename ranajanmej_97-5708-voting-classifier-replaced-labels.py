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


import warnings
warnings.filterwarnings("ignore")



train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


train_df.head()


cols_to_encode = ['Stage_fear','Drained_after_socializing']
for col in cols_to_encode:
    train_df[cols_to_encode] = train_df[cols_to_encode].replace({'Yes' : 1,'No' : 0})
    test_df[cols_to_encode] = test_df[cols_to_encode].replace({'Yes' : 1,'No' : 0})


train_df.head()


train_df['Time_spent_Alone'].unique()


train_df['Social_event_attendance'].unique()


train_df['Going_outside'].unique()


train_df['Post_frequency'].unique()


train_df[(train_df['Time_spent_Alone'] <= 1) & (train_df['Social_event_attendance'] >= 8)
        & (train_df['Going_outside'] >= 5) & (train_df['Friends_circle_size'] >= 11) &
        (train_df['Post_frequency'] >= 7) & (train_df['Personality'] == 'Introvert')].dropna()


act_Extrovert_ind = train_df[(train_df['Time_spent_Alone'] <= 1) & (train_df['Social_event_attendance'] >= 8)
        & (train_df['Going_outside'] >= 5) & (train_df['Friends_circle_size'] >= 11) &
        (train_df['Post_frequency'] >= 7) & (train_df['Personality'] == 'Introvert')].dropna().index


act_Extrovert_ind


train_df.loc[act_Extrovert_ind,'Personality'] = train_df.loc[act_Extrovert_ind,'Personality'].replace('Introvert','Extrovert')


train_df.iloc[act_Extrovert_ind]


train_df[(train_df['Time_spent_Alone'] >= 9) & (train_df['Social_event_attendance'] <= 2)
        & (train_df['Going_outside'] <= 2) & (train_df['Friends_circle_size'] <= 1) &
        (train_df['Post_frequency'] <= 1) & (train_df['Personality'] == 'Extrovert')].dropna()


act_introvert_ind = train_df[(train_df['Time_spent_Alone'] >= 9) & (train_df['Social_event_attendance'] <= 2)
        & (train_df['Going_outside'] <= 2) & (train_df['Friends_circle_size'] <= 1) &
        (train_df['Post_frequency'] <= 1) & (train_df['Personality'] == 'Extrovert')].dropna().index


train_df.loc[act_introvert_ind,'Personality'] = train_df.loc[act_introvert_ind,'Personality'].replace('Extrovert','Introvert')


train_df.iloc[act_introvert_ind]


X = train_df.drop(columns = 'Personality',axis = 1)
y = train_df['Personality']


from sklearn.experimental import enable_iterative_imputer  # Needed to enable IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge


imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=0)

# Fit and transform the data
train_df_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
test_df_imputed =  pd.DataFrame(imputer.fit_transform(test_df), columns=test_df.columns)


train_df_imputed.isnull().sum()


train_df_imputed = train_df_imputed.drop(columns = 'id',axis = 1)
test_df_imputed = test_df_imputed.drop(columns = 'id',axis = 1)


train_df_imputed.head(2)


from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()


cols_to_scale = ['Time_spent_Alone','Social_event_attendance','Going_outside','Friends_circle_size',
                'Post_frequency']


train_df_imputed[cols_to_scale] = scaler.fit_transform(train_df_imputed[cols_to_scale])


test_df_imputed[cols_to_scale] = scaler.transform(test_df_imputed[cols_to_scale])


train_df_imputed.head()


y.head()


y_enc = y.replace({'Extrovert' : 1,'Introvert' : 0})


import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(train_df_imputed, y_enc, test_size=0.2, random_state=42)

# Define base models
models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "LightGBM": LGBMClassifier(random_state=42)
}

# Evaluate each model using cross-validation
print("Cross-validation scores:")
best_model_name = None
best_score = 0.0
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    mean_score = scores.mean()
    print(f"{name}: {mean_score:.4f}")
    if mean_score > best_score:
        best_score = mean_score
        best_model_name = name

print(f"\nâœ… Best individual model: {best_model_name} (CV accuracy = {best_score:.4f})")

# Create Voting Classifier
voting_clf = VotingClassifier(
    estimators=[
        ('rf', models["RandomForest"]),
        ('xgb', models["XGBoost"]),
        ('lgbm', models["LightGBM"])
    ],
    voting='soft'  # or 'hard' for majority vote
)

# Train the voting classifier
voting_clf.fit(X_train, y_train)

# Evaluate on test data
y_pred = voting_clf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"\nðŸŽ¯ Voting Classifier Test Accuracy: {test_accuracy:.4f}")



voting_clf.fit(X_train,y_train)


y_preds = voting_clf.predict(test_df_imputed)


fin_pred = pd.DataFrame({
    'id' : test_df['id'],
    'Personality' : y_preds
})


fin_pred


fin_pred['Personality'] = fin_pred['Personality'].replace({1:'Extrovert',0: 'Introvert'})


fin_pred['Personality'].value_counts()


fin_pred.to_csv('fin_pred.csv',index = False)




