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


import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
sample = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")


train.columns


#y = train["loan_paid_back"]

#features = ["annual_income",'credit_score','loan_amount','interest_rate', 'employment_status']

#x = pd.get_dummies(train[features])
#x_test = pd.get_dummies(test[features])



#y = train.loan_paid_back

#features = ["annual_income",'credit_score','loan_amount','interest_rate', 'employment_status']

#X = train[features]



#X_train,X_valid,y_train,y_valid = train_test_split(X, y ,train_size=0.8, test_size= 0.2, random_state=0)


#X_train = pd.get_dummies(X_train)
#X_valid = pd.get_dummies(X_valid)


#X_train, X_valid = X_train.align(X_valid,join='left',axis=1,fill_value=0)



#print([X_train.shape],[y_train.shape])


#model = DecisionTreeRegressor()
#model.fit(X_train,y_train)
#preds = model.predict(X_valid)

#print("mae: \t" + str(mean_absolute_error(preds,y_valid)))


#models = RandomForestRegressor(n_estimators=100, random_state=0)
#models.fit(X_train,y_train)
#predicts = model.predict(X_valid)

#print("mae: \t" + str(mean_absolute_error(predicts,y_valid)))


#models = RandomForestClassifier(n_estimators=150, random_state=1)
#models.fit(X_train,y_train)
#predicyss = models.predict(X_valid)

#print("mae: \t" + str(mean_absolute_error(predicyss,y_valid)))


#modelsss = XGBRegressor(n_estimators = 250, learning_rate = 0.125)
#modelsss.fit(X_train,y_train,
#            early_stopping_rounds = 5,
#            eval_set = [(X_train,y_train)],
#            verbose = False)

#pre = modelsss.predict(X_valid)

#print("mae: \t" + str(mean_absolute_error(pre,y_valid)))


#pp = Pipeline(steps=[
#    ('impute',SimpleImputer(strategy= 'most_frequent')),
#    ('hot',OneHotEncoder(handle_unknown= 'ignore')),
#    ('mode',RandomForestRegressor(n_estimators=150,random_state=0))
#)

#pp.fit(X_train,y_train)

#cross = -1 * cross_val_score(pp,X,y,
#                       cv = 5,
#                       scoring = 'neg_mean_absolute_error')


#print(f"MAE SCORES: \t {[cross]}")


y = train.loan_paid_back

features = ["annual_income",'credit_score','loan_amount','interest_rate', 'employment_status']

X = train[features]
X_test = test[features]

X = pd.get_dummies(X)
X_test = pd.get_dummies(X_test)

X,X_test = X.align(X_test,join = 'left',axis =1)

X_test = X_test.fillna(0)


print([X.shape],[X_test.shape])


models = RandomForestClassifier(
    n_estimators=500,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=3,
    max_features='sqrt',
    random_state=1,
    n_jobs=-1
)
models.fit(X,y)

predict = models.predict_proba(X_test)[:,1]

print("prediction made successfully!")


scores = cross_val_score(models,X,y, cv=5 , scoring = 'roc_auc')

print(scores.mean())


submission = pd.DataFrame({
    'id': test['id'],
    'loan_paid_back' : predict
})

submission.to_csv('submission.csv',index = False)

