import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


train=pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


train.shape


test.shape





numeric=['int64','float64']
cat=['object']
for i in train:
    if (train[i].dtypes in numeric) and (train[i].isna().sum()!=0):
        mean_num=train[i].mean()
        train[i]=train[i].fillna(mean_num)
    elif (train[i].dtypes in cat) and (train[i].isna().sum()!=0):
        mode_cat=train[i].mode()[0]
        train[i]=train[i].fillna(mode_cat)
    else:
        print("Column doesn't match the above criteria",i)


numeric=['int64','float64']
cat=['object']
for i in test:
    if (test[i].dtypes in numeric) and (test[i].isna().sum()!=0):
        mean_num=test[i].mean()
        test[i]=test[i].fillna(mean_num)
    elif (test[i].dtypes in cat) and (test[i].isna().sum()!=0):
        mode_cat=test[i].mode()[0]
        test[i]=test[i].fillna(mode_cat)
    else:
        print("Column doesn't match the above criteria",i)


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
cat_cols=['Stage_fear','Drained_after_socializing']


for i in cat_cols:
    train[i]=le.fit_transform(train[i])
    test[i]=le.transform(test[i])


X=train.drop(columns=['id','Personality'])
y=train['Personality']

X_test=test.drop(columns=['id'])


model=LabelEncoder()
y=model.fit_transform(y)


from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier


X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2,random_state=42)


bagm=DecisionTreeClassifier()


from sklearn.ensemble import BaggingClassifier


bag=BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)
bag.fit(X_train,y_train)
bag.oob_score_


from sklearn.model_selection import StratifiedKFold,cross_validate


n_splits =10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


cv_results = cross_validate(
    estimator=bag,          # Your defined BaggingClassifier
    X=X,                    # Your feature data
    y=y,                    # Your target labels
    cv=skf,                 # The StratifiedKFold object handles the folds
    scoring=accuracy_score,# The metrics you want to evaluate
    return_train_score=True,# Set to True to get training scores for each fold (useful for bias/variance check)
    n_jobs=-1               # Use all available CPU cores for the cross-validation process
)


bag.fit(X_train,y_train)


y_pred=bag.predict(X_val)


accuracy_score(y_val,y_pred)


y_test=bag.predict(X_test)


y_test_labels = model.inverse_transform(y_test)

submission = test[['id']].copy()
submission['Personality'] = y_test_labels

submission.to_csv('submission.csv', index=False)


submission

