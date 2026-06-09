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


tr_df=pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")


tr_df.head(5)


tr_df.shape


tr_df.info()


tr_df.isnull().sum()


tr_df.duplicated().value_counts()


tr_df.describe()


# Add date features
tr_df['date'] = pd.to_datetime(tr_df['date'])
tr_df['day_of_week'] = tr_df['date'].dt.dayofweek
tr_df['is_weekend'] = (tr_df['day_of_week'] >= 5).astype(int)
tr_df['quarter'] = tr_df['date'].dt.quarter


tr_df.head()


tr_df.drop(["date"],axis=1,inplace=True)


tr_df['country'].unique()


from sklearn.preprocessing import LabelEncoder
x=LabelEncoder()
tr_df['country']=x.fit_transform(tr_df['country'])
tr_df['store']=x.fit_transform(tr_df['store'])
tr_df['product']=x.fit_transform(tr_df['product'])


#tr_df['num_sold'] = np.log1p(tr_df['num_sold'])



import seaborn as sns
sns.heatmap(data=tr_df.corr(),annot=True)


sns.boxplot(data=tr_df["num_sold"]);


import matplotlib.pyplot as plt
plt.hist(x=tr_df["num_sold"]);


tr_df["num_sold"].skew()


meed=tr_df["num_sold"].median()


tr_df.fillna(meed,inplace=True)


tst_df=pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")


tst_df.info()


tst_df.isnull().sum()


tst_df['date'] = pd.to_datetime(tst_df['date'])
tst_df['day_of_week'] = tst_df['date'].dt.dayofweek
tst_df['is_weekend'] = (tst_df['day_of_week'] >= 5).astype(int)
tst_df['quarter'] = tst_df['date'].dt.quarter


tst_df.drop(["date"],axis=1,inplace=True)


from sklearn.preprocessing import LabelEncoder
x=LabelEncoder()
tst_df['country']=x.fit_transform(tst_df['country'])
tst_df['store']=x.fit_transform(tst_df['store'])
tst_df['product']=x.fit_transform(tst_df['product'])


X=tr_df.drop(["num_sold"],axis=1)
y=tr_df["num_sold"]


from sklearn.model_selection import train_test_split,GridSearchCV
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=33)


# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LogisticRegression
# #model=RandomForestRegressor(random_state=43,max_depth=10,criterion='absolute_error',)
# model=LogisticRegression()
from xgboost import XGBRegressor
model = XGBRegressor(tree_method='hist', device='cuda', n_estimators=100, max_depth=10, learning_rate=0.1)


y_train


# model.fit(X_train,y_train)


# model.score(X_test,y_test)


# res = model.predict(tst_df)


from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators': [100, 200],
    'learning_rate': [0.03, 0.1],
    'max_depth': [3, 6],
    'min_child_weight': [1, 3],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.7, 1.0],
    'gamma': [0, 0.1],
}

# Step 3: Use RandomizedSearchCV (fastest)
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=25,                     
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=2,
    n_jobs=-1,                     # use all CPU threads 
    random_state=42
)

# Step 4: Fit with early stopping (MAJOR speed boost)



# Step 6: Fit the model using GridSearchCV
random_search.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train)],
    early_stopping_rounds=20,
    verbose=False
)

# Step 7: Get the best parameters and model
best_params = random_search.best_params_
best_model = random_search.best_estimator_




# Step 8: Evaluate the model on the test set
y_pred = best_model.predict(tst_df)
# mse = mean_squared_error(y_test, y_pred)

# Step 9: Print the results
print(f"Best Hyperparameters: {best_params}")
# print(f"Mean Squared Error on Test Set: {mse:.4f}")


submission_csv1 = tst_df.copy()
submission_csv1['num_sold'] = y_pred
submission_csv2 = submission_csv1[['id','num_sold']]
submission_csv2.head(10)


submission_csv2.to_csv("final_submission3.csv",index=False)







