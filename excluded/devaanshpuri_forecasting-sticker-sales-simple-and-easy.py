import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


train_dataset = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
train_dataset = train_dataset.drop('id', axis=1)
test_dataset = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
test_dataset = test_dataset.drop('id', axis=1)
train_dataset.head()


test_dataset.head()


train_dataset = train_dataset.drop_duplicates()
train_dataset = train_dataset.dropna()


train_dataset = train_dataset.set_index('date')
test_dataset = test_dataset.set_index('date')
train_dataset.plot(figsize = (15,5))


train_dataset.index


pd.to_datetime(train_dataset.index)


pd.to_datetime(test_dataset.index)


num_cols = list(train_dataset.select_dtypes(exclude=['object']).columns.difference(['num_sold']))
cat_cols = list(train_dataset.select_dtypes(include=['object']).columns)

num_cols_test = list(test_dataset.select_dtypes(exclude=['object']).columns.difference(['id']))
cat_cols_test = list(test_dataset.select_dtypes(include=['object']).columns)

print('num_cols:',num_cols)
print('cat_cols:',cat_cols)


from sklearn.preprocessing import LabelEncoder
label_encoders = {col: LabelEncoder() for col in cat_cols}

for col in cat_cols:
    combined_data = pd.concat([train_dataset[col], test_dataset[col]])
    le = LabelEncoder()
    le.fit(combined_data)
    train_dataset[col] = le.transform(train_dataset[col])
    test_dataset[col] = le.transform(test_dataset[col])


X = train_dataset.iloc[:,:-1]
y = train_dataset.iloc[:,-1]

from sklearn.model_selection import train_test_split
train_dataset['num_sold'] = np.log1p(train_dataset['num_sold'])
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state = 0 )


print(y_train.isnull().sum())
print((np.isinf(y_train)).sum())


import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error

lgb_model = lgb.LGBMRegressor()

param_grid = {
    'learning_rate': [0.08],
    'n_estimators': [1000],
    'max_depth': [12],
    'min_child_samples': [32],
    'subsample': [0.7],
    'colsample_bytree': [0.93],
    
}

grid_search = GridSearchCV(estimator=lgb_model, param_grid=param_grid, 
                           scoring=mean_squared_error, 
                           cv=5, verbose=1, n_jobs=-1)


grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best MAPE Score:", grid_search.best_score_)


best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)


mape = mean_absolute_percentage_error(y_test, y_pred)
print("Test MAPE:", mape)


y_test_pred = best_model.predict(test_dataset)


submission_df = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')
submission_df['num_sold'] = y_test_pred
submission_df.to_csv("submission.csv", index=False)
submission_df.head()

