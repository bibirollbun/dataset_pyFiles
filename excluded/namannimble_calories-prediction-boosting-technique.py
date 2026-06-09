import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.express as px 


from sklearn.preprocessing import  StandardScaler , MinMaxScaler
from sklearn.metrics import  mean_squared_error , mean_absolute_error
from sklearn.linear_model import  LinearRegression , Ridge
from sklearn.model_selection import  GridSearchCV , cross_val_score , RandomizedSearchCV


from catboost import  CatBoostRegressor
from xgboost import  XGBRegressor

from sklearn.metrics import make_scorer
from sklearn.pipeline import  Pipeline


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        

calorie_train = "/kaggle/input/playground-series-s5e5/train.csv"
calorie_test = "/kaggle/input/playground-series-s5e5/test.csv"


train = pd.read_csv(calorie_train)
test = pd.read_csv(calorie_test)


train.head()


train = train.rename(columns={
    "id":"id",
    "Sex": "sex",
    "Age": "age",
    "Height": "height",
    "Weight": "weight",
    "Duration": "duration",
    "Heart_Rate": "heart_rate",
    "Body_Temp": "body_temp",
    "Calories": "calories"
})



train.columns


train.info()


train.isnull().sum()


train.duplicated().sum()


train.describe()


train["bmi"] = train["weight"]/(train["height"]/(100)**2)
train['sex'] = train['sex'].map({'male': 1, 'female': 0})


X_train = train.drop(columns = ["calories","id"],axis=1)
y_train = train["calories"]


xgb_pipe = Pipeline([
    ("scaler1",StandardScaler()),
    ("model1",XGBRegressor())
])


cat_pipe = Pipeline([
    ("scaler2",StandardScaler()),
    ("model2",CatBoostRegressor())
])




def rmsle(y_true, y_pred):
    y_true = np.maximum(0, y_true)
    y_pred = np.maximum(0, y_pred)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true))**2))

rmsle_scorer = make_scorer(rmsle, greater_is_better=False)



cat_params = {
    'model2__iterations': [100, 200],
    'model2__depth': [4, 6, 8],
    'model2__learning_rate': [0.01, 0.1, 0.2],
    'model2__l2_leaf_reg': [1, 3, 5]
}

xgb_params = {
    'model1__n_estimators': [100, 200],
    'model1__max_depth': [3, 6, 8],
    'model1__learning_rate': [0.01, 0.1, 0.2],
    'model1__reg_lambda': [0.5, 1, 2]
}


cat_search = RandomizedSearchCV(
    cat_pipe,
    cat_params,
    n_iter=10, 
    scoring=rmsle_scorer, 
    cv=5,
    verbose=1, 
    n_jobs=-1,
    random_state=42
)


xgb_search = RandomizedSearchCV(
    xgb_pipe,
    xgb_params,
    n_iter=10,
    scoring = rmsle_scorer,
    cv=5,
    verbose=1, 
    n_jobs=-1,
    random_state=42
)





xgb_search.fit(X_train,y_train)
cat_search.fit(X_train,y_train)



cat_rmsle = -cat_search.best_score_
xgb_rmsle = -xgb_search.best_score_

print("CatBoost RMSLE:", cat_rmsle)
print("XGBoost RMSLE:", xgb_rmsle)


if cat_rmsle < xgb_rmsle:
    best_model = cat_search.best_estimator_
    best_name = "CatBoost"
else:
    best_model = xgb_search.best_estimator_
    best_name = "XGBoost"

print("Best Model Selected:", best_name)


test.head()



train.columns


test = test.rename(columns={
    "Sex": "sex",
    "Age": "age",
    "Height": "height",
    "Weight": "weight",
    "Duration": "duration",
    "Heart_Rate": "heart_rate",
    "Body_Temp": "body_temp",
}) 
test["bmi"] = test["weight"]/(test["height"]/(100)**2)
test["sex"] = test["sex"].map({'male': 1, 'female': 0})


X_test = test.drop(columns=["id"])


predictions = best_model.predict(X_test)


predictions


# submission file 

submission = pd.DataFrame({
    "id": test["id"],
    "Calories": predictions
})



submission


submission.to_csv('submission.csv', index=False)



print(xgb_rmsle)




