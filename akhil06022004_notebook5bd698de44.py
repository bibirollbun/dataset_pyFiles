import numpy as np
import pandas as pd
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv("/kaggle/input/ucs-654-kaggle-hack-lab-exam-2/train.csv")
df.head()


X_data = df.drop(columns=['target'])
Y_data = df['target']


X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)


base_models = [
    ('decision_tree', DecisionTreeRegressor(max_depth=5)),
    ('xgboost', XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42)),
    ('random_forest', RandomForestRegressor(n_estimators=50, random_state=42)),
    ('gbr' , GradientBoostingRegressor( n_estimators=700, learning_rate=0.1, max_depth=6, random_state=42, subsample=0.8)),
]


from sklearn.linear_model import Ridge
# meta_model = LinearRegression()
meta_model = Ridge(alpha=1.0)


# Stacking Regressor
stacking_regressor = StackingRegressor(estimators=base_models, final_estimator=meta_model)


stacking_regressor.fit(X_data,Y_data)


pr = stacking_regressor.predict(X_data)
print(pr)



#testing our model
df_test = pd.read_csv("/kaggle/input/ucs-654-kaggle-hack-lab-exam-2/test.csv")
df_test.head()


xtest_abc = df_test.drop(columns=['id'])
xtest_abc.head()


final = stacking_regressor.predict(xtest_abc)
final


final_df = pd.DataFrame(final,columns=['target'])
final_df.insert(0,'id',range(1,len(final_df)+1))
final_df.head()


final_df.to_csv('final_submission4.csv',index=False)
print('Data saved successfully in csv file')

