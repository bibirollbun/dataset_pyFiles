import pandas as pd
train_df = pd.read_csv("/kaggle/input/classwork-3-insurance-prediction/train.csv")


train_df.head()
# do some EDA (exploratory data analysis) here


X, y = train_df.drop(columns=['id', 'charges']), train_df[['charges']]
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


test_df = pd.read_csv("/kaggle/input/classwork-3-insurance-prediction/test.csv")


test_df.head()


from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor
import numpy as np
ct = ColumnTransformer(transformers=[
    ('onehot_encode', OneHotEncoder(), 
     make_column_selector(dtype_include='object'))
],
remainder='passthrough')
#reg = Ridge()
reg = RandomForestRegressor(max_depth=5) # random forest with maximum depth limit
#reg = HistGradientBoostingRegressor()
pf = PolynomialFeatures() # polynomial features for feature engineering
reg.fit(pf.fit_transform(ct.fit_transform(X_train)), y_train)
train_preds = reg.predict(pf.transform(ct.transform(X_train)))
print(f"Train R^2 for target 0")
print(r2_score(y_train, train_preds))
print(f"Val R^2 for target 0")
val_preds = reg.predict(pf.transform(ct.transform(X_val)))
print(r2_score(y_val, val_preds))


test_pred = reg.predict(pf.transform(ct.transform(test_df.drop(columns=['id']))))


subm_df = pd.read_csv("/kaggle/input/classwork-3-insurance-prediction/sample_submission.csv")
subm_df.head()


subm_df['charges'] = test_pred
subm_df.to_csv("submission.csv", index=False) # upload this file to submit to competition


subm_df




