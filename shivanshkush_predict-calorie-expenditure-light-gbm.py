# import libraries
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


# Shape
print("Shape of Train dataset")
display(train.shape)
print("\nShape of Test dataset")
display(test.shape)


# know about datatype 
print("Train info")
train.info()
print("\nTest info")
test.info()

# know the details of numeric data in dataset
print("Train details")
display(train.describe())
print("\nTest details")
display(test.describe())


# missing value
print(train.isnull().sum())
print(test.isnull().sum())


train.head(10)


test.head(10)


# encode the categorial column
train["Sex"]=LabelEncoder().fit_transform(train["Sex"])

test["Sex"]=LabelEncoder().fit_transform(test["Sex"])


train.head()


test.head()


# Add more features to increase accuracy and precision
train["BMI"]=train["Weight"]/math.pow((train["Height"].iloc[0]/100),2.0)
test["BMI"]=test["Weight"]/(test["Height"]/100)**2

# if(train["Sex"]==1):
#     train["BMR"]=(10 * train["Weight"]) + (6.25 * train["Height"] ) - (5 * train["Age"]) + 5
# else:
#     train["BMR"]=(10 * train["Weight"]) + (6.25 * train["Height"]) - (5 * train["Age"]) - 161
    


# train=BMR(train.copy())
# test=BMR(test.copy())
print(train.head())
test.head()


# for making output to be in range 
train['Calories'] = np.log1p(train['Calories'])


X=train.drop(["id","Calories"],axis=1)
y=train["Calories"]
X_test=test.drop(["id"],axis=1)


# Define RMSLE
def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(np.expm1(y_true), np.expm1(y_pred)))


FOLDS=5
kf = KFold(n_splits=5, shuffle=True, random_state=42)
lgb_predictions = np.zeros(len(X_test))

lgb_oof_predictions = np.zeros(len(X))

lgb_rmsle_scores = []
   
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    lgb = LGBMRegressor(objective='regression',
                        metric='rmse',
                        n_estimators=500,
                        learning_rate=0.01,
                        random_state=42,
                        subsample=0.7,
                        num_leaves=30)

    lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    lgb_val_preds = lgb.predict(X_val)
    lgb_oof_predictions[val_idx] = lgb_val_preds
    lgb_rmsle_scores.append(rmsle(y_val, lgb_val_preds))
    lgb_predictions += lgb.predict(X_test) / FOLDS
    


 print(f"LightGBM RMSLE: {np.mean(lgb_rmsle_scores):.4f}")
 print(f"OOF LightGBM RMSLE: {rmsle(y, lgb_oof_predictions):.4f}")


ans=pd.DataFrame({"id":test["id"],"Calories":np.expm1(lgb_predictions)})
ans.to_csv("submission.csv",index=False)
ans

