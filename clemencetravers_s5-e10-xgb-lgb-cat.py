import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import tensorflow as tf
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error 

from xgboost import XGBRegressor 
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
df_test=pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


df.head()


df.info()


plt.style.use('seaborn-whitegrid')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('animation', html='html5')



#fig, axes = plt.subplots(1, 2, figsize=(12, 5))

#sns.barplot(x=df.weather, y=df.accident_risk, ax=axes[0])
#axes[0].set_title("Accident risk selon la météo")

#sns.barplot(x=df.road_type, y=df.accident_risk, ax=axes[1])
#axes[1].set_title("Accident risk selon le type de route")


#plt.tight_layout()
#plt.show()


#fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#sns.barplot(x=df.lighting, y=df.accident_risk, ax=axes[0])
#axes[0].set_title("Accident risk selon la lumière")

#sns.barplot(x=df.time_of_day, y=df.accident_risk, ax=axes[1])
#axes[1].set_title("Accident risk selon le moment de la journée")
#plt.tight_layout()
#plt.show()


#plt.figure (figsize=(8,8))
#sns.scatterplot(x=df.curvature, y=df.accident_risk)

#plt.show()


#sns.boxplot(x=df.accident_risk)


OHE= OneHotEncoder(handle_unknown='ignore', sparse_output=False)


s = (df.dtypes == 'object') | (df.dtypes == 'bool') 
object_cols = list(s[s].index)



df[object_cols]


OH_cols = pd.DataFrame(OHE.fit_transform(df[object_cols]))

# One-hot encoding removed index; put it back
OH_cols.index = df.index

# Remove categorical columns (will replace with one-hot encoding)
num_df = df.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_df = pd.concat([num_df, OH_cols], axis=1)



X=OH_df.drop(['id', 'accident_risk'],axis=1)
y=OH_df.pop('accident_risk')


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)


train_X


model_XGB = XGBRegressor(n_estimators=300, learning_rate=0.1,early_stopping_rounds=10 )

model_XGB.fit(train_X, train_y,
           eval_set=[(val_X, val_y)]
)

pred_y_XGB = model_XGB.predict(val_X) 

print(mean_squared_error(val_y,pred_y_XGB,squared=False))


model_LGB= LGBMRegressor(num_leaves=50, min_child_samples= None, max_depth=20, learning_rate=0.1, n_estimators=400, force_row_wise= True) 

model_LGB.fit(train_X, train_y,
           eval_set=[(val_X, val_y)]
)

pred_y_LGB = model_LGB.predict(val_X) 

print(mean_squared_error(val_y,pred_y_LGB,squared=False))


model_cat= CatBoostRegressor(iterations= 500, depth=8, learning_rate=0.1)

model_cat.fit(train_X, train_y,
           eval_set=[(val_X, val_y)]
)

pred_y_cat = model_cat.predict(val_X) 

print(mean_squared_error(val_y,pred_y_cat,squared=False))


OH_cols_test = pd.DataFrame(OHE.fit_transform(df_test[object_cols]))

# One-hot encoding removed index; put it back
OH_cols_test.index = df_test.index

# Remove categorical columns (will replace with one-hot encoding)
num_df_test = df_test.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_df_test = pd.concat([num_df_test, OH_cols_test], axis=1)



id=OH_df_test.pop('id')


OH_df_test


predictions_xgb= model_XGB.predict(OH_df_test)
predictions_lgb= model_LGB.predict(OH_df_test)
predictions_cat=model_cat.predict(OH_df_test)


predictions= predictions_xgb * 0.4 + predictions_lgb * 0.3 + predictions_cat * 0.3


predictions = predictions.flatten()


output = pd.DataFrame({ 'id':id,
                       'Target': predictions})


output.set_index('id')


output.to_csv('submission.csv', index=False)

