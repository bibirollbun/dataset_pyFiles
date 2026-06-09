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


sample=pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv",sep=",")
train=pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv",sep=",")
test=pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv",sep=",")
extra=pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv",sep=",")


sample.head(4)


train.head()


test.head()


extra.head()


sample.shape,train.shape,extra.shape,test.shape


df=pd.concat([train,extra])


df.isna().sum()


import seaborn as sns
import matplotlib.pyplot as plt
# Plot using seaborn
plt.figure(figsize=(6, 4))
sns.countplot(x=train["Brand"], palette='pastel')

plt.xlabel('Categories')
plt.ylabel('Count')
plt.title('Histogram-like Bar Chart for Categorical Variable')
plt.show()



import seaborn as sns
import matplotlib.pyplot as plt
# Plot using seaborn
plt.figure(figsize=(6, 4))
sns.countplot(x=train["Material"], palette='pastel')

plt.xlabel('Categories')
plt.ylabel('Count')
plt.title('Histogram-like Bar Chart for Categorical Variable')
plt.show()


df


y=df["Price"]





y.isna().sum()


df.drop(["Price","id"],axis=1,inplace=True)
test.drop(["id"],axis=1,inplace=True)


df.shape,test.shape


from sklearn.impute import SimpleImputer 
k=SimpleImputer()
df["Weight Capacity (kg)"]=k.fit_transform(df[["Weight Capacity (kg)"]])


test["Weight Capacity (kg)"]=k.transform(test[["Weight Capacity (kg)"]])


df=pd.DataFrame(df)
tf=pd.DataFrame(test)



df


df_num=df.iloc[:,[3,8]]
tf_num=tf.iloc[:,[3,8]]


tf_num


df_cat=df.drop(df_num,axis=1)
tf_cat=tf.drop(tf_num,axis=1)


df_cat


df.isna().sum()





from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy="most_frequent")
df=imputer.fit_transform(df_cat)
tf=imputer.transform(tf_cat)





from sklearn.preprocessing import OrdinalEncoder,StandardScaler,RobustScaler


encoder=OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1)


df_cat=encoder.fit_transform(df)
tf_cat=encoder.transform(tf)


df_cat=pd.DataFrame(df_cat)
tf_cat=pd.DataFrame(tf_cat)


df_cat


df_num.columns=[x for x in range(7,len(df_num.columns)+7)]


tf_num.columns=[x for x in range(7,len(df_num.columns)+7)]


df_num[8]=np.log(df_num[8])


tf_num[8]=np.log(tf_num[8])


df_num[9]=df_num.mean(axis=1)
tf_num[9]=tf_num.mean(axis=1)


tf_num


df_cat = df_cat.reset_index(drop=True)
df_num = df_num.reset_index(drop=True)







dff=pd.concat([df_cat,df_num],axis=1)
tff=pd.concat([tf_cat,tf_num],axis=1)





dff


scaler=RobustScaler()
dff=scaler.fit_transform(dff)
tff=scaler.transform(tff)


from sklearn.preprocessing import PolynomialFeatures


poly=PolynomialFeatures(degree=2)


dff=poly.fit_transform(dff)
tff=poly.transform(tff)


dff.shape


from sklearn.decomposition import PCA


pca=PCA(n_components=0.95)


from sklearn.feature_selection import SelectKBest


# best=SelectKBest(k=40)



# dff=best.fit_transform(dff ,y)
# tff=best.transform(tff)


dff=pca.fit_transform(dff)
tff=pca.transform(tff)


dff.shape


from sklearn.linear_model import LinearRegression


from sklearn.linear_model import Ridge ,ElasticNet
net=ElasticNet()
net.fit(dff,y)


sol=net.predict(dff)


tsol=net.predict(tff)


from xgboost import XGBRegressor


from catboost import CatBoostRegressor


from lightgbm import LGBMRegressor


model1=LGBMRegressor()


# model1.fit(dff,y)


# lgbm=model1.predict(dff)


# tlgbm=model1.predict(tff)


# model=XGBRegressor()
# model=CatBoostRegressor()


from sklearn.ensemble import RandomForestRegressor 
# model=RandomForestRegressor()


# model=LinearRegression()
# model.fit(dff,y)


# model.score(dff,y)


import optuna
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Load your dataset (Replace this with your actual dataset)
# X, y = <your_data_loading_function()>

# Split data
X_train, X_test, y_train, y_test = train_test_split(dff, y, test_size=0.25, random_state=42)

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),  # Fewer steps
        'max_depth': trial.suggest_int('max_depth', 3, 8),  # Lower max depth
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2, step=0.05),  # No need for very small LR
        'subsample': trial.suggest_float('subsample', 0.7, 1.0, step=0.1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0, step=0.1),
    }

    model = xgb.XGBRegressor(**params, tree_method='hist', random_state=42)  # ✅ Fastest XGBoost method
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False, early_stopping_rounds=10)  # ✅ Early stopping

    y_pred = model.predict(X_test)
    return -mean_squared_error(y_test, y_pred)  # ✅ Direct test set evaluation (faster than cross-validation)

# Run Optuna with fewer trials
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10, show_progress_bar=True)  # ✅ Small n_trials

# Train final model with best params
best_params = study.best_params
final_model = xgb.XGBRegressor(**best_params, tree_method='hist', random_state=42)  # ✅ Faster tree method
final_model.fit(X_train, y_train)

# Evaluate on test set
y_pred = final_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Best Params: {best_params}')
print(f'Test MSE: {mse}')



# sol2=model.predict(dff)



# prediction=(sol2*0.6+lgbm*0.4)


p=final_model.predict(tff)


p


# ans=tlgbm*0.4+p*0.6


from sklearn.metrics import mean_squared_error


mse=mean_squared_error 


# r=mse(y,prediction)
# r


# np.sqrt(r)


import pandas as pd

# Sample data (Replace with your actual predictions)
data = {
    "id": [i for i in range(300000,len(test)+300000)],  # Replace with actual IDs
    "Price": p  # Replace with predicted prices
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save as CSV
df.to_csv("submission.csv", index=False)

print("Submission file 'submission.csv' created successfully!")



df










