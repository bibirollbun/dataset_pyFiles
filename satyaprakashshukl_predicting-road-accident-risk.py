import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor



df_train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
df_test  = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
df_sub = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')


df_train.head()


df_train.describe()


df_test.head()


df_sub.head()


df_train.drop(columns=['id'], inplace=True)
df_test.drop(columns=['id'], inplace=True)



df_train.shape,df_test.shape


df_train.isnull().sum()


df_test.isnull().sum()


df_train.shape,df_test.shape,df_sub.shape


plt.figure(figsize=(12,6))
sns.boxplot(x="road_type", y="accident_risk", data=df_train)
plt.title("Accident Risk Distribution by Road Type")
plt.show()

plt.figure(figsize=(12,6))
sns.boxplot(x="lighting", y="accident_risk", data=df_train)
plt.title("Accident Risk by Lighting Conditions")
plt.show()



sns.jointplot(x="curvature", y="accident_risk", data=df_train, kind="hex", cmap="viridis")
plt.show()

sns.jointplot(x="speed_limit", y="accident_risk", data=df_train, kind="reg", scatter_kws={'alpha':0.2})
plt.show()



pivot = df_train.pivot_table(values="accident_risk", 
                             index="time_of_day", 
                             columns="weather", 
                             aggfunc="mean")
plt.figure(figsize=(10,6))
sns.heatmap(pivot, annot=True, cmap="YlOrRd")
plt.title("Mean Accident Risk by Time of Day & Weather")
plt.show()



plt.figure(figsize=(10,6))
sns.scatterplot(x="num_reported_accidents", y="accident_risk", data=df_train, alpha=0.3)
sns.regplot(x="num_reported_accidents", y="accident_risk", data=df_train, scatter=False, color="red")
plt.title("Accident Risk vs Reported Accidents")
plt.show()



df_train.corr()


df_train.dtypes



y = df_train['accident_risk']
X = df_train.drop(columns=['accident_risk'])


X_test = df_test


X



df_sub.head()


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))



def encode_categoricals(X_train, X_test):
    X_train_enc = X_train.copy()
    X_test_enc = X_test.copy()
    encoders = {}
    
    for col in X_train_enc.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X_train_enc[col] = le.fit_transform(X_train_enc[col].astype(str))
        X_test_enc[col] = le.transform(X_test_enc[col].astype(str))
        encoders[col] = le
    return X_train_enc, X_test_enc, encoders

X_encoded, X_test_encoded, encoders = encode_categoricals(X, X_test)


xgb_params = {
    'n_estimators': 4000,
    'max_depth': 15,
    'learning_rate': 0.015216251287478466,
    'subsample': 0.6869288537951837,
    'colsample_bytree': 0.939877058046764,
    'reg_alpha': 4.883004566524303e-06,
    'reg_lambda': 0.0016477801911808878,
    'min_child_weight': 8,
    'gamma': 0.009579430708897484,
    'random_state': 42,
    'n_jobs': -1,
    'tree_method': 'hist'  
}


xgb_model = XGBRegressor(**xgb_params)
print("Training>>>>>>>")
xgb_model.fit(X_encoded, y)
print("Predicting>>>>>>>")
y_pred = xgb_model.predict(X_test_encoded)
print("Done! Predictions shape:", y_pred.shape)


from xgboost import plot_importance



plot_importance(xgb_model, importance_type='gain', max_num_features=15)
plt.title("XGBoost Feature Importance - Gain")
plt.show()

plot_importance(xgb_model, importance_type='cover', max_num_features=15)
plt.title("XGBoost Feature Importance - Cover")
plt.show()



train_pred = xgb_model.predict(X_encoded)
train_rmse = rmse(y, train_pred)
print("Train RMSE:", train_rmse)


y_pred


df_sub['accident_risk'] = y_pred


df_sub.to_csv('submission.csv', index=False)


df_sub.head()


df_sub['accident_risk'].hist()

