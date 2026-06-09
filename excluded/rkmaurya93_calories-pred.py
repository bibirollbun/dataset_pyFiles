import pandas as pd
import numpy as np


df=pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv",index_col=0)


df.head()


df.shape


df.info()


df.isnull().sum()


df.describe()


from sklearn.preprocessing import LabelEncoder


le=LabelEncoder()


df['Sex']=le.fit_transform(df['Sex'])


df.head()


import seaborn as sns
import matplotlib.pyplot as plt


sns.displot(df['Age'],kde=True)
plt.show()


sns.boxplot(df['Age'])


sns.displot(df['Height'],kde=True)


sns.boxplot(df['Height'])
plt.show()


df.columns


sns.displot(df['Weight'],kde=True)


sns.boxplot(df['Weight'])


sns.displot(df['Duration'],kde=True)


sns.boxplot(df['Duration'])


sns.displot(df['Heart_Rate'],kde=True)


sns.boxplot(df['Heart_Rate'])


sns.displot(df['Body_Temp'],kde=True)


sns.boxplot(df['Body_Temp'])



# Function to detect and clip outliers using IQR
def clip_outliers_iqr(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Clipping the column values to the lower and upper bounds
    return column.clip(lower=lower_bound, upper=upper_bound)

# Apply the clipping function to numerical columns
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df[col] = clip_outliers_iqr(df[col])

# Display the DataFrame after clipping
df.head()


sns.boxplot(df['Calories'])


df.corr()


correlation_matrix=df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix,annot=True,cmap='coolwarm',fmt='2f')
plt.title('correlation heatmap')
plt.show()


x=df.drop('Calories',axis=1)
y=df['Calories']


from sklearn.model_selection import train_test_split


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=41)


x_train.head()


from sklearn.preprocessing import StandardScaler


ss=StandardScaler()


x_train_transform=ss.fit_transform(x_train)
x_test_transform=ss.transform(x_test)


print('missing vlaue in y_train',y_train.isnull().sum())


print('missing values in y_test',y_test.isnull().sum())


x_train.isnull().sum()


print("Missing values in X_train:", x_train.isna().sum().sum())
print("Missing values in X_test:", x_test.isna().sum().sum())


print("NaNs in X_train after preprocessing:", np.isnan(x_train).sum().sum())
print("NaNs in X_test after preprocessing:", np.isnan(x_test).sum().sum())

print("Infs in X_train:", np.isinf(x_train).sum().sum())
print("Infs in X_test:", np.isinf(x_test).sum().sum())


x_test.isnull().sum()


x_train_transform


x_test_transform


from xgboost import XGBRegressor



xgb_model = XGBRegressor(
    n_estimators=300,     # Increase trees for better learning
    learning_rate=0.05,   # Reduce step size to prevent extreme predictions
    max_depth=10,          # Balance complexity
    min_child_weight=10,   # Avoid overfitting and unstable splits
    gamma=0.5,            # Add regularization
    subsample=0.8,        # Prevent reliance on certain features
    colsample_bytree=0.8, # Limit individual tree feature selection
    verbosity=2           # Enable debugging output
)


xgb_model.fit(x_train,y_train)


y_pred_xgb=xgb_model.predict(x_test)


print("Number of values close to zero:", np.sum(y_pred_xgb < 1e-6))


y_pred_xgb = np.where(y_pred_xgb < 1e-6, 1e-6, y_pred_xgb)  # Replace small values
y_pred_log = np.log(y_pred_xgb)  # Apply log safely


print("Min value in y_pred_xgb:", np.min(y_pred_xgb))
print("Number of negative or zero values:", np.sum(y_pred_xgb <= 0))


print("NaN in y_test_log:", np.isnan(y_test_log).sum())
print("NaN in y_pred_log:", np.isnan(y_pred_log).sum())


from sklearn.metrics import r2_score,mean_squared_error


y_test_log=np.log(y_test)


print("Number of NaN values in predictions:", np.isnan(y_test_log).sum())


rmsle=np.sqrt(mean_squared_error(y_test_log,y_pred_log))


rmsle


import matplotlib.pyplot as plt
import xgboost as xgb

xgb_model = XGBRegressor()
xgb_model.fit(x_train, y_train)

# Plot feature importance
xgb.plot_importance(xgb_model)
plt.show()


import optuna 
from sklearn.model_selection import cross_val_score


import optuna
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500)
    }

    model = XGBRegressor(**params)  # Ensure correct argument passing
    score = cross_val_score(model, x_train, y_train, scoring="neg_root_mean_squared_error", cv=5).mean()
    return -score  # Optuna minimizes, so we invert RMSE

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

print("Best Hyperparameters:", study.best_params_)


print("Best Hyperparameters:", study.best_trial.params)


best_params = study.best_trial.params
xgb_model = XGBRegressor(**best_params)
xgb_model.fit(x_train, y_train)


xgb_model_pred=xgb_model.predict(x_test)


y_pred_xgb = np.where(y_pred_xgb < 1e-6, 1e-6, xgb_model_pred)  # Replace small values
y_pred_log = np.log(xgb_model_pred)  # Apply log safely


y_test_log=np.log(y_test)


rmsle_xgb=np.sqrt(mean_squared_error(y_test_log,y_pred_log))


rmsle_xgb


from catboost import CatBoostRegressor


model = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, verbose=100)



model.fit(x_train,y_train)


pred_cat=model.predict(x_test)


y_pred_ = np.where(pred_cat < 1e-6, 1e-6, pred_cat)  # Replace small values
y_pred_cat = np.log(pred_cat)  # Apply log safely


from sklearn.metrics import mean_squared_error


y_test_log=np.log(y_test)


rmsle_cat=np.sqrt(mean_squared_error(y_test_log,y_pred_log))


rmsle_cat


from sklearn.ensemble import RandomForestRegressor


rf_model=RandomForestRegressor(n_estimators=200)


rf_model.fit(x_train,y_train)


rf_model_prd=rf_model.predict(x_test)


from sklearn.metrics import r2_score,mean_squared_error


y_test_log=np.log(y_test)
y_pred_log=np.log(rf_model_prd)


rmsle=np.sqrt(mean_squared_error(y_test_log,y_pred_log))


y_pred_xgb = np.where(y_pred_xgb < 1e-6, 1e-6, xgb_model_pred)  # Replace small values
y_pred_log = np.log(xgb_model_pred)  # Apply log safely


df_test=pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv",index_col=0)


df_test.head()





df_test['Sex']=le.transform(df_test['Sex'])


df_test.head()



# Function to detect and clip outliers using IQR
def clip_outliers_iqr(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Clipping the column values to the lower and upper bounds
    return column.clip(lower=lower_bound, upper=upper_bound)

# Apply the clipping function to numerical columns
for col in df_test.select_dtypes(include=['float64', 'int64']).columns:
    df_test[col] = clip_outliers_iqr(df_test[col])

# Display the DataFrame after clipping
df.head()


test_pred=model.predict(df_test)


test_pred


submission_df=pd.DataFrame({'id':df_test.index,'Calories':test_pred})


submission_df.head()


submission_df.to_csv('submission_rf.csv',index=False)




