import pandas as pd


df=pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv',index_col=0)


df.head()


df.shape


df.info()


df.isnull().sum()


cat_cols=df.select_dtypes(include='object').columns


df[cat_cols].head()


for i in cat_cols:
    print(i,df[i].value_counts())


df[cat_cols].nunique()


cat_cols


df.describe()


from sklearn.preprocessing import OrdinalEncoder


oe=OrdinalEncoder()


df[cat_cols]=oe.fit_transform(df[cat_cols])


df[cat_cols].head()


num_cols=df.select_dtypes(include='number').columns


df[num_cols].head()


df[num_cols].corr()


import seaborn as sns
import matplotlib.pyplot as plt


plt.figure(figsize=(10,6))
sns.boxplot(data=df[num_cols])
plt.show()


sns.displot(df['accident_risk'],kde=True)


sns.displot(df['curvature'],kde=True)


sns.displot(df['num_lanes'],kde=True)


sns.displot(df['num_reported_accidents'],kde=True)


sns.displot(df['speed_limit'],kde=True)


df.head()


bool_cols=df.select_dtypes(include='bool').columns
bool_cols


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for i in bool_cols:
    df[i]=le.fit_transform(df[i])


df.corr()


corr_matrix=df.corr()


plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix,annot=True,cmap='coolwarm')
plt.show()


x=df.drop('accident_risk',axis=1)
y=df['accident_risk']


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


x_train.head()


x_train.shape


y_train


from xgboost import XGBRegressor


model=XGBRegressor()


model.fit(x_train,y_train)


y_pred=model.predict(x_test)


from sklearn.metrics import mean_squared_error


mse=mean_squared_error(y_test,y_pred)


import numpy as np
rmse=np.sqrt(mse)


rmse


from sklearn.model_selection import RandomizedSearchCV, train_test_split
# Step 2: Define the model
xgb = XGBRegressor(objective='reg:squarederror', random_state=42)

# Step 3: Define hyperparameter grid
param_grid = {
    'n_estimators': [100, 300, 500, 800],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.3, 0.5],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [1, 1.5, 2]
}

# Step 4: Set up RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_grid,
    n_iter=50,
    scoring='neg_root_mean_squared_error',
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# Step 5: Fit the model
random_search.fit(x_train, y_train)

# Step 6: Evaluate
best_model = random_search.best_estimator_
y_pred = best_model.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Best Parameters:", random_search.best_params_)
print("Test RMSE:", rmse)


df_test=pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv',index_col=0)


x_train.head()


df_test.head()


df_test[cat_cols].head()


df_test[cat_cols].nunique()


df_test[cat_cols]=oe.fit_transform(df_test[cat_cols])


df_test[bool_cols].head()


df_test.isnull().sum()


for i in bool_cols:
    df_test[i]=le.fit_transform(df_test[i])


df_test.head()


random_search.predict(df_test) 


y_pred_test


pd.DataFrame(y_pred_test,columns=['accident_risk'],index=df_test.index).to_csv('submission.csv')




