import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor, early_stopping, log_evaluation


train=pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


test.info()


df=pd.concat([train,test],axis=0)
df.drop('accident_risk',axis=1,inplace=True)


df.drop('id',axis=1,inplace=True)


df.info()


df.describe()


fig, axes = plt.subplots(4, 2, figsize=(20, 20))  
axes = axes.flatten() 

cols = ['road_type','lighting','weather','time_of_day',
        'school_season','holiday','road_signs_present','public_road']

for i, c in enumerate(cols):
    sns.countplot(x=df[c], ax=axes[i])
    axes[i].set_title(f"Count of {c}")

plt.tight_layout()
plt.show()



for c in df.select_dtypes('number'):
    print(f"Unique Values in {c}:{df[c].nunique()}")
    print(f"Value Counts\n{df[c].value_counts()}\n--------------------------------")


for c in df.select_dtypes('bool'):
    df[c]=df[c].astype(int)


for c in df.select_dtypes("O"):
    le=LabelEncoder()
    df[c]=le.fit_transform(df[c])


x_train=df[:517754]
x_test=df[517754:]


y=train['accident_risk']
sample=pd.concat([x_train,y],axis=1)


plt.figure(figsize=(12,5))
sns.heatmap(sample.corr(),annot=True,cmap='coolwarm')


scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)


xt,xv,yt,yv=train_test_split(x_train,y,test_size=0.2,random_state=42)


lg=LinearRegression()
lg.fit(xt,yt)
y_lg=lg.predict(xv)


r2_score(yv,y_lg)
mean_squared_error(yv,y_lg,squared=False)


xgb=XGBRegressor()


xgb.fit(xt,yt)


y_xgb=xgb.predict(xv)


mean_squared_error(yv,y_xgb,squared=False)


cat=CatBoostRegressor(
    iterations=1000,          
    learning_rate=0.1,  
    loss_function='RMSE',
    depth=6,                   
    verbose=100,             
    random_state=42
)


cat.fit(xt,yt)


y_cat=cat.predict(xv)
mean_squared_error(yv,y_cat,squared=False)



y_sub=pd.DataFrame(cat.predict(x_test))
cat_sub=pd.concat([test['id'],y_sub],axis=1)


cat_sub = cat_sub.rename(columns={
    0: 'accident_risk',
})


cat_sub


cat_sub.to_csv('cat_sub.csv',index=False)


estimators = [
    ('cat',CatBoostRegressor(
    iterations=1000,          
    learning_rate=0.1,  
    loss_function='RMSE',
    depth=6,                   
    verbose=100,             
    random_state=42
)),
    ('xgb', XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=6))
]

final_estimator = Ridge(alpha=1.0)
stack_model = StackingRegressor(
    estimators=estimators,
    final_estimator=final_estimator,
    passthrough=True,   
    cv=5
)

stack_model.fit(xt, yt)

y_stack=stack_model.predict(xv)


mean_squared_error(yv,y_stack,squared=False)


y_sub1=pd.DataFrame(stack_model.predict(x_test))
stack_sub=pd.concat([test['id'],y_sub1],axis=1)
stack_sub = stack_sub.rename(columns={
    0: 'accident_risk',
})



stack_sub


stack_sub.to_csv('stack_sub.csv',index=False)


from sklearn.model_selection import GridSearchCV

xgb = XGBRegressor(random_state=42)

xgb_param_grid = {
    "n_estimators": [200, 500],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [4, 6, 8],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

xgb_grid = GridSearchCV(
    estimator=xgb,
    param_grid=xgb_param_grid,
    scoring="neg_root_mean_squared_error",  
    cv=3,
    verbose=2,
    n_jobs=-1
)

xgb_grid.fit(xt, yt)

print("Best XGB Params:", xgb_grid.best_params_)
print("Best XGB RMSE:", -xgb_grid.best_score_)



best_xgb=XGBRegressor(
    colsample_bytree=1.0,
    learning_rate=0.01,
    max_depth=8,
    n_estimators=500,
    subsample=0.8,
    random_state=42
)
best_xgb.fit(xt,yt)
y_bestxgb=pd.DataFrame(best_xgb.predict(x_test))
best_xgb_sub=pd.concat([test['id'],y_bestxgb],axis=1)
best_xgb_sub= best_xgb_sub.rename(columns={
    0: 'accident_risk',
})
best_xgb_sub.to_csv('submission.csv',index=False)

