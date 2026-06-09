import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


train_df=pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test_df=pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


train_df.sample(2)


test_df.sample(2)


X_train=train_df.drop('accident_risk',axis=1)
y_train=train_df['accident_risk']
X_test=test_df


train_df.info()


train_df.describe()


train_df.isna().sum()


test_df.isna().sum()


num_cols=train_df.drop(['id'],axis=1).select_dtypes(include=['int64','float64']).columns.tolist()
cat_cols=train_df.select_dtypes(include=['object','bool']).columns.tolist()


num_cols


cat_cols


no_of_plots=2
no_of_rows=(len(num_cols)+1)//no_of_plots

plt.figure(figsize=(12,4*no_of_rows))

for i,col in enumerate(num_cols):
    subplot_index=i+1
    plt.subplot(no_of_rows,no_of_plots,subplot_index)
    sns.histplot(train_df[col],bins=10,kde=True)
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.show()


no_of_plots=2
no_of_rows=(len(cat_cols)+1)//no_of_plots

plt.figure(figsize=(12,4*no_of_rows))

for i,col in enumerate(cat_cols):
    subplot_index=i+1
    plt.subplot(no_of_rows,no_of_plots,subplot_index)
    sns.countplot(data=train_df,x=col)
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.show()


from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
ohe=OneHotEncoder(sparse_output=False,drop='first')
oe=OrdinalEncoder(categories=[['foggy','rainy','clear'],['night','dim','daylight']])


ohe_enc_train=ohe.fit_transform(X_train[['road_type','road_signs_present','public_road','time_of_day','holiday','school_season']])
ohe_enc_test=ohe.transform(X_test[['road_type','road_signs_present','public_road','time_of_day','holiday','school_season']])


ohe_enc_train=pd.DataFrame(ohe_enc_train,columns=ohe.get_feature_names_out(['road_type','road_signs_present','public_road','time_of_day','holiday','school_season']))
ohe_enc_test=pd.DataFrame(ohe_enc_test,columns=ohe.get_feature_names_out(['road_type','road_signs_present','public_road','time_of_day','holiday','school_season']))


oe_enc_train=oe.fit_transform(X_train[['weather','lighting']])
oe_enc_test=oe.transform(X_test[['weather','lighting']])


oe_enc_train=pd.DataFrame(oe_enc_train,columns=oe.get_feature_names_out(['weather','lighting']))
oe_enc_test=pd.DataFrame(oe_enc_test,columns=oe.get_feature_names_out(['weather','lighting']))


X_train=pd.concat([X_train.reset_index(drop=True),ohe_enc_train.reset_index(drop=True),oe_enc_train.reset_index(drop=True)],axis=1)
X_test=pd.concat([X_test.reset_index(drop=True),ohe_enc_test.reset_index(drop=True),oe_enc_test.reset_index(drop=True)],axis=1)


X_train.drop(cat_cols,axis=1,inplace=True)
X_test.drop(cat_cols,axis=1,inplace=True)


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X_train_ss=ss.fit_transform(X_train.drop('id',axis=1))
X_test_ss=ss.transform(X_test.drop('id',axis=1))


X_train_ss=pd.DataFrame(X_train_ss,columns=X_train.drop('id',axis=1).columns)
X_test_ss=pd.DataFrame(X_test_ss,columns=X_test.drop('id',axis=1).columns)


X_train=pd.concat([X_train['id'].reset_index(drop=True),X_train_ss.reset_index(drop=True)],axis=1)
X_test=pd.concat([X_test['id'].reset_index(drop=True),X_test_ss.reset_index(drop=True)],axis=1)


X_train


y_train


def model_scores(model):
    nmse=cross_val_score(model,X_train.drop('id',axis=1),y_train,cv=KFold(n_splits=5,shuffle=True,random_state=42),scoring='neg_mean_squared_error',n_jobs=-1)
    r2score=cross_val_score(model,X_train.drop('id',axis=1),y_train,cv=KFold(n_splits=5,shuffle=True,random_state=42),scoring='r2',n_jobs=-1)
    rmse=np.sqrt(-nmse)
    print(model,rmse.mean(),r2score.mean())


from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.model_selection import cross_val_score,KFold,GridSearchCV


model_scores(LinearRegression())


model_scores(Lasso())


model_scores(Ridge())


import xgboost as xgb
from xgboost import XGBRegressor


model_scores(XGBRegressor(objective='reg:squarederror',random_state=42,n_jobs=-1))


param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
}
xgb_model = xgb.XGBRegressor(objective='reg:squarederror',
                           random_state=42,
                           n_jobs=-1)
grid_search = GridSearchCV(estimator=xgb_model,
                           param_grid=param_grid,
                           scoring='neg_mean_squared_error',
                           cv=3,
                           verbose=2,
                           n_jobs=-1)
grid_search.fit(X_train.drop('id',axis=1), y_train)

print("Best parameters:", grid_search.best_params_)
print("Best score (negative MSE):", grid_search.best_score_)


best_model = grid_search.best_estimator_


best_model


y_pred=best_model.predict(X_test.drop('id',axis=1))
y_pred


y_pred.shape


X_test.shape


output=pd.DataFrame({'id':X_test['id'],'accident_risk':y_pred})


output.sample(5)


output_path = "/kaggle/working/submission.csv"
output.to_csv(output_path,index=False)
print(f"✅ Saved: {output_path}")

