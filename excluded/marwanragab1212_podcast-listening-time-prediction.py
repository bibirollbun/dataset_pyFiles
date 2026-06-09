import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


train_df=pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test_df=pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
sumbission_df=pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


print(train_df.shape)
print(test_df.shape)


sumbission_df.head(5)


train_df.isna().sum()


train_df.isna().sum()


thr=len(train_df)*0.05
thr


cols_drop=train_df.columns[train_df.isna().sum()<=thr]
cols_drop


train_df.dropna(subset=cols_drop,inplace=True)


cols_missing=train_df.columns[train_df.isna().sum()>0]
cols_missing


for col in cols_missing:
  train_df[col].fillna(train_df[col].mode()[0],inplace=True)


train_df.head(5)


train_df.isna().sum()


for col in train_df:
  print(col,train_df[col].nunique())


train_df=train_df.drop(['id'],axis=1)


train_df.head(5)


test_df.isna().sum()


thr=len(test_df)*0.05
cols_drop=test_df.columns[test_df.isna().sum()<=thr]
test_df.dropna(subset=cols_drop,inplace=True)
cols_missing=test_df.columns[test_df.isna().sum()>0]
for col in cols_missing:
  test_df[col].fillna(test_df[col].mode()[0],inplace=True)


test_df.isna().sum()


corr = train_df.select_dtypes(["int64","float64"]).corr()
plt.figure(figsize =(12,8))
sns.heatmap(corr,annot = True,cmap ="coolwarm",fmt = ".3f")
plt.show()


sns.boxplot(test_df['Episode_Length_minutes'])
plt.show()



sns.boxplot(train_df,palette="deep")
plt.show()


sns.boxplot(test_df['Number_of_Ads'])
plt.show()


sns.boxplot(test_df.drop("id",axis=1),palette="deep")
plt.show()


test_df["Episode_Length_minutes"].describe()


seventy_pre=test_df["Episode_Length_minutes"].quantile(0.75)
twenty_pre=test_df["Episode_Length_minutes"].quantile(0.25)
IQR=seventy_pre-twenty_pre
IQR


upper=seventy_pre+(1.5*IQR)
lower=twenty_pre-(1.5*IQR)
print(upper,lower)


outlier = test_df["Episode_Length_minutes"] > upper


test_df.loc[outlier, "Episode_Length_minutes"] = test_df["Episode_Length_minutes"].mean()


sns.boxplot(train_df['Number_of_Ads'])
plt.show()


seventy_pre=train_df["Number_of_Ads"].quantile(0.75)
twenty_pre=train_df["Number_of_Ads"].quantile(0.25)
IQR=seventy_pre-twenty_pre
IQR


upper=seventy_pre+(1.5*IQR)
lower=twenty_pre-(1.5*IQR)
print(upper,lower)


outlire=train_df["Number_of_Ads"]>upper
train_df.loc[outlire,"Number_of_Ads"]=train_df["Number_of_Ads"].mean()


sns.boxplot(train_df['Number_of_Ads'])
plt.show()


seventy_pree=test_df["Number_of_Ads"].quantile(0.75)
twenty_pree=test_df["Number_of_Ads"].quantile(0.25)
IQR=seventy_pre-twenty_pre
IQR


upper=seventy_pree+(1.5*IQR)
lower=twenty_pree-(1.5*IQR)
print(upper,lower)


outlire=test_df["Number_of_Ads"]>upper
test_df.loc[outlire,"Number_of_Ads"]=test_df["Number_of_Ads"].mean()


sns.boxplot(test_df['Number_of_Ads'])
plt.show()


train_df.head(5)


X=train_df.drop(["Podcast_Name","Episode_Title","Listening_Time_minutes"],axis=1)
y=train_df.Listening_Time_minutes
x_test=test_df.drop(["id","Podcast_Name","Episode_Title"],axis=1)


from sklearn.preprocessing import StandardScaler
cols_num = X.select_dtypes(include=['float64', 'int64']).columns
cols_numt = x_test.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
X[cols_num] = scaler.fit_transform(X[cols_num])
x_test[cols_numt] = scaler.fit_transform(x_test[cols_numt])


cols_cg=X.select_dtypes("object").columns
X = pd.get_dummies(X, columns=cols_cg, drop_first=True, dtype=int)


cols_cg=x_test.select_dtypes("object").columns
x_test = pd.get_dummies(x_test, columns=cols_cg, drop_first=True, dtype=int)


from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor


# xgb_model = XGBRegressor(random_state = 42,colsample_bytree = 0.7,gamma =1,learning_rate = 0.05,max_depth= 10,n_estimators = 1000,subsample = 0.8,reg_lambda = 0.1,reg_alpha = 10)

# score=cross_val_score(xgb_model,X,y,cv=5,scoring='neg_mean_squared_error')

# rmse_scores = np.abs(score)
# print(f"Mean AUC: {rmse_scores.mean():.4f}")


from sklearn.linear_model import Lasso
lasso_model = Lasso(random_state = 42,alpha=  0.001, max_iter=1000, selection='cyclic', tol= 0.01)
scores = cross_val_score(lasso_model, X, y, cv=5, scoring='neg_root_mean_squared_error')
rmse_scores = np.abs(scores)
print(f"Mean AUC: {rmse_scores.mean():.4f}")


from sklearn.linear_model import Ridge
ridge_model = Ridge(alpha = 10,fit_intercept = True,solver = "lsqr")
scores = cross_val_score(ridge_model, X, y, cv=5, scoring='neg_root_mean_squared_error')
rmse_scores = np.abs(scores)
print(f"Mean AUC: {rmse_scores.mean():.4f}")


ann_model = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=500,
    alpha=0.001,
    batch_size=128,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=42
)


ann_model.fit(X,y)



y_pred=ann_model.predict(x_test)


sumission=pd.DataFrame({"id":test_df["id"],"Listening_Time_minutes":y_pred})
sumission.to_csv("submission.csv",index=False)

