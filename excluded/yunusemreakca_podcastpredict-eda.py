import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import VotingRegressor
import lightgbm as lgb
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import ExtraTreesRegressor

from warnings import filterwarnings
filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


train.head()


test.head()


train.info()


test.info()


train.describe().T


test.describe().T


plt.figure(figsize =(14,8))
sns.boxplot(train.drop(["id"],axis = 1) ,palette ="deep")
plt.show()


plt.figure(figsize =(10,6))
sns.boxplot(test.drop(["id","Episode_Length_minutes"],axis = 1) ,palette ="Paired")
plt.show()


plt.figure(figsize = (14,7))
sns.boxplot(x= "Publication_Day" ,y= "Listening_Time_minutes", hue = "Publication_Time", data = train)
plt.show()


plt.figure(figsize = (10,7))
sns.histplot(train["Listening_Time_minutes"])
plt.xlabel("Listening Time Minutes")
plt.ylabel("Frequency")
plt.show()


plt.figure(figsize = (10,7))
top_categories = train["Genre"].value_counts().nlargest(10)
sns.barplot(x=top_categories.index, y=top_categories.values,palette = "rocket")
plt.show()


plt.figure(figsize = (12,7))
top_categories = train["Podcast_Name"].value_counts().nlargest(10)
sns.barplot(x=top_categories.index, y=top_categories.values,palette = "viridis")
plt.show()


plt.figure(figsize = (12,7))
top_categories = train["Episode_Title"].value_counts().nlargest(10)
sns.barplot(x=top_categories.index, y=top_categories.values,palette = "deep")
plt.show()


train.hist(figsize=(12, 9), bins=15)
plt.suptitle("Feature Distributions")
plt.show()


corr = train.select_dtypes(["int64","float64"]).corr()
plt.figure(figsize =(12,8))
sns.heatmap(corr,annot = True,cmap ="coolwarm",fmt = ".3f")
plt.show()


sns.boxplot(test["Episode_Length_minutes"])
plt.show()


test[test["Episode_Length_minutes"] > 1e6]


Q3 = train["Episode_Length_minutes"].quantile(0.75)
Q1 = train["Episode_Length_minutes"].quantile(0.25)
IQR = Q3 - Q1
Upper_limit = Q3 + 1.5 * IQR


outlier = train["Episode_Length_minutes"] > Upper_limit
train.loc[outlier, "Episode_Length_minutes"] = train["Episode_Length_minutes"].mean()


Q3 = train["Number_of_Ads"].quantile(0.75)
Q1 = train["Number_of_Ads"].quantile(0.25)
IQR = Q3 - Q1
Upper_limit = Q3 + 1.5 * IQR


outlier = train["Number_of_Ads"] > Upper_limit
train.loc[outlier, "Number_of_Ads"] = train["Number_of_Ads"].mean()


Q3 = test["Number_of_Ads"].quantile(0.75)
Q1 = test["Number_of_Ads"].quantile(0.25)
IQR = Q3 - Q1
Upper_limit = Q3 + 1.5 * IQR


outlier = test["Number_of_Ads"] > Upper_limit
test.loc[outlier, "Number_of_Ads"] = test["Number_of_Ads"].mean()


Q3 = test["Episode_Length_minutes"].quantile(0.75)
Q1 = test["Episode_Length_minutes"].quantile(0.25)
IQR = Q3 - Q1
Upper_limit = Q3 + 3 * IQR


outlier = test["Episode_Length_minutes"] > Upper_limit
test.loc[outlier, "Episode_Length_minutes"] = test["Episode_Length_minutes"].mean()


plt.figure(figsize =(14,8))
sns.boxplot(train.drop(["id"],axis = 1) ,palette ="deep")
plt.show()


plt.figure(figsize =(14,8))
sns.boxplot(test.drop(["id"],axis = 1) ,palette ="deep")
plt.show()


print(train.isnull().sum())
print("----------------------------------------------------------")
print(test.isnull().sum())


print(train.isnull().sum()/ len(train) * 100)
print("----------------------------------------------------------")
print(test.isnull().sum()/ len(train) * 100)


train['Number_of_Ads'].fillna(train['Number_of_Ads'].median(), inplace=True)

train['Episode_Length_minutes'].fillna(train['Episode_Length_minutes'].median(), inplace=True)
test['Episode_Length_minutes'].fillna(test['Episode_Length_minutes'].median(), inplace=True)

train['Guest_Popularity_percentage'].fillna(train['Guest_Popularity_percentage'].median(), inplace=True)
test['Guest_Popularity_percentage'].fillna(test['Guest_Popularity_percentage'].median(), inplace=True)


train["Text_Lenght_Name"] = train["Podcast_Name"].apply(len)
test["Text_Lenght_Name"] = test["Podcast_Name"].apply(len)


def create_meaningful_arithmetic_features(df):
    epsilon = 1e-6
    
    df['Host_Popularity_per_Minute'] = df['Host_Popularity_percentage'] / (df['Episode_Length_minutes'] + epsilon)
    df['Guest_Popularity_per_Minute'] = df['Guest_Popularity_percentage'] / (df['Episode_Length_minutes'] + epsilon)

    df['Ads_per_Minute'] = df['Number_of_Ads'] / (df['Episode_Length_minutes'] + epsilon)

    df['Host_Guest_Popularity_Ratio'] = (df['Host_Popularity_percentage'] + epsilon) / (df['Guest_Popularity_percentage'] + epsilon)
    df['Guest_Host_Popularity_Ratio'] = (df['Guest_Popularity_percentage'] + epsilon) / (df['Host_Popularity_percentage'] + epsilon)

    df['Total_Popularity'] = df['Host_Popularity_percentage'] + df['Guest_Popularity_percentage']

    df['Popularity_Difference'] = df['Host_Popularity_percentage'] - df['Guest_Popularity_percentage']

    df['Ads_per_Host_Popularity'] = df['Number_of_Ads'] / (df['Host_Popularity_percentage'] + epsilon)
    df['Ads_per_Guest_Popularity'] = df['Number_of_Ads'] / (df['Guest_Popularity_percentage'] + epsilon)

    return df

train = create_meaningful_arithmetic_features(train.copy())
test = create_meaningful_arithmetic_features(test.copy())


train.head()


X = train.drop(["id","Listening_Time_minutes","Podcast_Name","Episode_Title"],axis=1)
y = train["Listening_Time_minutes"]
X_test = test.drop(["id","Podcast_Name","Episode_Title"],axis=1)

numeric_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X[numeric_cols])
X_test_scaled = scaler.transform(X_test[numeric_cols])

X_scaled_df = pd.DataFrame(X_scaled, columns=numeric_cols)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=numeric_cols)



train['Is_Episode_Length_Missing'] = train['Episode_Length_minutes'].isnull().astype(int)
test['Is_Episode_Length_Missing'] = test['Episode_Length_minutes'].isnull().astype(int)

train['Is_Guest_Popularity_Missing'] = train['Guest_Popularity_percentage'].isnull().astype(int)
test['Is_Guest_Popularity_Missing'] = test['Guest_Popularity_percentage'].isnull().astype(int)


categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
pref = ["Gen","Day","Time","Sent"]

X_train_encode = pd.get_dummies(data = X ,columns =  categorical_cols, prefix = pref,drop_first = True)
X_test_encode = pd.get_dummies(data = X_test,columns = categorical_cols,prefix = pref,drop_first=True)

bool_columns = X_train_encode.select_dtypes(include = "bool").columns
X_train_encode[bool_columns] = X_train_encode[bool_columns].astype(int)

bool_columns = X_test_encode.select_dtypes(include = "bool").columns
X_test_encode[bool_columns] = X_test_encode[bool_columns].astype(int)


X_categorical = X_train_encode.drop(numeric_cols, axis=1)
X_train_final = pd.concat([X_scaled_df, X_categorical.reset_index(drop=True)], axis=1)

X_categorical_test = X_test_encode.drop(numeric_cols, axis=1)
X_scaled_test_final = pd.concat([X_test_scaled_df, X_categorical_test.reset_index(drop=True)], axis=1)


xgb_model = XGBRegressor(random_state = 42,colsample_bytree = 0.7,gamma =1,learning_rate = 0.05,max_depth= 10,n_estimators = 1000,subsample = 0.8,reg_lambda = 0.1,reg_alpha = 10)


scores = cross_val_score(xgb_model, X_train_final, y, cv=5, scoring='neg_root_mean_squared_error')
rmse_scores = np.abs(scores)
print(f"Mean AUC: {rmse_scores.mean():.4f}")


xgb_model.fit(X_train_final , y)
y_test_pred = xgb_model.predict(X_scaled_test_final)


submission = pd.DataFrame({
    "id": test["id"],
    "Episode_Length_minutes": y_test_pred
})

submission.to_csv("submission_xgboost_podcast.csv", index=False)


lasso_model = Lasso(random_state = 42,alpha=  0.001, max_iter=1000, selection='cyclic', tol= 0.01)


scores = cross_val_score(lasso_model, X_train_final, y, cv=5, scoring='neg_root_mean_squared_error')
rmse_scores = np.abs(scores)
print(f"Mean AUC: {rmse_scores.mean():.4f}")


lasso_model.fit(X_train_final , y)
y_test_pred = lasso_model.predict(X_scaled_test_final)


submission = pd.DataFrame({
    "id": test["id"],
    "Episode_Length_minutes": y_test_pred
})

submission.to_csv("submission_lasso_podcast.csv", index=False)


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


ann_model.fit(X_train_final, y)
y_test_pred = ann_model.predict(X_scaled_test_final)


submission = pd.DataFrame({
    "id": test["id"],
    "Episode_Length_minutes": y_test_pred
})

submission.to_csv("submission_ann.csv", index=False)


ridge_model = Ridge(alpha = 10,fit_intercept = True,solver = "lsqr")


ridge_model.fit(X_train_final, y)
y_test_pred = ridge_model.predict(X_scaled_test_final)


scores = cross_val_score(ridge_model, X_train_final, y, cv=5, scoring='neg_root_mean_squared_error')
rmse_scores = np.abs(scores)
print(f"Mean AUC: {rmse_scores.mean():.4f}")


submission = pd.DataFrame({
    "id": test["id"],
    "Episode_Length_minutes": y_test_pred
})

submission.to_csv("submission_ridge.csv", index=False)


best_params = {'lr__fit_intercept': True, 'pca__n_components': 30, 'pca__whiten': False}


pca = PCA(n_components=best_params['pca__n_components'], whiten=best_params['pca__whiten'])
linear_regression = LinearRegression(fit_intercept=best_params['lr__fit_intercept'])


best_pcr_model = Pipeline([
    ('scaler', scaler),
    ('pca', pca),
    ('lr', linear_regression)
])


best_pcr_model.fit(X_train_final, y)
y_test_pred = best_pcr_model.predict(X_scaled_test_final)


scores = cross_val_score(best_pcr_model, X_train_final, y, cv=5, scoring='neg_root_mean_squared_error')
rmse_scores = np.abs(scores)
print(f"Mean AUC: {rmse_scores.mean():.4f}")


submission = pd.DataFrame({
    "id": test["id"],
    "Episode_Length_minutes": y_test_pred
})

submission.to_csv("submission_pcr.csv", index=False)


# {'min_samples_leaf': 50, 'max_iter': 750, 'max_depth': 20, 'max_bins': 255, 'learning_rate': 0.1, 'l2_regularization': 5.0}


hgb_model = HistGradientBoostingRegressor(min_samples_leaf =50,max_iter=750,max_depth=20,max_bins=255,learning_rate = 0.1, l2_regularization = 5.0)


scores = cross_val_score(hgb_model, X_train_final, y, cv=5, scoring='neg_root_mean_squared_error')
rmse_scores = np.abs(scores)
print(f"Mean AUC: {rmse_scores.mean():.4f}")


hgb_model.fit(X_train_final, y)
y_test_pred = hgb_model.predict(X_scaled_test_final)


submission = pd.DataFrame({
    "id": test["id"],
    "Episode_Length_minutes": y_test_pred
})

submission.to_csv("submission_hgb.csv", index=False)


xgb_model = XGBRegressor(random_state = 42,colsample_bytree = 0.7,gamma =1,learning_rate = 0.05,max_depth= 10,n_estimators = 1000,subsample = 0.8,reg_lambda = 0.1,reg_alpha = 10)
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


voting = VotingRegressor(
    estimators=[
        ('xgb', xgb_model),
        ('mlp', ann_model)
    ],
    n_jobs=-1
)

voting.fit(X_train_final, y)
y_test_pred = voting.predict(X_scaled_test_final)


submission = pd.DataFrame({
    "id": test["id"],
    "Episode_Length_minutes": y_test_pred
})

submission.to_csv("submission_voting_ann_xgb.csv", index=False)


lgb_params = {
    'objective': 'regression',  
    'metric': 'rmse', 
    'num_leaves': 31, 
    'max_depth': -1,  
    'learning_rate': 0.05, 
    'n_estimators': 1000, 
    'subsample_for_bin': 200000, 
    'subsample': 0.8,  
    'colsample_bytree': 0.8,  
    'min_child_weight': 1, 
    'min_child_samples': 20,  
    'max_bin': 255,  
    'bagging_fraction': 0.8,
    'bagging_freq': 5, 
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,  
    'n_jobs': -1,  
    'verbose': -1, 
}
lgbm_model = lgb.LGBMRegressor(**lgb_params)

lgbm_model.fit(X_train_final, y)



y_test_pred = lgbm_model.predict(X_scaled_test_final)


submission = pd.DataFrame({
    "id": test["id"],
    "Episode_Length_minutes": y_test_pred
})

submission.to_csv("submission_light.csv", index=False)


xgb_model = XGBRegressor(random_state = 42,colsample_bytree = 0.7,gamma =1,learning_rate = 0.05,max_depth= 10,n_estimators = 1000,subsample = 0.8,reg_lambda = 0.1,reg_alpha = 10)
lgb_params = {
    'objective': 'regression','metric': 'rmse','num_leaves': 31,'max_depth': -1,'learning_rate': 0.05,'n_estimators': 1000,'subsample_for_bin': 200000,'subsample': 0.8,'colsample_bytree': 0.8,
    'min_child_weight': 1,'min_child_samples': 20,'max_bin': 255,'bagging_fraction': 0.8,'bagging_freq': 5,'lambda_l1': 0.1,'lambda_l2': 0.1,'n_jobs': -1,'verbose': -1,}
lgbm_model = lgb.LGBMRegressor(**lgb_params)


voting = VotingRegressor(
    estimators=[
        ('xgb', xgb_model),
        ('lgbm', lgbm_model)
    ],
    n_jobs=-1
)

voting.fit(X_train_final, y)
y_test_pred = voting.predict(X_scaled_test_final)


submission = pd.DataFrame({
    "id": test["id"],
    "Episode_Length_minutes": y_test_pred
})

submission.to_csv("submission_light_xgb.csv", index=False)


elastic_model = ElasticNet(random_state = 42 , alpha =0.0004449700268707565 ,l1_ratio = 0.4722149251619493,fit_intercept =True ,max_iter = 1500,tol= 0.0013795402040204172,selection ="cyclic")
elastic_model.fit(X_train_final, y)

y_test_pred = elastic_model.predict(X_scaled_test_final)


submission = pd.DataFrame({
    "id": test["id"],
    "Episode_Length_minutes": y_test_pred
})

submission.to_csv("submission_elastic.csv", index=False)


et_model = ExtraTreesRegressor(
    n_estimators=500,            
    max_depth=50,               
    min_samples_split=5,        
    min_samples_leaf=2,     
    max_features='sqrt',        
    n_jobs=-1,
    verbose=3,
    random_state=42,             
)



et_model.fit(X_train_final, y)


y_pred = et_model.predict(X_scaled_test_final)


submission = pd.DataFrame({
    "id": test["id"],
    "Episode_Length_minutes": y_pred
})

submission.to_csv("submission_extratree.csv", index=False)


lm = LinearRegression()
model = lm.fit(X_train_final,y)
y_pred = model.predict(X_scaled_test_final)

print("Weights:", model.coef_)
print("Intercept:", model.intercept_)


submission = pd.DataFrame({
    "id": test["id"],
    "Episode_Length_minutes": y_pred
})

submission.to_csv("submission_linear.csv", index=False)




