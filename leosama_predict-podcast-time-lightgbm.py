import pandas as pd

df = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")


# !pip install -q lightgbm==3.3.3



df.head()


len(df["Podcast_Name"].unique())


df.isna().sum()


df["Guest_Popularity_percentage"].isna().sum() * 100 / len(df)


df["Episode_Length_minutes"].isna().sum() * 100 / len(df)


df['Episode_Length_minutes'] = df.groupby("Podcast_Name")['Episode_Length_minutes'].transform(lambda x: x.fillna(x.median()))



df['Guest_Popularity_percentage'] = df.groupby("Podcast_Name")['Guest_Popularity_percentage'].transform(lambda x: x.fillna(x.mean()))



df.dropna(inplace=True)


df.isna().sum()


import matplotlib.pyplot as plt

plt.figure(figsize=(14,6))

x = df.groupby(pd.qcut(df["Guest_Popularity_percentage"], q=5))["Listening_Time_minutes"].mean()

bars = plt.bar(x.index.astype(str), x.values)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, f'{height:.1f}', 
             ha='center', va='bottom', fontsize=10)





import matplotlib.pyplot as plt

plt.figure(figsize=(14,6))

x = df.groupby(pd.qcut(df["Host_Popularity_percentage"], q=5))["Listening_Time_minutes"].mean()

bars = plt.bar(x.index.astype(str), x.values)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, f'{height:.1f}', 
             ha='center', va='bottom', fontsize=10)





df["Number_of_Ads"] = df["Number_of_Ads"].clip(upper=3)


df["Number_of_Ads"].value_counts()





plt.figure(figsize=(14,6))

x = df.groupby(df["Number_of_Ads"])["Listening_Time_minutes"].mean()

bars = plt.bar(x.index.astype(str), x.values)


for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, f'{height:.1f}', 
             ha='center', va='bottom', fontsize=10)



df.head()


plt.figure(figsize=(14,6))

x = df.groupby(df["Genre"])["Listening_Time_minutes"].mean()

bars = plt.bar(x.index.astype(str), x.values)


for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, f'{height:.1f}', 
             ha='center', va='bottom', fontsize=10)



plt.figure(figsize=(14,6))

x = df.groupby(df["Publication_Day"])["Listening_Time_minutes"].mean()

bars = plt.bar(x.index.astype(str), x.values)


for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, f'{height:.1f}', 
             ha='center', va='bottom', fontsize=10)



df["Episode_Length_minutes"] = df["Episode_Length_minutes"].clip(upper=121.0)


import matplotlib.pyplot as plt

plt.figure(figsize=(20,6))

x = df.groupby(pd.qcut(df["Episode_Length_minutes"], q=10))["Listening_Time_minutes"].mean()

plt.xticks(rotation=90)
bars = plt.bar(x.index.astype(str), x.values)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, f'{height:.1f}', 
             ha='center', va='bottom', fontsize=10)





plt.figure(figsize=(10,6))

x = df.groupby(df["Episode_Sentiment"])["Listening_Time_minutes"].mean()

bars = plt.bar(x.index.astype(str), x.values)


for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, f'{height:.1f}', 
             ha='center', va='bottom', fontsize=10)




plt.subplot(1,2,1)
plt.boxplot(df["Episode_Length_minutes"])

plt.subplot(1,2,2)
plt.boxplot(df["Listening_Time_minutes"])

plt.show()


import seaborn as sns

plt.figure(figsize=(14,6))
sns.regplot(x="Episode_Length_minutes",y="Listening_Time_minutes",data=df,scatter_kws={'alpha':0.3})
plt.show()


from sklearn.linear_model import LinearRegression

filtered_df = df[df["Number_of_Ads"] == 3]

X = filtered_df[["Episode_Length_minutes"]]  
y = filtered_df["Listening_Time_minutes"]

model = LinearRegression()
model.fit(X, y)

print("Slope (m):", model.coef_[0])
print("Intercept (b):", model.intercept_)



df.head()


# df["Genre"] = df["Genre"].astype("category")
# df["Publication_Day"] = df["Publication_Day"].astype("category")
# df["Publication_Time"] = df["Publication_Time"].astype("category")
# df["Episode_Sentiment"] = df["Episode_Sentiment"].astype("category")





# import lightgbm as lgb
# from sklearn.model_selection import train_test_split

# X = df.drop(columns=["id","Podcast_Name","Episode_Title","Listening_Time_minutes"])
# y = df["Listening_Time_minutes"]


# X.info()


# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)




# from sklearn.metrics import accuracy_score,r2_score

# classifier = lgb.LGBMRegressor()

# classifier.fit(X,y,categorical_feature=["Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment"])





# df_test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
# df_test.head()


# df_test["Genre"] = df_test["Genre"].astype("category")
# df_test["Publication_Day"] = df_test["Publication_Day"].astype("category")
# df_test["Publication_Time"] = df_test["Publication_Time"].astype("category")
# df_test["Episode_Sentiment"] = df_test["Episode_Sentiment"].astype("category")


# X_test = df_test.drop(columns=["id","Podcast_Name","Episode_Title"])

# final_preds = classifier.predict(X_test)






# submission = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")
# submission['Listening_Time_minutes'] = final_preds
# submission.to_csv("submission.csv", index=False)
# print("ðŸš€ Submission saved.")


# from sklearn.metrics import mean_squared_error
# import numpy as np

# preds = classifier.predict(X_test)

# np.sqrt(mean_squared_error(y_test,preds))






# import lightgbm as lgb
# from sklearn.model_selection import train_test_split

# df["Genre"] = df["Genre"].astype("category")
# df["Publication_Day"] = df["Publication_Day"].astype("category")
# df["Publication_Time"] = df["Publication_Time"].astype("category")
# df["Episode_Sentiment"] = df["Episode_Sentiment"].astype("category")




# X = df.drop(columns=["id","Podcast_Name","Episode_Title","Listening_Time_minutes"])
# y = df["Listening_Time_minutes"]

# # classifier = lgb.LGBMRegressor()

# train_data = lgb.Dataset(X,label=y)





# params = {"objective":"regression",
#            'metric': 'rmse',
#            'learning_rate': 0.05,
#            'num_leaves': 31}


# cv_results = lgb.cv(
#     params,
#     train_data,
#     num_boost_round=3000,
#     stratified=False,
#     shuffle=True,
#     callbacks=[lgb.early_stopping(stopping_rounds=50)]

# )





# print('Best CV score:', cv_results['valid rmse-mean'][-1])



# best_num_boost_round = len(cv_results['valid rmse-mean'])
# best_num_boost_round


# final_model = lgb.train(
#     params,
#     train_data,
#     num_boost_round=best_num_boost_round
# )

# df_test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")

# df_test["Genre"] = df_test["Genre"].astype("category")
# df_test["Publication_Day"] = df_test["Publication_Day"].astype("category")
# df_test["Publication_Time"] = df_test["Publication_Time"].astype("category")
# df_test["Episode_Sentiment"] = df_test["Episode_Sentiment"].astype("category")

# X_test = df_test.drop(columns=["id","Podcast_Name","Episode_Title"])

# final_preds = final_model.predict(X_test)






# submission = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")
# submission['Listening_Time_minutes'] = final_preds
# submission.to_csv("submission.csv", index=False)
# print("ðŸš€ Submission saved.")





# import optuna


import lightgbm as lgb
from sklearn.model_selection import train_test_split

df["Genre"] = df["Genre"].astype("category")
df["Publication_Day"] = df["Publication_Day"].astype("category")
df["Publication_Time"] = df["Publication_Time"].astype("category")
df["Episode_Sentiment"] = df["Episode_Sentiment"].astype("category")



X = df.drop(columns=["id","Podcast_Name","Episode_Title","Listening_Time_minutes"])
y = df["Listening_Time_minutes"]


train_data = lgb.Dataset(X,label=y)


# def objective(trial):

#     param = {
#             'objective': 'regression',
#             'metric': 'rmse',
#             'boosting_type': 'gbdt',
#             'verbosity': -1,
#             'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
#             'num_leaves': trial.suggest_int('num_leaves', 15, 255),
#             'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
#             'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
#             'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
#             'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
#             'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 5.0),
#             'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 5.0)
#         }

#     cv_results = lgb.cv(
#         params,
#         train_data,
#         num_boost_round=3000,
#         stratified=False,
#         shuffle=True,
#         callbacks=[lgb.early_stopping(stopping_rounds=50),lgb.log_evaluation(0)]
    
#     )

#     return min(cv_results['valid rmse-mean'])



# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=10)


# best_params = study.best_trial

# best_params.params


# study.best_trial


best_params = {'learning_rate': 0.010727603199425518,
 'num_leaves': 154,
 'min_data_in_leaf': 81,
 'feature_fraction': 0.7326329461942478,
 'bagging_fraction': 0.665930857344637,
 'bagging_freq': 6,
 'lambda_l1': 0.6323954087710476,
 'lambda_l2': 4.233978144438282}

final_model = lgb.train(
    best_params,
    train_data,
    num_boost_round=3000
)

df_test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")

df_test["Genre"] = df_test["Genre"].astype("category")
df_test["Publication_Day"] = df_test["Publication_Day"].astype("category")
df_test["Publication_Time"] = df_test["Publication_Time"].astype("category")
df_test["Episode_Sentiment"] = df_test["Episode_Sentiment"].astype("category")

X_test = df_test.drop(columns=["id","Podcast_Name","Episode_Title"])

final_preds = final_model.predict(X_test)



submission = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")
submission['Listening_Time_minutes'] = final_preds
submission.to_csv("submission.csv", index=False)
print("ðŸš€ Submission saved.")




