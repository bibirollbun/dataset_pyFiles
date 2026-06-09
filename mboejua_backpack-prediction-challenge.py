import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings('ignore')
import xgboost as xgb
import random

from lightgbm  import LGBMRegressor
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import  make_scorer,  mean_squared_error
from sklearn.preprocessing import StandardScaler



train = pd.read_csv("train.csv", index_col="id")
train_extra = pd.read_csv("training_extra.csv", index_col="id")
test = pd.read_csv("test.csv", index_col="id")
train_c  = pd.concat([train, train_extra], axis=0,ignore_index=True)

y_train_c = train_c["Price"].copy()
x_train_c = train_c.copy()
x_train_c = train_c.drop(["Price"], axis="columns")

train_y = train["Price"].copy()
train_x = train.copy()
train_x = train_x.drop(["Price"], axis="columns")
test_x = test.copy()

train_x.info()


#Categorical and numeric columns
cat_col = [x for x, y in zip(train_x.columns, train_x.dtypes) if y in ["object", "category"] ]
num_col = [x for x, y in zip(train_x.columns, train_x.dtypes) if y not in ["object", "category"] ]

train_x[num_col].head(3)


def object_cat (df):
    for column, type in zip(df.columns,df.dtypes):
        if type == "object":
            df[column] = df[column].astype("category")

object_cat(train_x)
object_cat(test_x)

display(train_x.info())


palletes = ["coolwarm","pastel","Set2"]
fig, axes = plt.subplots(len(train_x.columns), 2, figsize=(60, 20*len(train_x.columns)), 
                         constrained_layout=True)

for axs, col, type in zip(axes, train_x.columns, train_x.dtypes):
    if type == "category":
        sns.countplot(data=train_x, x=col, ax=axs[0], palette=palletes[0])
        axs[0].set_title(f"Countplot of {col}", fontsize=25)
        axs[0].set_xlabel(col, fontsize=25)
        axs[0].set_ylabel("count" ,fontsize=25)
        axs[0].tick_params(axis='both', labelsize=20)

        sns.boxplot(data=train, y=col,x="Price", ax=axs[1], palette=palletes[1], orient="h")
        axs[1].set_title(f"Boxplot of {col}", fontsize=25)
        axs[1].set_xlabel(col, fontsize=25)
        #axs[1].set_ylabel("count" ,fontsize=25)
        axs[1].tick_params(axis='both', labelsize=20)
    else:
        sns.kdeplot(data=train, x=col, ax=axs[0])
        axs[0].set_title(f"Countplot of {col}", fontsize=25)
        axs[0].set_xlabel(col, fontsize=25)
        axs[0].set_ylabel("Price" ,fontsize=25)
        axs[0].tick_params(axis='both', labelsize=20)

        sns.histplot(data=train, x=col, bins = 10,ax=axs[1], palette=palletes[2], kde=True)
        axs[1].set_title(f"Boxplot of {col}", fontsize=25)
        axs[1].set_xlabel(col, fontsize=25)
        #axs[1].set_ylabel("count" ,fontsize=25)
        axs[1].tick_params(axis='both', labelsize=20)
        
        


##coded cat columns making suitable for ML
def object_cat (df):
    for column, type in zip(df.columns,df.dtypes):
        if type == "category":
            df[column] = df[column].astype("category").cat.codes
        elif type == "object":
            df[column] = df[column].astype("category").cat.codes

object_cat(train_x)
object_cat(test_x)

display(train_x.info())


#fill missing values with median
train_x[num_col].fillna(train_x[num_col].median(),inplace=True)
test_x[num_col].fillna(test_x[num_col].median(),inplace=True)


#Scale train_x test_x
scale = StandardScaler()
train_x_scaled = pd.DataFrame(scale.fit_transform(train_x), columns= train_x.columns)
test_x_scaled = pd.DataFrame(scale.transform(test_x), columns= train_x.columns)


x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=42)


param_grid = list(zip(
    # n_estimators
    [500,700,850, 1000, 1200],
    # learning_rate
    [0.1,0.075, 0.05, 0.025, 0.01],
    # max_depth
    [ 5, 6, 7, 8, 9],
    # subsample
    [0.6, 0.65, 0.7, 0.75, 0.8,],
    # colsample_bytree 
    [0.55, 0.6, 0.65, 0.7, 0.75]
))

rmse_scorer = make_scorer(mean_squared_error, squared=False)

test_acc =[]
for n_est, lr, max_d, sub_s, col_s in param_grid:
    cat_col = [f"name:{col}" for col in cat_col] 
    ls = LGBMRegressor(n_estimators=n_est, random_state=0,n_jobs=-1,
                       learning_rate=lr,max_depth=max_d,force_col_wise=True,
                       #categorical_feature=cat_col,
                       subsample=sub_s,subsample_freq=1,
                       verbose=-1, #supress logss due to vast iter
                         metric="rmse")
    # Cross-validation
    cv_rmse = cross_val_score(ls, train_x, train_y, cv=5, scoring=rmse_scorer)
    ls.fit(train_x,train_y)
    y_pred = ls.predict(x_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    test_acc.append(rmse)
    print(f"n_estimators: {n_est} | Training Mean RMSE: {cv_rmse.mean():.4f} | Test RMSE: {rmse} ")


best_index = np.argmax(test_acc)
params = param_grid[best_index]
params



#Best Model based on test_rmse

n_est, lr, max_d, sub_s, col_s = params
cat_col = [f"name:{col}" for col in cat_col] 
ls = LGBMRegressor(n_estimators=n_est, random_state=0,n_jobs=-1,
                       learning_rate=lr,max_depth=max_d,force_col_wise=True,
                       subsample=sub_s,subsample_freq=1,
                       verbose=-1, 
                         metric="rmse")
# Cross-validation
cv_rmse = cross_val_score(ls, train_x, train_y, cv=5, scoring=rmse_scorer)
ls.fit(train_x,train_y)
object_cat(x_train_c)
y_pred = ls.predict(x_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
y_pred_e = ls.predict(x_train_c)
rmse_e = mean_squared_error(y_train_c, y_pred_e, squared=False)
print(f"Params \n|n_estimators: {n_est} \n|Training Mean RMSE: {cv_rmse.mean():.4f} \n|Train_extra RMSE: {rmse_e} \n|Test RMSE: {rmse}")




# Plot feature importances
importance_xgb = ls.feature_importances_
sorted_idx = np.argsort(importance_xgb)[::-1]
features = train_x.columns

plt.figure(figsize=(10, 6))
plt.barh([features[i] for i in sorted_idx], importance_xgb[sorted_idx], color="greenyellow")
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Light GBM Regression Feature Importance')
plt.gca().invert_yaxis()  
plt.show()


y_pred_test = ls.predict(test_x)
submission = pd.DataFrame({'id': test_x.index, 'Price': y_pred_test})
submission.to_csv('submission.csv', index=False)
display(submission.head())

