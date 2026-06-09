import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


data_dir = "/kaggle/input/playground-series-s5e5/"
df_train = pd.read_csv(data_dir + "train.csv")
df_test = pd.read_csv(data_dir + "test.csv")


df_train


df_train.isna().sum()


df_train.describe()


features = ["Age","Height","Weight","Duration","Heart_Rate","Body_Temp","Sex"]
target = "Calories"


numerical_features = ["Age","Height","Weight","Duration","Heart_Rate","Body_Temp"]
categorical_features = ["Sex"]


row,col = 3,2
fig, axes = plt.subplots( row,col,  sharey = False, figsize = (20,18))

sns.histplot(df_train["Age"],kde=True,bins=30,ax=axes[0][0])    
sns.histplot(df_train["Weight"],kde=True,bins=40,ax=axes[0][1])
sns.histplot(df_train["Height"],kde=True,bins=30,ax=axes[1][0])
sns.histplot(df_train["Duration"],kde=True,bins=30,ax=axes[1][1])
sns.histplot(df_train["Heart_Rate"],kde=True,bins=30,ax=axes[2][0])
sns.histplot(df_train["Body_Temp"],kde=True,bins=30,ax=axes[2][1])


sns.countplot(data=df_train,x="Sex")
df_train["Sex"].value_counts()


from matplotlib.pyplot import figure

figure(figsize=(10, 8), dpi=80)
corr_matrix = df_train[[x for x in numerical_features if x not in ["id"]]].corr()
sns.heatmap(corr_matrix)


sns.pairplot(df_train[features])


row,col = 3,2
fig, axes = plt.subplots( row,col,  sharey = False, figsize = (20,18))


sns.kdeplot(df_train, x="Age", hue="Sex", fill=True,ax=axes[0][0]) 
sns.kdeplot(df_train, x="Height", hue="Sex", fill=True,ax=axes[0][1]) 
sns.kdeplot(df_train, x="Weight", hue="Sex", fill=True,ax=axes[1][0]) 
sns.kdeplot(df_train, x="Duration", hue="Sex", fill=True,ax=axes[1][1]) 
sns.kdeplot(df_train, x="Heart_Rate", hue="Sex", fill=True,ax=axes[2][0]) 
sns.kdeplot(df_train, x="Body_Temp", hue="Sex", fill=True,ax=axes[2][1]) 

   


sns.kdeplot(df_train, x="Calories", hue="Sex", fill=True)


row, col = 2, 3
fig, axes = plt.subplots(row, col, sharey=False, figsize=(20, 12))

sns.scatterplot(x=df_train["Age"], y=df_train["Calories"], ax=axes[0, 0])
sns.scatterplot(x=df_train["Height"], y=df_train["Calories"], ax=axes[0, 1])
sns.scatterplot(x=df_train["Weight"], y=df_train["Calories"], ax=axes[0, 2])
sns.scatterplot(x=df_train["Duration"], y=df_train["Calories"], ax=axes[1, 0])
sns.scatterplot(x=df_train["Heart_Rate"], y=df_train["Calories"], ax=axes[1, 1])
sns.scatterplot(x=df_train["Body_Temp"], y=df_train["Calories"], ax=axes[1, 2])

plt.tight_layout()  
plt.show()  


features = ["Age","Height","Weight","Duration","Heart_Rate","Body_Temp","Sex"]
target = "Calories"

numerical_features = ["Age","Height","Weight","Duration","Heart_Rate","Body_Temp"]
categorical_features = ["Sex"]


df_train["Sex"] = df_train["Sex"].map({"female":0, "male":1})
df_test["Sex"] = df_test["Sex"].map({"female":0, "male":1})


from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
import catboost as catboostmodel


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df_train[categorical_features+numerical_features] ,np.log1p(df_train[target]), test_size=0.2, random_state=2)

print(X_train.shape,X_test.shape)



cbm = catboostmodel.CatBoostRegressor(
    iterations=1000, 
    learning_rate=0.1, 
    depth=10, 
    loss_function='RMSE',
    l2_leaf_reg=5,
)

cbm.fit(X_train,y_train)


y_pred = cbm.predict(X_test)

val_rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))

print(f"RMSLE: {val_rmsle:.4f}")


predictions = cbm.predict(df_test[categorical_features+numerical_features])
df_test["Calories"] = np.expm1(predictions)
submission = df_test[["id","Calories"]]
submission.to_csv("submission.csv", index=False)


submission

