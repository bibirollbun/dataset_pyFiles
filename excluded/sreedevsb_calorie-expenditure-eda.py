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

# sns.kdeplot(df_train, x="Calories", hue="Sex", fill=True,ax=axes[2][1]) 
   


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

