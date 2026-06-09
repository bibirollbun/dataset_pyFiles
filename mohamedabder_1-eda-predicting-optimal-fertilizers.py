from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pickle
# ^^^ pyforest auto-imports - don't write above this line
from termcolor import colored
import pandas as pd
import numpy as np 

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import os


test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")



print(colored("Dimensions :","green",attrs=["bold"]))
print(train.shape,"\n")
print(colored("columns","green",attrs=["bold"]),colored("unique","light_yellow",attrs=["bold"]) , colored("dtypes","blue",attrs=["bold"]) , colored("%NaN","magenta",attrs=["bold"]))
[print(
        colored(f"{i:_^20} : ",'green') , 
        colored(f"{str(train[i].unique()[:3]):-<32}",'light_yellow'),
        colored(train[i].dtypes,"blue"),
        colored(f"{train[i].isnull().mean()*100} %",'magenta')
        ) 
for i in train]
train.head()


print(f'{colored("a","blue")}')


train.describe().T


train.nunique().sort_values(ascending=False)


target = train["Fertilizer Name"].name


var_cont = [col for col in train.select_dtypes(int) if col != "id"]
var_qual = [col for col in train.select_dtypes(exclude=int) if col not in ["id",var_cont]]


var_cont + var_qual


plt.figure(figsize=(16,20))

for idx,col in enumerate(var_cont):
    plt.subplot(6,2,idx*2+1)
    sns.histplot(data = train, x = col)
    plt.subplot(6,2,idx*2+2)
    sns.boxplot(data = train, x = col ,saturation = 1,
                color = "peachpuff",
                showmeans=True, 
            meanprops={"marker":"D","markeredgecolor":'black',
                      "markerfacecolor":'firebrick'} )

plt.tight_layout()
plt.show()


fig,axes = plt.subplots(1,2)
sns.countplot(y = var_qual[0], data=train, ax  = axes[0])
sns.countplot(y = var_qual[1], data=train, ax = axes[1])
plt.tight_layout()
plt.show()


sns.heatmap(data = train[var_cont].corr(), fmt = ".2f", annot = True, cbar= True, cmap = "Blues")
plt.show()


crosstab1 = pd.crosstab(train["Fertilizer Name"], train["Soil Type"], margins=False)
crosstab1
# ou écrire  :
# train.groupby(['Fertilizer Name', 'Soil Type']).size().unstack(fill_value=0)


crosstab2 = pd.crosstab(train["Crop Type"], train["Fertilizer Name"], margins=False)
crosstab2



plt.figure(figsize=(16,14))
for i, col in enumerate(var_qual,1):
    if col != target:
        plt.subplot(2,1,i)
        
        #pour colorer : 
        unique_val = train[target].unique()
        palette = sns.color_palette("tab10", n_colors=len(unique_val)) #tab10 = couleur de base de seaborn
        dic_palette = dict(zip(unique_val, palette))
        
        sns.countplot(x=col, hue = target, data = train, palette=dic_palette)
        plt.legend(bbox_to_anchor=(1,1))
        plt.title(f'{target} en fonction de {col.upper()}', fontdict={"color":"red"})
        plt.ylabel(f"{target}")
        plt.xlabel(f'{col}')
        plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


plt.figure(figsize=(22,14))

for i,col in enumerate(var_cont):
    plt.subplot(2,3,i+1)
    sns.histplot(data=train, x=col, hue=target, stat="count",binwidth=2.5)
# ax.legend(bbox_to_anchor=(1,1))
# plt.tight_layout()
plt.show()


stats_by_fertilizer = train.pivot_table(columns=target, values=var_cont, aggfunc=["mean","median","std"]).T
stats_by_fertilizer
#ou bien : 
# stats_by_fertilizer = train.groupby(target)[var_cont].agg(["mean","median","std"])
# stats_by_fertilizer


plt.figure(figsize=(22,14))
for i,col in enumerate(var_cont):
    plt.subplot(2,3,i+1)
    sns.boxplot(data=train, y=col, x=target)
# ax.legend(bbox_to_anchor=(1,1))
# plt.tight_layout()
plt.show()


import plotly.express as px
from sklearn.model_selection import train_test_split

#
#  l'échantillon (fraction)
sample_fraction = 0.01  # 1%

# utilisons le train_test_split pour l'échantillonnage stratifié avec 'stratify' pour s'assurer
#que les proportions de 'Fertilizer Name' sont maintenues.

_, train_sampled_stratified = train_test_split(
    train,
    test_size=sample_fraction, # la taille de l'échantillon que vous voulez
    stratify=train["Fertilizer Name"], # la colonne a stratifier
    random_state=42
)


px.scatter_3d(x = "Temparature", y = "Moisture", z = "Potassium", 
              color="Fertilizer Name", data_frame = train_sampled_stratified)


plt.figure(figsize=(26,12))
sns.boxplot(x="Temparature", y = "Moisture", hue = "Fertilizer Name", data = train_sampled_stratified)
plt.show()


#### Séparation des données : 
X,y = train.drop(target,axis=1), train[target]
X_train, X_test, y_train,y_test = train_test_split(X,y, stratify = y, test_size=.2)


std = StandardScaler()
std.fit(X_train[var_cont])
X_std = std.transform(X[var_cont])
X[var_cont] = X_std


for col in X_train.select_dtypes("object"):
    print(col)


var_qual_feat = [i for i in var_qual if i != target]


feature_encoder = {}

for column in var_qual_feat:
    feature_encoder[column]=LabelEncoder()
    feature_encoder[column].fit(X_train[column].astype(str))
    X[column]=feature_encoder[column].transform(X[column])
print(feature_encoder)
X[var_qual_feat].head()


target_encoder = LabelEncoder()
target_encoder.fit(y_train.astype(str))


y = pd.Series(target_encoder.transform(y),name=target)
y


train_preprocessed =  pd.concat([X,y],axis=1)


def save_file(name, file_save):
    file_path = f"{name}.pkl"
    with open(file_path,"wb") as f:
        pickle.dump(file_save, f)
    print(f'{colored(name,"red")} enregistré sous {colored(os.getcwd()+"/","green")}')



save_file("TargetEncoder", target_encoder)
save_file("feature_encoder", feature_encoder)
save_file("StandardScaler", std)



train_preprocessed.to_csv("train_preprocessed.csv")

