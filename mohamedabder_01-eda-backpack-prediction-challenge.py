from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
# ^^^ pyforest auto-imports - don't write above this line
import pandas as pd
import numpy as np
from termcolor import colored


import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline


from sklearn.impute import SimpleImputer

from scipy.stats import shapiro, kendalltau, kruskal, mannwhitneyu


import os
import warnings
warnings.filterwarnings("ignore")


path = 'D:\\etude_data_science\\Kaggle_competition\\13_Backpack_Prediction_Challenge\\dataset\\'


os.listdir(path)


train = pd.read_csv(path+"training_extra.csv")
train2 = pd.read_csv(path+"train.csv")
test = pd.read_csv(path+"test.csv")



print(train.shape)
train.head()


print(train2.shape)
train2.head()


df = train.copy()


df.head()


df["Compartments"] = df["Compartments"].astype(int)


df.isnull().mean()*100


#Récupération des colonnes avec NaN
col_NaN = df.isnull().sum()[df.isnull().sum()>0].index.tolist()


df[col_NaN].head(3)


df[col_NaN].dtypes


var_num_NaN = df[col_NaN].select_dtypes(exclude="object").columns.tolist()
var_qual_NaN =  [i for i in col_NaN if i not in var_num_NaN]


imputer_mean = SimpleImputer(strategy="mean")
imputer_mode = SimpleImputer(strategy="most_frequent")


imputer_mean.fit(df[var_num_NaN])
imputer_mode.fit(df[var_qual_NaN])


df[var_num_NaN] = imputer_mean.transform(df[var_num_NaN])
df[var_qual_NaN] = imputer_mode.transform(df[var_qual_NaN])


df.isnull().sum()


df.select_dtypes(exclude = "float").nunique()


var_num = df.select_dtypes(include=float).columns.tolist()
var_dis = df.drop("id", axis = 1).select_dtypes(include=int).columns.tolist()
var_qual = [i for i in df.drop("id", axis = 1) if i not in var_num+var_dis]


def varqual_viz(col,data, nrow,ncol,size=(14,8)):
    plt.figure(figsize=size)
    for i,k in enumerate(col):
        plt.subplot(nrow,ncol,i+1)
        ax = sns.countplot(x = k, data = data)
        lab = ax.get_xticklabels()
        ax.set_xticklabels(lab,rotation=45)
        plt.tight_layout()
    plt.show()



varqual_viz((var_qual+var_dis),data=df, nrow=2,ncol=4,size=(14,8))


def varnum_viz(col,data, nrow,ncol,size=(14,8)):
    plt.figure(figsize=size)
    for i,k in enumerate(col):
        plt.subplot(nrow,ncol,i*2+1)
        sns.histplot(x = k, data = data, color = "slateblue",kde=True, )

        plt.subplot(nrow,ncol,i*2+2)
        sns.boxplot(x = k, data = data, color="orange")
        plt.tight_layout()
    plt.show()


varnum_viz(var_num,data=df, nrow=2,ncol=2,size=(14,8))


def shapiro_test(list_col):
    """
    Cette fonction teste si list_col est une liste, si ce n'est pas le cas,
    elle convertie le texte en une liste avant de faire un test shapiro pour vérifier
    si la feature suit une loi normale ou non
    """
    if type(list_col) != list:
        list_col = [list_col]

    accepted,rejected = [], []

    for col in df[list_col]:
        stat, p_value = shapiro(df[col])  
        alpha = 0.05
        if p_value > alpha: 
            result = colored('Accepter', 'green')  
            accepted.append(col)
        else:
            result = colored('Rejet','red')        
            rejected.append(col)

        print(f'{col:-<50}\t Hypothèse: {result}')
    return accepted,rejected


print(colored("Test shapiro :", "blue"))
a,r = shapiro_test(var_num)


correlation = df[var_num].corrwith(df['Price']) #corrwith() permet de calculer la corrélation entre
#la target et chaque variable continue

# Trier les valeurs de corrélation par ordre décroissant
sorted_correlation = correlation.sort_values(ascending=False)
# plt.figure(figsize=(10, 20))
heatmap = sns.heatmap(pd.DataFrame(sorted_correlation), annot=True, cmap='RdYlGn',fmt=".2%", cbar=True)
heatmap.set_title('Corrélation entre la target et les variables continues')
plt.show()


plt.figure(figsize=(16,14))
for i,col in enumerate(var_qual+var_dis,1):
    plt.subplot(4,3,i)
    sns.boxplot(x=col, y = "Price", data = df)
plt.tight_layout()


df_temp = df.groupby(["Brand","Material"])["Price"].mean().reset_index()
print("dim :", df_temp.shape)
df_temp.head()


plt.figure(figsize=(14,4))
plt.subplot(1,2,1)
sns.barplot(x = "Brand", y = "Price", hue = "Material", data=df_temp, dodge=False, )
plt.subplot(1,2,2)
sns.barplot(x = "Brand", y = "Price", hue = "Material", data=df_temp, dodge=True, )
plt.show()


df_temp = df.groupby(["Size","Compartments","Waterproof"])["Price"].mean().reset_index()
sns.barplot(x = "Size", y = "Price", hue = "Waterproof", data=df_temp, dodge=True, )
plt.show()


df_temp = df.sample(10000, random_state=42) #récupération d'une partie des données


def violin_stripplot(col_x, col_hue):
    plt.figure(figsize=(16,6))
    plt.subplot(1,2,1)
    ax = sns.violinplot( x = col_x, y = "Price", hue = col_hue, data = df_temp)
    ax.set_xlabel(col_x, color="blue",fontweight="bold")
    ax.set_ylabel("Price", color="red",fontweight="bold")

    legend = plt.legend(bbox_to_anchor=(0, 1))  
    legend.set_title(col_hue)  # pour remettre le titre de la légende
    
    plt.subplot(1,2,2)
    ax = sns.stripplot( x = col_x, y = "Price", hue = col_hue, data = df_temp)
    ax.set_xlabel(col_x, color="blue",fontweight="bold")
    ax.set_ylabel("Price", color="red",fontweight="bold")

    legend = plt.legend(bbox_to_anchor=(1, 1))  
    legend.set_title(col_hue)
    plt.show()


violin_stripplot("Size","Laptop Compartment" )


violin_stripplot("Color","Style" )





def kendall(features, target,dataframe,retour=False):
    """
    Fonction qui effectue un test de Man Withney entre une variable qualitative et une variable continue
    features : liste de feature ou feature unique
    target : nom de la target
    dataframe : jeu de donnée contenant les features et la target
    renvoi un tuple de liste, le premier élément du tuple sont les variables à conserver, le second les variables à
    supprimer
    """
    var_a_conserver, var_a_supprimer = [],[]
    #si l'utilisateur n'entre qu'une seule feature, on la met en liste :
    if type(features) == str:
        features = [features]
    print(colored("P-values :","blue")) 
    print()

    
    #test correlation : 
    for feat in features:
        stat, pval = kendalltau(dataframe[feat],dataframe[target]) #test kendall
        alpha = 0.05
        if pval<alpha:
            print(f'{colored(feat,"green")} : {pval}')
            var_a_conserver.append(feat)
        else:
            print(f'{colored(feat,"red")} : {pval}')

            var_a_supprimer.append(feat)
    print()        
    print(colored("Variable a conserver ","green", attrs=["bold"]), var_a_conserver)
    print(colored("Variable a supprimer ","red", attrs=["bold"]), var_a_supprimer)
    if retour :
        return var_a_conserver, var_a_supprimer
        


kendall(var_num, "Price",df)


col_manwhitney, col_kruskal =  [], []

for i in var_qual+var_dis:
    if df[i].nunique() == 2:
        col_manwhitney.append(i)
    else:
        col_kruskal.append(i)


def manwithney(features, target,dataframe,retour=False):
    """
    Fonction qui effectue un test de Man Withney entre une variable qualitative et une variable continue
    features : liste de feature ou feature unique
    target : nom de la target
    dataframe : jeu de donnée contenant les features et la target
    renvoi un tuple de liste, le premier élément du tuple sont les variables à conserver, le second les variables à
    supprimer
    """
    var_a_conserver, var_a_supprimer = [],[]
    #si l'utilisateur n'entre qu'une seule feature, on la met en liste :
    if type(features) == str:
        features = [features]
    print(colored("P-values :","blue")) 
    print()

    #test correlation : 
    for feat in features:
            # Séparation en deux groupes basés sur la variable qualitative
        group1 = dataframe[dataframe[feat] == dataframe[feat].unique()[0]][target]
        group2 = dataframe[dataframe[feat] == dataframe[feat].unique()[1]][target]
            
        # test de Mann-Whitney
        stat, pval = mannwhitneyu(group1, group2)
        alpha = 0.05
        if pval<alpha:
            print(f'{colored(feat,"green")} : {pval}')
            var_a_conserver.append(feat)
        else:
            print(f'{colored(feat,"red")} : {pval}')

            var_a_supprimer.append(feat)
    print()        
    print(colored("Variable a conserver ","green", attrs=["bold"]), var_a_conserver)
    print(colored("Variable a supprimer ","red", attrs=["bold"]), var_a_supprimer)
    if retour:
        return var_a_conserver, var_a_supprimer

    
    
def kruskal_wallis(feature, target, dataframe,retour=False):
    var_a_conserver = []
    var_a_supprimer = []

    # Test si la feature est une liste ou simplement une feature
    if type(feature) == str:
        feature = [feature]
    print(colored("P-values :","blue")) 

    for feat in feature:
        # Séparer les données en fonction des différentes valeurs prises par la variable qualitatives
        grouped_data = [dataframe.loc[dataframe[target] == category, feat] for category in dataframe[target].unique()]

        # Effectuer le test de Kruskal-Wallis
        statistic, pval = kruskal(*grouped_data)

        # Interpréter les résultats
        alpha = 0.05  # Niveau de signification
        if pval<alpha:
            print(f'{colored(feat,"green")} : {pval}')
            var_a_conserver.append(feat)
        else:
            print(f'{colored(feat,"red")} : {pval}')

            var_a_supprimer.append(feat)
    print()        
    print(colored("Variable a conserver ","green", attrs=["bold"]), var_a_conserver)
    print(colored("Variable a supprimer ","red", attrs=["bold"]), var_a_supprimer)
    if retour :
        return var_a_conserver, var_a_supprimer


manwithney(col_manwhitney, "Price",df)


%time
kruskal_wallis(col_kruskal,  "Price",df)


test.isnull().sum()[test.isnull().sum()>0] #données manquantes pour les memes colonnes que le train


#  Utilisation des imputers qui ont été entrainés 
test[var_num_NaN] = imputer_mean.transform(test[var_num_NaN])
test[var_qual_NaN] = imputer_mode.transform(test[var_qual_NaN])


# il faut retirer la target de la standardisation :
target = df["Price"]
var_num.remove(target.name)


scaler = StandardScaler()
scaler.fit(df[var_num]) #entrainement
df[var_num] = scaler.transform(df[var_num]) #transformation
test[var_num] = scaler.transform(test[var_num]) #transformation


# Initialisation de l'encodeur OneHotEncoder
# - sparse=False : Retourne une matrice dense, pratique pour la manipulation directe.
# - drop='first' : Supprime la première catégorie pour chaque variable afin d'éviter la colinéarité.

encoder = OneHotEncoder(sparse=False, drop='first',handle_unknown='ignore')
# handle_unknown='ignore' permet de gérer les catégories inconnues dans les données de test


# Entraînement et transformation des variables qualitatives
encoded_columns = encoder.fit_transform(df[var_qual])

# Conversion des données encodées en DataFrame avec des noms explicites pour les colonnes
# - get_feature_names_out() : Génère des noms clairs comme 'Gender_Male' ou 'Color_Blue'.

var_qual_encoded = encoder.get_feature_names_out()

encoded_df = pd.DataFrame(encoded_columns,
                          columns=var_qual_encoded
                          )
encoded_df.head()


encoded_df.shape


#Remplacement des anciennes variables qualitatives par celles encodées :
df = df.drop(var_qual, axis=1) #suppression des anciennes variables 


df.head(3)


#Fusion des deux dataframes :
df = pd.concat([df, encoded_df], axis=1)
df.head(2)


#Suppression variable qualitatives et remplacement par le variables encodées :
test = test.drop(var_qual, axis = 1) 
test = pd.concat([test, df_test_encoded], axis = 1)
test.head()


test.to_csv("test_preprocessed.csv")


df.to_csv("df_preprocessed.csv")

