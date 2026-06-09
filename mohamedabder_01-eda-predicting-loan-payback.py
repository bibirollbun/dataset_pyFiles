import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import missingno as msno

import os
import warnings
warnings.filterwarnings("ignore")


train,test = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv"), pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
df = train.copy()


print("Dimensions du dataframe :", df.shape)
print("Header :")
df.head()


df.drop("id",axis=1).dtypes.value_counts()


for i in df.drop("id",axis=1):
    print(f"{i:_<20} : {str(df[i].unique()[:5]):_<65} ({df[i].dtypes}; {df[i].nunique() })")


df.credit_score = df.credit_score.astype(float)
df.loan_paid_back = df.loan_paid_back.astype(int)


var_qual = df.select_dtypes("object").columns.tolist()
df[var_qual]


msno.matrix(df,figsize=(12,4))
plt.show()


var_cont = df.select_dtypes(float).columns.tolist()
var_qual = df.select_dtypes(object).columns.tolist()
var_dis = df.drop("id", axis = 1).select_dtypes(int).columns.tolist()
target = df[var_dis]


#Vérifions qu'aucune colonne n'a été oubliée :
#le -1 pour retirer la colonne id
df.shape[1]-1 == len(var_dis+var_cont+var_qual)
#pas d'oublie


def histplot_viz(data, columns, nrow, ncol, stats="percent", figsize=(12,6)):
    plt.figure(figsize=figsize)
    for idx, col in enumerate(columns,1):
        plt.subplot(nrow,ncol,idx)
        plt.grid()

        ax = sns.histplot(data=data, x=col, stat=stats,
                     shrink=.95, color="orange") #shrink = écart entre les barres
        labels = [t.get_text()  for t in ax.get_xticklabels()] #Récupération des modalités de chaque colonne
        ax.set_xticklabels(labels=labels, rotation=45) #On applique une rotation de 45° pour faciliter la lecture
    plt.tight_layout()

    plt.show()


histplot_viz(data=df, columns = var_qual, nrow=2,ncol=3, figsize=(16,6))


plt.figure(figsize=(12,20))
for i,k in enumerate(var_cont):
#histplot :
    plt.subplot(6,2,i*2+1)
    sns.histplot(x = k, data = df, color = "slateblue",kde=True, )
    plt.title(f"Skew : {round(df[k].skew(),2)}")
    plt.axvline(df[k].mean(), ls="--", c = "red", label = "mean")
    plt.axvline(df[k].median(),ls=":", c = "red", label = "median")
    plt.legend()
        
        #boxplot
    plt.subplot(6,2,i*2+2)
    sns.boxplot(x = k, data = df, color="orange", showmeans=True)
plt.tight_layout()
plt.show()



from scipy.stats import kurtosis
for i in var_cont:
    print(f"{i} : {round(kurtosis(df[i]),2)}")



from scipy.stats import shapiro

for i in var_cont:
    # Effectuer le test de Shapiro-Wilk
    stat, p_value = shapiro(df[i])

# Afficher les résultats
    print(f"Statistique du test de Shapiro-Wilk : {stat}")
    print(f"Valeur p : {p_value}")

# Interprétation des résultats
    alpha = 0.05
    if p_value > alpha:
        print(f"Les données {i} suivent une distribution normale (on ne rejette pas H0)")
    else:
        print(f"Les données {i} ne suivent pas une distribution normale (on rejette H0)")




import scipy
plt.figure(figsize=(15, 10))  # taille globale

for i,k in enumerate(var_cont,1):
#histplot :
    plt.subplot(2,3,i)
    scipy.stats.probplot(df[k], dist="norm", plot=plt)
    plt.title(f"QQ Plot – {k}")
plt.show()



from scipy.stats import normaltest

for i in var_cont:
    # Test Agostino Pearson 
    stat, p_value = normaltest(df[i])

# Afficher les résultats
    print(f"Statistique du test de Shapiro-Wilk : {stat}")
    print(f"Valeur p : {p_value}")

# Interprétation des résultats
    alpha = 0.05
    if p_value > alpha:
        print(f"Les données {i} suivent une distribution normale (on ne rejette pas H0)")
    else:
        print(f"Les données {i} ne suivent pas une distribution normale (on rejette H0)")


def bivariate_boxplot(data, target,feature, nrow,ncol,figsize=(12,8)):
    plt.figure(figsize=figsize)
    for i,col in enumerate(feature,1):
        plt.subplot(nrow,ncol,i)
        sns.boxplot(y=col, x = target, data = data, palette="tab10")
    plt.tight_layout()
    plt.show()



bivariate_boxplot(data=df, target="loan_paid_back",feature=var_cont, nrow=3,ncol=2,figsize=(12,8))


def bivariate_countplot(data, columns,target, nrow, ncol,figsize=(12,16)):
    plt.figure(figsize=figsize)
    for idx, col in enumerate(columns,1):
        plt.subplot(nrow,ncol,idx)
        ax = sns.countplot(data=data, x=target, hue=col)
        ax.set_ylabel(col)
        plt.legend(bbox_to_anchor=(1,1))
    plt.tight_layout()
    plt.show()


bivariate_countplot(data=df, columns=var_qual,target="loan_paid_back", nrow=3, ncol=2,  figsize=(12,16))

