import warnings
warnings.filterwarnings("ignore")


!pip install skimpy
# !pip install imbalanced-learn


#Package de base
from termcolor import colored
import pandas as pd
import numpy as np
import skimpy 
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
#package visualisation

from scipy.stats import shapiro, kendalltau, kruskal, mannwhitneyu

#package preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split


## Rééchantillonge SMOTE sur la base d'apprentissage
from imblearn.over_sampling import SMOTE


path = "/kaggle/input/playground-series-s5e3/train.csv"
# pd.read_csv(path)
train = pd.read_csv(path)
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
# sample = pd.read_csv(path+"sample_submission.csv")


df = train.copy()
df.head()


skimpy.skim(df)


df.nunique()


print("Total day / 365 day =", int(df.shape[0]/max(df.day)),"years in this dataframe")


#1
df[df.day == 365]


df[df.id.max() == df.id]


#2
df.day.value_counts()


#we isolate the days that are repeated 7 times
anormal_days = df.day.value_counts()[df.day.value_counts() == 7].index


print("5 first anormal days : ", sorted(anormal_days)[:5])
#for example, take day = 3 :
df[df.day == 3]


df["day_cycle"] = (df.id % 365) +1


df[df.day==3]


print("Start day : 732")
print("Final day is start day+ a year : 732 + 365 =", 732+365)


df["day"] = df.day_cycle
df.drop("day_cycle", inplace=True, axis = 1) #remove


#3
def newyear_column(column):
    """function that converts 365 days into a year (provided the data is correctly ordered) """
    years = [] #will contain all row with the number of year
    cpt = 1  # Current year

    for i in df.day:
        years.append(cpt)  # we add current year to list
        if i == 365:  # if we reach the last day of the year
            cpt += 1  # we move into the new year
    df[column] = years  # Ajouter la liste comme colonne au DataFrame
newyear_column("year")


df.year.value_counts()


year_col = df.pop("year")
df.insert(2,"year", year_col)


df.head()


var_cont = [i for i in df.select_dtypes(float) if i !="id"] #continuous variables
var_dis = [i for i in df if i not in var_cont and i != "id"]


print(len(var_cont+var_dis)+1 ==  df.shape[1]) #+1 because we deleted "id"
#That displays ‘True’, so we haven't forgotten any variables


def var_cont_vizualisation(col,data, nrow,size=(14,20)):
    """
    col : list
    data : dataframe
    nrow : int
    For each feature, this returns two figures: histplot/boxplot with skew
    """
    
    plt.figure(figsize=size)
    for i,k in enumerate(col):
        #histplot :
        plt.subplot(nrow,2,i*2+1)
        sns.histplot(x = k, data = data, color = "slateblue",kde=True, )
        plt.title(f"Skew : {round(df[k].skew(),2)}")
        plt.axvline(df[k].mean(), ls="--", c = "red", label = "mean")
        plt.axvline(df[k].median(),ls=":", c = "green", label = "median")
        plt.legend()
        
        #boxplot
        plt.subplot(nrow,2,i*2+2)
        sns.boxplot(x = k, data = data, color="orange", showmeans=True)
        plt.tight_layout()
    plt.show()


var_cont_vizualisation(var_cont,df, 10,size=(14,20))


def shapiro_test(list_col):
    """
    This function test  if list_col is a list, if not,
    it converted the text in a list before a shapiro test to check 
    if the feature follow a normal distribution
    """
    if type(list_col) != list:
        list_col = [list_col]

    accepted,rejected = [], []

    for col in df[list_col]:
        stat, p_value = shapiro(df[col])  
        alpha = 0.05
        if p_value > alpha: 
            result = colored('Accepted', 'green')  
            accepted.append(col)
        else:
            result = colored('Rejected','red')        
            rejected.append(col)

        print(f'{col:-<50}\t Hypothesis: {result}')
    return accepted,rejected

print(colored("Test shapiro :", "blue"))
a,r = shapiro_test(var_cont)


def varqual_viz(col,data, nrow,ncol,size=(14,8)):
    plt.figure(figsize=size)
    for i,k in enumerate(col):
        plt.subplot(nrow,ncol,i+1)
        ax = sns.countplot(x = k, data = data)
        lab = ax.get_xticklabels()
        ax.set_xticklabels(lab,rotation=45)
        plt.tight_layout()
    plt.show()



varqual_viz((var_dis),data=df, nrow=1,ncol=3,size=(14,8))


df.rainfall.value_counts().plot(kind="pie")
plt.show()


pd.crosstab(df["year"],df["rainfall"],margins=True)


cross = pd.crosstab(df["year"],df["rainfall"])
cross = cross.reset_index()


sns.heatmap(cross, fmt = "d", cmap="Reds", annot=True)
plt.show()


cross = cross.melt(id_vars="year", var_name="rainfall", value_name="count")
sns.barplot(x = "year", y = "count", hue = "rainfall", data = cross)
plt.show()


def boxplot_2feat(list_col,nrow,ncol, size=(16,14)):
    plt.figure(figsize=size)
    for i,col in enumerate(list_col,1):
        plt.subplot(nrow,ncol,i)
        sns.boxplot(x= "rainfall", y=col, data = df)
    plt.tight_layout()    
    
boxplot_2feat(var_cont,4,3)


def manwithney(features, target, dataframe, retour=False):
    var_a_conserver, var_a_supprimer = [], []
    if isinstance(features, str):
        features = [features]

    print(colored("P-values :", "blue")) 
    print()

    # Test de Mann-Whitney sur chaque variable continue
    for feat in features:
        # Séparation en deux groupes selon `target`
        group1 = dataframe[dataframe[target] == dataframe[target].unique()[0]][feat]
        group2 = dataframe[dataframe[target] == dataframe[target].unique()[1]][feat]

        # Mann-Whitney test
        stat, pval = mannwhitneyu(group1, group2)
        alpha = 0.01  # threshold

        if pval < alpha:
            print(f'{colored(feat, "green")} : {pval}')
            var_a_conserver.append(feat)
        else:
            print(f'{colored(feat, "red")} : {pval}')
            var_a_supprimer.append(feat)

    print()
    print(colored("Feature to keep", "green", attrs=["bold"]), var_a_conserver)
    print(colored("Feature to delete", "red", attrs=["bold"]), var_a_supprimer)

    if retour:
        return var_a_conserver, var_a_supprimer



keepfeature, deletefeature = manwithney(var_cont, "rainfall",dataframe=df ,retour=True)


#1. Deleting column by exclusion : 
df = df[df.columns[~df.columns.isin(deletefeature)]] #we keep only columns which are not include in "deletefeature"
df.head()


#Remove in var_cont too
var_cont = [i for i in var_cont if i not in deletefeature]


df = df.drop(["day","year"], axis = 1)


#Split the data : 
X,y = df.drop("rainfall", axis = 1), df["rainfall"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train.head(3)


#Scaled the data
scaler = StandardScaler()
scaler.fit(X_train[var_cont]) #training on train set 

#Scaled to the dataset :
X[var_cont] = pd.DataFrame(scaler.transform(X[var_cont])) #transformation


X.head(3)


# We split the data in the same order as for preprocessing, with the same lines used for the train and the test.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.head(3)


# Initialization of SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)

# Apply smote on train set :
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Creation of a new dataframe with SMOTE data :
df_smote = pd.concat([X_train_smote, y_train_smote], axis=1)


plt.figure(figsize=(8,6))
plt.subplot(1,2,1)
y_train.value_counts().plot(kind="pie", autopct='%1.1f%%')
plt.xlabel(f"N = {len(y_train)}")
plt.title("Class distribution before SMOTE")
plt.subplot(1,2,2)
y_train_smote.value_counts().plot(kind="pie", autopct='%1.1f%%')
plt.xlabel(f"N = {len(y_train_smote)}")

plt.title("Class distribution after SMOTE")
plt.show()


print("before SMOTE (y_train):\n")
print(y_train.value_counts()) 


test[var_cont] = scaler.transform(test[var_cont]) #transformation
preprocessed_test = test[X_train_smote.columns]


preprocessed_test.head()


#Train SMOTE : 
df_smote.to_csv("train_SMOTE.csv")
#X_test,y_test : 
validation_set = pd.concat([X_test,y_test], axis = 1).to_csv("validation.csv")
#preprocessed test : 
preprocessed_test.to_csv("preprocessed_test.csv")

