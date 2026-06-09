# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder


path = "/kaggle/input/playground-series-s5e10/"


train_data = pd.read_csv(f"{path}train.csv", sep=",")
train_data


test_data = pd.read_csv(f"{path}test.csv", sep=",")
test_data


nb_lignes = train_data.shape[0]
nb_colonnes = train_data.shape[1]
print(f"La base d'entrainement est composée de {nb_lignes} lignes et de {nb_colonnes} colonnes")


print("Indices des colonnes et leurs noms (train dataset):\n")
dic = {}
for i,k in enumerate(list(train_data.columns)):
    dic[i]=k
    print(f"{i} : {k}")



print(f"Les colonnes et leurs types : \n\n{train_data.dtypes}")


print(f"Le nombre de valeurs manquantes au niveau de chaque colonne: \n\n{train_data.isna().sum()}")


nb_lignes2 = test_data.shape[0]
nb_colonnes2 = test_data.shape[1]
print(f"La base de test est composée de {nb_lignes2} lignes et de {nb_colonnes2} colonnes")


print("Indices des colonnes et leurs noms (test dataset):\n")
dic = {}
for i,k in enumerate(list(test_data.columns)):
    dic[i]=k
    print(f"{i} : {k}")



print(f"Les colonnes et leurs types : \n\n{test_data.dtypes}")


test_data.isna().sum()


train_data.describe()


colonnes_numeriques = train_data.select_dtypes(["number"]).columns
print(f"Les  colonnes numeriques : \n")
for k in colonnes_numeriques:
    print(f"- {k}")
    


dic


liste_indice_col_num = []
for ele in colonnes_numeriques:
    for cle, val in dic.items():
        if ele==val:
            liste_indice_col_num.append(cle)
print(f"Les indices de colonnes numériques : {liste_indice_col_num}")


num_data = train_data[[k for k in list(colonnes_numeriques)]]
num_data


num_cols = list(num_data.columns)
num_cols

plt.figure(figsize=(15,10))
for i, col in enumerate(num_cols,1):
    plt.subplot(2,3,i)
    sns.histplot(num_data[col], bins=30, color="purple")
    plt.title(f"Distribution de {col}")
plt.tight_layout()
plt.show()


num_features = ["curvature", "accident_risk"]

plt.figure(figsize=(10,5))
for i, col in enumerate(num_features,1):
    plt.subplot(1,2,i)
    sns.histplot(num_data, x=col, bins=50, color="#B65FCF")
    plt.title(f"Distribution de la variable '{col}'")
plt.tight_layout()
plt.show()



discrete_features = ['num_lanes', 'speed_limit', 'num_reported_accidents']

plt.figure(figsize=(15, 5))
for i, col in enumerate(discrete_features, 1):
    plt.subplot(1, 3, i)
    sns.countplot(data=num_data, x=col, palette="BuPu")
    plt.title(f"Répartition de {col}")
plt.tight_layout()
plt.show()



color = sns.color_palette("PuOr")
plt.figure(figsize=(15,10))
for i, col in enumerate(discrete_features, 1):
    plt.subplot(2,2,i)
    plt.pie(x=list(train_data[col].value_counts()), colors=color,labels=train_data[col].unique(),autopct='%1.1f%%')
    plt.title(f"Repartition de la variable {col}")
plt.show()


plt.figure()
sns.pairplot(num_data.sample(5000), diag_kind = "kde")
plt.suptitle("Relations entre les variables numeriques", y=1.02)
plt.show()


plt.figure()
sns.pairplot(num_data[num_features].sample(500), diag_kind = "kde")
plt.suptitle("Relations entre les variables numeriques", y=1.02)
plt.show()


plt.figure()
sns.heatmap(data= num_data.corr(), annot=True, cmap="YlGnBu" )
plt.suptitle("Heatmap")
plt.show()



colonnes_bool = train_data.select_dtypes(["bool"]).columns
print(f"Les  colonnes de booléens : \n")
for k in colonnes_bool:
    print(f"- {k}")
    


liste_indice_col_bool = []
for ele in colonnes_bool:
    for cle, val in dic.items():
        if ele==val:
            liste_indice_col_bool.append(cle)
print(f"La liste des indices de colonnes de booleens : {liste_indice_col_bool}")


color = sns.color_palette("crest")
plt.figure(figsize=(15,10))
for i, col in enumerate(colonnes_bool, 1):
    plt.subplot(2,2,i)
    plt.pie(x=list(train_data[col].value_counts()), colors=color,labels=[True, False],autopct='%1.1f%%')
    plt.title(f"Repartition de la variable '{col}'")
plt.show()


colonnes_categorielle = train_data.select_dtypes(["object"]).columns
print(f"Les  colonnes categorielles : \n")
for k in colonnes_categorielle:
    print(f"- {k}")
    



liste_indice_col_cat = []
for ele in colonnes_categorielle:
    for cle, val in dic.items():
        if ele==val:
            liste_indice_col_cat.append(cle)
print(f"Les indices de colonnes booleens : {liste_indice_col_cat}")


cat_data = train_data[[k for k in list(colonnes_categorielle)]]
cat_data


print("Valeurs uniques composant chaque colonne categorielles : \n")
for cat in colonnes_categorielle :
    print(f" - {cat} : {train_data[cat].unique()}")


plt.figure(figsize=(15,10))
color = sns.color_palette("BuPu")
for i, col in enumerate(colonnes_categorielle, 1):
    plt.subplot(2,2,i)
    label=train_data[col].unique()
    plt.pie(x=train_data[col].value_counts(), labels=label, colors=color, autopct='%1.1f%%')
    plt.title(f"Répartion de la variable {col}")
plt.show()


plt.figure(figsize=(10,6))
for i, col in enumerate(["num_lanes", "speed_limit", "num_reported_accidents"],1):
    plt.subplot(2,2,i)
    sns.boxplot(data=num_data, x=col, y="accident_risk")
plt.show()


train_data_encoded = train_data.copy()
train_data_encoded["curvature"] = pd.cut(
    train_data["curvature"],
    bins=3,
    labels=["faible", "moyenne", "forte"]
)



train_data_encoded["speed_limit"] = pd.cut(
    train_data["speed_limit"],
    bins=2,
    labels=["faible", "elevee"]
)



train_data_encoded


col_categorielle = list(colonnes_categorielle.copy())
col_categorielle.append("curvature")
col_categorielle.append("speed_limit")


col_categorielle



print(col_categorielle)


encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit(train_data_encoded[col_categorielle])
encoded = encoder.transform(train_data_encoded[col_categorielle])

train_encoded = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(col_categorielle))
train_encoded = pd.concat([train_data.drop(columns=col_categorielle).reset_index(drop=True), 
                          train_encoded.reset_index(drop=True)], axis=1)
train_encoded



train_encoded[colonnes_bool] = train_encoded[colonnes_bool].astype('int')
train_encoded



train_encoded["curvature_speed_interaction"] = train_data["curvature"] * train_data["speed_limit"]
train_encoded



train_encoded["accidents_per_lane"] = train_data["num_reported_accidents"] / (train_data["num_lanes"] + 1e-3)
train_encoded



train_encoded["curvature_per_lane"] = train_data["curvature"] / (train_data["num_lanes"] + 1e-3)
train_encoded


train_encoded["speed_per_lane"] = train_data["speed_limit"] / (train_data["num_lanes"] + 1e-3)
train_encoded


plt.figure(figsize=(10,6))
sns.heatmap(train_encoded.corr())
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
sns.heatmap(train_encoded.corr()[["accident_risk"]].sort_values(by="accident_risk", ascending=False), annot=True, center=0)
plt.title("Corrélation incluant les nouvelles features avec la variable cible")
plt.show()



def feature_engineering(data):

    data_encoded = data.copy()

    data_encoded["curvature"] = pd.cut(
        data["curvature"],
        bins=3,
        labels=["faible", "moyenne", "forte"]
    )

    data_encoded["speed_limit"] = pd.cut(
        data["speed_limit"],
        bins=2,
        labels=["faible", "elevee"]
    )

    col_categorielle = ['road_type', 'lighting', 'weather', 'time_of_day', 'curvature', 'speed_limit']

    encoded = encoder.transform(data_encoded[col_categorielle])
    data_final = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(col_categorielle))
    data_final = pd.concat([data.drop(columns=col_categorielle).reset_index(drop=True), 
                            data_final.reset_index(drop=True)], axis=1)

    
    colonnes_bool = ['road_signs_present', 'public_road', 'holiday', 'school_season']
    data_final[colonnes_bool] = data_final[colonnes_bool].astype('int')
    
    data_final["curvature_speed_interaction"] = data["curvature"] * data["speed_limit"]
    data_final["accidents_per_lane"] = data["num_reported_accidents"] / (data["num_lanes"] + 1e-3)
    data_final["curvature_per_lane"] = data["curvature"] / (data["num_lanes"] + 1e-3)
    data_final["speed_per_lane"] = data["speed_limit"] / (data["num_lanes"] + 1e-3)

    return data_final


test_data_encoded = feature_engineering(test_data)
test_data_encoded


s1 = pd.DataFrame(colonnes_categorielle)
s2 = pd.DataFrame(["curvature_bin", "speed_category"])
pd.concat([s1,s2]         ).reset_index



list(colonnes_categorielle)
s1 = pd.DataFrame(['road_type', 'lighting', 'weather', 'time_of_day'])
s2 = pd.DataFrame(["curvature_bin", "speed_category"])
pd.concat([s1,s2]         )


feature_engineering(train_data)


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression



X_train = train_encoded.drop("accident_risk",axis=1)
X_train 



y_train = train_encoded["accident_risk"]
y_train




X_test = test_data_encoded
X_test


models = {
    "RandomForest": RandomForestRegressor(),
    "CatBoost": CatBoostRegressor(verbose=0),
    "XGBRegressor": XGBRegressor(),
    "LGBMRegressor": LGBMRegressor(),
    "LinearRegression": LinearRegression(),
    "GradientBoosting": GradientBoostingRegressor()
}



df = pd.DataFrame()
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    df[name] = predictions


df


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

X, X_valid, y, y_valid = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)


best_rsme = float("inf")
best_model = None

for nom, modele in models.items():
    modele.fit(X,y)
    preds = modele.predict(X_valid)
    rmse = np.sqrt(mean_squared_error(y_valid, preds))
    print(f"RMSE pour le modele {nom} : {rmse:.5f}")

    if rmse < best_rsme:
        best_rmse = rmse
        best_model = modele
        nom_du_modele = nom


best_model.fit(X_train,y_train)



import joblib 

#sauvegarde du modele
joblib.dump(best_model, f'/kaggle/working/{nom_du_modele}.pkl')


#from IPython.display import FileLink
#FileLink(f"/kaggle/working/{nom_du_modele}.pkl'")


submission = pd.DataFrame({
    "id": X_test["id"],
    "accident_risk": df[f"{nom_du_modele}"]
})


submission


print(submission.head())
print(submission.shape)
print(X_test.shape)
print(submission.columns)
print(submission.isna().sum())


