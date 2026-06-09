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


# IMPORTAMOS LIBRERÍAS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, mean_squared_error




# CARGAMOS LOS DATOS

train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


# VISUALIZAMOS 
print("El conjunto de train tiene",train.shape[0],"filas y", train.shape[1],"columnas")
print("El conjunto de test tiene",test.shape[0],"filas y", test.shape[1],"columnas")
print("---- VISUALIZACIÓN DE LA PARTE DE TRAIN ----")
train.head()


train.info()


# FUNCIONES EDA

def cualitativa(df,columna) : 
    print(4*"-","Estamos analizando la columna",columna,4*"-")
    print(f"Valores nulos: {df[columna].isna().sum()} ") 
    print(f"Valores únicos: {df[columna].nunique()} ")
    if df[columna].nunique() > 10:
        print("CUIDADO: ALTA CARDINALIDAD")
    else:
        pass
    print(f"Categorías más frecuentes: {df[columna].value_counts().head(10)}")
    #REPRESENTACIÓN
    plt.figure(figsize = (8,4))
    sns.countplot(df, x = columna, order = df[columna].value_counts().index[:10])
    plt.title(f"Distribución de {columna}")
    plt.show()

def cuantitativa(df, columna):
    # ANÁLISIS
    print(5*"-", columna, 5*"-")
    print(f"Tiene {df[columna].isna().sum()} nulos")
    # REPRESENTACIÓN
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(df[columna], kde=True, ax=axes[0], color="skyblue")
    axes[0].set_title(f"Histograma + KDE de '{columna}'")
    axes[0].set_xlabel(columna)
    sns.boxplot(x=df[columna], ax=axes[1], color="lightgreen")
    axes[1].set_title(f"Boxplot de '{columna}'")
    axes[1].set_xlabel(columna)
    plt.tight_layout()
    plt.show()


# VAMOS A ANALIZAR LA VARIABLE "id" Y EN CASO DE QUE ESTÉ TODO EN ORDEN LA ELIMINAMOS
train["id"].value_counts() # NO HAY VALORES REPETIDOS


# ELIMINAMOS LA VARIABLE "id" YA QUE NO APORTA NADA AL MODELO
train = train.drop("id", axis = 1)


# LO MISMO CON LA PARTE DE TEST
test["id"].value_counts() # NO HAY VALORES REPETIDOS


# ELIMINO LA VARIABLE "id" DE LA PARTE DE TEST
test = test.drop("id", axis = 1)


# ANALIZAMOS LAS VARIABLES CUANTITATIVAS (TRAIN)
train.describe()

# NO PARECE HABER NINGÚN ERROR NI VALOR ATÍPICO EN ESTAS VARIABLES


# REPRESENTACIÓN

for i in train.select_dtypes(include = "number"):
    cuantitativa(train,i)


# VARIABLE CUALITATIVA ("Sex")
cualitativa(train,"Sex")


# REALIZAMOS LO MISMO PARA EL CONJUNTO DE TEST (NO REPRESENTAMOS YA QUE HAREMOS COMPARACIÓN DE DISTRIBUCIONES ENTRE TRAIN Y TES)
test.describe()


#¿TIENEN LAS VARIABLE DE TRAIN Y TEST DISTRIBUCIONES PARECIDAS?

for i in test.select_dtypes(include = "number"):
    warnings.filterwarnings('ignore')
    sns.kdeplot(data = train, x = i, label = "Train", fill = True)
    sns.kdeplot(data = test, x = i, label = "Test", fill = True)
    plt.title (f"Distribución de {i} en train y test")
    
    plt.show()
    


# ANÁLISIS BIVARIANTE
for i in train.select_dtypes(include = "number"):
    if i != "Calories":
      plt.scatter(train[i],train["Calories"])
      plt.title(f"{i} frente a Calories")
      plt.ylabel("Calories")
      plt.xlabel(f"{i}")
      plt.show()


mean_calories = train.groupby("Sex")["Calories"].mean()
mean_calories.plot(kind = "bar")
plt.title("Sex frente a Calories")
plt.ylabel("Calories")
plt.xlabel("Sex")
plt.show()


# CORRELACIONES CON RESPECTO A LA VARIABLE "target"

train.select_dtypes("number").corr()["Calories"]


# REPRESENTACIÓN
sns.heatmap(train.select_dtypes("number").corr(), cmap = "Greens")
plt.show()


#  Índice de Masa Corporal (IMC o BMI) 
df = train.copy()
plt.figure(figsize=(8, 5))
sns.scatterplot(x= df["Weight"] / (df["Height"] / 100) ** 2 , y=df["Calories"], alpha = 0.5)
plt.title("Relación entre IMC y Calorías Quemadas")
plt.xlabel("Índice de Masa Corporal (BMI)")
plt.ylabel("Calorías Quemadas")
plt.show()


# GRUPOS POR IMC
df = train.copy() # Copia para probar features antes de añadirlos a train y test
df["BMI"] = df["Weight"] / (df["Height"] / 100) ** 2
df["BMI_bin"] = pd.cut(df["BMI"], bins=np.arange(12, 42, 2))
plt.figure(figsize=(12, 6))
sns.boxplot(x="BMI_bin", y="Calories", data=df)
plt.xticks(rotation=45)
plt.title("Calorías por rango de IMC")
plt.show()


resumen = df.groupby("BMI_bin")["Calories"].agg(["count", "mean"]).reset_index()
print(resumen)


df["BMI_bin"] = df["BMI"].apply(lambda x: 0 if x <= 25 else 1)
df.groupby("BMI_bin")["Calories"].mean()


sns.boxplot(x="BMI_bin", y="Calories", data=df)
plt.xticks(rotation=45)
plt.title("Calorías por rango de IMC")
plt.show()


df["BMI_bin"] = df["BMI_bin"].astype("int")


df.select_dtypes(include = "number").corr()["Calories"]


# Edades por tramos
df["Age_group"] = pd.cut(df["Age"], bins=np.arange(20, 80, 5))
plt.figure(figsize=(12, 6))
sns.boxplot(x="Age_group", y="Calories", data=df)
plt.xticks(rotation=45)
plt.title("Calorias medias por Age")
plt.show()


resumen = df.groupby("Age_group")["Calories"].agg(["count", "mean"]).reset_index()
print(resumen)


bins = [0, 30, 60, 150]  # límites para jóvenes, adultos y mayores
labels = ['Joven (>=30)', 'Adulto (31-60)', 'Mayor (>60)']
df['Age_group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True)

plt.figure(figsize=(10, 4))
sns.boxplot(x="Age_group", y="Calories", data=df)
plt.xticks(rotation=45)
plt.title("Calorias medias por Age")
plt.show()


df = pd.get_dummies(df, columns = ["Age_group"])


df["Age_group_Mayor (>60)"] = df["Age_group_Mayor (>60)"].astype("int")
df["Age_group_Adulto (31-60)"] = df["Age_group_Adulto (31-60)"].astype("int")
df["Age_group_Joven (>=30)"] = df["Age_group_Joven (>=30)"].astype("int")


df.select_dtypes("number").corr()["Calories"]


# Intensidad del pulso 
df["HeartRate_per_Age"] = df["Heart_Rate"] / df["Age"]

plt.scatter(x = df["HeartRate_per_Age"], y = df["Calories"])
plt.show()


df.select_dtypes("number").corr()["HeartRate_per_Age"]


# Esfuerzo cardiovascular 
df["Workload_Index"] = df["Heart_Rate"] * df["Duration"]
plt.scatter(x = df["Workload_Index"], y = df["Calories"])
plt.show()


df.select_dtypes("number").corr()["Workload_Index"]


# 1. AÑADIMOS UNA DUMMIE CON EL IMC EN DOS GRUPOS IMC > 25 LO QUE INDICARÍA SOBREPESO - OBESIDAD Y IMC < 25 QUE INDICARÍA BAJO PESO - NORMAL
train["BMI"] = train["Weight"] / (train["Height"] / 100) ** 2
train["BMI_bin"] = train["BMI"].apply(lambda x: 0 if x <= 25 else 1)
train = train.drop("BMI", axis = 1)

test["BMI"] = test["Weight"] / (test["Height"] / 100) ** 2
test["BMI_bin"] = test["BMI"].apply(lambda x: 0 if x <= 25 else 1)
test = test.drop("BMI", axis = 1)


train.head()


# 2. AÑADIMOS VARIABLE "HeartRate_per_Age"
train["HeartRate_per_Age"] = train["Heart_Rate"] / train["Age"]
test["HeartRate_per_Age"] = test["Heart_Rate"] / test["Age"]


# LA ÚNICA VARIABLE QUE HAY QUE TRATAR ES "Sex" QUE LA CONVERTIMOS EN NUMÉRICA PARA UN MEJOR TRATAMIENTO POR PARTE DE LOS MODELOS
train["Sex"] = train["Sex"].apply(lambda x: 0 if x == "male" else 1)
test["Sex"] = test["Sex"].apply(lambda x: 0 if x == "male" else 1)


train.head()


test.head()


test.to_csv("test.csv", index = False)
train.to_csv("train.csv", index = False)

