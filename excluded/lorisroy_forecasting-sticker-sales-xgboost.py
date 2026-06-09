import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# For Model Training 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor , GradientBoostingRegressor,AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


#Load data
data = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
data_test=pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')

data.info()
target_str="num_sold"
target_array = ["num_sold"]


def grab_col_names(dataframe, cat_seuil=10, car_seuil=20):
    
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "object"] 

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_seuil and
                   dataframe[col].dtypes != "object"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_seuil and
                   dataframe[col].dtypes == "object"]

    cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car] 

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"] 

    num_cols = [col for col in num_cols if col not in num_but_cat] 
    
    print(f"Observations: {dataframe.shape[0]}") 
    print(f"Variables: {dataframe.shape[1]}") 
    print(f'cat_cols: {len(cat_cols)} : {cat_cols}') 
    print(f'num_cols: {len(num_cols)} : {num_cols}') 
    print(f'cat_but_car: {len(cat_but_car)} : {cat_but_car}') 
    print(f'num_but_cat: {len(num_but_cat)} : {num_but_cat}')

    return cat_cols, num_cols, cat_but_car, num_but_cat

cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(data)


#Permet de vÃ©rifier si les classes sont Ã©quilibrÃ©es
#ConnaÃ®tre la distribution des cat_cols lorsque y est manquant est important pour la qualitÃ© des prÃ©dictions
plt.figure(figsize=(16,12))

for i, col  in enumerate(cat_cols,1):
    plt.subplot(4,3,i)
    sns.countplot(x=data[col], data=data)
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


print(f"Valeurs manquantes dans le df d'entraÃ®nement : \n{data.isnull().sum()}")
print("\n\n")
print(f"Valeurs manquantes dans le df de test : \n{data_test.isnull().sum()}")


#Nous avons trop de diffÃ©rence parmi les classes donc prÃ©fÃ©rence de retirer ces lignes plutÃ´t que d'imputer une valeur
data = data.dropna(subset=["num_sold"])
data.isnull().sum()


for col in cat_cols:
    plt.figure(figsize=((8,2)))
    #print(f"\nMoyenne de target par la colonne {col} : {data.groupby(col)['num_sold'].mean()}")
    sns.barplot(x=data[col], y=data['num_sold'],data=data,estimator="mean")
    plt.title(f"Moyenne de target par cat de {col}")
    plt.show()


def format_date(df_train):
    # Ensure the 'date' column is a datetime object
    if 'date' in df_train.columns:
        df_train['date'] = pd.to_datetime(df_train['date'], errors='coerce')  # Convert to datetime
        if df_train['date'].isna().any():
            raise ValueError("The 'date' column contains invalid datetime values.")
    else:
        raise KeyError("The DataFrame does not have a 'date' column.")
    
    # Extract date-related components
    df_train['year'] = df_train['date'].dt.year
    df_train['month'] = df_train['date'].dt.month
    #df_train['day'] = df_train['date'].dt.day
    #df_train['dayOfYear'] = df_train['date'].dt.dayofyear
    df_train['weekday'] = df_train['date'].dt.weekday
    
    return df_train

data = format_date(data)
print(data.info())


data_test = format_date(data_test)
print(data_test.info())


date_cols=["year","month","weekday"]

for col in date_cols:
    plt.figure(figsize=((8,2)))
    sns.barplot(data=data,x=data[col],y=data[target_str],estimator="mean")
    plt.show()


sns.histplot(data=data,x=data[target_str],bins=15)
plt.show()


#La variable target est asymÃ©trique : on a intÃ©rÃªt a appliquer un log pour la rendre linÃ©aire et amÃ©liorer les perfs du modÃ¨le : effectuÃ© dans l'Ã©tape suivante pour y_train
print("Avant transformation : \n")
plt.figure(figsize=(5,2))
sns.histplot(data=data,x=data[target_str],bins=15)
plt.show()
print("\n\n AprÃ¨s transformation : \n")
plt.figure(figsize=(5,2))
sns.histplot(data=data,x=np.log1p(data[target_str]),bins=15)
plt.show()


useless_features = ["id","date"]

X = data.drop(columns=useless_features)
X = data.drop(columns=[target_str])
y = data[target_str]

data_test.drop(columns=useless_features)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#Je veux entraÃ®ner le modÃ¨le avec des donnÃ©es de target en log pour que ce soit davantage linÃ©aire
y_train = np.log1p(y_train)


models = [
    ('LinearRegression', LinearRegression()),
    ('DecisionTreeRegressor', DecisionTreeRegressor(random_state=42)),
    ('RandomForestRegressor', RandomForestRegressor(random_state=42)),
    ('KNeighborsRegressor', KNeighborsRegressor()),
    ('GradientBoostingRegressor', GradientBoostingRegressor(random_state=42)),
    ('XGBRegressor', XGBRegressor(random_state=42)),
    ('AdaBoostRegressor', AdaBoostRegressor(random_state=42))
]



model_scores = []


categorical_transformer = Pipeline(steps=[
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

categorical_transformer = Pipeline(steps=[("encoder",OneHotEncoder(handle_unknown="ignore"))])
preprocessor = ColumnTransformer(transformers=[("cat", categorical_transformer, cat_cols+date_cols)])


for name, model in models:
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])

    pipeline.fit(X_train,y_train)
    y_pred = pipeline.predict(X_test)
    #J'applique l'exponentielle afin de pouvoir faire des estimations Ã  la vraie Ã©chelle de donnÃ©e
    y_pred = np.expm1(y_pred)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    model_scores.append((name, mse, mae, r2, mape))
    
    print(f"{name} MSE: {mse:.2f}, MAE: {mae:.2f}, RÂ²: {r2:.2f}, MAPE: {mape:.2f}")
    print("-" * 50)


#Utilisation du meilleur modÃ¨le
best_model=XGBRegressor(random_state=42)
best_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", best_model)
    ])

best_pipeline.fit(X_train, y_train)

X_data_test=data_test.drop(columns=["id","date"])

y_test_pred = best_pipeline.predict(X_data_test)
y_test_pred= np.expm1(y_test_pred)
y_test_pred = np.ceil(y_test_pred)

submission = pd.DataFrame({'id' : data_test["id"], 'num_sold' : y_test_pred})


print(submission.head(20))
# Save submission file
submission.to_csv("submission.csv", index=False)

