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


#Importation des bibliothèques de base
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal


stock_price = pd.read_csv('/kaggle/input/jpx-tokyo-stock-exchange-prediction/train_files/stock_prices.csv', index_col=0, parse_dates=True)


stock_price.columns


# Afficher les cinq premiéres lignes du DataFrame df
stock_price.head()


# Afficher les cinq dernières lignes du DataFrame df
stock_price.tail()


stock_price.isna().sum()


stock_price["SupervisionFlag"].describe()


stock_price["AdjustmentFactor"].value_counts()


stock_price.drop('ExpectedDividend', axis=1, inplace=True)
stock_price.drop('SupervisionFlag', axis=1, inplace=True)



# Afficher des statistiques descriptives sur df avec arrondi à deux décimales
stock_price.info()
print (stock_price.describe().round(decimals=2))


stock_price["Date"]=pd.to_datetime(stock_price["Date"])


stock_list=pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/stock_list.csv")
stock_list


stock_price.isna().sum()


print("The training data begins on {} and ends on {}.\n".format(stock_price.Date.min(),stock_price.Date.max()))
display(stock_price.select_dtypes(include=["number"]).describe().style.format("{:,.2f}"))


stock_list.head()



stock_list.isna().sum()


# JPX average returns, closing price and volume between the time periods under observation

price_date = stock_price.Date.unique()
average_returns = stock_price.groupby("Date")["Target"].mean().mul(100).rename("Average Return")
close_average = stock_price.groupby("Date")["Close"].mean().rename("Average Closing Price")
volume_average = stock_price.groupby("Date")["Volume"].mean().rename("Volume")
fig = make_subplots(rows = 3, cols = 1, shared_xaxes = True)

for i, j in enumerate([average_returns, close_average, volume_average]):
    fig.add_trace(go.Scatter(x = price_date, y = j, mode = "lines", name = j.name), row = i + 1, col = 1)

fig.update_xaxes(rangeslider_visible = False, rangeselector = dict(
    buttons = list([dict(count = 1, step = "day", label = "1D", stepmode = "backward"),
                    dict(count = 5, step = "day", label = "5D", stepmode = "backward"),
                    dict(count = 1, step = "month", label = "1M", stepmode = "backward"),
                    dict(count = 6, step = "month", label = "6M", stepmode = "backward"),
                    dict(count = 1, step = "year", label = "1Y", stepmode = "backward"),
                    dict(count = 2, step = "year", label = "2Y", stepmode = "backward"),
                    dict(step = "all")])), row = 1, col = 1)
                  
fig.update_layout(title = "JPX Market Average Stock Return, Closing Price & Volume", hovermode = "x unified", height = 600, 
                  yaxis1 = dict(title = "Avg Stock Returns", ticksuffix = "%"),
                  yaxis2 = dict(title = "Avg Closing Price"),
                  yaxis3 = dict(title = "Avg Volume"), showlegend = False)


fig.show()


stock_list.rename(columns={"17SectorName": "Sector"}, inplace=True)
stock_list.head()


# Définir SecuritiesCode comme index 
stock_list.set_index('SecuritiesCode', inplace=True)
stock_price.set_index('SecuritiesCode', inplace=True)

# Effectuer la jointure avec join()                     
stock_price = stock_price.join(stock_list[["Name", "Sector"]], how="left")

# Réinitialiser l'index 
stock_price.reset_index(inplace=True)


stock_price


stock_price['Year'] = stock_price['Date'].dt.year
stock_price


stock_details = stock_price.groupby(['Year','Sector','Name']).agg(moy=('Target', 'mean')).reset_index()
stock_details['moy'] = stock_details['moy'] * 1000
stock_details


# Get the top 20 most profitable stocks per year
stock_details_sorted = stock_details.sort_values(by=['Year', 'moy'], ascending=[True, False])

top_stocks = stock_details_sorted.groupby('Year').head(20)
top_stocks


# Creating the bar plot

plt.figure(figsize=(8, 5))  
sns.barplot(data=top_stocks, x='Year', y='moy', hue = 'Sector')

# Add titles and labels
plt.title('Distribution of Targets by Year and Sector')
plt.xlabel('Year')
plt.ylabel('return average')

# Display the plot
plt.tight_layout()  
plt.show()


stock_quantity = stock_price.groupby(['Name','Sector']).agg(Quantity=('Volume', 'sum')).reset_index() 
stock_quantity


# Creating the bar plot

plt.figure(figsize=(8, 5))  # Set the figure size
sns.barplot(data=stock_quantity.head(10), x='Quantity', y='Name')

# Add titles and labels
plt.title('Distribution of Stocks by Volume ')
plt.xlabel('Stock Name')
plt.ylabel('Quantity')

# Display the plot
plt.tight_layout()  # Adjust layout to make room for labels
plt.show()


# Calculer les 5 dernières années
current_year = pd.to_datetime('today').year
years = [current_year - i for i in range(5)]
 
# Filtrer les données pour les 5 dernières années
df_last_5_years = stock_price[stock_price['Year'].isin(years)]
df_last_5_years


# Calculer les 5 dernières années
current_year = pd.to_datetime('today').year
years = [current_year - i for i in range(5)]
 
# Filtrer les données pour les 5 dernières années
df_last_5_years = stock_price[stock_price['Year'].isin(years)]
 
# Calculer le rendement total par stock
stock_returns = df_last_5_years.groupby('SecuritiesCode')['Target'].sum()
 
# Trier les stocks par rendement total (en ordre décroissant)
top_stocks = stock_returns.sort_values(ascending=False).head(10)  # Les 10 meilleurs stocks
 
# Tracer les rendements des meilleurs stocks
plt.figure(figsize=(12, 6))
top_stocks.plot(kind='bar', color='c')
 
# Ajouter des éléments au graphique
plt.title("Top 10 des stocks avec les meilleurs rendements sur les 5 dernières années")
plt.xlabel("SecuritiesCode")
plt.ylabel("Rendement total")
plt.xticks(rotation=45)
plt.tight_layout()
 
# Afficher le graphique
plt.show()


# Filtrer les données pour l'année 2021

df_2021 = stock_price[stock_price['Year'] == 2021]
 
# Calculer le rendement pour chaque stock

# Ici, on suppose que 'Return' est le rendement calculé sur chaque ligne (journalière)

# Si vous souhaitez un rendement global pour l'année, calculez le rendement total par stock.
 
# Groupby par 'SecuritiesCode' et calculer le rendement total pour chaque stock

stock_returns = df_2021.groupby('SecuritiesCode')['Target'].sum()
 
# Trouver le stock avec le meilleur rendement

best_stock = stock_returns.idxmax()  # Index (SecuritiesCode) du stock avec le meilleur rendement

best_stock_return = stock_returns.max()  # Meilleur rendement
 
# Affichage du résultat

print(f"Le meilleur stock en 2021 est le stock avec le SecuritiesCode {best_stock}, avec un rendement de {best_stock_return:.4f}.")
 


# Filtrer les données pour le stock avec le SecuritiesCode 7089 et l'année 2021
stock_7089 = stock_price[(stock_price['SecuritiesCode'] == 7089) & (stock_price['Year'] == 2021)]
 
# Tracer l'évolution du prix de clôture
plt.figure(figsize=(10,6))
plt.plot(stock_7089['Date'], stock_7089['Close'], label='Prix de clôture', color='b')
 
# Ajouter des éléments au graphique
plt.title("Évolution du prix de clôture du stock 7089 en 2021")
plt.xlabel("Date")
plt.ylabel("Prix de clôture")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.legend()
 
# Afficher le graphique
plt.show()


# Créer une copie du DataFrame stock_price pour éviter de modifier les données originales
df_model = stock_price.copy()
  
# Créer la colonne de rendement des actions (Return), basé sur les prix de clôture

df_model['Return'] = df_model.groupby('SecuritiesCode')['Close'].pct_change()

# Rendement : Mesure du pourcentage de variation du prix de clôture par rapport à la journée précédente.

# Objectif : Utilisé comme la variable cible pour prédire la performance future des actions.
 
# Moyenne mobile sur 5 jours (SMA_5)

df_model['SMA_5'] = df_model.groupby('SecuritiesCode')['Close'].transform(lambda x: x.rolling(window=5).mean())

# Moyenne mobile à court terme : Moyenne des prix de clôture des 5 derniers jours.

# Objectif : Capturer les tendances à court terme dans les prix des actions.
 
# Moyenne mobile sur 20 jours (SMA_20)

df_model['SMA_20'] = df_model.groupby('SecuritiesCode')['Close'].transform(lambda x: x.rolling(window=20).mean())

# Moyenne mobile à moyen terme : Moyenne des prix de clôture des 20 derniers jours.

# Objectif : Identifier les tendances à moyen terme dans les prix des actions.
 
# Moyenne mobile sur 50 jours (SMA_50)

df_model['SMA_50'] = df_model.groupby('SecuritiesCode')['Close'].transform(lambda x: x.rolling(window=50).mean())

# Moyenne mobile à long terme : Moyenne des prix de clôture des 50 derniers jours.

# Objectif : Analyser les tendances à long terme dans les prix des actions.
 
# Volatilité sur 5 jours (Volatility_5)

df_model['Volatility_5'] = df_model.groupby('SecuritiesCode')['Close'].transform(lambda x: x.rolling(window=5).std())

# Volatilité à court terme : Écart-type des prix de clôture des 5 derniers jours.

# Objectif : Mesurer les fluctuations à court terme dans les prix des actions.
 
# Volatilité sur 20 jours (Volatility_20)

df_model['Volatility_20'] = df_model.groupby('SecuritiesCode')['Close'].transform(lambda x: x.rolling(window=20).std())

# Volatilité à moyen terme : Écart-type des prix de clôture des 20 derniers jours.

# Objectif : Identifier les fluctuations à moyen terme dans les prix des actions.
 
# Moyenne mobile sur 5 jours des volumes (Volume_MA_5)

df_model['Volume_MA_5'] = df_model.groupby('SecuritiesCode')['Volume'].transform(lambda x: x.rolling(window=5).mean())

# Volume moyen à court terme : Moyenne des volumes échangés sur les 5 derniers jours.

# Objectif : Évaluer la liquidité à court terme de l'action.
 
# Différence entre les prix d'ouverture et de clôture (Open_Close_Diff)

df_model['Open_Close_Diff'] = df_model['Open'] - df_model['Close']

# Différence ouverture-clôture : Différence entre les prix d'ouverture et de clôture d'une journée donnée.

# Objectif : Identifier la direction et l'amplitude des mouvements intrajournaliers.
 
# Différence entre les prix haut et bas (High_Low_Diff)

df_model['High_Low_Diff'] = df_model['High'] - df_model['Low']

# Différence haut-bas : Différence entre les prix maximum et minimum d'une journée donnée.

# Objectif : Mesurer la volatilité intrajournalière.
 
# Changement relatif du volume (Volume_Change)

df_model['Volume_Change'] = df_model.groupby('SecuritiesCode')['Volume'].pct_change()

# Variation relative du volume : Taux de changement des volumes échangés par rapport à la journée précédente.

# Objectif : Identifier des anomalies ou des augmentations soudaines de l'activité de trading.
 
# Indicateur d'année (Year)

df_model['Year'] = df_model['Date'].dt.year

# Année : Extrait l'année à partir de la date.

# Objectif : Capturer les effets saisonniers ou les tendances à long terme.
 
# Indicateur de mois (Month)

df_model['Month'] = df_model['Date'].dt.month

# Mois : Extrait le mois à partir de la date.

# Objectif : Identifier des patterns saisonniers à moyen terme.
 
# Indicateur du jour de la semaine (Day_of_week)

df_model['Day_of_week'] = df_model['Date'].dt.weekday

# Jour de la semaine : Extrait le jour de la semaine (0 = lundi, 6 = dimanche).

# Objectif : Capturer les effets spécifiques au jour (par exemple, volatilité accrue le lundi).
 
# Supprimer les valeurs manquantes (NaN)

df_model = df_model.dropna()
 
# Sélectionner les features à utiliser pour l'entraînement du modèle

features = ['SMA_5', 'SMA_20', 'SMA_50', 'Volatility_5', 'Volatility_20', 

            'Volume_MA_5', 'Open_Close_Diff', 'High_Low_Diff', 'Volume_Change', 

            'Year', 'Month', 'Day_of_week']
 
# Créer la variable cible y (rendement)

df_model['Target']=(df_model['Target'] * 100)
target = 'Return'
 
# Créer X (features) et y (target) pour le modèle

X = df_model[features]

y = df_model[target]
 
# Afficher les premières lignes de X et y

print(X.head())

print(y.head())

 


pd.set_option('display.max_columns', None)
df_model





import pandas as pd

# 1. Create Target_Stock as the difference between Close and Open prices
df_model['Target_Stock'] = df_model['Close'] - df_model['Open']

# 2. Shift Target to create a prediction target for the next time step
df_model['Target'] = df_model['Target'].shift(-1)

# 3. Create a TargetClass for binary classification (1 if Target > 0, else 0)
df_model['TargetClass'] = (df_model['Target'] > 0).astype(int)

# 4. Add TargetNextClose as the shifted Adjusted Close for the next day
df_model['TargetNextClose'] = df_model['Close'].shift(-1)

# 5. Drop rows with NaN values created by shifting
df_model.dropna(inplace=True)

# 6. Reset index after dropping rows
df_model.reset_index(drop=True, inplace=True)

# 7. Drop unnecessary columns
df_model.drop(columns=['Volume', 'Close', 'Date'], inplace=True)

# Display the prepared DataFrame
print(df_model.head())



df_model[['Open','High','Low','Close','RSI','EMAF','EMAM','EMAS','Target','TargetClass','TargetNextClose']]


pd.set_option('display.max_columns', None)
df_model


df_model = df_model.iloc[:,:]#.values
pd.set_option('display.max_columns', None)

df_model.head(20)


import seaborn as sns
 
# Visualiser les 10 premières lignes du DataFrame X et y
print(X.head())
print(y.head())
 
# Visualiser la distribution du rendement
plt.figure(figsize=(10, 6))
sns.histplot(y, bins=50, kde=True, color='skyblue')
plt.title("Distribution du rendement")
plt.xlabel("Rendement")
plt.ylabel("Fréquence")
plt.tight_layout()
plt.show()
 
# Visualiser la moyenne mobile à 5 jours d'un stock spécifique (par exemple, pour le stock avec SecuritiesCode = 8713)
specific_stock = stock_price[stock_price['SecuritiesCode'] == 7089]
 
plt.figure(figsize=(10, 6))
plt.plot(specific_stock['Date'], specific_stock['SMA_5'], label="SMA_5", color='green')
plt.title("Moyenne mobile à 5 jours - Stock 7089")
plt.xlabel("Date")
plt.ylabel("Prix")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
 
# Visualiser l'évolution de la volatilité à 5 jours pour le même stock
plt.figure(figsize=(10, 6))
plt.plot(specific_stock['Date'], specific_stock['Volatility_5'], label="Volatilité_5", color='red')
plt.title("Volatilité à 5 jours - Stock 7089")
plt.xlabel("Date")
plt.ylabel("Volatilité")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
 
# Visualiser la corrélation entre certaines features et le rendement
correlation_matrix = stock_price[features].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Matrice de corrélation entre les features et le rendement")
plt.tight_layout()
plt.show()


plt.plot(x, y1, label='Data Series 1', color='blue', marker='o')
plt.plot(x, y2, label='Data Series 2', color='green', marker='s')
plt.plot(x, y3, label='Data Series 3', color='red', marker='^')


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

model = XGBRegressor(random_state=42)

# Train on the last fold (you can loop through all folds to evaluate performance)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate performance
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")



import matplotlib.pyplot as plt

# Plot feature importance
importance = model.feature_importances_
feature_names = features
plt.barh(feature_names, importance)
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Feature Importance Plot")
plt.show()



df_model


df_model_corr = df_model.drop(["AdjustmentFactor", "Name","Sector"], axis=1)
df_model_corr


df_model_corr


df_corr4=df_model[['Volume_Change', 'SMA_50', 'Volatility_5', 'Target']]
df_corr4



# Calculate the correlation matrix
correlation_matrix = df_model_corr.corr()

# Plot the correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
plt.title('Correlation Matrix of Selected Features')
plt.show()


df_model["Target"]

