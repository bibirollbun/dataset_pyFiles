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


stock_price


# Calculer les 5 dernières années
current_year = pd.to_datetime('today').year
years = [current_year - i for i in range(5)]

# Filtrer les données pour les 5 dernières années
df_last_5_years = stock_price[stock_price['Year'].isin(years)]

# Calculer le rendement total par stock
stock_returns = df_last_5_years.groupby('SecuritiesCode')['Return'].sum()

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
stock_returns = df_2021.groupby('SecuritiesCode')['Return'].sum()

# Trouver le stock avec le meilleur rendement
best_stock = stock_returns.idxmax()  # Index (SecuritiesCode) du stock avec le meilleur rendement
best_stock_return = stock_returns.max()  # Meilleur rendement

# Affichage du résultat
print(f"Le meilleur stock en 2021 est le stock avec le SecuritiesCode {best_stock}, avec un rendement de {best_stock_return:.4f}.")


# Filtrer les données pour le stock avec le SecuritiesCode 8713 et l'année 2021
stock_8713 = stock_price[(stock_price['SecuritiesCode'] == 8713) & (stock_price['Year'] == 2021)]

# Tracer l'évolution du prix de clôture
plt.figure(figsize=(10,6))
plt.plot(stock_8713['Date'], stock_8713['Close'], label='Prix de clôture', color='b')

# Ajouter des éléments au graphique
plt.title("Évolution du prix de clôture du stock 8713 en 2021")
plt.xlabel("Date")
plt.ylabel("Prix de clôture")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.legend()

# Afficher le graphique
plt.show()


# Trier les données par date (au cas où elles ne le seraient pas déjà)
stock_price = stock_price.sort_values(by=['SecuritiesCode', 'Date'])

# Créer la colonne de rendement des actions (Return), basé sur les prix de clôture
stock_price['Return'] = stock_price.groupby('SecuritiesCode')['Close'].pct_change()

# Créer des features basées sur les prix et volumes
# Moyenne mobile sur 5 jours (pour la tendance à court terme) 
stock_price['SMA_5'] = stock_price.groupby('SecuritiesCode')['Close'].transform(lambda x: x.rolling(window=5).mean())

# Moyenne mobile sur 20 jours (pour la tendance à moyen terme) 
stock_price['SMA_20'] = stock_price.groupby('SecuritiesCode')['Close'].transform(lambda x: x.rolling(window=20).mean())

# Moyenne mobile sur 50 jours (pour la tendance à long terme) 
stock_price['SMA_50'] = stock_price.groupby('SecuritiesCode')['Close'].transform(lambda x: x.rolling(window=50).mean())

# Calcul de l'écart-type sur 5 jours (volatilité à court terme)
stock_price['Volatility_5'] = stock_price.groupby('SecuritiesCode')['Close'].transform(lambda x: x.rolling(window=5).std())

# Calcul de l'écart-type sur 20 jours (volatilité à moyen terme)
stock_price['Volatility_20'] = stock_price.groupby('SecuritiesCode')['Close'].transform(lambda x: x.rolling(window=20).std())

# Ajouter le volume moyen sur 5 jours (liquidité)
stock_price['Volume_MA_5'] = stock_price.groupby('SecuritiesCode')['Volume'].transform(lambda x: x.rolling(window=5).mean())

# Créer des variables basées sur l'ouverture et la clôture
stock_price['Open_Close_Diff'] = stock_price['Open'] - stock_price['Close']
stock_price['High_Low_Diff'] = stock_price['High'] - stock_price['Low']

# Créer une colonne de changement relatif du volume
stock_price['Volume_Change'] = stock_price.groupby('SecuritiesCode')['Volume'].pct_change()

# Créer un indicateur d'année (pour la saisonnalité)
stock_price['Year'] = stock_price['Date'].dt.year

# Créer un indicateur de mois (pour la saisonnalité)
stock_price['Month'] = stock_price['Date'].dt.month

# Créer un indicateur de jour de la semaine (pour la saisonnalité)
stock_price['Day_of_week'] = stock_price['Date'].dt.weekday

# Supprimer les valeurs manquantes
stock_price = stock_price.dropna()

# Sélectionner les features à utiliser pour l'entraînement du modèle 
features = ['SMA_5', 'SMA_20', 'SMA_50', 'Volatility_5', 'Volatility_20', 'Volume_MA_5', 'Open_Close_Diff', 'High_Low_Diff', 'Volume_Change', 'Year', 'Month', 'Day_of_week']

# Créer le DataFrame final avec les features et la variable cible (le rendement)
X = stock_price[features]
y = stock_price['Return']

# Afficher les premières lignes du DataFrame préparé
print(X.head())
print(y.head())


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
specific_stock = stock_price[stock_price['SecuritiesCode'] == 8713]

plt.figure(figsize=(10, 6))
plt.plot(specific_stock['Date'], specific_stock['SMA_5'], label="SMA_5", color='green')
plt.title("Moyenne mobile à 5 jours - Stock 8713")
plt.xlabel("Date")
plt.ylabel("Prix")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Visualiser l'évolution de la volatilité à 5 jours pour le même stock
plt.figure(figsize=(10, 6))
plt.plot(specific_stock['Date'], specific_stock['Volatility_5'], label="Volatilité_5", color='red')
plt.title("Volatilité à 5 jours - Stock 8713")
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




