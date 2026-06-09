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


#DATA UNDERSTANDING 
# importing libraries and reading training data
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

training_data=pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/train_files/stock_prices.csv", parse_dates=["Date"])
stock_list=pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/stock_list.csv")

# creating anonomised dataset
training_data[["Open", "High", "Low", "Volume", "Close"]] = training_data[["Open", "High", "Low", "Volume", "Close"]] + 1


#Ce code est utilisé pour afficher des informations clés sur les données d'entraînement, notamment la période de temps couverte par les données et un résumé statistique des colonnes numériques. 
print("The training data begins on {} and ends on {}.\n".format(training_data.Date.min(),training_data.Date.max()))#Cette ligne affiche les dates de début et de fin des données d'entraînement.

display(training_data.select_dtypes(include=["number"]).describe().style.format("{:,.2f}")) #Cette ligne génère un résumé statistique des colonnes numériques et formate les résultats pour les rendre plus lisibles.




stock_list.head(5)


#Cette ligne calcule et affiche combien de titres financiers uniques (SecuritiesCode) sont présents dans le jeu de données training_data.
print("There are a total of {} unique stocks.\n".format(training_data.SecuritiesCode.nunique()))
display(stock_list.describe().style.format('{:,.2f}'))


training_data.head(5)


# l'analyse des rendements moyens, des prix de clôture et du volume des transactions d'un ensemble d'actions ou d'instruments financiers pendant une période spécifiqu
#
training_date = training_data.Date.unique() #1. Création des moyennes quotidiennes pour les rendements, les prix de clôture et le volume
average_returns = training_data.groupby("Date")["Target"].mean().mul(100).rename("Average Return")#groupby("Date") : Groupe les données par date.
close_average = training_data.groupby("Date")["Close"].mean().rename("Average Closing Price")#Calcule la moyenne des prix de clôture (Close) pour chaque date.
volume_average = training_data.groupby("Date")["Volume"].mean().rename("Volume")#Calcule la moyenne des volumes de transactions (Volume) pour chaque date.

fig = make_subplots(rows = 3, cols = 1, shared_xaxes = True)#2. Création de la figure avec plusieurs sous-graphiques

#3. Ajout de données aux sous-graphiques


for i, j in enumerate([average_returns, close_average, volume_average]):
    fig.add_trace(go.Scatter(x = training_date, y = j, mode = "lines", name = j.name), row = i + 1, col = 1)

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



#Analyse exploratoire des données

# rendements moyens annuels par secteur
#Les noms des secteurs et des actions sont nettoyés (retirer les espaces superflus, convertir en minuscules puis en majuscule pour la première lettre).
stock_list["Sector"] = [i.rstrip().lower().capitalize() for i in stock_list["17SectorName"]]
stock_list["Name"] = [i.rstrip().lower().capitalize() for i in stock_list["Name"]]
training_data = training_data.merge(stock_list[["SecuritiesCode", "Name", "Sector"]], on = "SecuritiesCode", how = "left") #Fusionne les informations du secteur et du nom des actions avec les données d'entraînement en utilisant SecuritiesCode comme clé de jointure. Cela permet d'ajouter les informations sur le secteur et le nom des actions au jeu de données d'entraînement.
#Crée une nouvelle colonne "Year" pour extraire l'année de la colonne "Date", facilitant ainsi l'analyse par année.
training_data["Year"] = training_data["Date"].dt.year
return_average = training_data.groupby(["Sector","Year"])["Target"].mean().mul(100).reset_index(name="AverageReturn")#Calcul des rendements moyens annuels par secteur
years = return_average["Year"].unique() #Récupère toutes les années uniques présentes dans les données, puis les trie.
years.sort()
#Visualisation avec plotly
fig = make_subplots(rows = 1, cols = 5, shared_yaxes = True)
#Ajout des barres pour chaque année
for i, year in enumerate(years):
    year_data = return_average[return_average["Year"] == year]
    negative = year_data["AverageReturn"] <= 0
    fig.add_trace(go.Bar(x = year_data["AverageReturn"][negative], y = year_data["Sector"][negative], orientation = "h",
                        text = year_data["AverageReturn"][negative], texttemplate = "%{text:.2f}%", textposition = "auto",
                        hovertemplate = "Average return in %{y} = %{x:.4f}%",
                        marker = dict(color = "red", opacity = 0.85), name = str(year)), row = 1, col = i + 1)
    
    fig.add_trace(go.Bar(x = year_data["AverageReturn"][~negative], y = year_data["Sector"][~negative], orientation = "h",
                        text = year_data["AverageReturn"][~negative], texttemplate = "%{text:.2f}%", textposition = "auto",
                        hovertemplate = "Average returns in %{y} = %{x:.4f}%",
                        marker = dict(color = "green", opacity = 0.85), name = str(year)), row = 1, col = i + 1)
    
    fig.update_xaxes(title = f"{year}", row = 1, col = i + 1)

fig.update_layout(template = "simple_white", title = "Yearly Average Returns by Sector", hovermode = "closest",
                 height = 600, showlegend = False)
                         
fig.show()


# 1. Calculer les prix de clôture moyens par secteur et par année

close_average = training_data.groupby(["Sector","Year"])["Close"].mean().reset_index(name="Close") #Regroupe les données par secteur (Sector) et par année (Year) ,Calcule la moyenne des prix de clôture (Close) pour chaque combinaison de secteur et d'année , Réinitialise l'index pour convertir le résultat en un DataFrame standard avec une colonne nommée Close.
years = close_average["Year"].unique() #2. Trouver les années uniques
years.sort()

fig = make_subplots(rows = 1, cols = 5, shared_yaxes = True) #3. Créer une figure avec plusieurs sous-graphiques

#4. Ajouter des barres pour chaque année

for i, year in enumerate(years):
    year_data = close_average[close_average["Year"] == year]
    fig.add_trace(go.Bar(x = year_data["Close"], y = year_data["Sector"], orientation = "h",
                        text = year_data["Close"], texttemplate = "%{text:.2s}", textposition = "auto",
                        hovertemplate = "Average closing price in %{y} = %{x:.4s}",
                        marker = dict(color = "green", opacity = 0.8), name = str(year)), row = 1, col = i + 1)
    fig.update_xaxes(title = f"{year}", row = 1, col = i + 1)

fig.update_layout(template = "simple_white", title = "Yearly Average Closing Price by Sector", hovermode = "closest",
                 height = 600, showlegend = False)
                         
fig.show()


# le volume moyen annuel des transactions 
#Ce code génère un graphique interactif composé de 5 sous-graphiques (un par année). Chaque sous-graphe montre les volumes moyens des transactions pour chaque secteur sous forme de barres horizontales.
#Les couleurs (bleues) et les infobulles facilitent l'interprétation des données, permettant d'analyser comment le volume des transactions a varié entre les secteurs et les années.
volume_average = training_data.groupby(["Sector","Year"])["Volume"].mean().reset_index(name="Volume")
years = volume_average["Year"].unique()
years.sort()

fig = make_subplots(rows = 1, cols = 5, shared_yaxes = True)

for i, year in enumerate(years):
    year_data = volume_average[volume_average["Year"] == year]
    fig.add_trace(go.Bar(x = year_data["Volume"], y = year_data["Sector"], orientation = "h",
                        text = year_data["Volume"], texttemplate = "%{text:.2s}", textposition = "auto",
                        hovertemplate = "Average volume in %{y} = %{x:.4s}",
                        marker = dict(color = "blue", opacity = 0.6), name = str(year)), row = 1, col = i + 1)
    fig.update_xaxes(title = f"{year}", row = 1, col = i + 1)

fig.update_layout(template = "simple_white", title = "Yearly Average Volume by Sector", hovermode = "closest",
                 height = 600, showlegend = False)
                         
fig.show()


# "Jours de bourse où le marché a performé positivement ou négativement selon les rendements moyens
#1. Calcul des rendements moyens quotidiens

average_returns = training_data.groupby("Date")["Target"].mean().mul(100).rename("Average Return)  
#2. Identification des jours positifs et négatifs

average_returns["Return Column"] = average_returns.apply(lambda x: "Positive" if x >= 0 else "Negative")
#3. Comptage des jours totaux, positifs et négatifs

count_total = len(average_returns)
count_positive = (average_returns["Return Column"] == "Positive").sum()
count_negative = (average_returns["Return Column"] == "Negative").sum()
#4. Création des données pour le graphique circulaire

pie_data = pd.DataFrame({
    "Return Column":["Positive", "Negative"],
    "Count":[count_positive, count_negative]
})
#5. Création du graphique circulaire

fig = px.pie(pie_data, names = "Return Column", color = "Return Column", values = "Count", 
            title = "Distribution of Stocks Averaging Positive or Negative Returns",
            color_discrete_map={"Positive": "green", "Negative": "red"},
            opacity = 0.85)

fig.update_traces(textinfo = "percent+label") #6. Mise à jour des étiquettes et mise en page


fig.update_layout(width = 600)

fig.show()


#Filtrage des données après le 23 décembre 2020 : 
#1. Filtrage des données à partir d'une date spécifique
#2. Affichage d'un résumé des nouvelles données filtrées
#3. Vérification de la taille des données après filtrage
#4. Vérification des valeurs manquantes dans la colonne "Target"
#Affichage des premières et dernières lignes des données

training_data = training_data[training_data.Date > "2020-12-23"]
print("New training data consists of 2000 stocks over a 231 day time frame (24/12/2020 to 03/12/2021).")
print(training_data.shape)
print("Missing values in column of interest, Target = ", training_data["Target"].isna().sum())

display(training_data.head(), training_data.tail())


# "Distribution de la colonne cible
#Calcul des percentiles 1% et 99% de la colonne "Target"
#Calcul des percentiles (1% et 99%) : Identifie les limites pour exclure les valeurs extrêmes.
#Filtrage des données : Supprime les valeurs extrêmes pour analyser les valeurs typiques.
#Visualisation : Produit un histogramme pour observer la répartition des rendements (Target) de manière claire et lisible.
upper_percentile = training_data["Target"].quantile(0.99)
lower_percentile = training_data["Target"].quantile(0.01)
df = training_data[(training_data["Target"] < upper_percentile) & (training_data["Target"] > lower_percentile)]
#Visualisation de la distribution de la colonne "Target"
sns.displot(df["Target"], aspect = 2.5, kde = False, bins = 10)



sns.boxplot(y = df["Target"]) #sns.boxplot() : Crée un diagramme en boîte (ou boîte à moustaches) à l'aide de la bibliothèque Seaborn.



# Returns distribution by sector boxplot
#Chaque boxplot représente un secteur :La distribution des rendements est visualisée pour chaque secteur sous forme de boîte à moustaches.
#Informations extraites :Médiane : Indique le rendement médian (ligne à l'intérieur de la boîte).
#Dispersion des rendements : Plus la boîte ou les moustaches sont grandes, plus les rendements varient au sein du secteur.
#Valeurs aberrantes : Les points individuels hors des moustaches représentent les rendements extrêmes.
#Comparaison entre secteurs :Permet de repérer les secteurs avec des rendements globalement élevés, faibles, ou très variables

sectors = training_data["Sector"].unique()

fig = go.Figure()

for sector in sectors:
    y = training_data[training_data["Sector"] == sector]["Target"] * 100
    fig.add_trace(go.Box(y = y, name = sector))
    
fig.update_layout(title = "Returns Distribution by Sector", yaxis = dict(title = "Return", ticksuffix = "%"), height = 700)

fig.show()


training_data


# Average close price comparison between start and end of time frame
#Barres comparatives pour chaque secteur :

#Chaque secteur dispose de deux barres représentant les prix moyens de clôture :
#Une pour le début de la période (24/12/2020). ,Une pour la fin de la période (03/12/2021) ,Analyse possible :

#Permet d'identifier les secteurs où le prix moyen de clôture a augmenté, diminué ou stagné.Donne une idée des performances générales des secteurs au fil du temps.Couplé à d'autres analyses :

#Peut être utilisé avec des rendements ou des volumes pour évaluer la dynamique des marchés sur la période.


fig = go.Figure()

for sector in sectors:
    start_close = training_data[(training_data["Sector"] == sector) & (training_data["Date"] == "2020-12-24")]["Close"].mean()
    end_close = training_data[(training_data["Sector"] == sector) & (training_data["Date"] == "2021-12-03")]["Close"].mean()
    
    fig.add_trace(go.Bar(x = [f"{sector} Start", f"{sector} End"], y = [start_close, end_close], name = sector))
    
    
fig.update_layout(title = "Average Close Price by Sector at Beginning and End Date",
                  yaxis = dict(title = "Close Price"), height = 700, width = 1080, barmode = "group")

fig.show()


#Feature Engineering 
def adjust_price(price):
    """
    Args:
        price (pd.DataFrame)  : pd.DataFrame include stock_price
    Returns:
        price DataFrame (pd.DataFrame): stock_price with generated AdjustedClose
    """
    # Conversion de la colonne "Date" en type datetime
    price.loc[: ,"Date"] = pd.to_datetime(price.loc[: ,"Date"], format="%Y-%m-%d")

    def generate_adjusted_close(df):  #fonction interne prend dataframe contenant les prix d'un seul titre t génère une colonne AdjustedClose (prix ajusté) pour ce titre.
        """
        Args:
            df (pd.DataFrame)  : stock_price for a single SecuritiesCode
        Returns:
            df (pd.DataFrame): stock_price with AdjustedClose for a single SecuritiesCode
        """
        # Les données sont triées par date de manière décroissante (du plus récent au plus ancien) pour préparer le calcul du facteur d'ajustement cumulatif.
        df = df.sort_values("Date", ascending=False)
        # Calcul du CumulativeAdjustmentFactor
        df.loc[:, "CumulativeAdjustmentFactor"] = df["AdjustmentFactor"].cumprod()
        # generate AdjustedClose
        df.loc[:, "AdjustedClose"] = (
            df["CumulativeAdjustmentFactor"] * df["Close"]
        ).map(lambda x: float(
            Decimal(str(x)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)
        ))
        # Réverser l'ordre des données
        df = df.sort_values("Date")
        # to fill AdjustedClose, replace 0 into np.nan
        df.loc[df["AdjustedClose"] == 0, "AdjustedClose"] = np.nan
        # forward fill AdjustedClose
        df.loc[:, "AdjustedClose"] = df.loc[:, "AdjustedClose"].ffill()
        return df

    # generate AdjustedClose
    price = price.sort_values(["SecuritiesCode", "Date"])
    price = price.groupby("SecuritiesCode").apply(generate_adjusted_close).reset_index(drop=True)

    price.set_index("Date", inplace=True)
    return price



df_price = pd.read_csv("/kaggle/input/jpx-tokyo-stock-exchange-prediction/train_files/stock_prices.csv")
df_price = adjust_price(df_price)


# will use these time periods for calculations of price change, moving averages, volatility, etc over the past 1, 5, 10, 20, 50, 100 and 200 previous
# trading days
periods = [1, 5, 10, 20, 50, 100, 200]
# first security in data frame selected to show visual examples of the new features created
example = df_price.loc[df_price["SecuritiesCode"] == 1301].copy()


# past price change

price_change = []
for period in periods:
    price_change.append(f"price_change {period}")
    example.loc[:, f"price_change {period}"] = example["AdjustedClose"].pct_change(period)


# visualising price change
example[price_change].plot(figsize = (14, 6), title = "Past Price Change, Securitiy 1301")


# future price change

future_price_change = []
for period in periods: 
    future_price_change.append(f"future_price_change {period}")
    example.loc[:, f"future_price_change {period}"] = example["AdjustedClose"].shift(-period) / example["AdjustedClose"] - 1


# visualising future price change
example[future_price_change].plot(figsize = (14, 6), title = "Future Price Change, Securitiy 1301")


#la variabilité ou les fluctuations des prix d'un actif 
volatility = []
for period in periods:
    volatility.append(f"volatility {period}")
    example.loc[:, f"volatility {period}"] = np.log(example["AdjustedClose"]).diff().rolling(period).std()


# visualisation of volatility
example[volatility].plot(figsize = (14, 6), title = "Historical Volatility, Securitiy 1301")


# Simple moving averages, 5, 10, 20, 50, 100

simple_moving_average = []
for period in periods:
    simple_moving_average.append(f"simple_moving_average {period}")
    example.loc[:, f"simple_moving_average {period}"] = example["AdjustedClose"].rolling(window = period).mean()


# simple moving averages visualisation
example[simple_moving_average].plot(figsize = (14, 6), title = "Simple Moving Averages, Securitiy 1301")


# Exponential moving averages, 5, 10, 20, 50, 100

exponential_moving_average = []
for period in periods:
    exponential_moving_average.append(f"exponential_moving_average {period}")
    example.loc[:, f"exponential_moving_average {period}"] = example["AdjustedClose"].ewm(span = period, adjust = False).mean()


# exponential moving averages visualisation
example[exponential_moving_average].plot(figsize = (14, 6), title = "Exponential Moving Averages, Securitiy 1301")


# RSI using simple moving average  l'Indice de Force Relative (RSI)Le RSI est un indicateur très utilisé dans l'analyse technique des actions. Il aide à identifier les conditions de surachat ou de survente d'une action
# removed 1 day calculation as it is not a very good representation
periods = [5, 10, 20, 50, 100, 200] #Initialisation des périodes Le RSI est un indicateur très utilisé dans l'analyse technique des actions. Il aide à identifier les conditions de surachat ou de survente d'une action
rsi_sma_periods = []
for period in periods:
    delta = example["AdjustedClose"].diff()
    up, down = delta.clip(lower = 0), -delta.clip(upper = 0)
    roll_up, roll_down = up.rolling(window = period).mean(), down.rolling(window = period).mean()
    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    rsi_sma_periods.append(f"rsi_sma_period {period}")
    example.loc[:, f"rsi_sma_period {period}"] = rsi


# visualisation of RSI SMA
example[rsi_sma_periods].plot(figsize = (14, 6), title = "Relative Strength Index using Simple Moving Average, Securitiy 1301")


# RSI using exponential moving average
# removed 1 day calculation as it is not a very good representation
periods = [5, 10, 20, 50, 100, 200]
rsi_ema_periods = []
for period in periods:
    delta = example["AdjustedClose"].diff()
    up, down = delta.clip(lower = 0), -delta.clip(upper = 0)
    roll_up, roll_down = up.ewm(span = period, adjust = False).mean(), down.ewm(span = period, adjust = False).mean()
    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    rsi_ema_periods.append(f"rsi_ema_periods {period}")
    example.loc[:, f"rsi_ema_periods {period}"] = rsi


# visualisation of RSI EMA
example[rsi_ema_periods].plot(figsize = (14, 6), title = "Relative Strength Index using Exponential Moving Average, Securitiy 1301")


#La principale différence entre la Moyenne Mobile Exponentielle (EMA) et la Moyenne Mobile Simple (SMA) réside dans la manière dont ces moyennes sont calculées et comment chaque type de moyenne réagit aux données récentes.


# On-balance volume

example["OBV"] = 0
example.loc[:, "OBV"] = example["Volume"].cumsum().where(example["AdjustedClose"].diff() > 0, -example["Volume"]).cumsum()



# visualisation of OBV On-Balance Volume (OBV) est un indicateur technique populaire utilisé pour mesurer l'accumulation ou la distribution d'un titre en fonction de son volume de transactions. 
example["OBV"].plot(figsize = (14, 6), title = "On-Balance Volume, Securitiy 1301")

