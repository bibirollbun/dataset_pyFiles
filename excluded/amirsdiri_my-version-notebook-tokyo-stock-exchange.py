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


stock_list = pd.read_csv("/kaggle/input/jpx-tokyo-stock-exchange-prediction/stock_list.csv")
stock_list


stock_price = pd.read_csv("/kaggle/input/jpx-tokyo-stock-exchange-prediction/train_files/stock_prices.csv")
stock_price


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


# Créer une copie du DataFrame stock_price pour éviter de modifier les données originales
df_model = stock_price.copy()

# Creating 1-day, 5-day, and 20-day lags for 'Close'
stock_price["Close_Lag_1"] = stock_price.groupby("SecuritiesCode")["Close"].shift(1)
stock_price["Close_Lag_5"] = stock_price.groupby("SecuritiesCode")["Close"].shift(5)
stock_price["Close_Lag_20"] = stock_price.groupby("SecuritiesCode")["Close"].shift(20)

# Creating lagged features for 'Volume'
stock_price["Volume_Lag_1"] = stock_price.groupby("SecuritiesCode")["Volume"].shift(1)
stock_price["Volume_Lag_5"] = stock_price.groupby("SecuritiesCode")["Volume"].shift(5)
stock_price["Volume_Lag_20"] = stock_price.groupby("SecuritiesCode")["Volume"].shift(20)

# Creating lagged features for 'Target'
stock_price["Target_Lag_1"] = stock_price.groupby("SecuritiesCode")["Target"].shift(1)
stock_price["Target_Lag_5"] = stock_price.groupby("SecuritiesCode")["Target"].shift(5)
stock_price["Target_Lag_20"] = stock_price.groupby("SecuritiesCode")["Target"].shift(20)


# V
stock_id = stock_price["SecuritiesCode"].iloc[0]  # Replace with a valid stock ID from your data
stock_data = stock_price[stock_price["SecuritiesCode"] == 7089]

# Plot 'Close' and its lagged features
plt.figure(figsize=(12, 6))
plt.plot(stock_data["Date"], stock_data["Close"], label="Close (Original)", color="blue")
plt.plot(stock_data["Date"], stock_data["Close_Lag_1"], label="Close (1-Day Lag)", color="orange", linestyle="--")
plt.plot(stock_data["Date"], stock_data["Close_Lag_5"], label="Close (5-Day Lag)", color="green", linestyle="--")
plt.plot(stock_data["Date"], stock_data["Close_Lag_20"], label="Close (20-Day Lag)", color="red", linestyle="--")

# Add labels, legend, and title
plt.xlabel("Date")
plt.ylabel("Price")
plt.title(f"Lagged Features for Stock {7089}")
plt.legend()
plt.tight_layout()
plt.show()


# Scatter plot: Close vs Close_Lag_1
plt.figure(figsize=(8, 6))
plt.scatter(stock_data["Close_Lag_1"], stock_data["Close"], alpha=0.6, color="purple")
plt.xlabel("Close (1-Day Lag)")
plt.ylabel("Close (Original)")
plt.title(f"Scatter Plot: Close vs Close_Lag_1 for Stock {7089}")
plt.grid()
plt.show()


# Compute correlation matrix for selected columns
lagged_cols = ["Close", "Close_Lag_1", "Close_Lag_5", "Close_Lag_20"]
correlation_matrix = stock_data[lagged_cols].corr()

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title(f"Correlation Between Original and Lagged Features for Stock {7089}")
plt.show()


# Filter the DataFrame for the specific stock 7089
stock_data = stock_price[stock_price["SecuritiesCode"] == 7089]

# Calculate the EMA for the 'Close' column with a specified window size
ema_window = 10  # You can change this to any desired window size
stock_data["EMA"] = stock_data["Close"].ewm(span=ema_window, adjust=False).mean()

# Display the DataFrame with the EMA
print(stock_data[["Date", "Close", "EMA"]])

# Plotting the Close prices and EMA
plt.figure(figsize=(12, 6))
plt.plot(stock_data["Date"], stock_data["Close"], label="Close Price", color="blue")
plt.plot(stock_data["Date"], stock_data["EMA"], label=f"EMA (span={ema_window})", color="orange")
plt.title(f"Close Price and EMA for Stock {7089}")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.show()


import yfinance as yf

# Step 1: Data Collection
ticker = '7089.T'  # Ensure the ticker is correct for the Japanese market
data = yf.download(ticker, start='2017-01-04', end='2021-12-03')

# Step 2: Calculate Daily Price Changes
data['Price Change'] = data['Close'].diff()

# Step 3: Separate Gains and Losses
data['Gain'] = np.where(data['Price Change'] > 0, data['Price Change'], 0)
data['Loss'] = np.where(data['Price Change'] < 0, -data['Price Change'], 0)

# Step 4: Calculate Average Gain and Average Loss
period = 14  # Commonly used period for RSI
data['Avg Gain'] = data['Gain'].rolling(window=period).mean()
data['Avg Loss'] = data['Loss'].rolling(window=period).mean()

# Step 5: Calculate Relative Strength (RS) and RSI
data['RS'] = data['Avg Gain'] / data['Avg Loss']
data['RSI'] = 100 - (100 / (1 + data['RS']))

# Step 6: Drop NaN values
data.dropna(inplace=True)

# Step 7: Visualization
plt.figure(figsize=(12, 6))
plt.plot(data['RSI'], label='RSI', color='blue')
plt.axhline(70, linestyle='--', alpha=0.5, color='red')  # Overbought line
plt.axhline(30, linestyle='--', alpha=0.5, color='green')  # Oversold line
plt.title(f'RSI for {ticker}')
plt.xlabel('Date')
plt.ylabel('RSI')
plt.legend()
plt.show()

# Display the last few rows of the DataFrame with RSI
print(data[['Close', 'RSI']].tail())


import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Step 1: Data Collection
ticker = '7089.T'  # Ensure the ticker is correct for the Japanese market
data = yf.download(ticker, start='2017-01-04', end='2021-12-03')

# Step 2: Calculate the Short-Term and Long-Term EMAs
short_window = 12  # Short-term EMA
long_window = 26   # Long-term EMA
signal_window = 9  # Signal line EMA

data['EMA_12'] = data['Close'].ewm(span=short_window, adjust=False).mean()
data['EMA_26'] = data['Close'].ewm(span=long_window, adjust=False).mean()

# Step 3: Calculate the MACD Line
data['MACD'] = data['EMA_12'] - data['EMA_26']

# Step 4: Calculate the Signal Line
data['Signal Line'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()

# Step 5: Calculate the MACD Histogram
data['MACD Histogram'] = data['MACD'] - data['Signal Line']

# Step 6: Drop NaN values
data.dropna(inplace=True)

# Step 7: Visualization
plt.figure(figsize=(14, 7))

# Plot MACD and Signal Line
plt.subplot(2, 1, 1)
plt.plot(data['MACD'], label='MACD', color='blue')
plt.plot(data['Signal Line'], label='Signal Line', color='red')
plt.title(f'MACD for {ticker}')
plt.xlabel('Date')
plt.ylabel('MACD')
plt.legend()

# Plot MACD Histogram
plt.subplot(2, 1, 2)
plt.bar(data.index, data['MACD Histogram'], label='MACD Histogram', color='gray')
plt.axhline(0, linestyle='--', color='black', linewidth=0.5)
plt.title('MACD Histogram')
plt.xlabel('Date')
plt.ylabel('Histogram')
plt.legend()

plt.tight_layout()
plt.show()

# Display the last few rows of the DataFrame with MACD
print(data[['Close', 'MACD', 'Signal Line', 'MACD Histogram']].tail())


import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Step 1: Data Collection
ticker = '7089.T'  # Ensure the ticker is correct for the Japanese market
data = yf.download(ticker, start='2017-01-04', end='2021-12-03')

# Step 2: Calculate the 20-day Simple Moving Average (SMA)
window = 20
data['SMA'] = data['Close'].rolling(window=window).mean()

# Step 3: Calculate the Standard Deviation
data['Std Dev'] = data['Close'].rolling(window=window).std()

# Step 4: Calculate the Upper and Lower Bollinger Bands
data['Upper Band'] = data['SMA'] + (data['Std Dev'] * 2)
data['Lower Band'] = data['SMA'] - (data['Std Dev'] * 2)

# Step 5: Drop NaN values
data.dropna(inplace=True)

# Step 6: Visualization
plt.figure(figsize=(14, 7))
plt.plot(data['Close'], label='Close Price', color='blue')
plt.plot(data['SMA'], label='20-Day SMA', color='orange')
plt.plot(data['Upper Band'], label='Upper Band', color='green')
plt.plot(data['Lower Band'], label='Lower Band', color='red')

# Fill the area between the upper and lower bands
plt.fill_between(data.index, data['Upper Band'], data['Lower Band'], color='lightgray', alpha=0.5)

plt.title(f'Bollinger Bands for {ticker}')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Display the last few rows of the DataFrame with Bollinger Bands
print(data[['Close', 'SMA', 'Upper Band', 'Lower Band']].tail())


df_model


data


stock_data


# Compute correlation matrix for selected columns
lagged_cols = ["Open", "High", "Low"	,"Close", "Adj Close","Volume","SMA","Std Dev","Upper Band","Lower Band"]
correlation_matrix = data[lagged_cols].corr()

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title(f"Correlation Between Original and Lagged Features for Stock {7089}")
plt.show()


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# Define features and target
features = ["Open", "High", "Low", "Adj Close", "SMA", "Std Dev", "Upper Band", "Lower Band"]
X = data[features]
y = data["Close"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = XGBRegressor(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate performance
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")




from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Predictions
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print("Model Evaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R² Score: {r2}")

# Residual analysis
residuals = y_test - y_pred
plt.figure(figsize=(10, 5))

# Residual plot
plt.subplot(1, 2, 1)
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")

# Actual vs Predicted
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Prediction vs Actual")

plt.tight_layout()
plt.show()


