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


stock_price


top_stocks


# Créer une copie du DataFrame stock_price pour éviter de modifier les données originales
df_model = stock_price.copy()

# Creating 1-day, 5-day, and 20-day lags for 'Close'
df_model["Close_Lag_1"] = df_model.groupby("SecuritiesCode")["Close"].shift(1)
df_model["Close_Lag_5"] = df_model.groupby("SecuritiesCode")["Close"].shift(5)
df_model["Close_Lag_20"] = df_model.groupby("SecuritiesCode")["Close"].shift(20)

# Creating lagged features for 'Volume'
df_model["Volume_Lag_1"] = df_model.groupby("SecuritiesCode")["Volume"].shift(1)
df_model["Volume_Lag_5"] = df_model.groupby("SecuritiesCode")["Volume"].shift(5)
df_model["Volume_Lag_20"] = df_model.groupby("SecuritiesCode")["Volume"].shift(20)

# Creating lagged features for 'Target'
df_model["Target_Lag_1"] = df_model.groupby("SecuritiesCode")["Target"].shift(1)
df_model["Target_Lag_5"] = df_model.groupby("SecuritiesCode")["Target"].shift(5)
df_model["Target_Lag_20"] = df_model.groupby("SecuritiesCode")["Target"].shift(20)

#Dropping the NAN values but it's not good because it could drop useful data 
df_model.dropna(inplace=True)

# Alternative: Fill NaN values using forward fill
#stock_price.fillna(method='ffill', inplace=True)


df_model


#choosing stock 7089 to work with 

stock_7089_lagged = df_model[df_model["SecuritiesCode"] == 7089]

# Plot 'Close' and its lagged features
plt.figure(figsize=(12, 6))
plt.plot(stock_7089_lagged["Date"], stock_7089_lagged["Close"], label="Close (Original)", color="blue")
plt.plot(stock_7089_lagged["Date"], stock_7089_lagged["Close_Lag_1"], label="Close (1-Day Lag)", color="orange", linestyle="--")
plt.plot(stock_7089_lagged["Date"], stock_7089_lagged["Close_Lag_5"], label="Close (5-Day Lag)", color="green", linestyle="--")
plt.plot(stock_7089_lagged["Date"], stock_7089_lagged["Close_Lag_20"], label="Close (20-Day Lag)", color="red", linestyle="--")

# Add labels, legend, and title
plt.xlabel("Date")
plt.ylabel("Price")
plt.title(f"Lagged Features for Stock {7089}")
plt.legend()
plt.tight_layout()
plt.show()


# Scatter plot: Close vs Close_Lag_1
plt.figure(figsize=(8, 6))
plt.scatter(stock_7089_lagged["Close_Lag_1"], stock_7089_lagged["Close"], alpha=0.6, color="purple")
plt.xlabel("Close (1-Day Lag)")
plt.ylabel("Close (Original)")
plt.title(f"Scatter Plot: Close vs Close_Lag_1 for Stock {7089}")
plt.grid()
plt.show()


# Compute correlation matrix for selected columns
lagged_cols = ["Close", "Close_Lag_1", "Close_Lag_5", "Close_Lag_20"]
correlation_matrix = stock_7089_lagged[lagged_cols].corr()

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title(f"Correlation Between Original and Lagged Features for Stock {7089}")
plt.show()


df_model 


# Calculate the EMA for the 'Close' column with a specified window size
ema_window = 10  # You can change this to any desired window size
df_model["EMA"] = df_model["Close"].ewm(span=ema_window, adjust=False).mean()


# Plotting the Close prices and EMA for all stocks
plt.figure(figsize=(12, 6))

# Plot for stock 7089
stock_7089_EMA = df_model[df_model["SecuritiesCode"] == 7089]

stock_7089_EMA = df_model[df_model["SecuritiesCode"] == 7089]
plt.plot(stock_7089_EMA["Date"], stock_7089_EMA["Close"], label=f"Close Price {7089}", alpha=0.5)
plt.plot(stock_7089_EMA["Date"], stock_7089_EMA["EMA"], label=f"EMA {7089} (span={ema_window})", linestyle='--')

plt.title("Close Price and EMA for 7089")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.show()


df_model


import numpy as np
import pandas as pd

# Calculate Daily Price Changes
#This is the diff between todays closing and yesterdays closing price
df_model['Price Change'] = df_model['Close'].diff()

# Separate Gains and Losses
df_model['Gain'] = np.where(df_model['Price Change'] > 0, df_model['Price Change'], 0)
df_model['Loss'] = np.where(df_model['Price Change'] < 0, -df_model['Price Change'], 0)

# Calculate Average Gain and Average Loss
period = 14  # Commonly used period for RSI
df_model['Avg Gain'] = df_model['Gain'].rolling(window=period).mean()
df_model['Avg Loss'] = df_model['Loss'].rolling(window=period).mean()

# Calculate Relative Strength (RS) and RSI
df_model['RS'] = df_model['Avg Gain'] / df_model['Avg Loss']
df_model['RSI'] = 100 - (100 / (1 + df_model['RS']))

# Drop NaN values (if necessary, but you may want to keep them for other calculations)
df_model.dropna(inplace=True)

#I think that the most important things here are, Avg Gain, Avg Loss


df_model


import matplotlib.pyplot as plt

# Ensure that the 'RSI' column exists in df_model for stock 7089
# Filter the DataFrame for stock 7089
stock_7089_RSI = df_model[df_model['SecuritiesCode'] == 7089]

# Set the figure size
plt.figure(figsize=(12, 6))

# Plot the RSI
plt.plot(stock_7089_RSI['Date'], stock_7089_RSI['RSI'], label='RSI', color='blue')

# Add horizontal lines for overbought and oversold levels
plt.axhline(70, linestyle='--', alpha=0.5, color='red', label='Overbought (70)')
plt.axhline(30, linestyle='--', alpha=0.5, color='green', label='Oversold (30)')

# Set the title and labels
plt.title('RSI for Stock 7089')
plt.xlabel('Date')
plt.ylabel('RSI')
plt.legend()

# Show the plot
plt.show()


import numpy as np
import pandas as pd

# Assuming df_model is your DataFrame that contains the stock data
# Calculate the Short-Term and Long-Term EMAs
short_window = 12  # Short-term EMA
long_window = 26   # Long-term EMA
signal_window = 9  # Signal line EMA

# Calculate the Short-Term and Long-Term EMAs
df_model['EMA_12'] = df_model['Close'].ewm(span=short_window, adjust=False).mean()
df_model['EMA_26'] = df_model['Close'].ewm(span=long_window, adjust=False).mean()

# Calculate the MACD Line
df_model['MACD'] = df_model['EMA_12'] - df_model['EMA_26']

# Calculate the Signal Line
df_model['Signal Line'] = df_model['MACD'].ewm(span=signal_window, adjust=False).mean()

# Calculate the MACD Histogram
df_model['MACD Histogram'] = df_model['MACD'] - df_model['Signal Line']

# Drop NaN values (if necessary, but you may want to keep them for other calculations)
df_model.dropna(inplace=True)


df_model


import matplotlib.pyplot as plt

# Filter the DataFrame for stock 7089
stock_7089_MACD = df_model[df_model['SecuritiesCode'] == 7089]

# Set up the plot
plt.figure(figsize=(14, 7))

# Plot MACD and Signal Line
plt.subplot(2, 1, 1)
plt.plot(stock_7089_MACD['Date'], stock_7089_MACD['MACD'], label='MACD', color='blue')
plt.plot(stock_7089_MACD['Date'], stock_7089_MACD['Signal Line'], label='Signal Line', color='red')
plt.title('MACD for Stock 7089')
plt.xlabel('Date')
plt.ylabel('MACD')
plt.legend()

# Plot MACD Histogram
plt.subplot(2, 1, 2)
plt.bar(stock_7089_MACD['Date'], stock_7089_MACD['MACD Histogram'], label='MACD Histogram', color='gray')
plt.axhline(0, linestyle='--', color='black', linewidth=0.5)
plt.title('MACD Histogram for Stock 7089')
plt.xlabel('Date')
plt.ylabel('Histogram')
plt.legend()

plt.tight_layout()
plt.show()


import numpy as np
import pandas as pd

# Calculate the 20-day Simple Moving Average (SMA)
window = 20

# Calculate the 20-day SMA
df_model['SMA'] = df_model['Close'].rolling(window=window).mean()

# Calculate the Standard Deviation
df_model['Std Dev'] = df_model['Close'].rolling(window=window).std()

# Calculate the Upper and Lower Bollinger Bands
df_model['Upper Band'] = df_model['SMA'] + (df_model['Std Dev'] * 2)
df_model['Lower Band'] = df_model['SMA'] - (df_model['Std Dev'] * 2)

# Drop NaN values (if necessary)
df_model.dropna(inplace=True)



df_model


import matplotlib.pyplot as plt

# Filter the DataFrame for stock 7089
stock_7089_bolinger = df_model[df_model['SecuritiesCode'] == 7089]

# Set up the plot
plt.figure(figsize=(14, 7))

# Plot Close Price, SMA, Upper Band, and Lower Band
plt.plot(stock_7089_bolinger['Date'], stock_7089_bolinger['Close'], label='Close Price', color='blue')
plt.plot(stock_7089_bolinger['Date'], stock_7089_bolinger['SMA'], label='20-Day SMA', color='orange')
plt.plot(stock_7089_bolinger['Date'], stock_7089_bolinger['Upper Band'], label='Upper Band', color='green')
plt.plot(stock_7089_bolinger['Date'], stock_7089_bolinger['Lower Band'], label='Lower Band', color='red')

# Fill the area between the upper and lower bands
plt.fill_between(stock_7089_bolinger['Date'], stock_7089_bolinger['Upper Band'], stock_7089_bolinger['Lower Band'], color='lightgray', alpha=0.5)

# Set the title and labels
plt.title('Bollinger Bands for Stock 7089')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()

# Show the plot
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Select relevant columns for correlation analysis
columns_of_interest = [
    'Close', 
    'Close_Lag_1', 
    'Close_Lag_5', 
    'Close_Lag_20', 
    'EMA', 
    'RSI', 
    'MACD', 
    'Signal Line', 
    'MACD Histogram', 
    'SMA', 
    'Upper Band', 
    'Lower Band'
]

# Create a new DataFrame with only the selected columns
correlation_data = df_model[columns_of_interest]

# Calculate the correlation matrix
correlation_matrix = correlation_data.corr()

# Set up the matplotlib figure
plt.figure(figsize=(12, 8))

# Create a heatmap
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})

# Set the title
plt.title('Correlation Heatmap of Features with Close Price')

# Show the plot
plt.show()


#BASED ON THE HEATMAP WE FOUND THAT THESE ARE THE IMPORTANT FEATURES TO ADD 'Close_Lag_1','Close_Lag_5','Close_Lag_20','EMA','SMA','Upper Band','Lower Band'

import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Define features and target
features = [
    'Close_Lag_1', 
    'Close_Lag_5', 
    'Close_Lag_20', 
    'EMA', 
    'SMA', 
    'Upper Band', 
    'Lower Band'
]
target = 'Close'  # Assuming 'Close' is the target variable

# Prepare the data
X = df_model[features]
y = df_model[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the XGBoost regressor
model = XGBRegressor(n_estimators=100, learning_rate=0.1)

# Fit the model to the training data
model.fit(X_train, y_train)

# predictions on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f'Mean Absolute Error: {mae:.2f}')
print(f'R-squared: {r2:.2f}')


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Define features and target
features = [
    'Close_Lag_1', 
    'Close_Lag_5', 
    'Close_Lag_20', 
    'EMA', 
    'SMA', 
    'Upper Band', 
    'Lower Band'
]
target = 'Close'  # Assuming 'Close' is the target variable

# Prepare the data
X = df_model[features].values
y = df_model[target].values

# Scale the data
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Reshape the data for LSTM [samples, time steps, features]
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Initialize the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Fit the model to the training data
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Make predictions on the test set
y_pred_scaled = model.predict(X_test)

# Inverse transform the predictions and actual values
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_inverse = scaler_y.inverse_transform(y_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test_inverse, y_pred)
r2 = r2_score(y_test_inverse, y_pred)

# Print evaluation metrics
print(f'Mean Absolute Error: {mae:.2f}')
print(f'R-squared: {r2:.2f}')


# Plotting the results
plt.figure(figsize=(14, 7))
plt.plot(y_test_inverse, label='Actual Prices', color='blue')
plt.plot(y_pred, label='Predicted Prices', color='orange')
plt.title('Actual vs Predicted Prices using LSTM')
plt.xlabel('Sample Index')
plt.ylabel('Stock Price')
plt.legend()
plt.grid()
plt.show()


df_model


# Compute correlation matrix for selected columns
features = ['Close_Lag_1', 'Close_Lag_5', 'Close_Lag_20', 
            'Volume_Lag_1', 'Volume_Lag_5', 'Volume_Lag_20', 
            'EMA', 'MACD', 'Signal Line', 'Upper Band', 'Lower Band']
feature_data = df_model[features]

correlation_matrix = features_data.corr()


# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title(f"Correlation Between Features and target for Stock {7089}")
plt.show()


df_model 


#Interesting error

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Assuming df_model is the DataFrame you created with all the features
# Step 1: Define Features and Target Variable
features = ['Close_Lag_1', 'Close_Lag_5', 'Close_Lag_20', 
            'Volume_Lag_1', 'Volume_Lag_5', 'Volume_Lag_20', 
            'EMA', 'MACD', 'Signal Line', 'Upper Band', 'Lower Band']
X = df_model[features]
y = df_model['Close']  # Target variable

# Step 2: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Model Selection and Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 4: Predictions
y_pred = model.predict(X_test)

# Step 5: Model Evaluation
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'R^2 Score: {r2}')

# Step 6: Visualization of Predictions
plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test, label='Actual Prices', color='blue')
plt.plot(y_test.index, y_pred, label='Predicted Prices', color='red')
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()


print("Columns in df_model:", df_model.columns)


# Check the columns in df_model
print("Columns in df_model:", df_model.columns)

# Define the features based on the actual columns present in df_model
# Adjust the feature names if necessary
features = ['Close_Lag_1', 'Close_Lag_5', 'Close_Lag_20', 
            'Volume_Lag_1', 'Volume_Lag_5', 'Volume_Lag_20', 
            'EMA', 'MACD', 'Signal Line', 'Upper Band', 'Lower Band']

# Check if all features are present in df_model
missing_features = [feature for feature in features if feature not in df_model.columns]
if missing_features:
    print("Missing features:", missing_features)
else:
    # Proceed with defining X and y
    X = df_model[features]
    y = df_model['Close']  # Target variable

    # Continue with the rest of the model implementation...


#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import yfinance as yf

#Data Collection for Stock 7089
ticker = '7089.T'  # Ensure the ticker is correct for the Japanese market
df_model = yf.download(ticker, start='2017-01-04', end='2021-12-03')

#Creating Features
df_model['Close_Lag_1'] = df_model['Close'].shift(1)

df_model['EMA'] = df_model['Close'].ewm(span=20, adjust=False).mean()

df_model.dropna(inplace=True)  # Drop rows with missing values

#Define Features and Target Variable
features = ['Close_Lag_1','EMA']

X = df_model[features]
y = df_model['Close']  # Target variable

#Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Model Selection and Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#Predictions
y_pred = model.predict(X_test)

#Model Evaluation
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'R^2 Score: {r2}')

#Visualization of Predictions
plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test, label='Actual Prices', color='blue')
plt.plot(y_test.index, y_pred, label='Predicted Prices', color='red')
plt.title('Stock Price Prediction for Stock 7089')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()


#XGBOOST

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import yfinance as yf

# Step 1: Data Collection for Stock 7089
ticker = '7089.T'  # Ensure the ticker is correct for the Japanese market
df_model = yf.download(ticker, start='2017-01-04', end='2021-12-03')

# Step 2: Creating Features
df_model['Close_Lag_1'] = df_model['Close'].shift(1)
df_model['Close_Lag_5'] = df_model['Close'].shift(5)
df_model['Close_Lag_20'] = df_model['Close'].shift(20)

df_model['Volume_Lag_1'] = df_model['Volume'].shift(1)
df_model['Volume_Lag_5'] = df_model['Volume'].shift(5)
df_model['Volume_Lag_20'] = df_model['Volume'].shift(20)

df_model['EMA'] = df_model['Close'].ewm(span=20, adjust=False).mean()

df_model['EMA_12'] = df_model['Close'].ewm(span=12, adjust=False).mean()
df_model['EMA_26'] = df_model['Close'].ewm(span=26, adjust=False).mean()
df_model['MACD'] = df_model['EMA_12'] - df_model['EMA_26']
df_model['Signal Line'] = df_model['MACD'].ewm(span=9, adjust=False).mean()

df_model['Middle Band'] = df_model['Close'].rolling(window=20).mean()
df_model['Upper Band'] = df_model['Middle Band'] + (df_model['Close'].rolling(window=20).std() * 2)
df_model['Lower Band'] = df_model['Middle Band'] - (df_model['Close'].rolling(window=20).std() * 2)

df_model.dropna(inplace=True)  # Drop rows with missing values

# Step 3: Define Features and Target Variable
features = ['Close_Lag_1', 'Close_Lag_5', 'Close_Lag_20', 
            'Volume_Lag_1', 'Volume_Lag_5', 'Volume_Lag_20', 
            'EMA', 'MACD', 'Signal Line', 'Upper Band', 'Lower Band']

X = df_model[features]
y = df_model['Close']  # Target variable

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model Selection and Training
model = XGBRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Predictions
y_pred = model.predict(X_test)

# Step 7: Model Evaluation
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'R^2 Score: {r2}')

# Step 8: Visualization of Predictions
plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test, label='Actual Prices', color='blue')
plt.plot(y_test.index, y_pred, label='Predicted Prices', color='red')
plt.title('Stock Price Prediction for Stock 7089 using XGBoost')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()


#Adjusted models : 
#We are going first to eliminate the unnecessary features ()
#Then we are going to adjust for overfitting 
#For random forest, we are going to reduce the number of trees (n_estimators), and limit the depth of the trees (max_depth)
#For the XGBOOST we are going to use regularization parameters like (alpha L1) and (lambda L2) to control complexity
#hyper parameters may not be optimal 
#We could use techniques like grid search or random search 
#key parameters to tune include n_estimators, max_depth, min_samples_split, and min_samples_leaf for Random Forest
#and learning_rate, max_depth, and subsample for XGBoost.
#Or then it might be a problem with the data cleaning 


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Assuming stock_price is your original DataFrame
# Create a copy of the DataFrame to avoid modifying the original data
df_model = stock_price.copy()

# Creating 1-day, 5-day, and 20-day lags for 'Close'
df_model["Close_Lag_1"] = df_model.groupby("SecuritiesCode")["Close"].shift(1)
df_model["Close_Lag_5"] = df_model.groupby("SecuritiesCode")["Close"].shift(5)
df_model["Close_Lag_20"] = df_model.groupby("SecuritiesCode")["Close"].shift(20)

# Drop rows with NaN values after creating lagged features
df_model.dropna(inplace=True)

# Step 1: Select features and target variable
features = ["Close_Lag_1", "Close_Lag_5", "Close_Lag_20"]
X = df_model[features]  # Features
y = df_model["Close"]    # Target variable (Close price)

# Step 2: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Build a Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Step 4: Predictions
rf_predictions = rf_model.predict(X_test)

# Step 5: Model Evaluation
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} - Mean Squared Error: {mse:.2f}, Mean Absolute Error: {mae:.2f}, R^2 Score: {r2:.2f}")

evaluate_model(y_test, rf_predictions, "Random Forest")

# Optional: Visualization of Predictions
plt.figure(figsize=(14, 7))
plt.plot(y_test.index, y_test, color='red', label='Actual Prices', linewidth=2)
plt.plot(y_test.index, rf_predictions, color='blue', label='Random Forest Predictions', linestyle='--')
plt.title('Random Forest Predictions vs Actual Prices for Stock 7089')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.grid()
plt.show()


import matplotlib.pyplot as plt
features = ['Close_Lag_1', 'Close_Lag_5', 'Close_Lag_20', 
            'Volume_Lag_1', 'Volume_Lag_5', 'Volume_Lag_20', 
            'EMA', 'MACD', 'Signal Line', 'Upper Band', 'Lower Band']

# Plot feature importance
importance = model.feature_importances_
feature_names = features
plt.barh(feature_names, importance)
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Feature Importance Plot")
plt.show()

