import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.compose import ColumnTransformer
from keras.layers import SimpleRNN, Dense
from plotly.subplots import make_subplots
from warnings import filterwarnings
filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test= pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')



train.head(10).T


train.describe()


train.info()



missing_values = train.isnull().sum()
print(f"Missing values:\n{missing_values}")


train['date'] = pd.to_datetime(train['date'])
train['num_sold'] = train['num_sold'].fillna(train['num_sold'].median())


train['year'] = train['date'].dt.year
train['month'] = train['date'].dt.month
train['day'] = train['date'].dt.day
train['day_of_week'] = train['date'].dt.dayofweek


test['date'] = pd.to_datetime(test['date'])
test['year'] = test['date'].dt.year
test['month'] = test['date'].dt.month
test['day'] = test['date'].dt.day
test['day_of_week'] = test['date'].dt.dayofweek


X = train.drop(['num_sold', 'date'], axis=1)
y = train['num_sold']


categorical_cols = ['country', 'store', 'product']
numerical_cols = ['year', 'month', 'day', 'day_of_week']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ]
)


X_train = preprocessor.fit_transform(X)

X_test = preprocessor.transform(test.drop(['date'], axis=1))


train['date']


sns.set(style="whitegrid")


plt.figure(figsize=(12, 6))
plt.plot(train['date'], train['num_sold'], color='blue')
plt.title('Original Time Series of Number Sold')
plt.xlabel('Date')
plt.ylabel('Number Sold')
plt.show()



rolling_mean = train['num_sold'].rolling(window=30).mean()
plt.figure(figsize=(12, 6))
plt.plot(train['date'], train['num_sold'], label='Number Sold', color='blue')
plt.plot(train['date'], rolling_mean, label='Rolling Mean', color='red')
plt.title('Rolling Mean of Number Sold')
plt.xlabel('Date')
plt.ylabel('Number Sold')
plt.legend()
plt.show()


rolling_std = train['num_sold'].rolling(window=30).std()
plt.figure(figsize=(12, 6))
plt.plot(train['date'], rolling_std, color='green')
plt.title('Rolling Standard Deviation of Number Sold')
plt.xlabel('Date')
plt.ylabel('Standard Deviation')
plt.show()


plt.figure(figsize=(12, 6))
sns.histplot(train['num_sold'], bins=30, kde=True)
plt.title('Distribution of Number Sold')
plt.xlabel('Number Sold')
plt.ylabel('Frequency')
plt.show()


plt.figure(figsize=(12, 6))
sns.boxplot(x=train['num_sold'])
plt.title('Boxplot of Number Sold')
plt.xlabel('Number Sold')
plt.show()


plt.figure(figsize=(12, 6))
sns.countplot(y='country', data=train, order=train['country'].value_counts().index)
plt.title('Count of Sales per Country')
plt.xlabel('Count')
plt.ylabel('Country')
plt.show()



plt.figure(figsize=(12, 6))
sns.countplot(y='store', data=train, order=train['store'].value_counts().index)
plt.title('Count of Sales per Store')
plt.xlabel('Count')
plt.ylabel('Store')
plt.show()


plt.figure(figsize=(12, 6))
sns.countplot(y='product', data=train, order=train['product'].value_counts().index)
plt.title('Count of Sales per Product')
plt.xlabel('Count')
plt.ylabel('Product')
plt.show()


plt.figure(figsize=(12, 6))
for country in train['country'].unique():
    subset = train[train['country'] == country]
    plt.plot(subset['date'], subset['num_sold'], label=country)
plt.title('Time Series of Number Sold by Country')
plt.xlabel('Date')
plt.ylabel('Number Sold')
plt.legend()
plt.show()


plt.figure(figsize=(12, 6))
for store in train['store'].unique():
    subset = train[train['store'] == store]
    plt.plot(subset['date'], subset['num_sold'], label=store)
plt.title('Time Series of Number Sold by Store')
plt.xlabel('Date')
plt.ylabel('Number Sold')
plt.legend()
plt.show()


plt.figure(figsize=(12, 6))
for product in train['product'].unique():
    subset = train[train['product'] == product]
    plt.plot(subset['date'], subset['num_sold'], label=product)
plt.title('Time Series of Number Sold by Product')
plt.xlabel('Date')
plt.ylabel('Number Sold')
plt.legend()
plt.show()


plt.figure(figsize=(12, 6))
plot_acf(train['num_sold'], lags=30)
plt.title('Autocorrelation Function')
plt.show()


plt.figure(figsize=(12, 6))
plot_pacf(train['num_sold'], lags=30)
plt.title('Partial Autocorrelation Function')
plt.show()


decomposition = seasonal_decompose(train['num_sold'], model='additive', period=30)
fig = decomposition.plot()
fig.set_size_inches(12, 10)
plt.show()


adf_result = adfuller(train['num_sold'])
print('ADF Statistic:', adf_result[0])
print('p-value:', adf_result[1])


plt.figure(figsize=(12, 6))
plt.plot(train['date'], train['num_sold'], label='Number Sold', color='blue')
plt.plot(train['date'], rolling_mean, label='Rolling Mean', color='red')
plt.title('Combined Time Series and Rolling Mean')
plt.xlabel('Date')
plt.ylabel('Number Sold')
plt.legend()
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(train['date'], train['num_sold'], label='Number Sold', color='blue')
plt.plot(train['date'], rolling_std, label='Rolling Std', color='green')
plt.title('Combined Time Series and Rolling Std')
plt.xlabel('Date')
plt.ylabel('Number Sold')
plt.legend()
plt.show()


plt.figure(figsize=(12, 6))
plt.hist(train['num_sold'], bins=30, alpha=0.5, label='Number Sold', color='blue')
plt.axvline(rolling_mean.mean(), color='red', linestyle='dashed', linewidth=1, label='Mean')
plt.title('Histogram with Mean Line')
plt.xlabel('Number Sold')
plt.ylabel('Frequency')
plt.legend()
plt.show()


plt.figure(figsize=(12, 6))
sns.boxplot(x='country', y='num_sold', data=train)
plt.title('Boxplot of Number Sold by Country')
plt.ylabel('Number Sold')
plt.xlabel('Country')
plt.show()


X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))



from keras.callbacks import EarlyStopping, ReduceLROnPlateau
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=100))
model.add(Dropout(0.2))
model.add(Dense(units=1))


model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()


early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

history = model.fit(X_train, y, epochs=10, batch_size=32, 
                    callbacks=[early_stopping, reduce_lr])


test_predictions = model.predict(X_test)


test_predictions = test_predictions.flatten()  


import pandas as pd
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')
if 'num_sold' in sample_submission.columns:
    sample_submission = sample_submission.drop(['num_sold'], axis=1)
else:
    print("Column 'num_sold' not found in the DataFrame.")
sample_submission = sample_submission.drop(['num_sold'], axis=1, errors='ignore')
print(sample_submission.head())



sample_submission['num_sold'] = test_predictions



test.to_csv("sample_submission.csv.csv", index=False)
print("submission saved!")


sample_submission


import pandas as pd

# اقرأ الملف
sample_submission = pd.read_csv('/kaggle/working/sample_submission.csv.csv')
sample_submission['num_sold'] = test_predictions
# احتفظ فقط بالأعمدة المطلوبة
sample_submission = sample_submission [['id', 'num_sold']]

# احفظ الملف الجديد
sample_submission .to_csv("cleaned_submission.csv", index=False)


sample_submission = pd.read_csv('/kaggle/working/cleaned_submission.csv')


sample_submission 

