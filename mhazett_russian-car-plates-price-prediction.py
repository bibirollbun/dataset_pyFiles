import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/train.csv')
df = df.dropna()
df = df.head(99)


print(df.info())
print(df.describe())
print("Missing value:\n", df.isnull().sum())
df['date'] = pd.to_datetime(df['date'])


plt.figure(figsize=(10,20))
plt.scatter(df['date'], df['price'], color='blue')
plt.xlabel('date')
plt.ylabel('price')
plt.title('price and date relationship')
plt.grid(True)


sns.histplot(data=df, x='price', kde=True)
plt.title('prices in histogram')


# feature engineer

df['plate_length'] = df['plate'].str.len()
print("Plate length:\n", df['plate_length'])

lucky_patterns = ['111','222', '333', '444', '555', '666', '777', '888', '999', '000']
df['has_lucky_number'] = df['plate'].apply(lambda x: int(any(p in x for p in lucky_patterns)))
print("Car plates that has lucky number:\n", df['has_lucky_number'])

df['has_repeated_digits'] = df['plate'].apply(lambda x: int(bool(re.search(r'(\d)\1{1,}', x))))
print("Car plates that has repeated digits:\n", df['has_repeated_digits'])

df['is_palindrome'] = df['plate'].apply(lambda x: int(x == x[::-1]))
print("Car plates that has palindrome:\n", df['is_palindrome'])


plt.figure(figsize=(14,10))
plt.plot(df.index, df['plate_length'], color='green', marker='o')
plt.title('all cars plate length')
plt.xlabel('row index')
plt.ylabel('plate length')
plt.grid(True)


plt.figure(figsize=(14,10))
plt.plot(df.index, df['has_lucky_number'], color='green', marker='o')
plt.title('all cars plate lucky number')
plt.xlabel('row index')
plt.ylabel('plate length')
plt.grid(True)


plt.figure(figsize=(14,10))
plt.plot(df.index, df['has_repeated_digits'], color='green', marker='o')
plt.title('all cars plate repeated digits')
plt.xlabel('row index')
plt.ylabel('plate length')
plt.grid(True)


plt.figure(figsize=(14,10))
plt.plot(df.index, df['is_palindrome'], color='green', marker='o')
plt.title('all cars plate palindrome')
plt.xlabel('row index')
plt.ylabel('plate length')
plt.grid(True)


features = ['plate_length', 'has_lucky_number', 'has_repeated_digits', 'is_palindrome']
X = df[features]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("R^2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

