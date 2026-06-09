import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')


df.info()


df.head()


df.isna().sum()


df.dropna(inplace=True)


df.info()


from sklearn.preprocessing import LabelEncoder


encoder = LabelEncoder()


df['Brand'] = encoder.fit_transform(df['Brand'])
df['Material'] = encoder.fit_transform(df['Material'])
df['Size'] = encoder.fit_transform(df['Size'])
df['Laptop Compartment'] = encoder.fit_transform(df['Laptop Compartment'])
df['Waterproof'] = encoder.fit_transform(df['Waterproof'])
df['Style'] = encoder.fit_transform(df['Style'])
df['Color'] = encoder.fit_transform(df['Color'])

print(df)


df.info()


X = df.drop(['Price','id'], axis=1)
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
train_data = X_train.join(y_train)


train_data.hist(figsize=(15, 8))


plt.figure(figsize=(15,8))
sns.heatmap(train_data.corr(numeric_only=True), annot=True, cmap="YlGnBu")


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train, y_train = train_data.drop(['Price'], axis=1), train_data['Price']
X_train_s = scaler.fit_transform(X_train)

reg = LinearRegression()

reg.fit(X_train, y_train)


reg.score(X_test, y_test)


from sklearn.metrics import mean_squared_error

y_pred = reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Model RMSE:", rmse)


y_pred


df_test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
df_test


df_test.info()


encoder = LabelEncoder()


df_test['Brand'] = encoder.fit_transform(df_test['Brand'])
df_test['Material'] = encoder.fit_transform(df_test['Material'])
df_test['Size'] = encoder.fit_transform(df_test['Size'])
df_test['Laptop Compartment'] = encoder.fit_transform(df_test['Laptop Compartment'])
df_test['Waterproof'] = encoder.fit_transform(df_test['Waterproof'])
df_test['Style'] = encoder.fit_transform(df_test['Style'])
df_test['Color'] = encoder.fit_transform(df_test['Color'])

print(df_test)


df_test.isna().sum()


df_test['Weight Capacity (kg)'] = df_test['Weight Capacity (kg)'].fillna(df_test['Weight Capacity (kg)'].mean())


df_test.isna().sum()


ids = df_test['id']  
df_test.drop(columns=['id'], inplace=True)
df_test.info()


df_test


predicted_prices = reg.predict(df_test)
predicted_prices


results = pd.DataFrame({'id': ids, 'Price': predicted_prices})
results.to_csv("predicted_prices.csv", index=False)
print("Predictions saved successfully!")




