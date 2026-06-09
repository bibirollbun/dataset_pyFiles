import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

pd.set_option('display.max_columns', None)


df_train = pd.read_csv('/kaggle/input/car-price-prediction-khazani-ahmed/train (1).csv')
df_test = pd.read_csv('/kaggle/input/car-price-prediction-khazani-ahmed/test (1).csv')


df_train.head()


df_train.info()


null_counts = df_train.isnull().sum()
null_counts[null_counts > 0]


object_columns = df_train.select_dtypes(include=['object']).columns.tolist()

numerical_columns = df_train.select_dtypes(include=['int64', 'float64']).columns.tolist()


object_columns.remove('CarName')
object_columns


numerical_columns.remove('price')
numerical_columns.remove('car_ID')
numerical_columns


for col in object_columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df_train, x=col)
    plt.title(f'Count of Cars by {col}')
    plt.xticks(rotation=90)
    plt.show()


for col in numerical_columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_train, y=col)
    plt.title(f'Box Plot of {col}')
    plt.ylabel(col)
    plt.grid(True)
    plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(y=df_train['price'])
plt.title('Box Plot of Car Prices')
plt.ylabel('Price')
plt.show()


df_train = df_train[df_train['price'] < 40000]


df_test.head()


df_test.info()


null_counts = df_test.isnull().sum()
null_counts[null_counts > 0]


df_train.symboling.unique()


df_train.fueltype.unique()


df_train.aspiration.unique()


df_train.doornumber.unique()


df_train.doornumber = df_train.doornumber.map({'four': 4, 'two': 2})
df_train.doornumber = df_train.doornumber.astype(int)


df_test.doornumber = df_test.doornumber.map({'four': 4, 'two': 2})
df_test.doornumber = df_test.doornumber.astype(int)


df_train.carbody.unique()


df_train.drivewheel.unique()


df_train.enginelocation.unique()


umnique = df_train.enginetype.unique()


df_train.enginetype.value_counts()


df_train


df_train.cylindernumber.unique()


cylinder_mapping = {
    'four': 4,
    'six': 6,
    'five': 5,
    'two': 2,
    'twelve': 12,
    'eight': 8,
    'three': 3
}
df_train.cylindernumber = df_train.cylindernumber.map(cylinder_mapping)
df_train.cylindernumber = df_train.cylindernumber.astype(int)


df_test.cylindernumber.unique()


df_test.cylindernumber = df_test.cylindernumber.map(cylinder_mapping)
df_test.cylindernumber = df_test.cylindernumber.astype(int)


df_train.fuelsystem.unique()


df_train = df_train.drop(columns=['car_ID','CarName', 'carbody', 'enginelocation',
                                  'enginetype', 'fuelsystem','fueltype','symboling',
                                  'aspiration', 'compressionratio','stroke','enginesize','drivewheel','cylindernumber'])


df_train.columns


object_columns = df_train.select_dtypes(include=['object']).columns.tolist()


object_columns


df_train = pd.get_dummies(df_train, columns=object_columns, drop_first=True)
df_test = pd.get_dummies(df_test, columns=object_columns, drop_first=True)


x = df_train.drop(columns=['price'])
y = df_train['price']


def train_regression_model(x, y):
    model = LinearRegression()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Linear Regression:")
    print(f"  Mean Squared Error: {mse}")
    print(f"  R^2 Score: {r2}")
    print("\n")

    residuals = y_test - y_pred

    plt.figure(figsize=(12, 6))
    sns.histplot(residuals, bins=30, kde=True)
    plt.title('Linear Regression Residuals Distribution')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.axvline(0, color='red', linestyle='--')
    plt.show()


train_regression_model(x,y)

