import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import warnings

import pickle

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.preprocessing import scale

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 100)


df_train=pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
df_train_extra=pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')


df_train.head()


df_train_extra.head()


df_train.shape, df_train_extra.shape


df_train.info()


df_train_extra.info()


null_values = df_train.isnull().sum()
null_values[null_values>0]


null_values = df_train_extra.isnull().sum()
null_values[null_values>0]


df = pd.concat([df_train, df_train_extra], ignore_index=True)


categorical_cols = df.columns[df.dtypes == 'object'].tolist()
numerical_cols = df.columns[df.dtypes != 'object'].tolist()

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

for col in numerical_cols[1:]:
     df[col] = df[col].fillna(df[col].median())


null_values = df.isnull().sum()
null_values[null_values>0]


plt.figure(figsize=(30,20))
corr = df.drop('id',axis=1).corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.3)
plt.title('Correlation Heatmap')
plt.show()


sns.scatterplot(data=df, x='Weight Capacity (kg)', y='Price', color='blue')
plt.title('Price vs. Weight Capacity')
plt.xlabel('Weight Capacity (kg)')
plt.ylabel('Price')
plt.grid(True)
plt.show()


avg_price_by_brand = df.groupby('Brand')['Price'].mean().reset_index()
sns.barplot(data=avg_price_by_brand, x='Brand', y='Price')
plt.title('Average Price by Brand')
plt.xlabel('Brand')
plt.ylabel('Average Price')
plt.show()


sns.histplot(df['Price'], bins=20, color='green', kde=True)
plt.title('Distribution of Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()


sns.boxplot(x='Brand', y='Price', data=df)
plt.title('Price Distribution by Brand')
plt.xlabel('Brand')
plt.ylabel('Price')
plt.show()


def add_features(df):
    df['Weight_per_Compartment'] = df['Weight Capacity (kg)'] / df['Compartments']

    binary_map = {'Yes': 1, 'No': 0}
    df['Laptop Compartment'] = df['Laptop Compartment'].map(binary_map)
    df['Waterproof'] = df['Waterproof'].map(binary_map)

    return df


df = add_features(df)


x=df.drop(['id','Price'],axis=1)
y=df[['Price']]


x = pd.get_dummies(x, drop_first=True)


x.head()


def train_regression_model(x, y):
    models = {
        'Linear Regression': LinearRegression(),
        # 'Random Forest': RandomForestRegressor(),
        # 'Support Vector Regressor': SVR(),
        # 'Decision Tree': DecisionTreeRegressor(),
        # 'K-Neighbors': KNeighborsRegressor(),
        # 'Ridge Regression': Ridge(),
        # 'Lasso Regression': Lasso(),
        # 'Gradient Boosting': GradientBoostingRegressor()
    }

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    for model_name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"{model_name}:")
        print(f"  Mean Squared Error: {mse}")
        print(f"  Root Mean Squared Error: {rmse}")
        print(f"  Mean Absolute Error: {mae}")
        print(f"  R^2 Score: {r2}")
        print("\n")

        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
            importance_df = pd.DataFrame({'Feature': x.columns, 'Importance': feature_importance})
            print(importance_df.sort_values(by='Importance', ascending=False))
            print("\n")

        residuals = y_test - y_pred

        plt.figure(figsize=(12, 6))
        sns.histplot(residuals, bins=30, kde=True)
        plt.title(f'{model_name} Residuals Distribution')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.axvline(0, color='red', linestyle='--')
        plt.show()
        print("\n")


train_regression_model(x, y)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

predict = model.predict(x_test)

mse = mean_squared_error(y_test, predict)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, predict)
r2 = r2_score(y_test, predict)
print(f"Mean Squared Error: {mse}, \n Root Mean Squared Error: {rmse}, \n Mean Absolute Error: {mae}, \n R^2 Score: {r2}")



with open('backpack_price.pkl', 'wb') as file:
    pickle.dump(model, file)


test_df=pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


test_df.head()


test_df.info()


null_values = test_df.isnull().sum()
null_values[null_values>0]


for col in categorical_cols:
    test_df[col] = test_df[col].fillna(test_df[col].mode()[0])

for col in numerical_cols[:-1]:
    test_df[col] = test_df[col].fillna(test_df[col].median())


pred_x=test_df.drop(['id'],axis=1)
pred_x = add_features(pred_x)
pred_x = pd.get_dummies(pred_x, drop_first=True)


predictions = model.predict(pred_x)


submision = pd.DataFrame()
submision['id'] = test_df['id']
submision['Price'] = predictions


submision.head()


submision.to_csv('submission.csv', index=False)

