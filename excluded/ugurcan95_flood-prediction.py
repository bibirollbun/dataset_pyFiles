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
from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.preprocessing import scale

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 100)


df=pd.read_csv('/kaggle/input/playground-series-s4e5/train.csv')


df.head()


df.shape


df.info()


null_values = df.isnull().sum()
null_values[null_values>0]


plt.figure(figsize=(30,20))
corr = df.drop('id',axis=1).corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.3)
plt.title('Correlation Heatmap')
plt.show()


plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='MonsoonIntensity', y='FloodProbability', hue='Urbanization', palette='viridis')
plt.title('Monsoon Intensity vs. Flood Probability')
plt.xlabel('Monsoon Intensity')
plt.ylabel('Flood Probability')
plt.legend(title='Urbanization Level')
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='TopographyDrainage', y='FloodProbability', palette='Set3')
plt.title('Flood Levels by Topography Drainage Type')
plt.xlabel('Topography Drainage Type')
plt.ylabel('Flood Probability')
plt.show()


plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x='AgriculturalPractices', y='FloodProbability', palette='muted')
plt.title('Flood Probability Distribution by Agricultural Practices')
plt.xlabel('Agricultural Practices')
plt.ylabel('Flood Probability')
plt.show()


plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Urbanization', y='FloodProbability', hue='Deforestation', palette='coolwarm')
plt.title('Urbanization vs. Flood Probability')
plt.xlabel('Urbanization Level')
plt.ylabel('Flood Probability')
plt.legend(title='Deforestation Level')
plt.show()


plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='DamsQuality', y='FloodProbability', hue='ClimateChange', palette='plasma')
plt.title('Dams Quality vs. Flood Probability')
plt.xlabel('Dams Quality Rating')
plt.ylabel('Flood Probability')
plt.legend(title='Climate Change Impact')
plt.show()


x=df.drop(['id','FloodProbability'],axis=1)
y=df[['FloodProbability']]


x.head()


def train_regression_model(x, y):
    if isinstance(y, pd.DataFrame):
        y = y.squeeze()


    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'XGBoost': XGBRegressor()
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
        print(f"  R^2 Score: {r2}\n")

        # Plot residuals
        residuals = y_test - y_pred
        plt.figure(figsize=(12, 6))
        sns.histplot(residuals, bins=30, kde=True)
        plt.title(f'{model_name} Residuals Distribution')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.axvline(0, color='red', linestyle='--')
        plt.show()


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



with open('flood_prediction.pkl', 'wb') as file:
    pickle.dump(model, file)


test_df=pd.read_csv('/kaggle/input/playground-series-s4e5/test.csv')


test_df.head()


test_df.info()


null_values = test_df.isnull().sum()
null_values[null_values>0]


pred_x=test_df.drop(['id'],axis=1)


predictions = model.predict(pred_x)


submision = pd.DataFrame()
submision['id'] = test_df['id']
submision['FloodProbability'] = predictions


submision.head()


submision.to_csv('submission.csv', index=False)

