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


df=pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')


df.head()


df.shape


df.info()


null_values = df.isnull().sum()
null_values[null_values>0]


df = df.dropna().reset_index(drop=True)


plt.figure(figsize=(12, 6))
sns.barplot(x='country', y='num_sold', data=df, estimator=sum, palette='viridis')
plt.title('Total Units Sold by Country')
plt.xlabel('Country')
plt.ylabel('Total Units Sold')
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(12, 6))
sns.barplot(x='product', y='num_sold', data=df, estimator=sum, palette='viridis')
plt.title('Total Units Sold by Product')
plt.xlabel('Product')
plt.ylabel('Total Units Sold')
plt.xticks(rotation=45)
plt.show()


df['date'] = pd.to_datetime(df['date'])

plt.figure(figsize=(30, 7))
sns.lineplot(x='date', y='num_sold', data=df, estimator='sum', ci=None)
plt.title('Units Sold Over Time')
plt.xlabel('Date')
plt.ylabel('Total Units Sold')
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(30, 7))
sns.lineplot(x='date', y='num_sold', data=df, estimator='sum', ci=None, hue='country')
plt.title('Units Sold Over Time by Country')
plt.xlabel('Date')
plt.ylabel('Total Units Sold')
plt.xticks(rotation=45)
plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')  # Adjust legend position
plt.show()


plt.figure(figsize=(12, 6))
sns.boxplot(x='product', y='num_sold', data=df, palette='coolwarm')
plt.title('Distribution of Units Sold by Product')
plt.xlabel('Product')
plt.ylabel('Units Sold')
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize = (12, 8))
ax = sns.barplot(data=df, x='country', y='num_sold', hue='product')


df['date'] = pd.to_datetime(df['date'])

df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['isWeekend'] = df['date'].dt.weekday >= 5


x = df.drop(['id','date','num_sold'],axis=1)
x = pd.get_dummies(x, drop_first=True)
y = df[['num_sold']]


x.head()


def train_regression_model(x, y):
    from joblib import Parallel, delayed

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso()
    }

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    def fit_and_evaluate(model_name, model):
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results = {
            'model_name': model_name,
            'mse': mse,
            'r2': r2,
            'feature_importance': None
        }

        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
            importance_df = pd.DataFrame({'Feature': x.columns, 'Importance': feature_importance})
            results['feature_importance'] = importance_df.sort_values(by='Importance', ascending=False)

        return results, y_test, y_pred

    results = Parallel(n_jobs=-1)(delayed(fit_and_evaluate)(name, model) for name, model in models.items())

    for result, y_test, y_pred in results:
        print(f"{result['model_name']}:")
        print(f"  Mean Squared Error: {result['mse']}")
        print(f"  R^2 Score: {result['r2']}\n")

        if result['feature_importance'] is not None:
            print(result['feature_importance'])
            print("\n")

        # Plot residuals for the first model only
        if result['model_name'] == 'Linear Regression':
            residuals = y_test - y_pred
            plt.figure(figsize=(12, 6))
            sns.histplot(residuals, bins=30, kde=True)
            plt.title(f'{result["model_name"]} Residuals Distribution')
            plt.xlabel('Residuals')
            plt.ylabel('Frequency')
            plt.axvline(0, color='red', linestyle='--')
            plt.show()
            print("\n")


train_regression_model(x, y)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(x_train, y_train)

predict = model.predict(x_test)

mse = mean_squared_error(y_test, predict)**0.5
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, predict)
r2 = r2_score(y_test, predict)
print(f"Mean Squared Error: {mse}, \n Root Mean Squared Error: {rmse}, \n Mean Absolute Error: {mae}, \n R^2 Score: {r2}")



with open('sticker_sales.pkl', 'wb') as file:
    pickle.dump(model, file)


test_df=pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


test_df.head()


test_df.info()


null_values = test_df.isnull().sum()
null_values[null_values>0]


test_df['date'] = pd.to_datetime(test_df['date'])

test_df['year'] = test_df['date'].dt.year
test_df['month'] = test_df['date'].dt.month
test_df['day'] = test_df['date'].dt.day
test_df['isWeekend'] = test_df['date'].dt.weekday >= 5


test_df


x = test_df.drop(['id','date'],axis=1)
pred_x = pd.get_dummies(x, drop_first=True)


pred_x.head()


predictions = model.predict(pred_x)


predictions = np.ceil(predictions).astype(int)
predictions


submision = pd.DataFrame()
submision['id'] = test_df['id']
submision['num_sold'] = predictions


submision.head()


submision.to_csv('submission.csv', index=False)

