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


df=pd.read_csv('/kaggle/input/playground-series-s3e25/train.csv')


df.head()


df.shape


df.info()


null_values = df.isnull().sum()
null_values[null_values>0]


plt.figure(figsize=(10,8))
corr = df.drop('id',axis=1).corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.3)
plt.title('Correlation Heatmap')
plt.show()


plt.figure(figsize=(10, 5))
sns.scatterplot(x='density_Total', y='allelectrons_Total', data=df, hue='Hardness', size='Hardness', sizes=(20, 200), palette='viridis', legend=None)
plt.title('Total Electrons vs. Density Total')
plt.xlabel('Density Total')
plt.ylabel('Total Electrons')
plt.show()


plt.figure(figsize=(30, 10))
sns.lineplot(x='density_Total', y='val_e_Average', data=df, marker='o')
plt.title('Average Valence Electrons by Density Total')
plt.xlabel('Density Total')
plt.ylabel('Average Valence Electrons')
plt.show()


plt.figure(figsize=(10, 5))
sns.scatterplot(
    x='density_Total',
    y='atomicweight_Average',
    data=df,
    hue='allelectrons_Total',
    palette='deep',
    size='Hardness',
    sizes=(20, 200),
    legend=False  # Disable legend
)
plt.title('Density Total vs. Average Atomic Weight')
plt.xlabel('Density Total')
plt.ylabel('Average Atomic Weight')
plt.show()


sns.histplot(df['Hardness'], bins=20, kde=True, color='blue')
plt.title('Distribution of Hardness')
plt.xlabel('Hardness')
plt.ylabel('Frequency');


def add_features(df):
    df = df[df['density_Total']<500]
    df = df[df['allelectrons_Total']<9000]

    return df


df = add_features(df)


x=df.drop(['id','Hardness'],axis=1)
y=df[['Hardness']]


x.head()


def train_regression_model(x, y):
    models = {
        'XGBRegressor': XGBRegressor(),
        'Random Forest': RandomForestRegressor(),
        'Gradient Boosting': GradientBoostingRegressor(),
        'Decision Tree': DecisionTreeRegressor(),
        'K-Neighbors': KNeighborsRegressor(),
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'Support Vector Regressor': SVR()
    }

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    results = []

    # Configure matplotlib to prevent text output
    plt.ioff()
    original_backend = plt.get_backend()
    plt.switch_backend('Agg')  # Switch to non-interactive backend

    try:
        for model_name, model in models.items():
            fig = None
            try:
                # Model training and prediction
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)

                # Store metrics
                results.append({
                    'Model': model_name,
                    'RÂ² Score': r2_score(y_test, y_pred),
                    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'MAE': mean_absolute_error(y_test, y_pred),
                    'MSE': mean_squared_error(y_test, y_pred)
                })

                # Feature importances
                if hasattr(model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'Feature': x.columns,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)

                    print(f"\nğŸ”� {model_name} Feature Importances:")
                    display(importance_df.style.format({'Importance': '{:.6f}'}).background_gradient(cmap='Blues'))

                # Create and display plot
                fig = plt.figure(figsize=(10, 5))
                sns.histplot(y_test - y_pred, bins=30, kde=True, color='skyblue')
                plt.axvline(0, color='red', linestyle='--')
                plt.title(f'{model_name} Residuals')
                plt.show()

            except Exception as e:
                print(f"")
            finally:
                # Ensure figure is always closed
                if fig is not None:
                    plt.close(fig)
                    del fig

    finally:
        plt.switch_backend(original_backend)
        plt.ion()

    # Display final results
    results_df = pd.DataFrame(results).sort_values('RÂ² Score', ascending=False)

    print("\n" + "="*60)
    print("ğŸ�† FINAL MODEL COMPARISON")
    print("="*60)

    display(results_df.style
           .background_gradient(subset=['RÂ² Score'], cmap='Greens')
           .format({
               'RÂ² Score': '{:.4f}',
               'RMSE': '{:.2f}',
               'MAE': '{:.2f}',
               'MSE': '{:.2f}'
           }))


train_regression_model(x, y)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(x_train, y_train)

predict = model.predict(x_test)

mse = mean_squared_error(y_test, predict)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, predict)
r2 = r2_score(y_test, predict)
print(f"Mean Squared Error: {mse}, \n Root Mean Squared Error: {rmse}, \n Mean Absolute Error: {mae}, \n R^2 Score: {r2}")



with open('mohs_hardness.pkl', 'wb') as file:
    pickle.dump(model, file)


test_df=pd.read_csv('/kaggle/input/playground-series-s3e25/test.csv')


test_df.head()


test_df.info()


null_values = test_df.isnull().sum()
null_values[null_values>0]


pred_x=test_df.drop(['id'],axis=1)


predictions = model.predict(pred_x)


submision = pd.DataFrame()
submision['id'] = test_df['id']
submision['Hardness'] = predictions


submision.tail()


submision.shape


submision.to_csv('submission.csv', index=False)

