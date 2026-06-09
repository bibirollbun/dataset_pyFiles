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


df=pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')


df.head()


df.shape


df.info()


null_values = df.isnull().sum()
null_values[null_values>0]


plt.figure(figsize=(30,20))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.3)
plt.title('Correlation Heatmap')
plt.show()


sns.countplot(x='season', data=df)
plt.title('Count of Rentals by Season')
plt.xlabel('Season')
plt.ylabel('Number of Rentals')
plt.xticks(ticks=[0, 1, 2, 3], labels=['Spring', 'Summer', 'Fall', 'Winter'])
plt.show()


plt.figure(figsize=(10, 6))
sns.scatterplot(x='temp', y='count', data=df)
plt.title('Rentals vs. Temperature')
plt.xlabel('Normalized Temperature')
plt.ylabel('Number of Rentals')
plt.show()


df['hour'] = pd.to_datetime(df['datetime']).dt.hour
plt.figure(figsize=(12, 6))
sns.lineplot(x='hour', y='count', data=df, estimator='mean')
plt.title('Average Rentals by Hour of the Day')
plt.xlabel('Hour of the Day')
plt.ylabel('Average Number of Rentals')
plt.xticks(range(24))
plt.show()



plt.figure(figsize=(10, 6))
sns.boxplot(x='weather', y='count', data=df)
plt.title('Rentals Distribution by Weather Condition')
plt.xlabel('Weather Condition')
plt.ylabel('Number of Rentals')
plt.xticks(ticks=[0, 1, 2, 3], labels=['Clear', 'Mist', 'Light Rain', 'Heavy Rain'])
plt.show()


x=df.drop(["count","datetime","registered","casual"],axis=1)
y=df[['count']]


x.head()


def train_regression_model(x, y):
    models = {
        'XGBRegressor': XGBRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
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

mse = mean_squared_error(y_test, predict)**0.5
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, predict)
r2 = r2_score(y_test, predict)
print(f"Mean Squared Error: {mse}, \n Root Mean Squared Error: {rmse}, \n Mean Absolute Error: {mae}, \n R^2 Score: {r2}")



with open('bike_sharing.pkl', 'wb') as file:
    pickle.dump(model, file)


test_df=pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')


test_df.head()


test_df.info()


null_values = test_df.isnull().sum()
null_values[null_values>0]


test_df['hour'] = pd.to_datetime(test_df['datetime']).dt.hour


test_df


pred_x=test_df.drop(["datetime"],axis=1)


predictions = model.predict(pred_x)


predictions


submision = pd.DataFrame()
submision['datetime'] = test_df['datetime']
submision['count'] = predictions


submision.head()


submision.to_csv('submission.csv', index=False)

