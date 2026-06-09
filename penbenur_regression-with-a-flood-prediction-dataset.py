import pandas as pd
import numpy as np 

import seaborn as sns 
import matplotlib.pylab as plt
%matplotlib inline

from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")


df_train=pd.read_csv('/kaggle/input/playground-series-s4e5/train.csv')
df_test=pd.read_csv('/kaggle/input/playground-series-s4e5/test.csv')


df_train.head()


df_train.info()


df_train.shape


df_train.isnull().sum()


df_train.describe()


df_train.corr()


df_train['FloodProbability'].value_counts()


# Convert all int64 columns to int32
df_train[df_train.select_dtypes(include=['int64']).columns] = df_train.select_dtypes(include=['int64']).astype('int32')


# Check for outliers and remove if needed
def remove_outliers(df):
    for column in df.select_dtypes(include=['number']).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
   
    return df

df_train = remove_outliers(df_train)


# Create new column with the sum of all other columns.
df_train['TotalSum'] = df_train.drop(columns=['FloodProbability']).sum(axis=1)


# Visualize the distribution of the target variable
plt.figure(figsize=(10, 6))
sns.histplot(df_train, bins=30, kde=True)
plt.title('Distribution of Flood Probability')
plt.xlabel('Flood Probability')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# Create a correlation heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = df_train.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# Split the dataset into features and target variable
X = df_train.drop(columns=['id','FloodProbability'])
y = df_train['FloodProbability']


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Train a Random Forest model
model = RandomForestRegressor(random_state=42)
model.fit(X_train_scaled, y_train)


# Predictions
y_pred = model.predict(X_test_scaled)


# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')


import joblib
# Save the model and the scaler
joblib.dump(model, 'random_forest_model.pkl')
joblib.dump(scaler, 'scaler.pkl')




