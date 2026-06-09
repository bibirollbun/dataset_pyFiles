# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_path = "/kaggle/input/playground-series-s3e1/train.csv"
test_path = "/kaggle/input/playground-series-s3e1/test.csv"
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
print("Train Data:")
print(train_df.head())
print("\nTest Data:")
print(test_df.head())


from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
print("Libraries imported successfully!")


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
print("Evaluation metric libraries imported successfully!")


def evaluate_models(X, y):
   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
        "ElasticNet": ElasticNet(),
        "Extra Trees": ExtraTreesRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "XGBoost": XGBRegressor(),
        "Decision Tree": DecisionTreeRegressor(),
        "K-Neighbors": KNeighborsRegressor(),
        "Support Vector Regression": SVR()
    }
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        results.append([name, mse, r2, mae])
    results_df = pd.DataFrame(results, columns=["Model", "MSE", "R² Score", "MAE"])    
    return results_df


numerical_features = train_df.select_dtypes(include=['int64', 'float64'])
X = numerical_features.drop(columns=['MedHouseVal'])
y = train_df['MedHouseVal']
results_table = evaluate_models(X, y)
print(results_table)


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
numerical_features_imputed = imputer.fit_transform(train_df.select_dtypes(include=['int64', 'float64']))
numerical_features_imputed_df = pd.DataFrame(numerical_features_imputed, columns=train_df.select_dtypes(include=['int64', 'float64']).columns)
train_df[train_df.select_dtypes(include=['int64', 'float64']).columns] = numerical_features_imputed_df
print(train_df.isnull().sum())


from sklearn.preprocessing import MinMaxScaler
numerical_features = train_df.select_dtypes(include=['int64', 'float64'])
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(numerical_features)
scaled_features_df = pd.DataFrame(scaled_features, columns=numerical_features.columns)
train_df[numerical_features.columns] = scaled_features_df
print(train_df.describe())


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
numerical_features = train_df.select_dtypes(include=['int64', 'float64'])
scaler = StandardScaler()
scaled_features = scaler.fit_transform(numerical_features)
pca = PCA(n_components=5)  # You can change the number of components here (e.g., 2 for 2D visualization)
pca_features = pca.fit_transform(scaled_features)
pca_features_df = pd.DataFrame(pca_features, columns=[f'PC{i+1}' for i in range(pca.n_components_)])
print("Explained Variance Ratio:", pca.explained_variance_ratio_)
print("Cumulative Explained Variance:", pca.explained_variance_ratio_.cumsum())
train_df_pca = pd.concat([train_df, pca_features_df], axis=1)
print(train_df_pca.head())

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.scatter(pca_features[:, 0], pca_features[:, 1])
plt.title('PCA - First Two Principal Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_scaled)
print("Explained Variance Ratio:", pca.explained_variance_ratio_)
print("Cumulative Explained Variance:", pca.explained_variance_ratio_.cumsum())
results_table = evaluate_models(X_pca, y)
print(results_table)
best_model = results_table.loc[results_table['R² Score'].idxmax()]
print("Best Model:", best_model)


from xgboost import XGBRegressor
X = train_df.drop(columns=['MedHouseVal', 'id'])
y = train_df['MedHouseVal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

model = XGBRegressor(random_state=42)
model.fit(X_train_pca, y_train)

X_test_full = test_df.drop(columns=['id'])
X_test_full_scaled = scaler.transform(X_test_full)
X_test_full_pca = pca.transform(X_test_full_scaled)
predictions = model.predict(X_test_full_pca)
output = pd.DataFrame({
    'Id': test_df['id'],
    'MedHouseVal': predictions
})
output.to_csv('predictions.csv', index=False)
print("Predictions saved to 'predictions.csv'")

