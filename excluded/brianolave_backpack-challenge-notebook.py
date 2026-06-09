import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


train_df.head()


test_df.head()


# EDA
print(train_df.info())
print(train_df.describe())
print(train_df.isnull().sum())


# Visualize missing values
sns.heatmap(train_df.isnull(), cbar=False, cmap='viridis')
plt.show()


# Handling missing values and categorical encoding
categorical_features = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
numerical_features = ['Compartments', 'Weight Capacity (kg)']

ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
scaler = StandardScaler()
imp_cat = SimpleImputer(strategy='most_frequent')
imp_num = SimpleImputer(strategy='mean')

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', imp_num), ('scaler', scaler)]), numerical_features),
    ('cat', Pipeline([('imputer', imp_cat), ('ohe', ohe)]), categorical_features)
])



# Splitting dataset
X = train_df.drop(columns=['id', 'Price'])
y = train_df['Price']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Baseline model - Linear Regression
lr = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

lr.fit(X_train, y_train)
y_pred = lr.predict(X_val)
print(f'Linear Regression RMSE: {np.sqrt(mean_squared_error(y_val, y_pred))}')


# Random Forest Model
rf = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_val)
print(f'Random Forest RMSE: {np.sqrt(mean_squared_error(y_val, y_pred_rf))}')


# Generate predictions for test set
X_test = test_df.drop(columns=['id'])
test_predictions = rf.predict(X_test)

# Prepare submission
submission = pd.DataFrame({'id': test_df['id'], 'Price': test_predictions})
submission.to_csv('submission.csv', index=False)

