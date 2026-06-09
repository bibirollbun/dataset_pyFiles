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


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier  # or DecisionTreeRegressor
from sklearn import metrics
import warnings
warnings.simplefilter(action='ignore')

from sklearn.preprocessing import MinMaxScaler

#read the local csv files
file_path_train = '/kaggle/input/playground-series-s4e4/train.csv'
df_train = pd.read_csv(file_path_train)

file_path_test = '/kaggle/input/playground-series-s4e4/test.csv'
df_test = pd.read_csv(file_path_test)


# Count missing values in DataFrames
print("Train: ", df_train.isnull().sum())
print("Test: ", df_test.isnull().sum())


print(df_train.head())


print(df_test.head())


#describe the dataset details
df_train.info()


#describe the dataset details
df_test.info()


#describing statistical info of dataset: number of records, mean, STD and five-number summary for all numerical data
df_train.describe()


df_test.describe()


#Convert the Sex attribute into numerical categories.
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_train['Sex'] = le.fit_transform(df_train['Sex'])
df_test['Sex'] = le.transform(df_test['Sex'])


df_train.describe()


df_train=df_train.reindex()


df_train.describe().T


df_test.describe().T


#Heatmap for classes/categories correlation undestanding
# Correlation heatmap for train dataset
numeric_df = df_train.select_dtypes(include=[np.number])
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


#Data Splitting for training
from sklearn.model_selection import train_test_split
X = df_train.drop(columns=['Rings'])
y = df_train['Rings']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#undestanding what we are getting here
print('X_train:', X_train.shape)
print('X_test: ', X_test.shape)
print('y_train:', y_train.shape)
print('y_test: ', y_test.shape)


from sklearn.ensemble import RandomForestRegressor # will use regressor not classified
#rf_model = RandomForestRegressor() #not a RandomForestClassifier()
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
#random_state=42 , if not resulst can vary between runs, n_estimator - number of decisiosn tree in estimator
rf_model.fit(X_train, y_train)


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Predictions from the regressor, used in graph, use test of validation X
rf_pred = rf_model.predict(X_train)


#Validation
# Calculate Metrics
rf_mse = mean_squared_error(y_train, rf_pred)
rf_mae = mean_absolute_error(y_train, rf_pred)
rf_r2 = r2_score(y_train, rf_pred)


# Evaluate the regression model

print("Mean Squared Error (MSE):", rf_mse) #Measures the average squared difference between predictions and actual values. Large errors penalized more because of square
print("Mean Absolute Error (MAE):", rf_mae) #Measures the average magnitude of errors in predictions. Provides an interpretable error in the same units as the target variable
print("RÂ² Score:", rf_r2) #Indicates the proportion of variance explained by the model (ranges from -âˆ� to 1). r2=1 - Perfect prediction, r2=0 -Model performs no better than predicting the mean., r2<0 Model performs worse than predicting the mean



#Visualization

#Visualizing feature importance helps to understand which features contribute the most to the predictions.

import numpy as np

# Get feature importances
importances = rf_model.feature_importances_
feature_names = X.columns

# Sort and plot
sorted_idx = np.argsort(importances)
plt.figure(figsize=(8, 6))
plt.barh(range(len(sorted_idx)), importances[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), feature_names[sorted_idx])
plt.xlabel('Feature Importance')
plt.title('Feature Importance in RandomForestRegressor')
plt.show()



# plot actual vs. predicted values as a line plot for better visualization of trends.
# Line plot of actual vs predicted
plt.figure(figsize=(20, 6))
plt.plot(range(len(y_train)), y_train, label='Actual Values', color='blue', marker='o')
plt.plot(range(len(rf_pred)), rf_pred, label='Predicted Values', color='orange', linestyle='--', marker='x')
plt.xlabel('Abalones')
plt.ylabel('Size')
plt.title('Actual vs. Predicted Ring sizes in RandomForestRegression Model')
plt.legend()
plt.show()


# Create residuals visualization
rf_residuals = y_train - rf_pred

plt.figure(figsize=(20, 6))
plt.scatter(rf_pred, rf_residuals, alpha=0.6, color='green')
plt.axhline(0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals Plot in RandomForestRegression Model')
plt.show()


#Another visual for actual vs predicted

import matplotlib.pyplot as plt

plt.figure(figsize=(20, 6))
plt.scatter(y_train, rf_pred, alpha=0.6, color='blue')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', linewidth=2)  # Perfect fit line
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Ring size in RandomForestRegression Model')
plt.show()


#Data Splitting
from sklearn.model_selection import train_test_split
X = df_train.drop(columns=['Rings'])
y = df_train['Rings']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Choose a Regressor:
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
lr_model_fit = lr_model.fit(X_train, y_train)


# Make predictions
lr_pred = lr_model.predict(X_train)



#Validation mentrics
#Let's find RÂ² score - how much the criterion data can be expleined by predictor (closer to 1 is better fit)
lr_r2_score = lr_model.score(X_train,y_train)
lr_mae = mean_absolute_error(y_train, lr_pred)
lr_mse = mean_squared_error(y_train, lr_pred)

print("Mean Squared Error (MSE):", lr_mse) #Measures the average squared difference between predictions and actual values. Large errors penalized more because of square
print("Mean Absolute Error (MAE):", lr_mae) #Measures the average magnitude of errors in predictions. Provides an interpretable error in the same units as the target variable
print("RÂ² Score:", lr_r2_score) #Indicates the proportion of variance explained by the model (ranges from -âˆ� to 1). r2=1 - Perfect prediction, r2=0 -Model performs no better than predicting the mean., r2<0 Model performs worse than predicting the mean



#Visualization
# plot actual vs. predicted values as a line plot for better visualization of trends.
# Line plot of actual vs predicted
plt.figure(figsize=(20, 6))
plt.plot(range(len(y_train)), y_train, label='Actual Values', color='blue', marker='o')
plt.plot(range(len(lr_pred)), lr_pred, label='Predicted Values', color='orange', linestyle='--', marker='x')
plt.xlabel('Abalones')
plt.ylabel('Size')
plt.title('Actual vs. Predicted Ring sizes in LinearRegression model')
plt.legend()
plt.show()


# Calculate residuals
lr_residuals = y_train - lr_pred

plt.figure(figsize=(20, 6))
plt.scatter(lr_pred, lr_residuals, alpha=0.6, color='green')
plt.axhline(0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals Plot in LinearRegression Model')
plt.show()


import matplotlib.pyplot as plt

plt.figure(figsize=(20, 6))
plt.scatter(y_train, lr_pred, alpha=0.6, color='blue')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', linewidth=2)  # Perfect fit line
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Ring size in LinearRegression Model')
plt.show()


# Summary Table

metrics_data = {
    "Model": ["Random Forest", "Linear Regression", ],
    "Mean Squared Error(MSE) ": [
        rf_mse,
        lr_mse,
    ],
    "Mean Absolute Error(MAE) ": [
        rf_mae,
        lr_mae,
    ],
    "RÂ² Score:                ": [
        rf_r2,
        lr_r2_score,
    ],
}

# Create a DataFrame
metrics_df = pd.DataFrame(metrics_data)

#Summary of findings

print("Random Forest Model shows better predicting results than Linear Regression Model based on 'Actual vs. Predicted Ring sizes' visualization and metric below")
print("and Random Forest Model will be preffered for this task ")
print("MSE - average of squred difference between predictions and actual values. Large errors penalized more because of square")
print("MAE - average magnitude of errors in predictions. Provides an interpretable error in the same units as the target variable")
print("MSE and MAE with lower value demostrate a better model performcance.")
print("RÂ² Score - proportion of variance explained by the model (ranges from -âˆ� to 1). r2=1 - Perfect prediction, r2=0 -Model performs no better than predicting the mean.,")
print("r2<0 Model performs worse than predicting the mean")
# Display the table
print("")
print("SUMMARY METRICS TABLE:")
print(metrics_df)


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import datasets 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import Ridge, Lasso, ElasticNet 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


numeric_df = df_train.select_dtypes(include=[np.number])
y = numeric_df['Rings']
X = numeric_df.drop(columns=['Rings']) #all 9 components
scaler=StandardScaler() 


# Train-Test Split same as with Linera regression before: 20/80 ttest/train size;  random_state = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize Models
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.1)
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)


# Train, Evaluate and Compare results metrics for ridge, lasso, elastic
for model in [ridge, lasso, elastic]:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{model.__class__.__name__} MSE: {mse:.2f}")
    print(f"{model.__class__.__name__} MAE: {mae:.2f}")
    print(f"{model.__class__.__name__} r2: {r2:.2f}")


#Another alternative way to Calculate, Evaluate and Compare ridge, lasso, elastic. MSE is same.
mycolumns=X_train.columns
#Run Models with ridge, lasso, elastic 
for model in [ridge, lasso, elastic]:
    model.fit(X_train,y_train) 
    y_pred = model.predict(X_test) 
    MSE=np.mean((y_pred - y_test)**2) 
    MSE=np.mean((y_pred - y_test)**2) 
    print()
    print(f"{model.__class__.__name__} MSE: {MSE:.2f}")
    mod1_coeff = pd.DataFrame() 
    #mod1_coeff[f"{model} Columns"]= mycolumns
    mod1_coeff["Columns"]= mycolumns 
    mod1_coeff['Coefficient Estimate'] = pd.Series(model.coef_) 
    print(mod1_coeff)


from sklearn.decomposition import PCA 
from sklearn import datasets, linear_model 
regr = linear_model.LinearRegression() 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Correlation heatmap
numeric_df = df_train.select_dtypes(include=[np.number])
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


y = numeric_df['Rings']
X = numeric_df.drop(columns=['Rings']) #all 9 components
X


# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Apply PCA
components = 4
#pca = PCA() #default -all components
pca = PCA(n_components=components) # top 2-9 principal components can be selected (from 9 total)
X_pca = pca.fit_transform(X_scaled)


# Train-Test Split same as with Linera regression before: 20/80 ttest/train size;  random_state = 42
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Linear Regression on Principal Components
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predictions
y_pred = regressor.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"number of components: {components}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"RÂ² Score: {r2:.2f}")

