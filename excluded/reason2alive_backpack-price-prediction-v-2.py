# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Additional libraries
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import warnings
warnings.simplefilter('ignore')


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#files loading
test_df = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv', index_col='id')
train_df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv', index_col='id')
extra_df = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv', index_col='id')


test_df.isnull().sum().sort_values()


train_df.isnull().sum().sort_values()


extra_df.isnull().sum().sort_values()


train_df.describe()


extra_df.describe()


for col in train_df:
    print(f'{col} : {train_df[col].nunique()} values\n{train_df[col].unique()}\n')


plt.figure(figsize=(20,30))

plt.subplot(4,2,1)
sns.boxplot(data=train_df, x='Price', y='Brand', hue='Material', palette='coolwarm')
plt.title('Price distribution by Brand')

plt.subplot(4,2,2)
sns.boxplot(data=train_df, x='Price', y='Brand', hue='Size', palette='coolwarm')
plt.title('Price distribution by Material')

plt.subplot(4,2,3)
sns.boxplot(data=train_df, x='Price', y='Brand', hue='Compartments', palette='coolwarm')
plt.title('Price distribution by Compartments')

plt.subplot(4,2,4)
sns.boxplot(data=train_df, x='Price', y='Brand', hue='Style', palette='coolwarm')
plt.title('Price distribution by Bag style')

plt.subplot(4,2,5)
sns.boxplot(data=train_df, x='Price', y='Brand', hue='Color', palette='coolwarm')
plt.title('Price distribution by Color')

plt.subplot(4,2,6)
sns.boxplot(data=train_df, x='Price', y='Brand', hue='Waterproof', palette='coolwarm')
plt.title('Price distribution by Waterproof availability')

plt.subplot(4,2,7)
sns.boxplot(data=train_df, x='Price', y='Brand', hue='Laptop Compartment', palette='coolwarm')
plt.title('Price distribution by Laptop compartment availability')

plt.tight_layout()
plt.show()


plt.figure(figsize=(20,30))

plt.subplot(4,2,1)
sns.boxplot(data=extra_df, x='Price', y='Brand', hue='Material', palette='coolwarm')
plt.title('Price distribution by Brand')

plt.subplot(4,2,2)
sns.boxplot(data=extra_df, x='Price', y='Brand', hue='Size', palette='coolwarm')
plt.title('Price distribution by Material')

plt.subplot(4,2,3)
sns.boxplot(data=extra_df, x='Price', y='Brand', hue='Compartments', palette='coolwarm')
plt.title('Price distribution by Compartments')

plt.subplot(4,2,4)
sns.boxplot(data=extra_df, x='Price', y='Brand', hue='Style', palette='coolwarm')
plt.title('Price distribution by Bag style')

plt.subplot(4,2,5)
sns.boxplot(data=extra_df, x='Price', y='Brand', hue='Color', palette='coolwarm')
plt.title('Price distribution by Color')

plt.subplot(4,2,6)
sns.boxplot(data=extra_df, x='Price', y='Brand', hue='Waterproof', palette='coolwarm')
plt.title('Price distribution by Waterproof availability')

plt.subplot(4,2,7)
sns.boxplot(data=extra_df, x='Price', y='Brand', hue='Laptop Compartment', palette='coolwarm')
plt.title('Price distribution by Laptop compartment availability')

plt.tight_layout()
plt.show()


#Need function to clean data, encode data, and evaluate model
def cln(df):
    cat_columns = [col for col in df.columns if df.dtypes[col] == 'object']
    num_columns = [col for col in df.columns if df.dtypes[col] == 'float64']

    for col in cat_columns:
        df[col] = df[col].fillna('unknown')

    for col in num_columns:
        df[col] = df[col].fillna(df[col].mean())
    return df

def enc(df):
    cat_columns = [col for col in df.columns if df.dtypes[col] == 'object']
    lab_enc = {}
    for col in cat_columns:
        lab_enc[col] = LabelEncoder()
        df[col] = lab_enc[col].fit_transform(df[col])        
    return df
    
def eva(y_test, y_predict, model):
    rmse = np.sqrt(mean_squared_error(y_test, y_predict))
    r2 = r2_score(y_test, y_predict)
    print(f'{model}\nRSME : {rmse:.4f}\nR2 score : {r2:.4f}')


#combine train data and train extra to train model
combined_df = pd.concat([train_df, extra_df], ignore_index=True)
combined_df = cln(combined_df)
print(combined_df.head())

#encode the data for model
encoded_df = enc(combined_df)
print(encoded_df.head())
print(encoded_df.isnull().sum().sort_values())


#split the data
X = encoded_df.iloc[:, :-1].values
y = encoded_df.iloc[:,-1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

print(f'[X shape]\nX_train : {X_train.shape}\nX_test : {X_test.shape}')
print(f'[Y shape]\ny_train : {y_train.shape}\ny_test : {y_test.shape}')


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#Train linear regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)


#Evaluate model
y_pred_lin = lin_reg.predict(X_test_scaled)
eva(y_test, y_pred_lin, '[Linear Regression Model]')


#Try more on polynomial linear regression model
poly_reg = PolynomialFeatures(degree=3)
X_poly = poly_reg.fit_transform(X_train_scaled)
X_test_poly = poly_reg.transform(X_test_scaled)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y_train)


#Evaluate model 
y_pred_pol = lin_reg2.predict(poly_reg.transform(X_test_scaled))
eva(y_test, y_pred_pol, '[Polynomial Regression Model]')


xgb_reg = xgb.XGBRegressor()
# Hyperparameter tuning using GridSearchCV
param_grid_xgb = {
    'n_estimators': [5, 7],
    'max_depth': [None, 3],
    'learning_rate': [0.01, 0.1]
}
grid_search_xgb = GridSearchCV(xgb_reg, param_grid_xgb, cv=5, scoring='r2')

#train XGBoost
grid_search_xgb.fit(X_train_scaled, y_train)
best_xgb_model = grid_search_xgb.best_estimator_


y_pred_xgb = best_xgb_model.predict(X_test)
eva(y_test, y_pred_xgb, 'XGBoost Model')


#prepare the test data
test_dataset = enc(cln(test_df))
test_dataset.isnull().sum().sort_values()


test_dataset_scaled = scaler.transform(test_dataset)
#Chose XGBoost for prediction
y_predict_test = best_xgb_model.predict(test_dataset_scaled)


submit = pd.DataFrame({'id' : test_dataset.index, 'Price':y_predict_test})
submit


#submission.csv create
submit.to_csv('submission.csv',index=False)

