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


import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import zscore


train_df=pd.read_csv('/kaggle/input/playground-series-s4e4/train.csv')
test_df=pd.read_csv('/kaggle/input/playground-series-s4e4/test.csv')


train_df.head()


test_df.head()


train_df=train_df.drop('id', axis=1)
test_id=test_df['id']
test_df=test_df.drop('id', axis=1)


train_df.isnull().sum()


train_df.describe()


train_df.info()


train_df.shape


train_df.corr(numeric_only=True)


train_df['Sex'].value_counts()


# Volume Approximation
train_df['Volume']=train_df['Length']*train_df['Diameter']*train_df['Height']


test_df['Volume']=test_df['Length']*test_df['Diameter']*test_df['Height']


train_df['Meat_to_Shell_Ratio']=train_df['Whole weight']/train_df['Shell weight'] 


test_df['Meat_to_Shell_Ratio']=test_df['Whole weight']/test_df['Shell weight'] 


train_df['Surface_Area']=2 * 3.1416 * (train_df['Diameter']/2) * (train_df['Height']/2)


test_df['Surface_Area']=2 * 3.1416 * (test_df['Diameter']/2) * (test_df['Height']/2)


# Correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(train_df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# Distribution of Rings
plt.figure(figsize=(10, 6))
sns.histplot(train_df['Rings'], kde=True)
plt.title('Distribution of Rings')
plt.show()


# Explore relationship between Sex and Rings
plt.figure(figsize=(10, 6))
sns.boxplot(x='Sex', y='Rings', data=train_df)
plt.title('Rings by Sex')
plt.show()

sex_rings_stats = train_df.groupby('Sex')['Rings'].agg(['mean', 'median', 'std', 'count'])
print(sex_rings_stats)


# Boxplots to visualize outliers for numeric columns only
plt.figure(figsize=(12, 8))
numeric_cols = train_df.select_dtypes(include=[np.number]).columns  # Get numeric columns
for i, col in enumerate(numeric_cols):  # Loop through numeric columns
    plt.subplot(3, 4, i + 1)
    sns.boxplot(train_df[col])
    plt.title(col)
plt.tight_layout()
plt.show()


train_df


train_df=pd.get_dummies(train_df, columns=['Sex'], drop_first=True)


X = train_df.drop('Rings', axis=1)
y = train_df['Rings']


# Feature Engineering: Adding Polynomial Features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)


# Predictions
y_pred = model.predict(X_test)


# Evaluation
print(f'RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}')
print(f'R²: {r2_score(y_test, y_pred)}')


test_df=pd.get_dummies(test_df, columns=['Sex'], drop_first=True)


# Feature Engineering: Adding Polynomial Features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
test_df_poly = poly.fit_transform(test_df)

# Scale features
scaler = StandardScaler()
test_df_scaled = scaler.fit_transform(test_df_poly)


prediction=model.predict(test_df_scaled).astype(int)


prediction


prediction.shape


test_id.shape


submission=pd.DataFrame({'id': test_id, 'Rings': prediction})


submission.to_csv('submission.csv')


joblib.dump(model, 'model.pkl')


joblib.dump(scaler, 'scaler.pkl')




