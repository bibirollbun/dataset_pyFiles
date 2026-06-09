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


df = pd.read_csv('/kaggle/input/ucs-654-kaggle-hack-lab-exam-1/train.csv')
test = pd.read_csv('/kaggle/input/ucs-654-kaggle-hack-lab-exam-1/test.csv')


df.head()


df.info()


test.info()


import seaborn as sns


df1 = df.dropna()
print(f"Original DataFrame shape: {df.shape}")
print(f"Cleaned DataFrame shape: {df1.shape}")


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
sns.boxplot(data=df)

plt.title('Boxplot of All Features')
plt.xlabel('Features')
plt.ylabel('Values')
plt.xticks(rotation=45)
plt.show()






df.boxplot(figsize=(12, 8))
plt.title('Boxplot of All Features')
plt.xlabel('Features')
plt.ylabel('Values')
plt.xticks(rotation=45)
plt.show()

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_pre = df[~((df < lower_bound) | (df > upper_bound)).any(axis=1)]

print(f"Original DataFrame shape: {df.shape}")
print(f"DataFrame shape after removing outliers: {df_pre.shape}")



import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
sns.boxplot(data=df_pre)

plt.title('Boxplot of All Features')
plt.xlabel('Features')
plt.ylabel('Values')
plt.xticks(rotation=45)
plt.show()



import pandas as pd
import matplotlib.pyplot as plt


for column in df_pre.select_dtypes(include=[np.number]).columns:
    plt.figure() 
    df[column].hist(bins=20)  
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.grid(False) 
    plt.show()


df_pre.drop(['f3'] , axis = 1 , inplace = True)


df_pre.head()


test.drop(['f3'] , axis = 1 , inplace = True)


import pandas as pd
from sklearn.preprocessing import StandardScaler


target_column = 'target'  
features = df_pre.drop(target_column, axis=1)

scaler = StandardScaler()

scaled_features = scaler.fit_transform(features)
scaled_df = pd.DataFrame(scaled_features, columns=features.columns)

scaled_df[target_column] = df_pre[target_column]

print(scaled_df.head())


import pandas as pd
from sklearn.preprocessing import StandardScaler

target_column = 'id' 
features = test.drop(target_column, axis=1)

scaler = StandardScaler()

scaled_features = scaler.fit_transform(features)

scaled_df_test = pd.DataFrame(scaled_features, columns=features.columns)

scaled_df_test[target_column] = test[target_column]

print(scaled_df_test.head())


from sklearn.model_selection import train_test_split
X = scaled_df.drop(columns=['target']) 
y = scaled_df['target']  

X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

# Print the shapes of the splits
print(f"Training set size: {X_train.shape[0]} rows")
print(f"Validation set size: {X_val.shape[0]} rows")


import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from skopt import BayesSearchCV
from skopt.space import Integer, Real
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline

X = df_pre.drop('target', axis=1)
y = df_pre['target']

k_best_features = SelectKBest(score_func=f_regression, k='all') 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_space = {
    'rf__n_estimators': Integer(100, 300),         
    'rf__max_depth': Integer(10, 50),             
    'rf__min_samples_split': Integer(2, 15),      
    'rf__min_samples_leaf': Integer(1, 10),        
    'rf__max_features': Real(0.5, 1.0, prior='uniform') 
}

rf_regressor = RandomForestRegressor(random_state=42)
pipeline = Pipeline(steps=[
    ('feature_selection', k_best_features),  
    ('rf', rf_regressor)
])

bayes_search = BayesSearchCV(
    estimator=pipeline,
    search_spaces=param_space,
    n_iter=50,  
    cv=5,      
    random_state=42,
    n_jobs=-1
)

bayes_search.fit(X_train, y_train)

print("Best Hyperparameters:", bayes_search.best_params_)

best_pipeline = bayes_search.best_estimator_

y_pred = best_pipeline.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")



test = pd.read_csv('/kaggle/input/ucs-654-kaggle-hack-lab-exam-1/test.csv')


test.drop(['f3'] , axis = 1 , inplace = True)





tr = pd.DataFrame(best_pipeline.predict(features), columns = ['target'])


features = test.drop('id', axis=1)


tr


final_df = pd.concat([test['id'],tr],axis = 1)


final_df


final_df.to_csv('om.csv', index = False)




