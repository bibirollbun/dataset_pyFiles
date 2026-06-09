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


import pandas as pd
import numpy  as np 
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn


train_df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
train_dfex = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


train_df.shape, train_dfex.shape


train_df = pd.concat([train_df, train_dfex], axis=0).reset_index(drop=True)
train_df.shape


train_df.head()


train_df.info()


train_df.duplicated().sum()


train_df = train_df.drop_duplicates()


train_df.isnull().sum()


train_df.value_counts().sum()


num_features = train_df.select_dtypes(['int64', 'float64'] )
num_features


cat_features = train_df.select_dtypes('object')
cat_features


print('Brand')
print(train_df['Brand'].value_counts())
print('Material')
print(train_df['Material'].value_counts())
print('Size')
print(train_df['Size'].value_counts())
print('Compartment')
print(train_df['Laptop Compartment'].value_counts())
print('WaterProof')
print(train_df['Waterproof'].value_counts())
print('Style')
print(train_df['Style'].value_counts())
print('Color')
print(train_df['Color'].value_counts())


sns.heatmap(train_df.isnull(), cmap='tab20c_r') 
plt.show()


X_train = train_df.drop('Price', axis=1)
y_train = train_df['Price']

X_test = test_df.copy()
y_test = None



# Re-extract categorical and numerical feature names after dropping 'Price'
X_train_cat_features_names = X_train.select_dtypes(include=['object']).columns.to_list()
X_train_num_features_names = X_train.select_dtypes(include=['float64', 'int64']).columns.to_list()

X_test_cat_features_names = X_test.select_dtypes(include=['object']).columns.to_list()
X_test_num_features_names = X_test.select_dtypes(include=['float64', 'int64']).columns.to_list()


X_train_cat_features_names


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Define preprocessing steps for numerical and categorical features
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Apply transformations
preprocessor = ColumnTransformer([
    ('num', num_pipeline, X_train_num_features_names),
    ('cat', cat_pipeline, X_train_cat_features_names)
], remainder='passthrough')

# Fit and transform
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)



from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor()
gbr.fit(X_train_transformed, y_train)

gbr_predictions = gbr.predict(X_test_transformed)

gbr_predictions = pd.DataFrame({
    'id': test_df['id'],
    'Price': gbr_predictions
})

gbr_predictions.to_csv('gbr_predictions.csv', index=False)
print("File saved successfully")


gbr_predictions.to_csv('/kaggle/working/gbr_predictions.csv', index=False)


