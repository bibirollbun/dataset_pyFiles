import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv', index_col=0)
training_extra = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv', index_col=0)


train.head()


train.dtypes


print(train.describe())
print(train.describe(include='object'))


print('Total missing data:', np.sum(train.isna().sum(axis=0))*100/train.size)
train.isna().sum(axis=0)*100/len(train)


train.duplicated().sum()


X_train = train[train.columns[:-1]]
y_train = train['Price']


X_train[X_train.isna().any(axis=1)].reset_index().isna().sum(axis=1).hist()


from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy="most_frequent")


columns = X_train.columns
X_train = pd.DataFrame(imp.fit_transform(X_train), columns = columns)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


X_train['Compartments'] = StandardScaler().fit_transform(X_train[['Compartments']])
X_train['Weight Capacity (kg)'] = StandardScaler().fit_transform(X_train[['Weight Capacity (kg)']])


X_train['Weight Capacity (kg)'].hist()


from sklearn.preprocessing import OrdinalEncoder

ord_encoder = OrdinalEncoder(categories=[['Small', 'Medium', 'Large']])
X_train['Size'] = ord_encoder.fit_transform(X_train[['Size']])


X_train['Laptop Compartment'] = X_train['Laptop Compartment'].apply(lambda x: 1 if x == 'Yes' else 0)
X_train['Waterproof'] = X_train['Waterproof'].apply(lambda x: 1 if x == 'Yes' else 0)


from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(sparse=False)

categorical_columns = X_train.select_dtypes(include=['object']).columns
print(categorical_columns)
onehot_columns = onehot_encoder.fit_transform(X_train[categorical_columns])

onehot_df = pd.DataFrame(onehot_columns, columns = onehot_encoder.get_feature_names_out(categorical_columns))
print(onehot_df.columns)
X_train.drop(categorical_columns, axis=1, inplace=True)
X_train = pd.concat([X_train, onehot_df], axis=1)


X_train


from sklearn.model_selection import train_test_split 

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.33, random_state=42)


!pip install lightgbm


import lightgbm as lgb

model = lgb.LGBMRegressor()
model.fit(X=X_train, y=y_train)
np.mean((model.predict(X_val) - y_val)**2)


from sklearn.model_selection import train_test_split 

X_train = train[train.columns[:-1]]
y_train = train['Price']
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.33)


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer


# preprocessing pipelines
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')), 
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), 
    ('scaler', OneHotEncoder(handle_unknown='ignore'))
])

ordinal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), 
    ('scaler', OrdinalEncoder(categories=[['Small', 'Medium', 'Large']]))
])

def binary_transform(x):
    # Convert to numpy array if it isn't already
    x_array = np.array(x)
    # Transform without reshaping
    return np.where(x_array == 'yes', 1, 0)
    
binary_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), 
    ('converter', FunctionTransformer(binary_transform, validate=0))
])


numerical_features = X_train.select_dtypes(include='float64').columns
ordinal_features = ['Size']
categorical_features = ['Brand', 'Material', 'Style', 'Color']
binary_features = ['Laptop Compartment', 'Waterproof']


# Combine transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('ordinal', ordinal_transformer, ordinal_features),
        ('binary', binary_transformer, binary_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

preprocessor.fit(X_train, y_train)


# Final pipeline: data preprocessing and model 
pipe = Pipeline([
    ('preprocess', preprocessor), 
    ('model', lgb.LGBMRegressor())]
)


pipe.fit(X_train, y_train)


np.mean(abs(pipe.predict(X_val) - y_val))

