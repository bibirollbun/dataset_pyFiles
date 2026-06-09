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


train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")


train


train_extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")


train_extra


## concatinating both train and train_extra
df = pd.concat([train,train_extra], axis=0, ignore_index=True)


df


df.shape


test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")


test


sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")


sample_submission


df.head()


df.shape


df.sample(5)


df.isnull().sum()


df.info()


df.describe()


## id    


df['id'].dtype


df['id'].nunique()


df.drop('id',axis=1,inplace=True)


## Brand


df['Brand'].dtype


df['Brand'].isnull().sum()


df['Brand'].value_counts()


### importing libraries for visualization
import seaborn as sns 
import matplotlib.pyplot as plt 


## Not repeating visualization already did in previous versions


df.isnull().sum()


## dropping all the rows which contains null values then training 
df.dropna(inplace=True)


df.isnull().sum()


df.shape


df.dtypes


df.sample(10)


X = df.drop('Price',axis=1) ## Independent features
y = df['Price'] ## target feature


X


y


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=42)


X_train


X_test


X_train.isnull().sum()


X_test.isnull().sum()


## categorical feature list
cat_col_list = [feature for feature in X.columns if X[feature].dtype=='O']
num_col_list = [feature for feature in X.columns if X[feature].dtype!='O']


cat_col_list


num_col_list


from sklearn.preprocessing import OneHotEncoder,LabelEncoder,OrdinalEncoder,FunctionTransformer
from sklearn.preprocessing import StandardScaler


from sklearn.compose import ColumnTransformer


from sklearn.impute import SimpleImputer


from sklearn.pipeline import Pipeline


# Define categorical and numerical columns
cat_ohe_cols = ['Brand', 'Style','Color']  # Columns where One-Hot Encoding will be applied
cat_label_cols = ['Laptop Compartment', 'Waterproof']  # Columns where Label Encoding will be applied
cat_ordinal_cols = ['Material', 'Size'] # Columns where Ordinal Encoding will be applied
num_cols = ['Compartments', 'Weight Capacity (kg)']  # numerical columns


train.head()


# Function to apply Label Encoding
def label_encode(df, columns):
    for col in columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df


# One-Hot Encoding Pipeline
cat_ohe_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore', dtype=int))
])

# Label Encoding Pipeline
cat_label_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('label_encoder', OrdinalEncoder(dtype=int)) ## applying Ordinal Encodin on these cat_label_cols 
])

# Ordinal Encoding Pipeline
cat_ordinal_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal_encoder', OrdinalEncoder(categories=[['Polyester', 'Nylon', 'Canvas', 'Leather'],['Small','Medium','Large']], dtype=int))
])

# Numerical Pipeline
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler',StandardScaler())
])

# # Numerical Pipeline
# num_impute_pipeline = Pipeline([
#     ('imputer', SimpleImputer(strategy='mean'))
# ])

# num_scalling_pipeline = Pipeline([
#     ('scaler', StandardScaler())
# ])


# Column Transformer
transformer = ColumnTransformer(transformers=[
    ('cat_ohe', cat_ohe_pipeline, cat_ohe_cols),
    ('cat_label', cat_label_pipeline, cat_label_cols),
    ('cat_ordinal', cat_ordinal_pipeline, cat_ordinal_cols),
    ('num_pipeline', num_pipeline, num_cols),
    # ('numscalling_pipeline', num_scalling_pipeline, scaling_cols),
], remainder='passthrough')  # Keeps other columns as they are


transformer


transformer.fit_transform(X_train)


transformer.fit_transform(X_train).shape


# pd.DataFrame(transformer.fit_transform(X_train))


X_train_trf = transformer.fit_transform(X_train)
X_test_trf = transformer.transform(X_test)


X_train_trf


pd.DataFrame(X_train_trf)


pd.DataFrame(X_train_trf).isnull().sum()


X_test_trf


test.head()


test.isnull().sum()


## dropping the 'id' feature
test.drop('id',axis=1,inplace=True)


test.head(3)


test_trf = transformer.transform(test)


test_trf


## Model Training and Model Selection
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor

import xgboost 
from xgboost import XGBRegressor

from lightgbm import LGBMRegressor


## Creating a function to evaluat model
def evaluate_model(true, predicted):
    mae=mean_absolute_error(true,predicted)
    mse=mean_squared_error(true,predicted)
    rmse=np.sqrt(mse)
    r2=r2_score(true,predicted)
    print("R2 Score:{:.4f}".format(r2))
    print("MSE:{:.4f}".format(mse))
    print("RMSE:{:.4f}".format(rmse))
    print("MAE:{:.4f}".format(mae))
    # ---------
    return 0


sample_submission.head()


id_column = sample_submission['id']


pd.DataFrame(X_train_trf)


## Model training
models={
    "Linear_Regression":LinearRegression(),
    "Lasso":Lasso(),
    "Ridge":Ridge(),
    "ElasticNet":ElasticNet()
}

for i in range(len(list(models))):
    model_name = list(models.keys())[i]
    model=list(models.values())[i]
    model.fit(X_train_trf,y_train) ## Train Model on X_train_trf (encoded as well as scaled)

    ## Make Predictions
    y_train_pred=model.predict(X_train_trf)
    y_test_pred=model.predict(X_test_trf)

    print(model_name,"=============>")
    print()
    print("Evaluating Train Dataset")
    evaluate_model(y_train,y_train_pred)
    print(f"\n{'-'*50}\n")
    print("Evaluating Test Dataset")
    evaluate_model(y_test,y_test_pred)
    print("="*50)
    print("\n")

    ## prediction
    prediction = model.predict(test_trf)

    result = pd.DataFrame(
    {
        'id':id_column,
        'Price':prediction
    }
    )

    result.to_csv('{}_prediction.csv'.format(model_name),index=False)
    print("File saved as '{}_prediction.csv'....".format(model_name))
    print()


## Model training
models={
    "DecisionTreeRegressor":DecisionTreeRegressor(),
    # "RandomForest":RandomForestRegressor(),
    "AdaBoost":AdaBoostRegressor(),
    "GradientBoost":GradientBoostingRegressor(),
    "XGBRegressor":XGBRegressor(),
    "LGBMRegressor":LGBMRegressor(),
}

for i in range(len(list(models))):

    model_name = list(models.keys())[i]
    model=list(models.values())[i]
    model.fit(X_train_trf,y_train) ## Train Model on X_train_trf (encoded)

    ## Make Predictions
    y_train_pred=model.predict(X_train_trf)
    y_test_pred=model.predict(X_test_trf)

    print(model_name,"=============>")
    print()
    print("Evaluating Train Dataset")
    evaluate_model(y_train,y_train_pred)
    print(f"\n{'-'*50}\n")
    print("Evaluating Test Dataset")
    evaluate_model(y_test,y_test_pred)
    print("="*50)
    print("\n")

    ## prediction
    prediction = model.predict(test_trf)

    result = pd.DataFrame(
    {
        'id':id_column,
        'Price':prediction
    }
    )

    result.to_csv('{}_prediction.csv'.format(model_name),index=False)
    print("File saved as '{}_prediction.csv'....".format(model_name))
    print()




