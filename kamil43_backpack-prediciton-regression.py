import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict,GridSearchCV
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer


train_df=pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_df=pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


train_df_ids=train_df['id']
test_df_ids=test_df['id']

train_df.drop('id',axis=1, inplace=True)
test_df.drop('id',axis=1, inplace=True)


train_df.head()


train_df.info()


train_df.describe().T


train_df.hist(figsize=(12, 10), bins=10)
plt.show()


for column in train_df.select_dtypes(include='number').columns:
    plt.figure(figsize=(6, 4))  # Create a new figure for each plot
    sns.boxplot(y=train_df[column])
    plt.title(f"Boxplot of {column}")
    plt.show()


#No outliers :) Lets check for null values}


print('Null values:')
print('*'*30)
print('Training set:')
print(train_df.isna().sum())
print('*'*30)
print(test_df.isna().sum())


#Columns with null values: Brand, Material, Size, Laptop Comparatment, Waterproof, Style, Color, Weight Capacity (kg)


cat_columns=[x for x in train_df.columns if train_df[x].dtype==object]
for col in cat_columns:
    print(train_df[col].value_counts())
    print('*'*30)


#Numerical columns we can try to impute using SimpleImputer with strategy mean
mean_imputer=SimpleImputer(strategy='mean')


#Lets try to impute categorical valeus with Unknown
def impute_with_unknown(df, cat_columns):
    for col in cat_columns:
        if col=='Size':
            df[col]=df[col].fillna('Medium')
        else:
            df[col]=df[col].fillna('Unknown')
impute_with_unknown(train_df, cat_columns)


print('Training set:')
print(train_df.isna().sum())
print('*'*30)
print(test_df.isna().sum())


#ENCOODING CATEGORY COLUMNS:
#OneHotEncoding: Brand, Material, Style, Color, Laptop Compartment, Waterproof 
#OrdinaryEncoding: Size
onehot_columns=['Brand', 'Material', 'Style', 'Color','Laptop Compartment', 'Waterproof']
binary_columns=[]
ordinary_columns=['Size']
onehot=OneHotEncoder(handle_unknown='ignore')
ordinal = OrdinalEncoder(categories=['Small','Medium','Large'])



def rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))


import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer

# Define transformations
log_transformer = FunctionTransformer(np.log1p)  # Apply log(1 + x)
num_pipeline = make_pipeline(mean_imputer,log_transformer, StandardScaler())

onehot = OneHotEncoder(handle_unknown='ignore')  # One-hot encoding

ordinal = OrdinalEncoder(categories=[['Small', 'Medium', 'Large']])
# Define preprocessing pipeline
preprocessing = ColumnTransformer([
    ('num', num_pipeline, ['Weight Capacity (kg)']),
    ('onehot', onehot, onehot_columns),
    ('ordinal', ordinal, ['Size'])
])



#Splitting data
target=train_df['Price']
train_df.drop('Price',axis=1, inplace=True)

X_train,X_valid,y_train,y_valid=train_test_split(train_df,target,test_size=0.2,random_state=42)


#Model selection
models = {
    'linear_regression': LinearRegression(),
    'ridge': Ridge(),
    'lasso': Lasso(),
    'sgd': SGDRegressor(),
    'tree': DecisionTreeRegressor(),
    'rf': RandomForestRegressor(n_jobs=-1),
    'xgb': XGBRegressor()
}

pipelines = {}

for name, model in models.items():
    pipelines[name] = Pipeline([
        ('preprocessing', preprocessing),  
        ('model', model)                    
    ])


for name, pipeline in pipelines.items():
    print(name)
    preds=cross_val_predict(pipeline,X_train[:10000],y_train[:10000], cv=3)
    print(rmse(y_train[:10000],preds))
    print('*'*30)


grid_param = {
    'model__alpha': [0, 0.1, 0.3, 0.5,],  # Regularization strength
    'model__tol': [0.0001, 0.001],  # Convergence tolerance
    'model__max_iter': [None, 300, 500],  # Maximum iterations
    'model__solver': ['auto', 'svd', 'cholesky'],  # Solver for optimization
}
grid_search = GridSearchCV(estimator=pipelines['ridge'], param_grid=grid_param, cv=5, n_jobs=-1, verbose=3,scoring='neg_root_mean_squared_error')
grid_search.fit(X_train,y_train)


print(-grid_search.best_score_)
print(grid_search.best_params_)
final_estimator=grid_search.best_estimator_


final_estimator.fit(train_df,target)


impute_with_unknown(test_df, cat_columns)


preds=final_estimator.predict(test_df)


results = pd.DataFrame({
    'id': test_df_ids,
    'Price': preds
})
results.to_csv('predictions.csv', index=False)




