import pandas as pd
import numpy as np


train_basic_df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
train_extra_df = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')

# Merge train_basic_df and train_extra_df
train_df = pd.concat([train_basic_df,train_extra_df])
# Reset index after merging
train_df = train_df.reset_index(drop=True)


# Define constants here
SEED = 0


train_df.head()


# Check if merging was done properly:
print(f'Basic train {train_basic_df.shape}')
print(f'Extra train {train_extra_df.shape}')
print(f'Final merged train {train_df.shape}')


train_df.info()


# See how many missing value in each column
print('Null count in each column')
train_df.isnull().sum()


# Present the null value in percentage
null_distribution = train_df.isnull().mean()
print('Null distribution in %')
round(null_distribution*100,2)


# See summary of numerical col
train_df.describe()


# Visualize overview of numerical col distribution with histogram
train_df.hist(bins=20, figsize=(15,10))


# See summary of categorical col
train_df.describe(include = 'object')


def print_pivot_table(group,target):
    divider_line = '='*60
    print(divider_line)
    print(train_df.groupby(by = group)[target].aggregate([('TotalSoldQty','count'),('AvgPrice','mean')]))


categorical_col = train_df.select_dtypes(include = 'O').columns
for cat in categorical_col:
    print_pivot_table(cat,'Price')


# See the correlation of features in regard to target
train_df_encoded = pd.get_dummies(train_df)
train_df_encoded.corr()['Price'].sort_values(ascending = False)


# Make a copy of training data
train_df_copy = train_df.copy()

# Looking at the correlation, it seems like weight capacity and material polyester affects the price most
# Study relationship between weight capacity and price
import seaborn as sns
import matplotlib.pyplot as plt
weight_bins = [0,10,20,np.inf]
weight_label = ['0-10','10-20','20-inf']
train_df_copy['BinnedWeight'] = pd.cut(train_df_copy['Weight Capacity (kg)'],bins = weight_bins, labels=weight_label,right = True,include_lowest=False)
sns.boxplot(train_df_copy,x='BinnedWeight',y='Price')
plt.show()

# Study relationship between material and price
sns.boxplot(x=train_df['Material'], y=train_df['Price'])


# Import necessary libraries for data preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


# Split data into feature and target
X = train_df.drop('Price', axis = 1)
y = train_df['Price']


# Create custom column transformer to integrate into pipeline
from sklearn.base import BaseEstimator,TransformerMixin

# Encapsulate data imputer and feature creation
# Reason for not using imputer at pipeline is that couldn't properly retain dataframe structure after imputing
class ImputerAndFeatureCreationTransformer(BaseEstimator,TransformerMixin):
    def __init__(self,strategy = 'mean'):
        self.strategy = strategy
        self.num_imputer = SimpleImputer(strategy='mean').set_output(transform='pandas')
        self.cat_imputer = SimpleImputer(strategy='most_frequent').set_output(transform='pandas')
        self.num_col = None
        self.cat_col = None
        self.custom_impute_cat_col = ['Brand']
        self.ignore_num_col = ['id']
    
    def fit(self, X, y = None):
        self.num_col = X.select_dtypes(include = 'number').columns
        self.num_col = [col for col in self.num_col if col not in self.ignore_num_col]
        self.cat_col = X.select_dtypes(include = 'object').columns
        self.cat_col = [col for col in self.cat_col if col not in self.custom_impute_cat_col]
        self.num_imputer.fit(X[self.num_col])
        self.cat_imputer.fit(X[self.cat_col])
        return self

    def transform(self, X):
        # Apply imputers
        X = self.apply_imputer(X)
        # Create relationship between brands and Style
        # X['Brand_Style'] = X['Brand'].astype(str) + '_' + X['Style'].astype(str)
        # Binning of weight capacity
        # X['WeightCapacity_Binned'] = pd.cut(X['Weight Capacity (kg)'],bins = 3,labels = ['LightDuty','MediumDuty','HeavyDuty'])
        X['WeightCapacity_Binned'] = X['Weight Capacity (kg)'].apply(self.get_weight_limit)
        X['Material_x_Weight'] = X['WeightCapacity_Binned'].astype(str) + '_' + X['Material'].astype(str)
        return X

    def apply_imputer(self,X):
        X = X.copy()

        # Num col
        X[self.num_col] = self.num_imputer.transform(X[self.num_col])

        # Cat col
        X[self.cat_col] = self.cat_imputer.transform(X[self.cat_col])
        
        for col in self.custom_impute_cat_col:
            if col == 'Brand':
                X[col] = X[col].fillna('Others')
        
        return X
    
    def get_weight_limit(self, weight_capacity):
        if weight_capacity <= 10:
            return 'LightDuty'
        elif weight_capacity > 10 and weight_capacity <= 20:
            return 'MediumDuty'
        else:
            return 'HeavyDuty'
        


# Separate columns based on numerical and categorical
num_col = X.select_dtypes(include = 'number').columns
cat_col = X.select_dtypes(include = 'object').columns
col_to_drop = ['id']

# Remove the col_to_drop in both num and cat columns
num_col = [col for col in num_col if col not in col_to_drop]
cat_col = [col for col in cat_col if col not in col_to_drop]

# cat_feature_creation_col = ['Brand_Style','WeightCapacity_Binned','Material_x_Weight']
cat_feature_creation_col = ['WeightCapacity_Binned','Material_x_Weight']

cat_col = cat_col + cat_feature_creation_col

# Create preprocessor column transformer
preprocessor_imputer_feature_creation = ImputerAndFeatureCreationTransformer()

preprocessor_transformer = ColumnTransformer(transformers=[('num_transformer_scaling',StandardScaler(),num_col),
                                                      ('cat_transformer_imputer',OneHotEncoder(),cat_col),
                                                               ('drop_col','drop',col_to_drop)])


# Build pipeline:
# Impute missing val -> Create feature -> Drop irrelevant col -> Scaling & Encoding -> Baseline model (Linear Regression)
from sklearn.linear_model import LinearRegression

pipeline = Pipeline(steps = [('imputer_feature_creation',preprocessor_imputer_feature_creation),
                            ('transformer',preprocessor_transformer),
                            ('model',LinearRegression())])
pipeline
# print(pipeline)


from sklearn.metrics import mean_squared_error
def evaluate_model(X, y, pipeline, dataset_name):
    y_pred = pipeline.predict(X)
    model_name = type(pipeline.named_steps['model']).__name__
    rmse_score = mean_squared_error(y,y_pred,squared = False)
    print(f'({model_name}) {dataset_name} prediction score: {rmse_score}')


from sklearn.model_selection import train_test_split
# Split the data to prevent data leakage during validation stage
X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2,random_state=SEED)


# Fit the data into pipeline
# Again ensure that pipeline is using linear regression as base model
pipeline.set_params(model = LinearRegression())

pipeline.fit(X_train,y_train)
print('Successfully fit the data into pipeline')


# Evaluate training and validation score
evaluate_model(X_train,y_train,pipeline,'Training')
evaluate_model(X_val,y_val,pipeline,'Validation')


from sklearn.linear_model import Lasso
pipeline.set_params(model=Lasso(random_state = SEED)) 
pipeline.fit(X_train,y_train)
print('Successfully fit the data into pipeline')


# Evaluate training and validation score
evaluate_model(X_train,y_train,pipeline,'Training')
evaluate_model(X_val,y_val,pipeline,'Validation')


from sklearn.linear_model import Ridge
pipeline.set_params(model=Ridge(random_state = SEED)) 
pipeline.fit(X_train,y_train)
print('Successfully fit the data into pipeline')


# Evaluate training and validation score
evaluate_model(X_train,y_train,pipeline,'Training')
evaluate_model(X_val,y_val,pipeline,'Validation')


from xgboost import XGBRegressor
pipeline.set_params(model = XGBRegressor(random_state=SEED,n_estimators = 20,max_depth = 3,n_jobs=-1))
pipeline.fit(X_train,y_train)
print('Successfully fit the data into pipeline')


# Evaluate training and validation score
evaluate_model(X_train,y_train,pipeline,'Training')
evaluate_model(X_val,y_val,pipeline,'Validation')


from sklearn.model_selection import RandomizedSearchCV

params = [{
    'model':[XGBRegressor(random_state = SEED, n_jobs = -1)],
    'model__n_estimators':[750,1000,1250,1500],
    'model__max_depth':[3,5],
    'model__learning_rate':[0.1,0.15,0.2]
}]

grid_search = RandomizedSearchCV(pipeline,params, scoring='neg_root_mean_squared_error', cv = 5, n_iter = 5)
grid_search.fit(X_train,y_train)
print('Successfully fit the data into pipeline')


# Use the best performing hyperparam model
best_pipeline = grid_search.best_estimator_

# Print out the best hyperparam
print(grid_search.best_params_)


# Evaluate training and validation score
evaluate_model(X_train,y_train,best_pipeline,'Training')
evaluate_model(X_val,y_val,best_pipeline,'Validation')


# Import test data here
test_data = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
test_data


test_data.index


# Here we create a function to encapsulate the prediction and exporting process
def PredictAndExport(pipeline, X_test, filePath, unique_identifier, useIndex = False):
    test_data_id = test_data[unique_identifier]
    y_test = pipeline.predict(X_test)
    final_output = {'id':test_data_id, 'Price':y_test}
    test_result_df = pd.DataFrame(final_output)
    print('Final prediction:')
    print(test_result_df)
    # Write the final prediction result to csv
    test_result_df.to_csv(path_or_buf=filePath, index = useIndex)
    print('Successfully exported as csv')


# Predict and export the result
filePath = '/kaggle/working/submission.csv'
PredictAndExport(best_pipeline, test_data, filePath, 'id')

