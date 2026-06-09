# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import missingno
from sklearn_pandas import DataFrameMapper
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error as MSE
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingClassifier, BaggingClassifier, RandomForestRegressor, AdaBoostClassifier, GradientBoostingRegressor

import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv') 
train_extra = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')

display(train.head())
display(train_extra.head())
display(test.head())


train = train.drop('id',axis=1)
train_extra=train_extra.drop('id',axis=1)
test = test.drop('id', axis=1)


display(train.info())
display(train_extra.info())
display(test.info())


sorted_index = train_extra.isnull().sum().sort_values(ascending=False).index
pd.DataFrame({'Null #':train_extra.isnull().sum().sort_values(ascending=False),'Null %':(train_extra.isnull().mean()*100).round(3).sort_values(ascending=False),'Type':train_extra.dtypes.reindex(sorted_index)})



sorted_index = train.isnull().sum().sort_values(ascending=False).index   # ['Color', 'Brand', 'Material', 'Style', 'Laptop Compartment', 'Waterproof', 'Size', 'Weight Capacity (kg)', 'Compartments', 'Price']

missing_df = pd.DataFrame({
    'Null #': train.isnull().sum().sort_values(ascending=False),
    'Null %': (train.isnull().mean()*100).round(3).sort_values(ascending=False),
    'type': train.dtypes.reindex(sorted_index)
})
missing_df


missing_cols = ['Price','Compartments','Color','Brand','Material','Style','Laptop Compartment','Waterproof','Size','Weight Capacity (kg)']
missingno.matrix(train[missing_cols].sort_values(by='Price', ascending=False))


missing_cols = ['Price','Compartments','Color','Brand','Material','Style','Laptop Compartment','Waterproof','Size','Weight Capacity (kg)']
missingno.matrix(train_extra[missing_cols].sort_values(by='Price', ascending=False))


weight_null = train[train['Weight Capacity (kg)'].isnull()]
weight_notnull = train[~train['Weight Capacity (kg)'].isnull()]

display(train.describe())
display(weight_null.describe())
display(weight_notnull.describe())
weight_null['Color'].value_counts()


df = pd.concat([train,train_extra],axis=0)
df.info()


X, y = df.drop('Price', axis=1), df['Price']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# I did not end up using this code cause the final format is csr_matrix

# cat_col_mask = df.dtypes == 'object'
# cat_cols = df.columns[cat_col_mask].tolist()
# num_cols = df.columns[~cat_col_mask].tolist()

# cat_imputation_mapper = DataFrameMapper([(cat_cols, SimpleImputer(strategy='most_frequent'))], input_df=True, df_out=True)
# num_imputation_mapper = DataFrameMapper([(num_cols, SimpleImputer(strategy='mean'))], input_df=True, df_out=True)

# num_cat_union = FeatureUnion([('cat_mapper', cat_imputation_mapper), ('num_mapper', num_imputation_mapper)])

# to_dict = FunctionTransformer(lambda x: [dict(zip(range(x.shape[1]), row)) for row in x], validate=False)

# preprocessing_pipeline = Pipeline([('featureunion',num_cat_union), ('todict', to_dict), ('vectorizer',DictVectorizer(sort=False))])
# preprocessed_df = preprocessing_pipeline.fit_transform(df)


cat_col_mask = X_train.dtypes == 'object'
cat_cols = X_train.columns[cat_col_mask].tolist()
num_cols = X_train.columns[~cat_col_mask].tolist()

cat_pipeline = Pipeline([('imputater', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore',drop='if_binary'))])
num_pipeline = Pipeline([('imputater', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
preprocessor = ColumnTransformer([('num', num_pipeline, num_cols), ('cat', cat_pipeline, cat_cols)])

preprocessed_train_array = preprocessor.fit_transform(X_train)  
preprocessed_val_array = preprocessor.transform(X_val) 
preprocessed_test_array = preprocessor.transform(test)
# scikit-learn's transformers and pipelines output numpy arrays by default. that means spliting data into train and test based on feature names will be impossible!
# options: do spliting before preprocessing, or retrieve feature names from the transformer


feature_names = preprocessor.get_feature_names_out()
preprocessed_X_train = pd.DataFrame(preprocessed_train_array, columns=feature_names)
preprocessed_X_val = pd.DataFrame(preprocessed_val_array, columns=feature_names)
preprocessed_test = pd.DataFrame(preprocessed_test_array, columns=feature_names)


print(preprocessed_test.columns)


# grid = {'max_depth':np.arange(2,8,1), 'subsample':np.arange(0.3,0.9,0.1), 'n_estimators':np.arange(100,600,100), 'learning_rate':np.arange(0.01,0.2,0.01), 'gamma':np.arange(0.1,0.6,0.1), 'colsample_bytree':np.arange(0.6,0.9,0.1),'min_child_weight': [1, 3, 5, 7],'reg_alpha': np.arange(0, 1, 0.2),'reg_lambda': np.arange(0, 1, 0.2)}

# xgb_reg = xgb.XGBRegressor()
# xgb_cv = RandomizedSearchCV(estimator=xgb_reg, param_distributions=grid, n_iter=200 ,cv=3, scoring='neg_mean_squared_error', verbose=1) # Don't specify n_jobs=-1 cause it has the risk of running out of disc space
# xgb_cv.fit(preprocessed_X_train, y_train)
# print(xgb_cv.best_score_)
# print(xgb_cv.best_params_)


!df -h


# After running the RandomSearchCV and getting the best values for each parameter, we run XGBoost again with those parameters.
xgb_reg = xgb.XGBRegressor(subsample=0.9000000000000001,reg_lambda=0.2,reg_alpha=0.2,n_estimators=400,min_child_weight=5,max_depth=3,learning_rate=0.11,gamma=0.5,colsample_bytree=0.8999999999999999)
xgb_reg.fit(preprocessed_X_train, y_train)
preds = xgb_reg.predict(preprocessed_X_val)
rmse = np.sqrt(MSE(y_val,preds))
print(rmse)


# Combine preprocessed_X_train and preprocessed_X_val for final prediction
X_train = pd.concat([preprocessed_X_train, preprocessed_X_val], axis=0)
y_train = pd.concat([y_train, y_val], axis=0)


print(X_train.shape)
print(y_train.shape)


xgb_reg = xgb.XGBRegressor(subsample=0.9000000000000001,reg_lambda=0.2,reg_alpha=0.2,n_estimators=400,min_child_weight=5,max_depth=3,learning_rate=0.11,gamma=0.5,colsample_bytree=0.8999999999999999)
xgb_reg.fit(X_train, y_train) 
preds = xgb_reg.predict(preprocessed_test)
print(preds)


test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
ids = test['id']
sub = pd.DataFrame({'id':ids, 'Price':preds})
print(sub.shape)
sub.head()




sub.to_csv('submission.csv', index=False)
sub

