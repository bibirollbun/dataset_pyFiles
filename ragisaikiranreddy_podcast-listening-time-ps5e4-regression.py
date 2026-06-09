import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm.auto import tqdm

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df_train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv',index_col= 'id')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv',index_col='id')


import warnings
warnings.filterwarnings('ignore')


# info of the data
print("\n <==Train Data Info==>")
print(df_train.info())

# print("\n <==Test Data Info==>")
# print(df_test.info())

# head of data
print("\n <==Head of Train Data==>")
display(df_train.head())

print("\n <===Head of Test Data==>")
display(df_test.head())

target = list(set(df_train.columns)- set(df_test.columns))[0]
print(f'\n Target column is: {target}')


# missing values in data
print("\n <==Missing Values in train data==>")
print(df_train.isna().sum())

# print("\n <==Missing Values in test data===>")
# print(df_test.isna().sum())


# Categorical and Numerical Columns in Data
cat_cols = [col for col in df_train.columns if df_train[col].dtype=='object' ]
num_cols = [col for col in df_train.columns if df_train[col].dtype in ['int64','float64'] and col is not target]

print("\n <==Categorical columns in train data==>")
print(cat_cols)

print("\n <==Numerical columns in train data==>")
print(num_cols)

# check if duplicate rows 
print("\nNo.of Duplicate Rows in Train Data:")
print(df_train.duplicated().sum())

print("\n <==correlation of Numerical columns with target==>")

# as target not added to num_cols
# df_train[num_cols].corrwith(df_train[target]).sort_values(ascending=False)

df_train.select_dtypes(include=['int64','float64']).corr()['Listening_Time_minutes'].sort_values(ascending=False)


# print('shape of train data: ', df_train.shape)
# print('shape of test data: ', df_test.shape)


df_train.describe()


# df_train['Guest_Popularity_percentage'].value_counts().sort_index(ascending=False)


def feature_eng(df):
    df['Is_weekend'] = df['Publication_Day'].isin(['Saturday','Sunday']).astype(int)
    df['Is_High_Host_Popularity'] = (df['Host_Popularity_percentage'] > 70).astype(int)
    # df['Is_High_Guest_Popularity'] = (df['Guest_Popularity_percentage'] > 70).astype(int)

    # df['Host_Guest_Popularity_Gap'] = df['Host_Popularity_percentage'] - df['Guest_Popularity_percentage']
    
    # df['Is_Long_Episode'] = (df['Episode_Length_minutes'] > 65).astype(int)

    # df['Episode_Length_per_Ad']  = df['Episode_Length_minutes']/(1+df['Number_of_Ads'])
    return df

df_train = feature_eng(df_train)
df_test = feature_eng(df_test)


df_train.head()


df_train.shape, df_test.shape


df_train.isna().sum()


df_train.select_dtypes(include=['int64','float64']).corr()['Listening_Time_minutes'].sort_values(ascending=False)


df_train.select_dtypes(include=['int64','float64']).corr()['Listening_Time_minutes'].sort_values(ascending=True).plot(kind='barh')


y = df_train[target]
X = df_train.drop(target,axis=1)

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train,y_valid = train_test_split(X,y, test_size=0.2, random_state=1)
print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)


# Categorical and Numerical Columns in Data
cat_cols = [col for col in df_train.columns if df_train[col].dtype=='object' ]
num_cols = [col for col in df_train.columns if df_train[col].dtype in ['int64','float64'] and col is not target]


bool_cols = [col for col in df_train.columns
             if df_train[col].dropna().isin([0, 1]).all()]
num_cols = [col for col in num_cols if col not in bool_cols]
num_cols


sorted(df_test.columns) == sorted(cat_cols+bool_cols+num_cols)


print("\n <==Unique values in each Categorical columns of X_train==>")
print(X_train[cat_cols].nunique())

print("\n <==Unique values in each Categorical columns of X_valid==>")
print(X_valid[cat_cols].nunique())

low_cardinality = [ col for col in cat_cols if X_train[col].nunique()<=10]
print("\n <==columns with low cardinality:")
print(low_cardinality)

print("\n <==value counts for low cardinality columns:")
for col in low_cardinality:
    print(f"\n  Value count for {col}")
    print(X_train[col].value_counts())


# skewness
print("\n <==skeweness of Numerical columns in X_train data==>")
print(X_train[num_cols].skew())

skewness = abs(df_train[num_cols].skew())
less_skewed_cols = [col for col in skewness.index if skewness[col]<0.5]
more_skewed_cols = [col for col in skewness.index if skewness[col]>0.5]
print('\n less skewed columns: ')
print(less_skewed_cols)
print('\n more skewed columns: ')
print(more_skewed_cols)

print("\n <==correlation plot of Numerical columns with target==>")
# X_train[num_cols].corrwith(y_train).sort_values(ascending=False)
abs(X_train[num_cols].corrwith(y_train)).sort_values(ascending=True).plot(kind='barh')


# X_train['Podcast_Name'].value_counts()


ordinal_cols = [col for col in low_cardinality if col!='Genre']
nominal_cols = ['Genre']

day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
time_order = ['Morning', 'Afternoon', 'Evening', 'Night']
sentiment_order = ['Negative', 'Neutral', 'Positive']
custom_categories = [day_order, time_order, sentiment_order]
ordinal_cols


print('Numerical==>: ')

print(f'less_skewed columns:{less_skewed_cols}')
print(f'more_skewed columns:{more_skewed_cols}')
print(f'bool columns:{bool_cols}')

print('\nCategorical==>: ')
print(f'ordinal columns:{ordinal_cols}')
print(f'nominal columns:{nominal_cols}')


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer



# Pipelines
num_pipeline_mean = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

num_pipeline_median = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

bool_imputer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    # ('scaler', StandardScaler())
])

ordinal_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(categories=custom_categories))
])

nominal_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Column transformer
preprocessor = ColumnTransformer(
    transformers= [
            ('num_mean', num_pipeline_mean, less_skewed_cols),
            ('num_median', num_pipeline_median, more_skewed_cols),
            ('num_bool', bool_imputer, bool_cols),
            ('ordinal_cat', ordinal_pipeline, ordinal_cols),
            ('nominal_cat', nominal_pipeline, nominal_cols)
        ],
    remainder='drop' ## default 
    )


preprocessor


# X_train_transformed= pd.DataFrame(preprocessor.fit_transform(X_train), columns= preprocessor.get_feature_names_out())


# preprocessor.get_feature_names_out()


# X_train_transformed.head()


# X_valid_transformed = pd.DataFrame(preprocessor.transform(X_valid), columns= preprocessor.get_feature_names_out())


# X_train_transformed.shape


# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# rf_model = RandomForestRegressor(n_estimators = 300,n_jobs=-1, random_state=1)

# print(rf_model)
# rf_model.fit(X_train_transformed,y_train)
# print('model trained!\n')

# train_preds = rf_model.predict(X_train_transformed)
# rmse = np.sqrt(mean_squared_error(y_train,train_preds))
# print('root mean sqaured error(train): ',rmse)


# valid_preds = rf_model.predict(X_valid_transformed)
# print('mean absolute error(valid):', mean_absolute_error(y_valid,valid_preds))
# rmse = np.sqrt(mean_squared_error(y_valid,valid_preds))
# print('root mean sqaured error(valid): ',rmse)


from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators = 300,
                                 max_depth= 12,
                                 min_samples_leaf =3,
                                 n_jobs=-1, random_state=1)


# from sklearn.model_selection import cross_val_score
# rmses = -1*cross_val_score(dt_model,X_train_transformed, y_train,
#                           scoring = 'neg_root_mean_squared_error',cv=5)
# print(rmses)
# pd.Series(rmses).describe()

# can lead to data leakage as each cv fold sets used in cross val score are also used in the fit of preprocessor to get X_train_transformed
# fitting the preprocessor on all of X_train, before doing cross-validation.


full_pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('model', rf_model)
])


full_pipeline





from sklearn.metrics import mean_squared_error, mean_absolute_error

full_pipeline.fit(X_train,y_train)

train_preds = full_pipeline.predict(X_train)
rmse_train = np.sqrt(mean_squared_error(y_train,train_preds))
print('root mean sqaured error(train): ',rmse_train)
print('\n')

valid_preds = full_pipeline.predict(X_valid)
print('mean absolute error(valid):', mean_absolute_error(y_valid,valid_preds))
rmse_valid = np.sqrt(mean_squared_error(y_valid,valid_preds))
print('root mean sqaured error(valid): ',rmse_valid)


from sklearn.model_selection import cross_val_score
rmses_p = -1*cross_val_score(full_pipeline,X_train, y_train,
                          scoring = 'neg_root_mean_squared_error',cv=3)
print(rmses_p)
pd.Series(rmses_p).describe()


# pip install pactools


from sklearn.model_selection import GridSearchCV

# from pactools.grid_search import GridSearchCVProgressBar

param_grid = [
    {          
        'model__n_estimators': [300, 350],     
        'model__max_depth': [15, 30],           
        'model__min_samples_leaf': [2, 4],      
        # 'model__max_features': ['sqrt', 0.8]    
    }
]

grid_search = GridSearchCV(estimator= full_pipeline,
                           param_grid= param_grid,
                           cv=2,
                           scoring = 'neg_root_mean_squared_error',
                           return_train_score=True,
                           verbose=5,
                           n_jobs=-1)


# grid_search.fit(X_train,y_train)
# grid_search.fit(X,y)


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from time import time


param_distribs = {
    'model__n_estimators': randint(300, 500),         # Randomly sample between 300–600 trees
    'model__max_depth': randint(10, 31),              # Sample depth between 10 and 30
    'model__min_samples_leaf': randint(2, 5),         # Small leaves to avoid overfitting
    'model__max_features': ['sqrt', 0.8]              # Sample from two solid values
}


random_search = RandomizedSearchCV(
    estimator=full_pipeline,
    param_distributions=param_distribs,
    n_iter=4,
    cv=2,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    random_state=42,
    return_train_score=True,
    verbose=2
)

random_search.fit(X_train,y_train)





print("Best Parameters:", random_search.best_params_)
print("Best RMSE:", -random_search.best_score_)


random_search.best_estimator_


random_search.cv_results_


cv_res = pd.DataFrame(random_search.cv_results_)
cv_res.sort_values(by="mean_test_score", ascending=False,inplace=True)
cv_res.head()


# preprocess X_test
# X_test_transformed = pd.DataFrame(preprocessor.transform(df_test), 
#                                   columns= preprocessor.get_feature_names_out(), index = df_test.index)
# print(X_test_transformed.shape)
# X_test_transformed.head()

# y_preds = full_pipeline.predict(df_test)
# y_preds.shape


# creating submission file for best estimator (RandomForest Regressor)
y_best_preds = random_search.best_estimator_.predict(df_test)
y_best_preds.shape


pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')
# df_test


# Save test predictions to file
output = pd.DataFrame({'id': df_test.index,
                       'Listening_Time_minutes': y_best_preds})
output.to_csv('rf_randomsearch_submission.csv', index=False)
print('output file created Sucessfully!')
output.head()




