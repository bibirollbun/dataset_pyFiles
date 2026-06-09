import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv',index_col='id')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv',index_col='id')


train_df.info()


train_df.head()


train_df.describe()


train_df.isna().sum()


test_df.isna().sum()


print(train_df.shape)
print(test_df.shape)


target = list(set(train_df.columns) -set(test_df.columns))[0]
train_df.select_dtypes(['int64','float64']).corr()[target].sort_values(ascending=False)


import seaborn as sns
num_cols = [col for col in train_df.columns if train_df[col].dtype in ['int64','float64'] and col!=target]

for feature in num_cols:
    plt.figure(figsize=(5,5))
    sns.histplot(train_df[feature], kde=True, bins=30)
    
    # sns.scatterplot(data=train_df,x=feature,y='Calories')
    plt.title(f"Hist Plot between {feature}")
    plt.xlabel(feature)
    plt.ylabel(f"frequency of {feature}")
    plt.show()


# sns.histplot(train_df['Body_Temp'], kde=True, bins=30)


def feature_eng(df):
    # df['BMI'] = df['Weight'] / (df['Height'] ** 2)
    df['Cardio_Load'] = df['Heart_Rate'] * df['Duration']
    df['Exertion_effect'] = df['Duration'] * df['Body_Temp']
    df['Heart_Efficiency_proxy'] = df['Heart_Rate'] / df['Age']
    # df['Fever_Flag'] = (df['Body_Temp'] > 38).astype(int)
    
    # min, max = 37.1, 41.5
    df['Temp_Level'] = df['Body_Temp'].apply(lambda x: 'low' if x < 39.0 else ('normal' if x < 40.0 else 'high'))

    df['Duration_per_age'] = df['Duration'] / (df['Age'] + 1)
    df['Duration_per_weight'] = df['Duration']/df['Weight']
    df['Total_Duration'] = df['Duration_per_age']+ df['Duration_per_weight']

    # df['intensity_index0'] = df['Heart_Rate']+ df['Body_Temp']

    df['Intensity_index'] = df['Heart_Rate'] * df['Body_Temp']
    # df['intensity_index2'] = (df['Heart_Rate'] - df['Heart_Rate'].mean()) * (df['Body_Temp'] - df['Body_Temp'].mean())
    df['Heart_Temp_ratio'] = df['Heart_Rate'] / df['Body_Temp']   #mphasizes heart response relative to body temperature rise.
    
    return df

train_df = feature_eng(train_df)
test_df = feature_eng(test_df)


print(train_df.shape)
train_df.head()


cat_cols= [col for col in train_df.columns if train_df[col].dtype=='object' ]
num_cols = [col for col in train_df.columns if train_df[col].dtype in ['int64','float64'] and col!=target]
# num_cols.remove('Fever_Flag')
# bool_col =['Fever_Flag']


print(cat_cols)
print(num_cols)


for feature in num_cols:
    plt.figure(figsize=(5,5))
    sns.scatterplot(data=train_df,x=feature,y='Calories')
    plt.title(f"Scatter Plot between {feature} and Calories")
    plt.xlabel(feature)
    plt.ylabel("Calories")
    plt.show()


corr=  train_df[num_cols+[target]].corr()
sns.heatmap(corr, annot=True,annot_kws={"size":7.5})


train_df.select_dtypes(['int64','float64']).corr()[target].sort_values(ascending=True).plot(kind='barh')


y = train_df[target]
X = train_df.drop(target,axis=1)

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size=0.2)
print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)


X_train[num_cols].corrwith(y_train).sort_values(ascending=False)


X_train[num_cols].skew().sort_values(ascending=False)


skewness = abs(train_df[num_cols].skew()).sort_values(ascending=False)
# skewness
less_skewed_cols = [col for col in skewness.index if skewness[col]<0.5]
more_skewed_cols = [col for col in skewness.index if skewness[col]>0.5]


print('\n categorical columns: ')
print(cat_cols)
print('\n less skewed columns: ')
print(less_skewed_cols)
print('\n more skewed columns: ')
print(more_skewed_cols)


# from sklearn.preprocessing import OneHotEncoder, StandardScaler

# oh_encoder = OneHotEncoder(handle_unknown='ignore',sparse=False,drop='if_binary')
# oh_train_encoded = pd.DataFrame(oh_encoder.fit_transform(X_train[cat_cols]),columns=oh_encoder.get_feature_names_out(),index=X_train.index)
# oh_valid_encoded = pd.DataFrame(oh_encoder.transform(X_valid[cat_cols]),columns=oh_encoder.get_feature_names_out(),index=X_valid.index)

# scaler = StandardScaler()
# scaler_train_encoded = pd.DataFrame(scaler.fit_transform(X_train[num_cols]),columns=scaler.feature_names_in_,index=X_train.index)
# scaler_valid_encoded = pd.DataFrame(scaler.transform(X_valid[num_cols]),columns=scaler.feature_names_in_,index=X_valid.index)

# train_encoded = pd.concat([oh_train_encoded,scaler_train_encoded],axis=1)
# valid_encoded = pd.concat([oh_valid_encoded,scaler_valid_encoded],axis=1)


# print(train_encoded.shape, y_train.shape)
# print(valid_encoded.shape, y_valid.shape)


# train_encoded.head()





from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer

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
])

nominal_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore',drop='if_binary'))
])

# Column transformer
preprocessor = ColumnTransformer(
    transformers= [
            ('num_mean', num_pipeline_mean, less_skewed_cols),
            ('num_median', num_pipeline_median, more_skewed_cols),
            # ('num_bool', bool_imputer, bool_col),
            ('nominal_cat', nominal_pipeline, cat_cols)
        ],
    remainder='drop' ## default 
    )


X_train_transformed= pd.DataFrame(preprocessor.fit_transform(X_train), columns= preprocessor.get_feature_names_out())
X_valid_transformed = pd.DataFrame(preprocessor.transform(X_valid), columns= preprocessor.get_feature_names_out())


print(X_train_transformed.shape, y_train.shape)
print(X_valid_transformed.shape, y_valid.shape)


X_train_transformed.head()


plt.figure(figsize=(15,7))
plt.title(" Outlier present in Train Data")
sns.boxplot(X_train_transformed)
plt.xticks(rotation=45)
plt.show()


# XGB Regressor
from xgboost import XGBRegressor

model_xgb = XGBRegressor(n_estimators=1000, learning_rate=0.01, n_jobs=-1)

model_xgb.fit(X_train_transformed,y_train,
             early_stopping_rounds=5, 
             eval_set=[(X_valid_transformed, y_valid)], 
             verbose=False)


# full_pipeline = Pipeline([
#     ('preprocessing', preprocessor),
#     ('model', rf_model)
# ])


feature_names = X_train_transformed.columns

raw_importance = model_xgb.feature_importances_
normalized_importance = raw_importance / raw_importance.sum()
importance_df = pd.DataFrame({
    'feature': feature_names,
    'IMP': normalized_importance
}).sort_values(by='IMP', ascending=False)
importance_df = importance_df.round(6)
print(importance_df)


from sklearn.metrics import mean_squared_log_error

train_preds = model_xgb.predict(X_train_transformed)
# train_preds = np.clip(train_preds, 1, None)
train_preds = np.maximum(0, train_preds)
print('root mean squared log error(train):', np.sqrt(mean_squared_log_error(y_train,train_preds)))

valid_preds = model_xgb.predict(X_valid_transformed)
# valid_preds = np.clip(valid_preds, 1, None)
valid_preds = np.maximum(0, valid_preds)
print('root mean squared log error(valid):', np.sqrt(mean_squared_log_error(y_valid,valid_preds)))


from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

model_rf = RandomForestRegressor(n_estimators = 200,n_jobs=-1, random_state=1)
model_lgb= LGBMRegressor(n_jobs=-1, random_state=1)


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error


def get_scores(model,X_train_transformed,X_valid_transformed,y_train,y_valid):
    print(model)
    model.fit(X_train_transformed,y_train)
    print('model trained!\n')

    print("feature importance:\n")
    
    feature_names = X_train_transformed.columns
    raw_importance = model.feature_importances_
    normalized_importance = raw_importance / raw_importance.sum()
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'IMP': normalized_importance
    }).sort_values(by='IMP', ascending=False)
    importance_df = importance_df.round(6)
    print(importance_df)

    print("\n")

    train_preds = model.predict(X_train_transformed)
    train_preds = np.maximum(0, train_preds)

    rmse = np.sqrt(mean_squared_error(y_train,train_preds))
    print(f'root mean sqaured error: {rmse}' )
    print('mean absolute error:', mean_absolute_error(y_train,train_preds))
    print('root mean squared log error(train):', np.sqrt(mean_squared_log_error(y_train,train_preds)))
    
    valid_preds = model.predict(X_valid_transformed)
    valid_preds = np.maximum(0, valid_preds)
    
    rmse = np.sqrt(mean_squared_error(y_valid,valid_preds))
    print(f'root mean sqaured error: {rmse}' )
    print('mean absolute error:', mean_absolute_error(y_valid,valid_preds))
    print('root mean squared log error(valid):', np.sqrt(mean_squared_log_error(y_valid,valid_preds)))


models = [model_lgb,model_rf]

for model in models:
    print('-------------------------------')
    get_scores(model,X_train_transformed,X_valid_transformed,y_train,y_valid)


# preprocess test_df

# oh_test_encoded = pd.DataFrame(oh_encoder.transform(test_df[cat_cols]),columns=oh_encoder.get_feature_names_out(),index=test_df.index)
# scaler_test_encoded = pd.DataFrame(scaler.transform(test_df[num_cols]),columns=scaler.feature_names_in_,index=test_df.index)
# test_encoded = pd.concat([oh_test_encoded,scaler_test_encoded],axis=1)
# test_encoded.head()


X_test_transformed= pd.DataFrame(preprocessor.transform(test_df), columns= preprocessor.get_feature_names_out())
print(X_test_transformed.shape)
X_test_transformed.head()


y_pred = model_xgb.predict(X_test_transformed)
# y_pred = np.clip(y_pred, 1, None)
y_pred = np.maximum(0, y_pred)


output =pd.DataFrame({'id': test_df.index,
                   'Calories': y_pred})

print('\n predictions of test_df: \n',output.head())

# output.to_csv(f'XGB_fe4_ps5e5_submission.csv',index= False)
print(f'\nSubmission file using XGB Regressor model created successfully!')
print('-----------------------------------\n')


better_models ={
    'XGBRegressor' : model_xgb,
    'RandomForest': model_rf,
    'LGBMRegressor' :model_lgb,
}

for model_name in better_models:
    model = better_models[model_name]
    y_pred = model.predict(X_test_transformed)
    y_pred = np.maximum(0, y_pred)

    
    output =pd.DataFrame({'id': test_df.index,
                          'Calories': y_pred})

    print('\n predictions of test_df: \n',output.head())
    
    output.to_csv(f'{model_name}_fe_ps5e5_submission.csv',index= False)
    print(f'\nSubmission file using {model_name} Regressor model created successfully!')
    print('-----------------------------------\n')




