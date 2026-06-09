import numpy as np 
import pandas as pd
import seaborn as sns


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv',index_col='id')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv',index_col='id')


train_df.info()


train_df.head()


train_df.isna().sum()


test_df.isna().sum()


print(train_df.shape)
print(test_df.shape)


target = list(set(train_df.columns) -set(test_df.columns))[0]
train_df.select_dtypes(['int64','float64']).corr()[target].sort_values(ascending=False)


def feature_eng(df):
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)
    df['Cardio_Load'] = df['Heart_Rate'] * df['Duration']
    df['Exertion_effect'] = df['Duration'] * df['Body_Temp']
    df['Heart_Efficiency_proxy'] = df['Heart_Rate'] / df['Age']
    # df['Fever_Flag'] = (df['Body_Temp'] > 37.5).astype(int)
    return df

train_df = feature_eng(train_df)
test_df = feature_eng(test_df)


print(train_df.shape)
train_df.head()


cat_cols= [col for col in train_df.columns if train_df[col].dtype=='object' ]
num_cols = [col for col in train_df.columns if train_df[col].dtype in ['int64','float64'] and col!=target]


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


X_train[num_cols].skew().sort_values(ascending=True).plot(kind='barh')


from sklearn.preprocessing import OneHotEncoder, StandardScaler

oh_encoder = OneHotEncoder(handle_unknown='ignore',sparse=False,drop='if_binary')
oh_train_encoded = pd.DataFrame(oh_encoder.fit_transform(X_train[cat_cols]),columns=oh_encoder.get_feature_names_out(),index=X_train.index)
oh_valid_encoded = pd.DataFrame(oh_encoder.transform(X_valid[cat_cols]),columns=oh_encoder.get_feature_names_out(),index=X_valid.index)

scaler = StandardScaler()
scaler_train_encoded = pd.DataFrame(scaler.fit_transform(X_train[num_cols]),columns=scaler.feature_names_in_,index=X_train.index)
scaler_valid_encoded = pd.DataFrame(scaler.transform(X_valid[num_cols]),columns=scaler.feature_names_in_,index=X_valid.index)


train_encoded = pd.concat([oh_train_encoded,scaler_train_encoded],axis=1)
valid_encoded = pd.concat([oh_valid_encoded,scaler_valid_encoded],axis=1)



print(train_encoded.shape, y_train.shape)
print(valid_encoded.shape, y_valid.shape)


train_encoded.head()


# RandomForest Regressor
from sklearn.ensemble import RandomForestRegressor

model_rf = RandomForestRegressor()
model_rf.fit(train_encoded,y_train)


# XGB Regressor
from xgboost import XGBRegressor

model_xgb = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=-1)

model_xgb.fit(train_encoded,y_train,
             early_stopping_rounds=5, 
             eval_set=[(valid_encoded, y_valid)], 
             verbose=False)


# LGBMegressor
from lightgbm import LGBMRegressor
model_lgb = LGBMRegressor(n_jobs=-1, random_state=1)
model_lgb.fit(train_encoded,y_train)


models = [model_rf, model_xgb, model_lgb]


from sklearn.metrics import mean_squared_log_error

def get_score(model,train_encoded,y_train,valid_encoded,y_valid):

    train_preds = model.predict(train_encoded)
    train_preds = np.clip(train_preds, 1, None)
    print('root mean squared log error(train):', np.sqrt(mean_squared_log_error(y_train,train_preds)))
    
    valid_preds = model.predict(valid_encoded)
    valid_preds = np.clip(valid_preds, 1, None)
    print('root mean squared log error(valid):', np.sqrt(mean_squared_log_error(y_valid,valid_preds)))


for model in models:
    print("---------------------------\n")
    print(model)
    get_score(model,train_encoded,y_train,valid_encoded,y_valid)


# preprocess test_df
oh_test_encoded = pd.DataFrame(oh_encoder.transform(test_df[cat_cols]),columns=oh_encoder.get_feature_names_out(),index=test_df.index)
scaler_test_encoded = pd.DataFrame(scaler.transform(test_df[num_cols]),columns=scaler.feature_names_in_,index=test_df.index)
test_encoded = pd.concat([oh_test_encoded,scaler_test_encoded],axis=1)
test_encoded.head()


submission_df = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
submission_df.head()


better_models ={
    'RandomForest': model_rf,
    'XGBREgressor' : model_xgb,
    'LGBMRegressor' :model_lgb,
}

for model_name in better_models:
    print(f"<-------{model_name}------->")
    model = better_models[model_name]
    y_pred = model.predict(test_encoded)
    y_pred = np.clip(y_pred, 1, None)
    
    output =pd.DataFrame({'id': test_df.index,
                       'Calories': y_pred})

    print('\n predictions of test_df: \n',output.head())
    
    output.to_csv(f'{model_name}_ps5e5_submission.csv',index= False)
    print(f'\nSubmission file using {model_name} Regressor model created successfully!')
    print('-----------------------------------\n')




