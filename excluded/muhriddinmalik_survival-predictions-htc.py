import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split as Split
import xgboost as xgb

ordin = OrdinalEncoder()
ohe = OneHotEncoder(sparse_output=False)
warnings.filterwarnings("ignore")


df = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
df


df_test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")
df_test


df.info()


df.isna().sum()


def clean_data(df,threshold=50):
    # Step 1 : Drop columns with >50% nan values
    nan_percentage = df.isna().mean() * 100
    # We need an average fixed percentage that can affect the model.
    # Now we can analyse the columns with nan values more than 50%
    columns_to_drop = nan_percentage[nan_percentage > threshold].index
    #  Drop the columns
    df_dropped = df.drop(columns=columns_to_drop)
    
    # Step 2: Extract numerical and categorical columns
    categorical_cols = df_dropped.select_dtypes(include=['object']).columns
    numerical_cols = df_dropped.select_dtypes(include=['int','float']).columns
    # Step 3 : Replace numerical columns with mean values and categoricals with most frequent values
    
    # Categorical
    most_freq = df_dropped[categorical_cols].select_dtypes(include=['object']).mode().iloc[0]
    df_dropped[categorical_cols] = df_dropped[categorical_cols].fillna(most_freq)
    
    # Numerical
    mean_val = df_dropped[numerical_cols].select_dtypes(include=['float','int']).mean()
    df_dropped[numerical_cols] = df_dropped[numerical_cols ].fillna(mean_val)
    
    return df_dropped


df_train = clean_data(df)
df_train.info()


def encoder(df):
    categorical_clm = df.select_dtypes(include=['object']).columns.tolist()
    
    
    df[categorical_clm] = ordin.fit_transform(df[categorical_clm])
    
    return  df


df_train_enc = encoder(df_train)
df_train_enc


df_train_enc.isna().sum()


Y = df_train_enc['efs'].values
X = df_train_enc.drop(['efs','efs_time','ID'],axis=1)
X_tr = X.values
X_train,X_test,y_train,y_test = Split(X_tr,Y,test_size=0.25,random_state=42)


df_test = clean_data(df_test,threshold=100)
df_test


df_test.isna().sum().max(
)


IDs = df_test['ID'].values
TEST = encoder(df_test.drop('ID',axis=1))
TEST = TEST[X.columns.to_list()]
TEST.info()


# XGBoost Parameters
xgb_km_params = {
    'max_depth': 2,
    'learning_rate': 0.012887726635046637,
    'n_estimators': 5759,
    'reg_lambda': 0.014550241891247515,
    'random_state': 25,
    'objective': 'reg:squarederror',
    'enable_categorical': True
}


model2 = xgb.XGBRegressor(**xgb_km_params)
model2.fit(X_train,y_train)
model2.score(X_test,y_test)


preds = model2.predict(TEST)
preds


submission = pd.DataFrame({
    "ID" : IDs,
    "prediction" : preds
})
submission


submission.to_csv("submission.csv",index=False)

