import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
import supplemental_english as supplement

from xgboost import XGBRegressor
from sklearn.metrics import  confusion_matrix, accuracy_score, mean_absolute_percentage_error
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler , LabelEncoder
from sklearn.pipeline import Pipeline



train_df = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/train.csv')
print(f'train dataset {train_df.shape}')
test_df = pd.read_csv('/kaggle/input/russian-car-plates-prices-prediction/test.csv')
print(f'test dataset {test_df.shape}')



train_ds = train_df.copy()
test_ds = test_df.copy()


train_df.head(5)


train_df.info()


train_df.isnull().sum()


test_df.isnull().sum()


numeric_col = train_df.drop(columns=['id','price']).select_dtypes(include=['int64']).columns.tolist()
categorcal_col = train_df.drop(columns=['id','price']).select_dtypes(include=['object']).columns.tolist()


numeric_transformer = SimpleImputer(strategy='mean')
categorcal_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_col),
        ('cat', categorcal_transformer, categorcal_col)
    ]
)


X = train_df.drop(columns=['id','price'])
y = train_df['price']

X = preprocessor.fit_transform(X)

X_test = test_df.drop(columns=['id'])
X_test = preprocessor.transform(X_test)


X.shape


X


X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=42, test_size=0.2)

print(X_train.shape)


randomReg = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=3, ccp_alpha=0.5, random_state=42)


#randomReg.fit(X_train,y_train)


#X_pred = randomReg.predict(X_val)
#randomReg.score(X_pred, y_val)



# mape = mean_absolute_percentage_error(y_val, X_pred)
# print(f'MAPE : {mape:.2f}')


# test_pred = randomReg.predict(X_test)


# tree_reg = DecisionTreeRegressor(splitter='best', random_state=42, max_depth=10, min_samples_leaf=2, ccp_alpha=1)

# tree_reg.fit(X_train, y_train)


# X_pred = tree_reg.predict(X_val)
# mape = mean_absolute_percentage_error(y_val, X_pred)
# print(f'MAPE : {mape:.2f}')


# test_pred_2 = tree_reg.predict(X_test)


# adaboost = AdaBoostRegressor(n_estimators=100, learning_rate=0.0001,random_state=42, loss='exponential')
# adaboost.fit(X_train,y_train)


# X_pred_3 = adaboost.predict(X_val)
# rmse_3 = mean_absolute_percentage_error(y_val, X_pred_3)
# print(f'RMSE : {rmse_3:.2f}')


# test_pred_3 = adaboost.predict(X_test)


# xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.005 , tree_method="hist", device="cuda")
# xgb_model.fit(X_train, y_train, 
#              eval_set=[(X_val, y_val)], verbose=False)


# X_pred_4 = xgb_model.predict(X_val)
# rmse_4 = mean_absolute_percentage_error(y_val, X_pred_4)
# print(f'RMSE : {rmse_4:.2f}')


train_ds.head(5), test_ds.head(5)


train_ds.info()


gov_rows =[]

for (letters, (num_from, num_to), region_code), (desc, forbidden, advantage, significance) in supplement.GOVERNMENT_CODES.items():
    gov_rows.append({
        'letters' : letters,
        'num_from' : num_from,
        'num_to' : num_to,
        'region_code' : region_code,
        'forbidden_to_buy': bool(forbidden),
        'road_advantage': bool(advantage),
        'significance_level' : significance
    })

region_rows = []

for region , codes in supplement.REGION_CODES.items():
    for code in codes:
        region_rows.append({'region_code' :code})


df_region = pd.DataFrame(region_rows)
df_gov = pd.DataFrame(gov_rows)


df_gov.head(5)


df_region.head(5)


df_govs = pd.merge(df_gov, on='region_code', how='left', right=df_region)
df_govs.replace(to_replace=True,value=1,inplace=True)
df_govs.replace(to_replace=False,value=0,inplace=True)
df_govs.shape


new_train = train_ds.copy()
new_test = test_ds.copy()
new_train.head()


def extract_plate_feature(df):
    df = df.copy()

    df['plate_str'] = df['plate'].astype('str')

    # region code: last 2–3 digits
    df['region_code'] = df['plate_str'].str.extract(r'(\d{2,3}$)')[0]

    # prefix letters: first 1–3 characters
    
    df['prefix'] = df['plate_str'].str.extract(r'^([A-ZА-Я]{1,3})')[0]

    # numeric block: three digits
    
    df['number'] = df['plate_str'].str.extract(r'([0-9]{3})')[0]

    # Date parts

    df['date'] = pd.to_datetime(df['date'])

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day']    = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday


    # government‐plate flag
    df['is_gov'] = df['prefix'].isin(supplement.GOVERNMENT_CODES).astype(int)


    # numeric region code directly

    df['region_num'] = pd.to_numeric(df['region_code'], errors='coerce').fillna(0).astype(int)

    return df


new_train = extract_plate_feature(new_train)
new_test = extract_plate_feature(new_test)


new_train.info()


sns.pairplot(new_test)


# find object columns
obj_data=new_train.select_dtypes(include=['O'])
obj_data
obj_data.columns


le_pref = LabelEncoder()
le_reg  = LabelEncoder()
ohe_num = OneHotEncoder()

new_train['pref_enc'] = le_pref.fit_transform(new_train['prefix'])
new_train['plate_enc'] = le_pref.fit_transform(new_train['plate'])
new_train['plate_str_enc'] = le_pref.fit_transform(new_train['plate_str'])
new_train['reg_enc'] = le_reg.fit_transform(new_train['region_code'])
new_train['number_enc'] = le_reg.fit_transform(new_train['number'])
# new_train['letters_enc'] = le_pref.fit_transform(new_train['letters'])


new_test['pref_enc'] = le_pref.fit_transform(new_test['prefix'])
new_test['plate_enc'] = le_pref.fit_transform(new_test['plate'])
new_test['plate_str_enc'] = le_pref.fit_transform(new_test['plate_str'])
new_test['reg_enc'] = le_reg.fit_transform(new_test['region_code'])
new_test['number_enc'] = le_reg.fit_transform(new_test['number'])
# new_test['letters_enc'] = le_pref.fit_transform(new_test['letters'])


new_train.head(5)


new_train.info()


# test_X['number'] = test_X['number'].astype(int)
# test_X.info()


new_train['number'] = new_train['number'].astype(int)
new_train.info()


X = new_train.drop(columns=['plate', 'plate_str', 'region_code', 'prefix','date','price',],axis=1)
y = new_train['price']

test_X = new_test.drop(columns=['plate', 'plate_str', 'region_code', 'prefix','date','price',],axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


test_X.info()


st= MinMaxScaler()
st.fit(X_train)


sc_x_train=st.transform(X_train)
sc_x_test=st.transform(X_test)


sns.distplot(sc_x_test)


randomReg_model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=3, ccp_alpha=0.5, random_state=42)


# randomReg_model.fit(sc_x_train, y_train)
# randomReg_model.score(sc_x_test, y_test)


# predicted_randReg = randomReg_model.predict(test_X)


Random_classi_model = RandomForestClassifier(
    n_estimators=500, 
    criterion='entropy', 
    min_samples_split=2, 
    random_state=42, 
    n_jobs=1, 
    max_depth=10, 
    ccp_alpha=0.05
)


# Random_classi_model.fit(sc_x_train, y_train)
# Random_classi_model.score(sc_x_test, y_test)


#predict_rand_class = Random_classi_model.predict(test_X)


tree_model = DecisionTreeClassifier(splitter='best', criterion='entropy', max_depth=10,  min_samples_split=3, ccp_alpha=1.0, )

tree_model.fit(sc_x_train, y_train)

tree_model.score(sc_x_test, y_test)


predict_tree = tree_model.predict(test_X)


logit_reg = LogisticRegression(penalty='l2', C=0.8, tol=0.0001, random_state=42, n_jobs=1, max_iter=100, fit_intercept=True)


# logit_reg.fit(sc_x_train, y_train)
# logit_reg.score(sc_x_test,y_test)


# pred_test_X = logit_reg.predict(test_X)


# xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.005 , tree_method="hist", device="cuda")
# xgb_model.fit(sc_x_train, y_train)
             


# xgb_model.score(sc_x_test,y_test)


#predicted_xgb = xgb_model.predict(test_X)


submission = pd.DataFrame({
    'id' : test_df['id'],
    'price' : predict_tree
})

submission.to_csv('submission.csv', index=False)

submission.head(5)




