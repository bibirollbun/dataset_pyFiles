import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn


df=pd.read_csv('/kaggle/input/bluebook-for-bulldozers/TrainAndValid.csv',low_memory=False)


df.info()


df.isna().sum()


df.saledate


fig,ax=plt.subplots()
ax.scatter(df['saledate'],df['SalePrice'])


df.SalePrice.plot.hist(bins=100)


df=pd.read_csv('/kaggle/input/bluebook-for-bulldozers/TrainAndValid.csv',low_memory=False,parse_dates=['saledate'])


df.saledate


fig,ax=plt.subplots()
ax.scatter(df['saledate'],df['SalePrice'])


df=df.sort_values(by=['saledate'],ascending=True)
df.saledate


df_temp=df.copy()


df_temp


df_temp['saleyear']=df_temp.saledate.dt.year
df_temp['salemonth']=df_temp.saledate.dt.month
df_temp['saleday']=df_temp.saledate.dt.day
df_temp['saledayofweek']=df_temp.saledate.dt.dayofweek
df_temp['saledayofyear']=df_temp.saledate.dt.dayofyear


df_temp


df_temp=df_temp.drop('saledate',axis=1)


# df_temp.saledate


df_temp.state.value_counts()


df_temp.dtypes


pd.api.types.is_string_dtype(df_temp['state'])


for label,content in df_temp.items():
    if pd.api.types.is_object_dtype(content):
        print(label)


for label,content in df_temp.items():
    if pd.api.types.is_object_dtype(content):
        df_temp[label]=content.astype('category').cat.as_ordered()


df_temp.info()


df_temp.state.cat.categories


df_temp.state.cat.codes


print(df_temp.isna().sum()/len(df_temp)*100)


for label,content in df_temp.items():
    if pd.api.types.is_numeric_dtype(content):
        print(label)


for label,content in df_temp.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            print(label)


for label,content in df_temp.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            df_temp[label+'_is_missing']=pd.isnull(content)
            df_temp[label]=content.fillna(content.median())


df_temp.auctioneerID_is_missing.value_counts()


for label,content in df_temp.items():
    if not pd.api.types.is_numeric_dtype(content):
        print(label)


pd.Categorical(df_temp['state']).codes


for label,content in df_temp.items():
    if not pd.api.types.is_numeric_dtype(content):
        df_temp[label+'_is_missing']=pd.isnull(content)
        df_temp[label]=pd.Categorical(content).codes+1


pd.Categorical(df_temp['UsageBand']).codes


df_temp.info()


df_temp.T


df_temp.isna().sum().sum()


for label,content in df_temp.items():
    if not pd.api.types.is_numeric_dtype(content):
        print(label)


from sklearn.ensemble import RandomForestRegressor


%%timeit
model=RandomForestRegressor(n_jobs=-1,random_state=42)
model.fit(df_temp.drop('SalePrice',axis=1),df_temp['SalePrice'])


df_val=df_temp[df_temp.saleyear==2012]
df_train=df_temp[df_temp.saleyear!=2012]
len(df_val),len(df_train)


X_train,y_train=df_train.drop('SalePrice',axis=1),df_train.SalePrice
X_valid,y_valid=df_val.drop('SalePrice',axis=1),df_val.SalePrice

X_train.shape,y_train.shape,X_valid.shape,y_valid.shape


from sklearn.metrics import mean_squared_log_error

def root_mean_squared_log_error(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

def show_scores(model):
    train_preds=model.predict(X_train)
    val_preds=model.predict(X_valid)
    scores={'training rmsle': root_mean_squared_log_error(y_train,train_preds),
           'valid rsmle': root_mean_squared_log_error(y_valid,val_preds)}
    return scores


model=RandomForestRegressor(n_jobs=-1,random_state=42,max_samples=10000)


%%time
model.fit(X_train,y_train)


show_scores(model)


from sklearn.model_selection import RandomizedSearchCV

rf_grid={'n_estimators':np.arange(10,100,10),
         'max_depth':[None,3,5,100,],
         'min_samples_split':np.arange(2,20,2),
         'min_samples_leaf':np.arange(1,20,2),
         'max_features':[0.5,1,'sqrt','auto'],
         'max_samples':[10000]}

rs_model=RandomizedSearchCV(RandomForestRegressor(n_jobs=-1,
                                                  random_state=42),
                            param_distributions=rf_grid)

rs_model.fit(X_train,y_train)


rs_model.best_params_


show_scores(rs_model)


model100=RandomForestRegressor(n_estimators=40,
                               min_samples_leaf=1,
                               min_samples_split=14,
                               max_features=0.5,
                               n_jobs=-1,
                               random_state=42)

model100.fit(X_train,y_train)


show_scores(model100)


df_test=pd.read_csv('/kaggle/input/bluebook-for-bulldozers/Test.csv',
                    low_memory=False,
                    parse_dates=['saledate'])
df_test


# test_preds=model100.predict(df_test)


def preprocess_data(df):
    df['saleyear']=df.saledate.dt.year
    df['salemonth']=df.saledate.dt.month
    df['saleday']=df.saledate.dt.day
    df['saledayofweek']=df.saledate.dt.dayofweek
    df['saledayofyear']=df.saledate.dt.dayofyear

    df=df.drop('saledate',axis=1)

    for label,content in df.items():
        if pd.api.types.is_numeric_dtype(content):
            if pd.isnull(content).sum():
                df[label+'_is_missing']=pd.isnull(content)
                df[label]=content.fillna(content.median())

        if not pd.api.types.is_numeric_dtype(content):
            df[label+'_is_missing']=pd.isnull(content)
            df[label]=pd.Categorical(content).codes+1

    return df
        


df_test=preprocess_data(df_test)
df_test


set(X_train.columns)-set(df_test.columns)


df_test['auctioneerID_is_missing']=False
df_test


df_all=pd.read_csv('/kaggle/input/bluebook-for-bulldozers/TrainAndValid.csv',
                   low_memory=False,
                   parse_dates=['saledate'])

df_all=preprocess_data(df_all)

X_all = df_all.drop("SalePrice", axis=1)
y_all = df_all["SalePrice"]

final_model = RandomForestRegressor(
    n_estimators=40,
    min_samples_leaf=1,
    min_samples_split=14,
    max_features=0.5,
    n_jobs=-1,
    random_state=90
)

final_model.fit(X_all, y_all)


# test_preds=final_model.predict(df_test)


df_test=df_test.reindex(columns=X_all.columns, fill_value=0)


test_preds=final_model.predict(df_test)
test_preds


df_preds=pd.DataFrame()
df_preds['SalesID']=df_test['SalesID']
df_preds['SalesPrice']=test_preds
df_preds


df_preds.to_csv('test_predictions.csv',index=False)


final_model.feature_importances_


def plot_features(columns, importances, n=20):
    df=(pd.DataFrame({'features':columns,
                     'feature_importances':importances})
       .sort_values('feature_importances',ascending=False)
       .reset_index(drop=True))

    fig,ax=plt.subplots()
    ax.barh(df['features'][:n],df['feature_importances'][:n])
    ax.set_ylabel('features')
    ax.set_xlabel('feature importance')
    ax.invert_yaxis()


plot_features(X_all.columns,final_model.feature_importances_)


from joblib import dump, load

dump(final_model,'buldozer_price_model_1.joblib')




