import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import make_column_transformer
from sklearn.model_selection import KFold
from itertools import combinations
from warnings import simplefilter
import xgboost as xgb
import numpy as np


lt_df_train = pd.read_csv(r"/kaggle/input/playground-series-s5e4/train.csv",index_col='id')
lt_df_test = pd.read_csv(r"/kaggle/input/playground-series-s5e4/test.csv",index_col='id')


outlier_indexes_ep_length_syn_train = lt_df_train[lt_df_train['Episode_Length_minutes']>250].index
lt_df_train.loc[outlier_indexes_ep_length_syn_train,['Episode_Length_minutes']] = lt_df_train['Episode_Length_minutes'].drop(outlier_indexes_ep_length_syn_train).mean(skipna=True).round(1)
display(lt_df_train.loc[outlier_indexes_ep_length_syn_train])


outlier_indexes_ep_length_syn_test = lt_df_test[lt_df_test['Episode_Length_minutes']>250].index
lt_df_test.loc[outlier_indexes_ep_length_syn_test,['Episode_Length_minutes']] = lt_df_train['Episode_Length_minutes'].mean(skipna=True).round(1)
display(lt_df_test.loc[outlier_indexes_ep_length_syn_test])


outlier_indexes_num_ads_syn_train = lt_df_train[lt_df_train['Number_of_Ads']>10].index
lt_df_train.loc[outlier_indexes_num_ads_syn_train,['Number_of_Ads']] = lt_df_train['Number_of_Ads'].drop(outlier_indexes_num_ads_syn_train).mean(skipna=True).round(2)
display(lt_df_train.loc[outlier_indexes_num_ads_syn_train])


outlier_indexes_num_ads_syn_test = lt_df_test[lt_df_test['Number_of_Ads']>10].index
lt_df_test.loc[outlier_indexes_num_ads_syn_test,['Number_of_Ads']] = lt_df_train['Number_of_Ads'].mean(skipna=True).round(2)
display(lt_df_test.loc[outlier_indexes_num_ads_syn_test])


lt_df_train['Episode_Length_minutes'] = lt_df_train.groupby(['Podcast_Name','Episode_Title'])['Episode_Length_minutes'].transform(lambda x: x.fillna(x.mean()))
display(lt_df_train[lt_df_train['Episode_Length_minutes'].isnull()].head())


lt_df_train['Guest_Popularity_percentage'] = lt_df_train.groupby(['Podcast_Name','Episode_Title'])['Guest_Popularity_percentage'].transform(lambda x: x.fillna(x.mean()))
display(lt_df_train[lt_df_train['Guest_Popularity_percentage'].isnull()].head())


lt_df_train['Number_of_Ads'] = lt_df_train.groupby(['Podcast_Name','Episode_Title'])['Number_of_Ads'].transform(lambda x: x.fillna(x.mean()))
display(lt_df_train[lt_df_train['Number_of_Ads'].isnull()].head())


lt_df_test['Episode_Length_minutes'] = lt_df_test.groupby(['Podcast_Name','Episode_Title'])['Episode_Length_minutes'].transform(lambda x: x.fillna(x.mean()))
display(lt_df_test[lt_df_test['Episode_Length_minutes'].isnull()].head())


lt_df_test['Guest_Popularity_percentage'] = lt_df_test.groupby(['Podcast_Name','Episode_Title'])['Guest_Popularity_percentage'].transform(lambda x: x.fillna(x.mean()))
display(lt_df_test[lt_df_test['Guest_Popularity_percentage'].isnull()].head())


X_train = lt_df_train.drop(['Listening_Time_minutes'],axis=1)
y_train = lt_df_train[['Listening_Time_minutes']]


ohe_columns = X_train.columns[X_train.dtypes=='object'] # select only object columns to One-Hot Encode.


enc = OneHotEncoder(sparse_output=False)
transformer = make_column_transformer((enc, ohe_columns), remainder='passthrough', verbose_feature_names_out=False)

# We convert the resulting arrays to DataFrames
transformed=transformer.fit_transform(X_train)
X_train = pd.DataFrame(
    transformed, 
    columns=transformer.get_feature_names_out(),
    index=X_train.index
)
lt_df_test = pd.DataFrame(
    transformer.transform(lt_df_test),
    columns=transformer.get_feature_names_out(),
    index=lt_df_test.index
)


scaler = MinMaxScaler()
scaler.set_output(transform="pandas")
X_train = scaler.fit_transform(X_train)
lt_df_test = scaler.transform(lt_df_test)


seed1 = 42
cv = KFold(7, random_state=seed1, shuffle=True)
pred_test = np.zeros((250000,))

# XGBoost learning_rate schedule
def lr_decay(epoch):
    if epoch < 115:
        return 0.05
    else:
        return 0.01
callbacks = xgb.callback.LearningRateScheduler(lr_decay)

# XGBoost parameters
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'seed': seed1,
    'max_depth': 19,
    'learning_rate': 0.03,
    'min_child_weight': 50,
    'reg_alpha': 5,
    'reg_lambda': 1,
    'subsample': 0.85,
    'colsample_bytree': 0.6,
    'colsample_bynode': 0.5,
    'device': "cuda"
}

for idx_train, idx_valid in cv.split(X_train):
    
    X_train_inner, y_train_inner = X_train.iloc[idx_train], y_train.iloc[idx_train]
    X_valid, y_valid = X_train.iloc[idx_valid], y_train.iloc[idx_valid]
    X_test = lt_df_test[X_train.columns].copy()

    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train_inner, label=y_train_inner)
    dval = xgb.DMatrix(X_valid, label=y_valid)
    dtest = xgb.DMatrix(X_test)

    # Train the model with early stopping
    model = xgb.train(
        params, 
        dtrain, 
        num_boost_round=1000000, 
        evals=[(dtrain, 'train'), (dval, 'validation')], 
        early_stopping_rounds=30, 
        verbose_eval=500,
        callbacks=[callbacks]
    )

    # Evaluate on validation set
    predictions = model.predict(dval)

    # Generate predictions for test set and save submission
    pred_test += np.maximum(0, np.minimum(120, model.predict(dtest)))
    print("----------------------------------------------------------------")

pred_test /= 7


sub_df = pd.DataFrame(
    pred_test,
    columns=['Listening_Time_minutes'],
    index=lt_df_test.index
)
sub_df.to_csv(r'/kaggle/working/submission.csv',index=True)

