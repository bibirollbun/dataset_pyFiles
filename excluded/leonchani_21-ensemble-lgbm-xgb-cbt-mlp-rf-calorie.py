# Table manipulation, calculating
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 100) # increase the maximum number of columns

# Saving model
import xgboost as xgb
import joblib

# Ignore all warnings
import warnings
warnings.simplefilter("ignore")


df_train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


df_train


df_test


mapping = {'male': 1.5, 'female': 1.0}
df_train['Sex'] = df_train['Sex'].replace(mapping)
df_test['Sex']  = df_test['Sex'].replace(mapping)


def standardize_dataframe(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Standardizes the columns of the specified DataFrame and returns the updated original DataFrame.
    
    Args:
        df (pd.DataFrame): The DataFrame to standardize.
        cols (list[str]): A list of column names to standardize.
    
    Returns:
        pd.DataFrame: The DataFrame with the specified columns standardized (modifies the original DataFrame).

    """
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(df[cols])
    scaled_values = scaler.transform(df[cols])
    df[cols] = scaled_values

    return df


# columns_to_standardize = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads']
columns_to_drop = ['id', 'Calories']
columns_to_standardize = df_train.copy().drop(columns=columns_to_drop).columns

standardize_dataframe(df_train, columns_to_standardize)
standardize_dataframe(df_test, columns_to_standardize)


# Loading Models
light_gbm     = joblib.load('/kaggle/input/model_for_ensemble/scikitlearn/default/1/LightGBM.joblib')
xgboost       = joblib.load('/kaggle/input/model_for_ensemble/scikitlearn/default/1/XGBoost.joblib')
catboost      = joblib.load('/kaggle/input/model_for_ensemble/scikitlearn/default/1/CBT.joblib')
mlp           = joblib.load('/kaggle/input/model_for_ensemble/scikitlearn/default/1/MLP.joblib')
random_forest = joblib.load('/kaggle/input/model_for_ensemble/scikitlearn/default/1/RandomForest.joblib')


test_id = df_test["id"]
test = df_test.drop(columns=['id'])

lgbm_submit_score = []
xgb_submit_score  = []
cbt_submit_score  = []
mlp_submit_score  = []
rf_submit_score   = []


for fold_, model in enumerate(light_gbm):
    pred_ = model.predict(test)
    lgbm_submit_score.append(pred_)

lgbm_pred = np.mean(lgbm_submit_score, axis=0)
lgbm_pred = np.expm1(lgbm_pred)


# for fold_, model in enumerate(xgboost):
#     dtest = xgb.DMatrix(test)
#     pred_ = model.predict(dtest)
#     xgb_submit_score.append(pred_)

# xgb_pred = np.mean(xgb_submit_score, axis=0)


for fold_, model in enumerate(catboost):
    pred_ = model.predict(test)
    cbt_submit_score.append(pred_)

cbt_pred = np.mean(cbt_submit_score, axis=0)
cbt_pred = np.expm1(cbt_pred)


# for fold_, model in enumerate(mlp):
#     pred_ = model.predict(test)
#     mlp_submit_score.append(pred_)

# mlp_pred = np.mean(mlp_submit_score, axis=0)
# mlp_pred = np.expm1(mlp_pred)


# for fold_, model in enumerate(random_forest):
#     pred_ = model.predict(test)
#     rf_submit_score.append(pred_)

# rf_pred = np.mean(rf_submit_score, axis=0)
# rf_pred = np.expm1(rf_pred)


# Merge Predict Score
df_lgbm_score = pd.DataFrame(lgbm_pred, columns=['lgbm_score']).reset_index()
df_cbt_score  = pd.DataFrame(cbt_pred, columns=['cbt_score']).reset_index()
# df_mlp_score  = pd.DataFrame(mlp_pred, columns=['mlp_score']).reset_index()
# df_rf_score   = pd.DataFrame(rf_pred, columns=['rf_score']).reset_index()


df_score = pd.merge(df_lgbm_score, df_cbt_score, how = 'inner', on = 'index')
# df_score = pd.merge(df_score, df_mlp_score, how = 'inner', on = 'index')
# df_score = pd.merge(df_score, df_rf_score, how = 'inner', on = 'index')
df_score


# df_score['pred'] = (df_score['lgbm_score'] + df_score['cbt_score'] + df_score['mlp_score'] + df_score['rf_score']) / 4
df_score['pred'] = (df_score['lgbm_score'] + df_score['cbt_score']) / 2
df_score


submission = pd.DataFrame({
    'id': test_id,
    'Calories': df_score['pred']
})

submission['Calories'] = submission['Calories'].apply(lambda x: 0 if x < 0 else x)

# Save
submission.to_csv('submission.csv', index=False)

submission


!pip install watermark


%load_ext watermark
%watermark -n -u -v -iv -w -p pytensor,aeppl,xarray

