# %load_ext cudf.pandas


from tqdm import tqdm
from itertools import combinations
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
import gc 
import lightgbm as lgb
import warnings
from xgboost import XGBRegressor
import pickle

warnings.simplefilter('ignore')

train = pd.DataFrame(pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv').drop(columns = ['id']))
test = pd.DataFrame(pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv').drop(columns = ['id']))
original = pd.DataFrame(pd.read_csv('/kaggle/input/podcast-listening-time-prediction-dataset/podcast_dataset.csv'))
sub = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')

print(train.shape)
print(test.shape)

assert list(train.columns) == list(original.columns) ,f"Error: Column mismatch between original and train data"


original_clean = original.dropna(subset=['Listening_Time_minutes']).drop_duplicates()
train = pd.concat([train, original_clean], axis=0, ignore_index=True)


num_cols = test.select_dtypes(exclude=['object', 'category']).columns

# Find columns with cardinality less than num_rows / 100
low_cardinality_numerical = [
    col for col in num_cols
    if train[col].round(0).nunique() < len(train) / 100
]

low_cardinality_numerical


def feature_eng(df,original):
    podc_dict = {'Mystery Matters': 0, 'Joke Junction': 1, 'Study Sessions': 2, 'Digital Digest': 3, 'Mind & Body': 4, 'Fitness First': 5, 'Criminal Minds': 6, 'News Roundup': 7, 'Daily Digest': 8, 'Music Matters': 9, 'Sports Central': 10, 'Melody Mix': 11, 'Game Day': 12, 'Gadget Geek': 13, 'Global News': 14, 'Tech Talks': 15, 'Sport Spot': 16, 'Funny Folks': 17, 'Sports Weekly': 18, 'Business Briefs': 19, 'Tech Trends': 20, 'Innovators': 21, 'Health Hour': 22, 'Comedy Corner': 23, 'Sound Waves': 24, 'Brain Boost': 25, "Athlete's Arena": 26, 'Wellness Wave': 27, 'Style Guide': 28, 'World Watch': 29, 'Humor Hub': 30, 'Money Matters': 31, 'Healthy Living': 32, 'Home & Living': 33, 'Educational Nuggets': 34, 'Market Masters': 35, 'Learning Lab': 36, 'Lifestyle Lounge': 37, 'Crime Chronicles': 38, 'Detective Diaries': 39, 'Life Lessons': 40, 'Current Affairs': 41, 'Finance Focus': 42, 'Laugh Line': 43, 'True Crime Stories': 44, 'Business Insights': 45, 'Fashion Forward': 46, 'Tune Time': 47}
    genr_dict = {'True Crime': 0, 'Comedy': 1, 'Education': 2, 'Technology': 3, 'Health': 4, 'News': 5, 'Music': 6, 'Sports': 7, 'Business': 8, 'Lifestyle': 9}
    week_dict = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    time_dict = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
    sent_dict = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    
    df['Episode_Num'] = df['Episode_Title'].str[8:].astype('category')
    
    df['Genre'] = df['Genre'].replace(genr_dict)
    df['Podcast_Name'] = df['Podcast_Name'].replace(podc_dict)
    df['Publication_Day'] = df['Publication_Day'].replace(week_dict)
    df['Publication_Time'] = df['Publication_Time'].replace(time_dict)
    df['Episode_Sentiment'] = df['Episode_Sentiment'].replace(sent_dict)
    
    df['Genre'] = df['Genre'].astype('category')
    df['Podcast_Name'] = df['Podcast_Name'].astype('category')
    df['Publication_Day'] = df['Publication_Day'].astype('category')
    df['Publication_Time'] = df['Publication_Time'].astype('category')
    df['Episode_Sentiment'] = df['Episode_Sentiment'].astype('category')
    
    df['Episode_Length_minutes_missing']=df['Episode_Length_minutes'].isna().astype(int)
    df['Guest_Popularity_percentage_missing'] = df['Guest_Popularity_percentage'].isna().astype(int)

    if test['Number_of_Ads'].isna().sum() > 50:
        df['Number_of_Ads_missing'] = df['Number_of_Ads'].isna().astype(int)
    else:
        df = df[df['Number_of_Ads'].isna() == False]

    df['Guest_outlier'] = (df['Guest_Popularity_percentage'] > 100).astype(int)
    df['Host_outlier'] = (df['Host_Popularity_percentage'] > 100).astype(int)
    df['Episode_Length_outlier'] = (df['Episode_Length_minutes'] >= 150).astype(int)
    df['Ads_outlier'] =(df['Number_of_Ads'] > 10).astype(int)

    df.loc[df['Guest_Popularity_percentage'] > 100, 'Guest_Popularity_percentage'] = np.nan
    df.loc[df['Host_Popularity_percentage'] > 100, 'Host_Popularity_percentage'] = np.nan
    df.loc[df['Episode_Length_minutes'] >= 150, 'Episode_Length_minutes'] = np.nan
    df.loc[df['Number_of_Ads'] > 100, 'Guest_Popularity_percentage'] = np.nan

    
    # df['Episode_Length_minutes'] = df['Episode_Length_minutes'].fillna(original['Episode_Length_minutes'].median())
    # df['Guest_Popularity_percentage'] = df['Guest_Popularity_percentage'].fillna(original['Guest_Popularity_percentage'].median())
    # df['Host_Popularity_percentage'] = df['Host_Popularity_percentage'].fillna(original['Host_Popularity_percentage'].median())
    # df['Number_of_Ads'] = df['Number_of_Ads'].fillna(1)

    df['weekend'] = df['Publication_Day'].isin([5,6]).astype(int)
    df['host_guest_pop'] = (df['Host_Popularity_percentage']+ df['Guest_Popularity_percentage'])/2

    df["Ads_per_minute"] = (df["Number_of_Ads"] / (1+df["Episode_Length_minutes"]) ).astype('float64')
    df['Has_Ads'] = df['Number_of_Ads'].apply(lambda x: 1 if x > 0 else 0).astype(int)
    df['Is_Long_Episode'] = ((df['Episode_Length_minutes'] > 60) & (df['Episode_Length_minutes'] <= 90)).astype(int)
    df['Is_High_Host_Popularity'] = ((df['Host_Popularity_percentage'] > 70) & (df['Host_Popularity_percentage'] <= 90)).astype(int)
    df['Is_High_Guest_Popularity'] = ((df['Guest_Popularity_percentage'] > 70) & (df['Guest_Popularity_percentage'] <= 70)).astype(int)

    df['Is_Very_Long_Episode'] = (df['Episode_Length_minutes'] > 90).astype(int)
    df['Is_Very_High_Host_Popularity'] = (df['Host_Popularity_percentage'] > 90).astype(int)
    df['Is_Very_High_Guest_Popularity'] = (df['Guest_Popularity_percentage'] > 90).astype(int)  
    df['Length_Category'] = pd.qcut(df['Episode_Length_minutes'], q=20, labels=False, duplicates='drop')
    df['Host_Category'] = pd.qcut(df['Host_Popularity_percentage'], q=20, labels=False, duplicates='drop')
    df['Guest_Category'] = pd.qcut(df['Guest_Popularity_percentage'], q=20, labels=False, duplicates='drop')

    # df = df.drop(columns=['Episode_Title'])

    return df

train = feature_eng(train,original)
test = feature_eng(test,original)



from joblib import Parallel, delayed
from tqdm import tqdm
from itertools import combinations
import gc

encode_columns = ['Episode_Num','Episode_Sentiment', 'Publication_Day', 'Publication_Time','Podcast_Name']


interact_cols = pd.Series(encode_columns + low_cardinality_numerical).unique().tolist()

encoded_columns = []
pair_size = [3,4,2]

def process_columns(cols, train, test):
    new_col_name = '_'.join(cols)
    train_col = train[list(cols)].astype(str).agg('_'.join, axis=1).astype('category')
    test_col = test[list(cols)].astype(str).agg('_'.join, axis=1).astype('category')
    return new_col_name, train_col, test_col

results = Parallel(n_jobs=-1)(
    delayed(process_columns)(cols, train, test)
    for r in pair_size
    for cols in tqdm(list(combinations(interact_cols, r)), desc=f"Processing pairs of size {r}")
)
gc.collect()
# Add new columns to train and test
for new_col_name, train_col, test_col in results:
    train[new_col_name] = train_col
    test[new_col_name] = test_col
gc.collect()
# Save column names
encoded_columns = [name for name, _, _ in results]
del results


for k in range(7,10):
    n = f"round{k}"
    train[n] = train["Episode_Length_minutes"].round(k)
    test[n] = test["Episode_Length_minutes"].round(k)

X = train.drop(columns=['Listening_Time_minutes','Episode_Title'])
y = train['Listening_Time_minutes']
test = test.drop(columns = ['Episode_Title'])


features = list(X)


KFOLD = 20
RANDOM_STATE = 42
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
pd.options.mode.copy_on_write = True
from cuml.preprocessing import TargetEncoder
import joblib
import os
from sklearn.metrics import mean_squared_error

cv = KFold(KFOLD, random_state=RANDOM_STATE, shuffle=True)
y_pred_lgb = np.zeros(len(sub))
oof_lgb = np.zeros(len(train))

y_pred_xgb = np.zeros(len(sub))
oof_xgb = np.zeros(len(train))

i = 0
for j, (idx_train, idx_valid) in enumerate(cv.split(X, y)):
    print(f"#Fold {i+1} #")  
    i += 1
    start = time.time()
    X_train, y_train = train.iloc[idx_train][features + ['Listening_Time_minutes']], y.iloc[idx_train]
    X_valid, y_valid = X.iloc[idx_valid], y.iloc[idx_valid]
    X_test = test[X.columns].copy()

    encoder = TargetEncoder(n_folds=7, seed=42, stat="mean")

    for col in encoded_columns:
        X_train[col] = encoder.fit_transform(X_train[[col]], y_train).astype('float32')
        X_valid[col] = encoder.transform(X_valid[[col]]).astype('float32')
        X_test[col] = encoder.transform(X_test[[col]]).astype('float32')
        
    gc.collect()
    del encoder

    X_train = X_train.drop(['Listening_Time_minutes'],axis=1)
    
    model_lgb = lgb.LGBMRegressor(
        n_iter=1000,
        max_depth=-1,
        num_leaves=1024,
        colsample_bytree=0.25,
        subsample = 0.9,
        learning_rate=0.03,
        objective='l2',
        metric='rmse', 
        verbosity=-1,
        max_bin=1024,
        # device='gpu',
    )

    model_lgb.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(100)],
    ) 
    oof_lgb[idx_valid] = model_lgb.predict(X_valid)
    y_pred_lgb += model_lgb.predict(X_test)
    
    del model_lgb
    gc.collect()
    
    model_xgb = XGBRegressor(
        tree_method='gpu_hist',   
        predictor='gpu_predictor',
        max_depth=15,
        colsample_bytree=0.25,
        subsample=0.9,
        n_estimators=10_000,
        learning_rate=0.03,
        enable_categorical=True,
        early_stopping_rounds=100,  
        min_child_weight=10,
        device='cuda'  
    )

    model_xgb.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=100)
    
    oof_xgb[idx_valid] = model_xgb.predict(X_valid)
    y_pred_xgb += model_xgb.predict(X_test)
    
    print(np.sqrt(mean_squared_error(y_valid,oof_xgb[idx_valid])))
    gc.collect()
    end = time.time()
    print(f"Time Elapsed: {end-start} seconds")

    del X_train, y_train,X_valid, y_valid, X_test,model_xgb
    

y_pred_lgb /= KFOLD   
y_pred_xgb /= KFOLD




from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge

df_ensemble_train = pd.DataFrame({
    'lgb': oof_lgb,
    'xgb': oof_xgb,
    # 'cat': oof_cat
})

df_ensemble_test = pd.DataFrame({
    'lgb': y_pred_lgb,
    'xgb': y_pred_xgb,
    # 'cat': y_pred_cat
})

model_ridge = Ridge(alpha=0.0)
model_ridge.fit(df_ensemble_train, y)


y_pred = model_ridge.predict(df_ensemble_train)
y_pred_test = model_ridge.predict(df_ensemble_test)


rmse_train_ensemble = np.sqrt(mean_squared_error(y_pred, y))
rmse_train_xgb = np.sqrt(mean_squared_error(oof_xgb,y))
rmse_train_lgb = np.sqrt(mean_squared_error(oof_lgb,y))


print(f"Lgbm Train RMSE: {rmse_train_lgb}")
print(f"Xgboost Train RMSE: {rmse_train_xgb}")
print(f"Ensemble Train RMSE: {rmse_train_ensemble}") 





sub['Listening_Time_minutes'] = y_pred_test

df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv', dtype={'Episode_Length_minutes': str})
num_digits = df["Episode_Length_minutes"].astype(str).str.split(".", expand=True)[1].apply(lambda x: len(x) if x is not None else None)

mask = num_digits > 2
sub.loc[mask, 'Listening_Time_minutes'] = test.loc[mask, 'Episode_Length_minutes']

sub.to_csv('submission.csv', index=False)
sub.head()


