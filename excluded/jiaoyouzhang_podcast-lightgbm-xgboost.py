import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

import gc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from itertools import combinations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from itertools import combinations

SEED = 42
n_splits = 5
n_estimators=5000
early_stopping_rounds = 100


train_data = pd.read_csv(r"/kaggle/input/playground-series-s5e4/train.csv")
test_data = pd.read_csv(r"/kaggle/input/playground-series-s5e4/test.csv")
data = pd.read_csv(r"/kaggle/input/playground-series-s5e4/sample_submission.csv")

print("train_data shape :",train_data.shape)
print("test_data shape :",test_data.shape)
print("data shape :",data.shape)


#train_data.head()


#train_data.isna().sum().sort_values(ascending=False)


#train_data.dtypes



vectorizer = TfidfVectorizer()
tfidf_train = vectorizer.fit_transform(train_data['Podcast_Name'])
feature_names = vectorizer.get_feature_names_out()
tfidf_train_df = pd.DataFrame(tfidf_train.toarray(), columns=feature_names)
tfidf_test = vectorizer.transform(test_data['Podcast_Name'])
tfidf_test_df = pd.DataFrame(tfidf_test.toarray(), columns=feature_names)



def data_process(df):
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

    df['Episode_Length_minutes'].fillna(df['Episode_Length_minutes'].median(), inplace=True)
    
    df = df.drop(columns=['Episode_Title'])
    return df

train_data=data_process(train_data)
test_data=data_process(test_data)
train_data['Difference'] = train_data['Listening_Time_minutes'] -train_data['Episode_Length_minutes'] 


cat_cols = train_data.select_dtypes(exclude=['number']).columns.tolist()
pair_size = [2, 3]

for r in pair_size:
    for cols in list(combinations(cat_cols, r)):
        new_col_name = '_'.join(cols)
        
        train_data[new_col_name] = train_data[list(cols)].astype(str).agg('_'.join, axis=1)
        train_data[new_col_name] = train_data[new_col_name].astype('category')
        
        test_data[new_col_name] = test_data[list(cols)].astype(str).agg('_'.join, axis=1)
        test_data[new_col_name] = test_data[new_col_name].astype('category')


train_data = pd.concat([train_data, tfidf_train_df], axis=1)
test_data = pd.concat([test_data, tfidf_test_df], axis=1)


X = train_data.drop(['id','Difference','Listening_Time_minutes'],axis=1)
y = train_data['Difference']
test = test_data.drop(['id'],axis=1)

print('true y min:',y.min())
print('true y max:',y.max())
print('true y mean:',y.mean())
print('true y median:',y.median())


kf = KFold(n_splits, shuffle=True, random_state=42)
kf_splits = kf.split(X)
scores1 = []
test_preds1 = []

lgbm_params1 = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': n_estimators,
    'learning_rate': 0.08,
    #'max_depth': 15,
    'num_leaves': 2048, 
    'reg_alpha' : 1,
    'reg_lambda' : 8,
    'colsample_bytree' : 0.7,
    'subsample' : 1.0,
    'subsample_freq' : 6,
    'seed': SEED,
    'verbose': -1,
    'device' : 'cpu' 
}

for i, (train_idx, val_idx) in enumerate(kf_splits):
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
    
    callbacks = [lgb.early_stopping(stopping_rounds=early_stopping_rounds),lgb.log_evaluation(period=1000)]
    model = lgb.LGBMRegressor(**lgbm_params1)
    model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)],eval_metric='rmse', categorical_feature=cat_cols, callbacks=callbacks)
    
    val_pred = model.predict(X_val_fold, num_iteration=model.best_iteration_)
    score = mean_squared_error(y_val_fold, val_pred,squared=False)
    scores1.append(score)
    
    test_pred = np.maximum(model.predict(test, num_iteration=model.best_iteration_),0)
    test_preds1.append(test_pred)

    
    print(f'LightGBM Fold {i + 1} rmse: {score}')
print(f'LightGBM rmse: {np.mean(scores1):.5f};');



y_preds_lgb = np.mean(test_preds1, axis=0)
y_preds_lgb = y_preds_lgb
print('predict mean :',y_preds_lgb.mean())
print('predict median :',np.median(y_preds_lgb))
y_preds_lgb = np.clip(y_preds_lgb,0,119.97)
print('predict mean final:',y_preds_lgb.mean())
print('predict median final:',np.median(y_preds_lgb))


kf = KFold(n_splits, shuffle=True, random_state=43)
kf_splits = kf.split(X)
scores1 = []
test_preds1 = []

xgb_params1 = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'n_estimators': n_estimators,
    'learning_rate': 0.08,
    #'max_depth': 15,
    'max_leaves': 0,
    'gamma': 0.5,
    'subsample': 1.0,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.8,
    'reg_lambda': 4,
    'max_bin' : 1024,
    'seed': SEED,
    'tree_method': 'hist',  # 使用 hist 方法加速训练
    'device': 'cpu'
}

for i, (train_idx, val_idx) in enumerate(kf_splits):
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

    model = xgb.XGBRegressor(**xgb_params1, enable_categorical=True)
    model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], early_stopping_rounds=early_stopping_rounds, verbose=1000)

    val_pred = model.predict(X_val_fold)
    score = mean_squared_error(y_val_fold, val_pred, squared=False)
    scores1.append(score)

    test_pred = np.maximum(model.predict(test), 0)
    test_preds1.append(test_pred)

    print(f'XGBoost Fold {i + 1} rmse: {score}')
print(f'XGBoost rmse: {np.mean(scores1):.5f};')


y_preds_xgb = np.mean(test_preds1, axis=0)
y_preds_xgb = y_preds_xgb
print('predict mean :',y_preds_xgb.mean())
print('predict median :',np.median(y_preds_xgb))
y_preds_xgb = np.clip(y_preds_xgb,0,119.97)
print('predict mean final:',y_preds_xgb.mean())
print('predict median final:',np.median(y_preds_xgb))


y_preds = y_preds_lgb*0.8 + y_preds_xgb*0.2 
# Save predictions for submission
submission = pd.DataFrame({'id': test_data['id'], 'Listening_Time_minutes': y_preds+test_data['Episode_Length_minutes']})
submission.to_csv('submission.csv', index=False)
print(submission.head())

