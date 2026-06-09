# %load_ext cudf.pandas
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', 100)

TARGET = 'Listening_Time_minutes'
CATS = ['Podcast_Name', 'Episode_Num', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
NUMS = ['Episode_Length_minutes', 'Host_Popularity_percentage', 
        'Guest_Popularity_percentage', 'Number_of_Ads']

FEATURES = CATS + NUMS

oof_pred_name = 'imputer'


import warnings
warnings.simplefilter('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv', index_col='id')
print(f"Train shape: {train.shape}")
print(f"Test  shape: {test.shape}")
train.head(3)


def feature_eng(df):
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
    
    df = df.drop(columns=['Episode_Title'])
    return df

train = feature_eng(train)
test = feature_eng(test)

train.head(3)


import gc

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from xgboost import XGBRegressor
from copy import deepcopy


from copy import deepcopy
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

FOLDS = 5

impute_methods = {
    'none': None,
    'mean': SimpleImputer(strategy='mean'),
    'median': SimpleImputer(strategy='median'),
    'iterative': IterativeImputer(random_state=42)
}

results = {}

for name, imputer in impute_methods.items():
    print(f"\n==== Imputation Method: {name} ====")

    train_cp = deepcopy(train)
    test_cp = deepcopy(test)

    if imputer is not None:
        train_cp[FEATURES] = imputer.fit_transform(train_cp[FEATURES])
        test_cp[FEATURES] = imputer.transform(test_cp[FEATURES])
    # else: keep original missing values for 'none'

    FEATURES = [col for col in train.columns if col != TARGET]

    oof = np.zeros(len(train))
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

    for i, (train_idx, valid_idx) in enumerate(kf.split(train_cp)):
        print(f"--- Fold {i+1} / {FOLDS} ---")

        X_train = train_cp.loc[train_idx, FEATURES].reset_index(drop=True)
        y_train = train_cp.loc[train_idx, TARGET].reset_index(drop=True)
        X_valid = train_cp.loc[valid_idx, FEATURES].reset_index(drop=True)
        y_valid = train_cp.loc[valid_idx, TARGET].reset_index(drop=True)

        model = XGBRegressor(
            n_estimators=10000,
            learning_rate=0.1,
            max_depth=10,
            colsample_bytree=0.7,
            objective='reg:squarederror',
            tree_method='hist',
            random_state=42,
            enable_categorical=True
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            early_stopping_rounds=100,
            verbose=500
        )

        oof[valid_idx] = model.predict(X_valid)
        del X_train, y_train, X_valid, y_valid, model
        gc.collect()

    rmse_score = mean_squared_error(train[TARGET], oof, squared=False)
    results[name] = rmse_score
    print(f"RMSE ({name}): {rmse_score:.5f}")

print("\n==== Summary RMSE ====")
for method, score in results.items():
    print(f"{method}: {score:.5f}")




