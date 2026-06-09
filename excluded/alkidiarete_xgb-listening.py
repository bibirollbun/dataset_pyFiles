# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd       
import matplotlib as mat
import matplotlib.pyplot as plt    
import numpy as np
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import KFold
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


import warnings
warnings.simplefilter('ignore')


train= pd.read_csv(r"/kaggle/input/playground-series-s5e4/train.csv")
test= pd.read_csv(r"/kaggle/input/playground-series-s5e4/test.csv")

print("train shape :",train.shape)
print("test shape :",train.shape)


train.drop(columns=['id'], inplace=True)
test.drop(columns=['id'], inplace=True)


train = train[train['Number_of_Ads']<10]

features_to_impute = [
    'Episode_Length_minutes',
    'Guest_Popularity_percentage'
]

for feature in features_to_impute:
    value = train[feature].mean()
    train[feature] = train[feature].fillna(value)
    test[feature] = test[feature].fillna(value)


day_map = {'Sunday': 0, 'Monday': 1, 'Tuesday': 2, 'Wednesday': 3,
           'Thursday': 4, 'Friday': 5, 'Saturday': 6}
time_map = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
sentiment_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
podc_map = {'Mystery Matters': 0, 'Joke Junction': 1, 'Study Sessions': 2, 'Digital Digest': 3, 'Mind & Body': 4, 'Fitness First': 5, 'Criminal Minds': 6, 'News Roundup': 7, 'Daily Digest': 8, 'Music Matters': 9, 'Sports Central': 10, 'Melody Mix': 11, 'Game Day': 12, 'Gadget Geek': 13, 'Global News': 14, 'Tech Talks': 15, 'Sport Spot': 16, 'Funny Folks': 17, 'Sports Weekly': 18, 'Business Briefs': 19, 'Tech Trends': 20, 'Innovators': 21, 'Health Hour': 22, 'Comedy Corner': 23, 'Sound Waves': 24, 'Brain Boost': 25, "Athlete's Arena": 26, 'Wellness Wave': 27, 'Style Guide': 28, 'World Watch': 29, 'Humor Hub': 30, 'Money Matters': 31, 'Healthy Living': 32, 'Home & Living': 33, 'Educational Nuggets': 34, 'Market Masters': 35, 'Learning Lab': 36, 'Lifestyle Lounge': 37, 'Crime Chronicles': 38, 'Detective Diaries': 39, 'Life Lessons': 40, 'Current Affairs': 41, 'Finance Focus': 42, 'Laugh Line': 43, 'True Crime Stories': 44, 'Business Insights': 45, 'Fashion Forward': 46, 'Tune Time': 47}
genr_map = {'True Crime': 0, 'Comedy': 1, 'Education': 2, 'Technology': 3, 'Health': 4, 'News': 5, 'Music': 6, 'Sports': 7, 'Business': 8, 'Lifestyle': 9}

def preprocessing(df):
    df['Episode_Title'] = df['Episode_Title'].str.replace('Episode', '', regex=False).astype(int)
    df['Publication_Day'] = df['Publication_Day'].map(day_map)
    df['Publication_Time'] = df['Publication_Time'].map(time_map)
    df['Podcast_Name'] = df['Podcast_Name'].map(podc_map)
    df['Genre'] = df['Genre'].map(genr_map )
    df['Is_Weekend'] = df['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)
    df['Is_High_Host_Popularity'] = (df['Host_Popularity_percentage'] > 70).astype(int)
    df['Is_High_Guest_Popularity'] = (df['Guest_Popularity_percentage'] > 70).astype(int)
    df['Host_Guest_Popularity_Gap'] = df['Host_Popularity_percentage'] - df['Guest_Popularity_percentage']
    df['Is_Long_Episode'] = (df['Episode_Length_minutes'] > 60).astype(int)
    
    if 'Episode_Sentiment' in df.columns:
        df['Episode_Sentiment'] = df['Episode_Sentiment'].map(sentiment_map)

    return df

train = preprocessing(train)
test = preprocessing(test)


SEED = 42
N_SPLITS = 5


X = train.drop(columns=['Listening_Time_minutes'])
y = train['Listening_Time_minutes']

X['Episode_Length_minutes'] = np.log1p(X['Episode_Length_minutes'])
X['Host_Guest_Popularity_Gap'] = np.log1p(np.abs(X['Host_Guest_Popularity_Gap']))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


best_params = {
    'n_estimators': 769,
    'learning_rate': 0.024025766173931488,
    'max_depth': 17,
    'min_child_weight': 5,
    'subsample': 0.666496972140272,
    'colsample_bytree': 0.6379829756397679,
    'gamma': 3.663052996753313,
    'reg_alpha': 9.089264890544545,
    'reg_lambda': 9.072490097887034,
    'random_state': 42,
    'objective': 'reg:squarederror',
    'tree_method': 'gpu_hist'
}


kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

scores = []
models = []

for fold, (train_index, valid_index) in enumerate(tqdm(kf.split(X), total=N_SPLITS)):
    X_train, X_val = X.iloc[train_index], X.iloc[valid_index]
    y_train, y_val = y.iloc[train_index], y.iloc[valid_index]
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    evals_result = {}
    model = xgb.train(
        params=best_params,
        dtrain=dtrain,
        num_boost_round=1500,
        evals=[(dtrain, "train"), (dval, "valid")],
        early_stopping_rounds=100,
        evals_result=evals_result,
        verbose_eval=0
    )

    y_pred_val = model.predict(dval, iteration_range=(0, model.best_iteration))

    score = np.sqrt(mean_squared_error(y_val, y_pred_val))
    print(f'Fold: {fold+1} RMSE score: {np.mean(score):.5f}') 

    scores.append(score)
    models.append(model)


test = test.copy()
test['Episode_Length_minutes'] = np.log1p(test['Episode_Length_minutes'])
test['Host_Guest_Popularity_Gap'] = np.log1p(np.abs(test['Host_Guest_Popularity_Gap']))


dtest = xgb.DMatrix(test)
preds = np.mean([model.predict(dtest, iteration_range=(0, model.best_iteration)) for model in models], axis=0)

sub = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")
output = pd.DataFrame({
    "id": sub['id'],
    "Listening_Time_minutes": preds
})
output.to_csv("submission_ensemble.csv", index=False)
output.head()

