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


df=pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')


df.info()


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
sample_sub = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


def feature_engineering(df):
    # Handle missing values
    df['Guest_Popularity_percentage'].fillna(0, inplace=True)
    df['Episode_Length_minutes'].fillna(df['Episode_Length_minutes'].median(), inplace=True)
    df['Number_of_Ads'].fillna(df['Number_of_Ads'].median(), inplace=True)
    
    # Fixed Time Features - Handle text-based time descriptions
    time_mapping = {
        'Morning': 8,
        'Afternoon': 14,
        'Evening': 18,
        'Night': 22,
        'Midnight': 0,
        'Noon': 12
    }
    df['Publication_Time'] = df['Publication_Time'].map(time_mapping)
    
    # Day encoding
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['Publication_Day'] = pd.Categorical(df['Publication_Day'], categories=day_order).codes
    
    # Interaction features
    df['Host_Guest_Interaction'] = df['Host_Popularity_percentage'] * df['Guest_Popularity_percentage']
    df['Ads_Length_Ratio'] = df['Number_of_Ads'] / (df['Episode_Length_minutes'] + 1)
    
    # Text features
    df['Title_Length'] = df['Episode_Title'].str.len()
    df['Title_Word_Count'] = df['Episode_Title'].str.split().str.len()
    
    # Popularity bins
    df['Host_Pop_Bin'] = pd.cut(df['Host_Popularity_percentage'], bins=5, labels=False)
    df['Guest_Pop_Bin'] = pd.cut(df['Guest_Popularity_percentage'], bins=5, labels=False)
    
    # Sentiment encoding
    sentiment_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
    df['Episode_Sentiment'] = df['Episode_Sentiment'].map(sentiment_map)
    
    # Drop columns that won't be used
    df.drop(['id', 'Episode_Title'], axis=1, inplace=True, errors='ignore')
    
    return df


# Apply fixed feature engineering
train_fe = feature_engineering(train.copy())
test_fe = feature_engineering(test.copy())


# Target and features
y = train_fe['Listening_Time_minutes']
X = train_fe.drop('Listening_Time_minutes', axis=1)


# TF-IDF for Podcast Names
podcast_names = pd.concat([X['Podcast_Name'], test_fe['Podcast_Name']])
tfidf = TfidfVectorizer(max_features=50)
tfidf.fit(podcast_names)


X_podcast = tfidf.transform(X['Podcast_Name'])
test_podcast = tfidf.transform(test_fe['Podcast_Name'])


svd = TruncatedSVD(n_components=10, random_state=42)
X_podcast_svd = svd.fit_transform(X_podcast)
test_podcast_svd = svd.transform(test_podcast)


X = pd.concat([
    X.drop('Podcast_Name', axis=1),
    pd.DataFrame(X_podcast_svd, columns=[f'Podcast_SVD_{i}' for i in range(10)])
], axis=1)


test_fe = pd.concat([
    test_fe.drop('Podcast_Name', axis=1),
    pd.DataFrame(test_podcast_svd, columns=[f'Podcast_SVD_{i}' for i in range(10)])
], axis=1)


X = pd.get_dummies(X, columns=['Genre'])
test_fe = pd.get_dummies(test_fe, columns=['Genre'])


missing_cols = set(X.columns) - set(test_fe.columns)
for col in missing_cols:
    test_fe[col] = 0
test_fe = test_fe[X.columns]


# Fixed LightGBM training function
def run_lgb(X, y, test_fe, n_folds=5):
    oof_preds = np.zeros(X.shape[0])
    test_preds = np.zeros(test_fe.shape[0])
    scores = []
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        
        lgb_params = {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'min_child_samples': 20,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'n_estimators': 2000,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_valid = lgb.Dataset(X_valid, y_valid)
        
        model = lgb.train(
            lgb_params,
            lgb_train,
            valid_sets=[lgb_valid],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=True),
                lgb.log_evaluation(100)
            ]
        )
        
        # Predictions
        oof_preds[valid_idx] = model.predict(X_valid)
        test_preds += model.predict(test_fe) / n_folds
        
        # Score
        fold_score = mean_absolute_error(y_valid, oof_preds[valid_idx])
        scores.append(fold_score)
        print(f"Fold {fold+1} MAE: {fold_score:.4f}")
    
    print(f"Overall MAE: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    return oof_preds, test_preds


oof_preds, test_preds = run_lgb(X, y, test_fe)


submission = pd.DataFrame({
    'id': test['id'],
    'Listening_Time_minutes': test_preds
})


target_mean = y.mean()
target_std = y.std()
submission['Listening_Time_minutes'] = submission['Listening_Time_minutes'].clip(
    target_mean - 3*target_std,
    target_mean + 3*target_std
)


submission.to_csv('submission.csv', index=False)
print("Submission file created!")




