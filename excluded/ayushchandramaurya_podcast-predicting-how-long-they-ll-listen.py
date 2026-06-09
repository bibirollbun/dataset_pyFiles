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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy import stats


train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


train.head()


test.head()


train.info()


test.info()


train.describe()


print(train.duplicated().sum())
print(test.duplicated().sum())


target_column = "Listening_Time_minutes"

y_train = train[target_column]
X_train = train.drop(columns=[target_column])
X_test = test


numerical_features = X_train.select_dtypes(include=['number']).columns.tolist()
categorical_features =X_train.select_dtypes(exclude=['number']).columns.tolist()



print("Numerical Features:", numerical_features)  
print("Categorical Features:", categorical_features)


X_train[numerical_features] = X_train[numerical_features].fillna(X_train[numerical_features].median())
X_train[categorical_features] = X_train[categorical_features].fillna(X_train[categorical_features].mode().iloc[0])

X_test[numerical_features] = X_test[numerical_features].fillna(X_test[numerical_features].median())
X_test[categorical_features] = X_test[categorical_features].fillna(X_test[categorical_features].mode().iloc[0])



X_train.isnull().sum()


X_test.isnull().sum()


from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

def engineer_features(X_train, X_test):
    combined = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)

    # 1. Ad Density
    combined['ads_per_minute'] = combined['Number_of_Ads'] / (combined['Episode_Length_minutes'] + 1e-3)

    # 2. Is Weekend
    combined['is_weekend'] = combined['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)

    # 3. Time of Day Features
    combined['is_morning'] = (combined['Publication_Time'] == 'Morning').astype(int)
    combined['is_night'] = (combined['Publication_Time'] == 'Night').astype(int)

    # 4. Episode Length Buckets
    combined['length_bucket'] = pd.cut(combined['Episode_Length_minutes'], bins=[0, 30, 60, 90, 200],
                                       labels=['short', 'medium', 'long', 'very_long'])
     # --- New Features Around Episode_Length_minutes ---
    combined['length_times_popularity'] = combined['Episode_Length_minutes'] * combined['Guest_Popularity_percentage']
    combined['length_div_host_popularity'] = combined['Episode_Length_minutes'] / (combined['Host_Popularity_percentage'] + 1e-3)
    combined['length_ads_ratio'] = combined['ads_per_minute'] * combined['Episode_Length_minutes']

    combined['length_bucket_sentiment'] = combined['length_bucket'].astype(str) + "_" + combined['Episode_Sentiment'].astype(str)
    combined['length_bucket_genre'] = combined['length_bucket'].astype(str) + "_" + combined['Genre'].astype(str)

    podcast_avg_length = combined.groupby('Podcast_Name')['Episode_Length_minutes'].transform('mean')
    combined['relative_length'] = combined['Episode_Length_minutes'] / (podcast_avg_length + 1e-3)

    combined['is_above_avg_length'] = (combined['Episode_Length_minutes'] > podcast_avg_length).astype(int)
    combined['length_zscore'] = (combined['Episode_Length_minutes'] - podcast_avg_length) / (combined['Episode_Length_minutes'].std() + 1e-3)
    combined['is_length_outlier'] = (combined['length_zscore'].abs() > 2).astype(int)


    # 5. Sentiment Ordinal Mapping
    sentiment_map = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
    combined['sentiment_score'] = combined['Episode_Sentiment'].map(sentiment_map)

    # 6. Host-Guest Popularity Ratio
    combined['popularity_ratio'] = combined['Guest_Popularity_percentage'] / (
        combined['Host_Popularity_percentage'] + 1e-3)

    # 7. Host-Guest Interaction Strength
    combined['interaction_strength'] = combined['Guest_Popularity_percentage'] * combined['Host_Popularity_percentage']

    # 8. Episode Number from Title
    combined['episode_number'] = combined['Episode_Title'].str.extract(r'(\d+)').astype(float)

    # 9. Genre + Sentiment Interaction
    combined['genre_sentiment'] = combined['Genre'].astype(str) + "_" + combined['Episode_Sentiment'].astype(str)

    # 10. Ad Density Category
    combined['ad_density_cat'] = pd.cut(combined['ads_per_minute'], bins=[0, 0.2, 0.5, 1, 2, 5],
                                        labels=['very_low', 'low', 'medium', 'high', 'very_high'])

    # 11. Publication Day Cyclical Encoding
    day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
               'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    combined['pub_day_num'] = combined['Publication_Day'].map(day_map)
    combined['pub_day_sin'] = np.sin(2 * np.pi * combined['pub_day_num'] / 7)
    combined['pub_day_cos'] = np.cos(2 * np.pi * combined['pub_day_num'] / 7)

    
    # 13. Cluster-Based Features
    cluster_data = combined[['Episode_Length_minutes', 'Guest_Popularity_percentage', 'sentiment_score']].fillna(0)
    combined['content_cluster'] = KMeans(n_clusters=5, random_state=42).fit_predict(cluster_data)

    # --- Handle Missing Values ---
    for col in ['Episode_Length_minutes', 'Guest_Popularity_percentage']:
        combined[col] = combined.groupby('Genre')[col].transform(lambda x: x.fillna(x.mean()))

 
     # --- Encode Categorical Features ---
    categorical_cols = ['Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Day',
                        'Publication_Time', 'Episode_Sentiment', 'length_bucket',
                        'genre_sentiment', 'length_bucket_sentiment', 'length_bucket_genre','ad_density_cat']
    for col in categorical_cols:
        le = LabelEncoder()
        combined[col] = le.fit_transform(combined[col].astype(str))

    # --- Split back ---
    X_train_fe = combined.iloc[:len(X_train)].reset_index(drop=True)
    X_test_fe = combined.iloc[len(X_train):].reset_index(drop=True)

    return X_train_fe, X_test_fe



X_train_fe, X_test_fe = engineer_features(X_train, X_test)



X_train_fe


X_test_fe


X = X_train_fe.drop(['id'],axis=1)
y = y_train
test_id = X_test_fe['id']
test = X_test_fe.drop(['id'],axis=1)


from xgboost import XGBRegressor
from sklearn.model_selection import KFold
import numpy as np

# Best XGB parameters
xgb_params = {
    'n_estimators': 425,
    'max_depth': 15,
    'learning_rate': 0.051564535401996674,
    'subsample': 0.6816345671807827,
    'colsample_bytree': 0.9977810444050708,
    'gamma': 1.4032650461122345,
    'reg_alpha': 2.7815627866713517,
    'reg_lambda': 3.780137117381534,
    'random_state': 42,
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor'
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
xgb_oof = np.zeros(len(X))

for train_idx, val_idx in kf.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    xgb_model = XGBRegressor(**xgb_params)
    xgb_model.fit(X_train, y_train)
    xgb_oof[val_idx] = xgb_model.predict(X_val)



from lightgbm import LGBMRegressor

lgbm_model = LGBMRegressor(device='gpu', n_estimators=425, max_depth=15, learning_rate=0.0515,
                           subsample=0.68, colsample_bytree=0.998,
                           reg_alpha=2.78, reg_lambda=3.78, random_state=42)

lgbm_oof = np.zeros(len(X))

for train_idx, val_idx in kf.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    lgbm_model.fit(X_train, y_train)
    lgbm_oof[val_idx] = lgbm_model.predict(X_val)



from catboost import CatBoostRegressor

cat_model = CatBoostRegressor(task_type='GPU', iterations=425, depth=15, learning_rate=0.0515,
                               reg_lambda=3.78, random_seed=42, verbose=0)

cat_oof = np.zeros(len(X))

for train_idx, val_idx in kf.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    cat_model.fit(X_train, y_train)
    cat_oof[val_idx] = cat_model.predict(X_val)



from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Stack OOF predictions as features
stacked_X = np.column_stack((xgb_oof, lgbm_oof, cat_oof))

# Train meta-model
meta_model = Ridge()
meta_model.fit(stacked_X, y)

# Predict and evaluate
stacked_preds = meta_model.predict(stacked_X)
stacked_rmse = np.sqrt(mean_squared_error(y, stacked_preds))
print(f"Final Stacked RMSE: {stacked_rmse:.4f}")



# Predict with each base model on the test set
xgb_test_pred = xgb_model.predict(test)
lgbm_test_pred = lgbm_model.predict(test)
cat_test_pred = cat_model.predict(test)



# Stack predictions from base models
stacked_test = np.column_stack((xgb_test_pred, lgbm_test_pred, cat_test_pred))

# Predict with the meta model (Ridge)
final_test_preds = meta_model.predict(stacked_test)



submission = pd.DataFrame({
    'id': test_id,
    'Listening_Time_minutes': final_test_preds
})
submission.to_csv('submission.csv', index=False)



submission.head()




