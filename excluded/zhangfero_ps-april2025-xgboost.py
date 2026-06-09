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


import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


# === STEP 1: ç‰¹å¾�å·¥ç¨‹å‡½æ•°å°�è£… ===
def basic_feature_engineering(df):
    df = df.copy()
    
    # ç¼ºå¤±å¡«å……
    df['Episode_Length_minutes'] = df['Episode_Length_minutes'].fillna(df['Episode_Length_minutes'].median())
    df['Guest_Popularity_percentage'] = df['Guest_Popularity_percentage'].fillna(0)
    df['Number_of_Ads'] = df['Number_of_Ads'].fillna(0)
    
    # æ–°è¡�ç”Ÿç‰¹å¾�
    df['Ads_per_min'] = df['Number_of_Ads'] / (df['Episode_Length_minutes'] + 1e-3)
    df['Host_Guest_popularity'] = df['Host_Popularity_percentage'] * df['Guest_Popularity_percentage']
    df['Is_Weekend'] = df['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)

    # æ—¶é—´æ��å�–
    df['Publication_Time'] = pd.to_datetime(df['Publication_Time'], format='%H:%M:%S', errors='coerce')
    df['Pub_Hour'] = df['Publication_Time'].dt.hour.fillna(-1)

    # Episode ç¼–å�·æ��å�– + å‘¨æœŸç‰¹å¾�
    df['Episode_Title_num'] = df['Episode_Title'].str.extract(r'(\d+)').astype(float)
    df['sin_Episode'] = np.sin(2 * np.pi * df['Episode_Title_num'] / 8)
    df['cos_Episode'] = np.cos(2 * np.pi * df['Episode_Title_num'] / 8)
    df['sin_Minutes'] = np.sin(2 * np.pi * df['Episode_Length_minutes'] / 60)
    df['cos_Minutes'] = np.cos(2 * np.pi * df['Episode_Length_minutes'] / 60)

    # ç±»åˆ«ç»„å�ˆç‰¹å¾�
    combination_pairs = [
        ('Genre', 'Episode_Sentiment'),
        ('Publication_Day', 'Episode_Sentiment'),
        ('Publication_Day', 'Genre'),
    ]
    for col1, col2 in combination_pairs:
        new_col = f'{col1}_{col2}'
        df[new_col] = df[col1].astype(str) + "_" + df[col2].astype(str)

    return df

# FunctionTransformer å°�è£…
fe_transformer = FunctionTransformer(basic_feature_engineering)


train_filepath = "/kaggle/input/playg-202504/train.csv"
test_filepath = "/kaggle/input/playg-202504/test.csv"

train = pd.read_csv(train_filepath)
test = pd.read_csv(test_filepath)

# å�»é™¤ train ä¸­å®Œå…¨é‡�å¤�çš„è¡Œ
train = train.drop_duplicates()

y = train['Listening_Time_minutes']


# ç‰¹å¾�å·¥ç¨‹å¤„ç�†
train = basic_feature_engineering(train)
test = basic_feature_engineering(test)

# è�šå�ˆç‰¹å¾�
podcast_stats = train.groupby('Podcast_Name')['Listening_Time_minutes'].agg(['mean', 'std']).reset_index()
podcast_stats.columns = ['Podcast_Name', 'Podcast_avg_time', 'Podcast_std_time']
train = train.merge(podcast_stats, on='Podcast_Name', how='left')
test = test.merge(podcast_stats, on='Podcast_Name', how='left')

# Label Encoding
# æ‰€æœ‰éœ€è¦�ç¼–ç �çš„åˆ—ï¼ˆå�Ÿå§‹ + ç»„å�ˆï¼‰
cat_cols = ['Podcast_Name', 'Genre', 'Episode_Sentiment', 'Publication_Day',
            'Genre_Episode_Sentiment', 'Publication_Day_Episode_Sentiment', 'Publication_Day_Genre']

for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])


# æ�„é€ æœ€ç»ˆç‰¹å¾�é›†å�ˆ
drop_cols = ['id', 'Episode_Title', 'Listening_Time_minutes', 'Publication_Time']
X = train.drop(columns=drop_cols)
X_test = test.drop(columns=['id', 'Episode_Title', 'Publication_Time'])


# === STEP 4: æ�„å»ºæ¨¡å�‹è®­ç»ƒ + KFold éªŒè¯� ===
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=14,
        learning_rate=0.035,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method='hist',
        n_jobs=-1
    )

    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              early_stopping_rounds=30,
              verbose=False)

    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict(X_test) / kf.n_splits

rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"âœ… KFold CV RMSE: {rmse:.4f}")





# ä¿�å­˜é¢„æµ‹æ–‡ä»¶
submission = pd.DataFrame({
    "id": test["id"],
    "Listening_Time_minutes": test_preds
})
submission.to_csv("submission_leibiezuhe&eposidedivide8.csv", index=False)
print("ğŸ“� å·²ä¿�å­˜é¢„æµ‹ç»“æ�œè‡³ submission.csv")


# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import KFold
# from sklearn.metrics import mean_squared_error
# import xgboost as xgb
# # ç›®æ ‡å�˜é‡�
# y = train['Listening_Time_minutes']

# # æ�„é€ ç‰¹å¾�
# for df in [train, test]:
#     # ç¼ºå¤±å€¼å¡«å……
#     df['Episode_Length_minutes'] = df['Episode_Length_minutes'].fillna(df['Episode_Length_minutes'].median())
#     df['Guest_Popularity_percentage'] = df['Guest_Popularity_percentage'].fillna(0)
#     df['Number_of_Ads'] = df['Number_of_Ads'].fillna(0)

#     # æ–°å¢�ç‰¹å¾�
#     df['Ads_per_min'] = df['Number_of_Ads'] / (df['Episode_Length_minutes'] + 1e-3)
#     df['Host_Guest_popularity'] = df['Host_Popularity_percentage'] * df['Guest_Popularity_percentage']
#     df['Is_Weekend'] = df['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)
    
#     # æ—¶é—´æ��å�–
#     df['Publication_Time'] = pd.to_datetime(df['Publication_Time'], format='%H:%M:%S', errors='coerce')
#     df['Pub_Hour'] = df['Publication_Time'].dt.hour.fillna(-1)

# # è�šå�ˆç‰¹å¾�
# podcast_stats = train.groupby('Podcast_Name')['Listening_Time_minutes'].agg(['mean', 'std']).reset_index()
# podcast_stats.columns = ['Podcast_Name', 'Podcast_avg_time', 'Podcast_std_time']

# train = train.merge(podcast_stats, on='Podcast_Name', how='left')
# test = test.merge(podcast_stats, on='Podcast_Name', how='left')

# # Label Encoding
# categorical_cols = ['Podcast_Name', 'Genre', 'Episode_Sentiment', 'Publication_Day']
# for col in categorical_cols:
#     le = LabelEncoder()
#     train[col] = le.fit_transform(train[col])
#     test[col] = le.transform(test[col])



# # æ�„é€ è®­ç»ƒç”¨ç‰¹å¾�
# drop_cols = ['id', 'Episode_Title', 'Listening_Time_minutes', 'Publication_Time']
# X = train.drop(columns=drop_cols)
# X_test = test.drop(columns=['id', 'Episode_Title', 'Publication_Time'])

# # XGBoost + KFold éªŒè¯�
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# oof_preds = np.zeros(len(train))
# test_preds = np.zeros(len(test))

# for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
#     X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
#     y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

#     model = xgb.XGBRegressor(
#         n_estimators=400,
#         max_depth=14,#14
#         learning_rate=0.035, #0.01-0.3
#         subsample=0.8,
#         colsample_bytree=0.8,
#         random_state=42,
#         tree_method='hist',
#         n_jobs=-1
#     )
#     model.fit(X_train, y_train, 
#               eval_set=[(X_val, y_val)], 
#               early_stopping_rounds=30,
#               verbose=False)
    
#     oof_preds[val_idx] = model.predict(X_val)
#     test_preds += model.predict(X_test) / kf.n_splits

# # æ¨¡å�‹æ€§èƒ½
# rmse = np.sqrt(mean_squared_error(y, oof_preds))
# rmse


# # è¿›è¡Œé¢„æµ‹
# test_preds = model.predict(X_test)

# # æ�„å»ºæ��äº¤æ–‡ä»¶
# submission = pd.DataFrame({
#     'id': test['id'],
#     'Listening_Time_minutes': test_preds
# })
# submission.to_csv("submission.csv", index=False)
# print("âœ… é¢„æµ‹å®Œæˆ�ï¼Œæ–‡ä»¶å·²ä¿�å­˜ä¸º submission.csv")

