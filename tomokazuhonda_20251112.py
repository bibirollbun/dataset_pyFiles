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
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import warnings
import gc # ãƒ¡ãƒ¢ãƒªè§£æ”¾ï¼ˆã‚¬ãƒ™ãƒ¼ã‚¸ã‚³ãƒ¬ã‚¯ã‚·ãƒ§ãƒ³ï¼‰ã�®ã�Ÿã‚�ã�®ãƒ©ã‚¤ãƒ–ãƒ©ãƒª
warnings.filterwarnings('ignore') 

# --- 1. ãƒ‡ãƒ¼ã‚¿ãƒ­ãƒ¼ãƒ‰ã�¨ãƒ¡ãƒ¢ãƒªæœ€é�©åŒ– ---

# âœ… ãƒ•ã‚¡ã‚¤ãƒ«ãƒ‘ã‚¹: å®Ÿè¡Œç’°å¢ƒã�«å�ˆã‚�ã�›ã�¦å¤‰æ›´ã�—ã�¦ã��ã� ã�•ã�„
DATA_PATH = '../input/m5-forecasting-accuracy/'

def reduce_mem_usage(df):
    """ãƒ‡ãƒ¼ã‚¿ï¼ˆè¡¨ï¼‰ã‚’è»½ã��ã�—ã�¦ãƒ‘ã‚½ã‚³ãƒ³ã�¸ã�®è² æ‹…ã‚’æ¸›ã‚‰ã�™é–¢æ•°"""
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object: 
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8) 
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else: # floatå�‹
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16) 
    return df

print("ãƒ‡ãƒ¼ã‚¿ã�®ãƒ­ãƒ¼ãƒ‰ã‚’é–‹å§‹...")
sales_df = pd.read_csv(DATA_PATH + 'sales_train_validation.csv')
cal_df = pd.read_csv(DATA_PATH + 'calendar.csv')               
price_df = pd.read_csv(DATA_PATH + 'sell_prices.csv')          
submission_df = pd.read_csv(DATA_PATH + 'sample_submission.csv') 

# ãƒ¡ãƒ¢ãƒªãƒ€ã‚¤ã‚¨ãƒƒãƒˆã‚’å®Ÿè¡Œ
sales_df = reduce_mem_usage(sales_df)
cal_df = reduce_mem_usage(cal_df)
price_df = reduce_mem_usage(price_df)

# --- 2. ãƒ‡ãƒ¼ã‚¿å½¢å¼�ã�®å¤‰æ�›ã�¨ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿æ� ã�®ä½œæˆ� ---
print("ãƒ¯ã‚¤ãƒ‰ -> ãƒ­ãƒ³ã‚°å½¢å¼�ã�¸å¤‰æ�›ã�—ã€�ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿æ� ã‚’ä½œæˆ�...")

# è¨“ç·´æœŸé–“ï¼ˆé��å�»ã�®ãƒ‡ãƒ¼ã‚¿ï¼‰ã‚’ç¸¦é•·ã�®è¡¨ã�«å¤‰æ�›
dates = [f'd_{i}' for i in range(1, 1914)] 
train_df = pd.melt(
    sales_df,
    id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], 
    value_vars=dates, 
    var_name='d', 
    value_name='sales' 
)

# äºˆæ¸¬ã�—ã�Ÿã�„æœªæ�¥ã�®æ—¥ä»˜ã�®ãƒªã‚¹ãƒˆã‚’ä½œæˆ�
TEST_DAYS = 28
TEST_VALIDATION_DAYS = [f'd_{i}' for i in range(1914, 1914 + TEST_DAYS)] 
TEST_EVALUATION_DAYS = [f'd_{i}' for i in range(1942, 1942 + TEST_DAYS)] 
future_dates = TEST_VALIDATION_DAYS + TEST_EVALUATION_DAYS

test_ids = sales_df[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']].copy()

# å…¨ã�¦ã�®IDã�¨æœªæ�¥ã�®æ—¥ä»˜ã‚’çµ„ã�¿å�ˆã‚�ã�›ã�Ÿãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿æ� ã‚’ä½œæˆ�
future_data_base = []
for d_id in future_dates:
    temp_df = test_ids.copy()
    temp_df['d'] = d_id
    temp_df['sales'] = np.nan
    future_data_base.append(temp_df)

future_df = pd.concat(future_data_base, ignore_index=True)

# è¨“ç·´ãƒ‡ãƒ¼ã‚¿ã�¨æœªæ�¥ã�®äºˆæ¸¬ãƒ‡ãƒ¼ã‚¿æ� ã‚’ä¸€ã�¤ã�®å·¨å¤§ã�ªè¡¨ã�«çµ�å�ˆ
df = pd.concat([train_df, future_df], ignore_index=True)
df['sales'] = df['sales'].astype(np.float16)


# ğŸš¨ 16GB ãƒ¡ãƒ¢ãƒªå¯¾ç­–ï¼šä¸�è¦�ã�«ã�ªã�£ã�Ÿå…ƒã�®å¤§ã��ã�ªDataFrameã‚’å‰Šé™¤ã�—ã�¦ãƒ¡ãƒ¢ãƒªã‚’ç©ºã�‘ã‚‹ï¼�
del sales_df, train_df, future_df, test_ids, future_data_base, temp_df
gc.collect() 
print(f"ç�¾åœ¨å‡¦ç�†ä¸­ã�®ãƒ‡ãƒ¼ã‚¿ã‚µã‚¤ã‚ºï¼š {df.memory_usage().sum() / 1024**3:.2f} GB") 


# --- 3. ãƒ‡ãƒ¼ã‚¿çµ�å�ˆã�¨å‰�å‡¦ç�† ---
print("ã‚«ãƒ¬ãƒ³ãƒ€ãƒ¼æƒ…å ±ã�¨å€¤æ®µæƒ…å ±ã‚’çµ�å�ˆã�—ã€�å‰�å‡¦ç�†ã‚’å®Ÿè¡Œ...")
df = df.merge(cal_df, on='d', how='left') 
df = df.merge(price_df, on=['store_id', 'item_id', 'wm_yr_wk'], how='left') 
df.drop(columns=['wm_yr_wk'], inplace=True) 

# ãƒ©ãƒ™ãƒ«ã‚¨ãƒ³ã‚³ãƒ¼ãƒ‡ã‚£ãƒ³ã‚°ï¼ˆæ–‡å­—ã‚’æ•°å­—ã�«ç›´ã�™ï¼‰
for col in ['item_id', 'store_id', 'state_id', 'cat_id', 'dept_id', 'event_name_1', 'event_type_1']:
    df[col] = df[col].fillna('None') 
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col]).astype(np.int16)

# ğŸš¨ 16GB ãƒ¡ãƒ¢ãƒªå¯¾ç­–ï¼šã‚«ãƒ¬ãƒ³ãƒ€ãƒ¼ã�¨ä¾¡æ ¼ã�®ãƒ‡ãƒ¼ã‚¿ã‚‚å‰Šé™¤
del cal_df, price_df, encoder
gc.collect()


# --- 4. ç‰¹å¾´é‡�è¨­è¨ˆ (FE) ---
print("ç‰¹å¾´é‡�ï¼ˆãƒ©ã‚°/ãƒ­ãƒ¼ãƒªãƒ³ã‚°ï¼‰ã‚’ä½œæˆ�...")

# **ãƒ©ã‚°ç‰¹å¾´é‡�ï¼ˆé��å�»ã�®å£²ä¸Šï¼‰**
df['lag_28'] = df.groupby(['id'])['sales'].shift(28) 

# **ãƒ­ãƒ¼ãƒªãƒ³ã‚°ç‰¹å¾´é‡�ï¼ˆç§»å‹•å¹³å�‡ï¼‰**
df['roll_mean_28_7'] = df.groupby(['id'])['lag_28'].transform(
    lambda x: x.rolling(window=7).mean()
)

# **æ™‚é–“ç‰¹å¾´é‡�**
df['dayofweek'] = df['d'].apply(lambda x: int(x.split('_')[1]) % 7) 
df['month'] = df['month'].astype(np.int8)

# è¨“ç·´é–‹å§‹æ—¥ã‚’åˆ¶é™�ï¼ˆå…¨æœŸé–“ã‚’ä½¿ã‚�ã�ªã�„ã�“ã�¨ã�§ãƒ¡ãƒ¢ãƒªã�¨å­¦ç¿’æ™‚é–“ã‚’ç¯€ç´„ï¼‰
START_TRAIN_DAY = 350 
df = df[df['d'].str.replace('d_', '').astype(int) >= START_TRAIN_DAY].copy()


# --- 5. äºˆæ¸¬ãƒ¢ãƒ‡ãƒ«ã�®è¨“ç·´ï¼ˆLightGBMï¼‰ ---
print("LightGBMã�®å­¦ç¿’ãƒ‡ãƒ¼ã‚¿æº–å‚™...")
TARGET = 'sales' 
FEATURES = [ 
    'item_id', 'store_id', 'state_id', 'cat_id', 'dept_id',
    'sell_price', 'lag_28', 'roll_mean_28_7', 'dayofweek', 'month',
    'event_name_1', 'event_type_1' 
]
CATEGORICAL_FE = ['item_id', 'store_id', 'state_id', 'cat_id', 'dept_id', 'event_name_1', 'event_type_1']

# è¨“ç·´ãƒ‡ãƒ¼ã‚¿ã�¨æ¤œè¨¼ãƒ‡ãƒ¼ã‚¿ã�«åˆ†ã�‘ã‚‹
VALID_DAYS = 28 
TEST_DAYS = 28
max_d = df['d'].str.replace('d_', '').astype(int).max() - TEST_DAYS * 2 
train_mask = df['d'].str.replace('d_', '').astype(int) <= (max_d - VALID_DAYS)
valid_mask = (df['d'].str.replace('d_', '').astype(int) > (max_d - VALID_DAYS)) & \
             (df['d'].str.replace('d_', '').astype(int) <= max_d) 

train_df = df[train_mask].copy()
valid_df = df[valid_mask].copy()

# ğŸš¨ 16GB ãƒ¡ãƒ¢ãƒªå¯¾ç­–ï¼šå­¦ç¿’å‰�ã�«å…ƒã�®å·¨å¤§ã�ªDFã‚’å‰Šé™¤ã�—ã�¦ãƒ¡ãƒ¢ãƒªã‚’ç¢ºä¿�ï¼�
del train_mask, valid_mask
# dfã�¯äºˆæ¸¬ï¼ˆrecursive_forecastï¼‰ã�§å¿…è¦�ã�«ã�ªã‚‹ã�Ÿã‚�ã€�ã�“ã�“ã�§ã�¯å‰Šé™¤ã�›ã�šæ®‹ã�™
# del df # ã�“ã�“ã‚’ã‚³ãƒ¡ãƒ³ãƒˆã‚¢ã‚¦ãƒˆã�™ã‚‹ã�“ã�¨ã�§ã€�äºˆæ¸¬ãƒ•ã‚§ãƒ¼ã‚ºã�§ã�®NameErrorã‚’é˜²ã��
gc.collect()

print("LightGBMã�®å­¦ç¿’é–‹å§‹...")
params = {
    'objective': 'poisson', 
    'metric': 'rmse',
    'n_estimators': 1000,
    'learning_rate': 0.075,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42
}

model = lgb.LGBMRegressor(**params)
model.fit(
    train_df[FEATURES], train_df[TARGET],
    eval_set=[(valid_df[FEATURES], valid_df[TARGET])], 
    eval_metric='rmse',
    callbacks=[lgb.early_stopping(100)], 
    categorical_feature=CATEGORICAL_FE
)

# ğŸš¨ 16GB ãƒ¡ãƒ¢ãƒªå¯¾ç­–ï¼šå­¦ç¿’ç”¨ãƒ»æ¤œè¨¼ç”¨ã�®DFã‚‚å‰Šé™¤
del train_df, valid_df
gc.collect()


# --- 6. äºˆæ¸¬ã�®å®Ÿè¡Œï¼ˆå†�å¸°çš„äºˆæ¸¬ï¼‰ ---
print("å†�å¸°çš„äºˆæ¸¬ã‚’å®Ÿè¡Œä¸­...")

def recursive_forecast(model, features_list, target_col, all_data_df):
    """
    å†�å¸°çš„äºˆæ¸¬ã�®ä»•çµ„ã�¿ï¼šäºˆæ¸¬ã�—ã�Ÿçµ�æ�œã‚’æ¬¡ã�®æ—¥ã�®äºˆæ¸¬ã�«ä½¿ã�†
    
    Args:
        all_data_df: å­¦ç¿’å¾Œã�®ãƒ‡ãƒ¼ã‚¿ã�Œæ ¼ç´�ã�•ã‚Œã�ŸDFï¼ˆãƒ¡ãƒ¢ãƒªã�«æ®‹ã�£ã�¦ã�„ã‚‹dfã�®ã�“ã�¨ï¼‰
    """
    
    # äºˆæ¸¬å¯¾è±¡æœŸé–“ã�®ãƒ‡ãƒ¼ã‚¿ï¼ˆd_1914 ä»¥é™�ï¼‰ã�®ã�¿ã‚’ãƒ•ã‚£ãƒ«ã‚¿
    # dfå…¨ä½“ã�¯ãƒ¡ãƒ¢ãƒªã‚’å¤§é‡�ã�«æ¶ˆè²»ã�™ã‚‹ã�Ÿã‚�ã€�äºˆæ¸¬æœŸé–“ã� ã�‘ã‚’æŠ½å‡º
    pred_df = all_data_df[all_data_df['d'].str.replace('d_', '').astype(int) > 1913].copy()
    
    # æœªæ�¥ã�®å…¨æ—¥ï¼ˆd_1914 ã�‹ã‚‰ d_1969ï¼‰ã‚’1æ—¥ã�šã�¤ãƒ«ãƒ¼ãƒ—ã�™ã‚‹
    for day in range(1914, 1914 + TEST_DAYS * 2):
        
        d_id = f'd_{day}'
        
        # äºˆæ¸¬ã�—ã�Ÿã�„æ—¥ã�®ãƒ‡ãƒ¼ã‚¿ã� ã�‘ã‚’æŠ½å‡º
        day_to_predict = pred_df[pred_df['d'] == d_id].copy()
        
        if not day_to_predict.empty:
            predictions = model.predict(day_to_predict[features_list])
            predictions = np.maximum(0, predictions)
            
            # äºˆæ¸¬çµ�æ�œã‚’ all_data_df ã�«å��æ˜ ã�•ã�›ã€�æ¬¡ã�®æ—¥ã�®ãƒ©ã‚°ç‰¹å¾´é‡�ã�¨ã�—ã�¦ä½¿ã�ˆã‚‹ã‚ˆã�†ã�«ã�™ã‚‹
            # ã�“ã�®æ“�ä½œã�¯é��å¸¸ã�«ãƒ¡ãƒ¢ãƒªæ¶ˆè²»ã�Œå¤§ã��ã�„ã�§ã�™ã�Œã€�å†�å¸°çš„äºˆæ¸¬ã�«ã�¯å¿…é ˆã�§ã�™
            all_data_df.loc[all_data_df['d'] == d_id, target_col] = predictions
            pred_df.loc[pred_df['d'] == d_id, target_col] = predictions
            
    return pred_df

# ãƒ¡ãƒ¢ãƒªã�«æ®‹ã�£ã�¦ã�„ã‚‹dfã‚’ä½¿ç”¨ã�—ã�¦ã€�äºˆæ¸¬ã‚’å®Ÿè¡Œ
try:
    final_predictions_df = recursive_forecast(model, FEATURES, TARGET, df)
except NameError:
    # ã‚‚ã�—dfã�Œãƒ¡ãƒ¢ãƒªè§£æ”¾ã�«ã‚ˆã‚Šå‰Šé™¤ã�•ã‚Œã�¦ã�„ã�Ÿå ´å�ˆã�®ç·Šæ€¥å›�é�¿ç­–
    print("äºˆæ¸¬ã�«å¿…è¦�ã�ªå…ƒã�®ãƒ‡ãƒ¼ã‚¿ï¼ˆdfï¼‰ã�Œãƒ¡ãƒ¢ãƒªè§£æ”¾ã�«ã‚ˆã‚Šè¦‹ã�¤ã�‹ã‚Šã�¾ã�›ã‚“ã�§ã�—ã�Ÿã€‚æ��å‡ºãƒ•ã‚¡ã‚¤ãƒ«ã‚’0ã�§åŸ‹ã‚�ã�¾ã�™ã€‚")
    final_predictions_df = pd.DataFrame() # ç©ºã�®DataFrameã‚’ä½œæˆ�
    pass
except Exception as e:
    print(f"äºˆæ¸¬ä¸­ã�«äºˆæœŸã�›ã�¬ã‚¨ãƒ©ãƒ¼ã�Œç™ºç”Ÿã�—ã�¾ã�—ã�Ÿ: {e}")
    final_predictions_df = pd.DataFrame()
    pass


# --- 7. æ��å‡ºãƒ•ã‚¡ã‚¤ãƒ«ã�®ä½œæˆ�ï¼ˆçµ�æ�œã‚’å°‚ç”¨ã�®ç´™ã�«æ¸…æ›¸ã�™ã‚‹ï¼‰ ---
print("æ��å‡ºãƒ•ã‚¡ã‚¤ãƒ«ã‚’æ•´å½¢ä¸­...")

# äºˆæ¸¬çµ�æ�œã‚’æ��å‡ºç”¨ã�®è¡¨ï¼ˆsubmission_dfï¼‰ã�«æ›¸ã��è¾¼ã‚€
for i, col in enumerate([f'F{i}' for i in range(1, TEST_DAYS + 1)]):
    
    # äºˆæ¸¬çµ�æ�œã�Œç”Ÿæˆ�ã�•ã‚Œã�¦ã�„ã‚‹å ´å�ˆã�®ã�¿å‡¦ç�†
    if not final_predictions_df.empty:
        d_v = f'd_{1914 + i}'
        d_e = f'd_{1942 + i}'
        
        # Validation
        submission_df.loc[submission_df['id'].str.contains('validation'), col] = \
            final_predictions_df[final_predictions_df['d'] == d_v]['sales'].values

        # Evaluation
        submission_df.loc[submission_df['id'].str.contains('evaluation'), col] = \
            final_predictions_df[final_predictions_df['d'] == d_e]['sales'].values
    else:
        # final_predictions_dfã�Œç©ºã�®å ´å�ˆã�¯ã€�æ��å‡ºãƒ•ã‚¡ã‚¤ãƒ«ã�®äºˆæ¸¬å€¤ã‚’0ã�§åŸ‹ã‚�ã‚‹
        submission_df.loc[:, col] = 0.0

# çµ�æ�œã‚’ 'submission.csv' ã�¨ã�„ã�†ãƒ•ã‚¡ã‚¤ãƒ«ã�«ä¿�å­˜
submission_df.to_csv('submission.csv', index=False)
print("submission.csv ã�Œå‡ºåŠ›ã�•ã‚Œã�¾ã�—ã�Ÿï¼�")

