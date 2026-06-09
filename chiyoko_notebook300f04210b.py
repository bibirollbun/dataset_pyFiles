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


#åŸºæœ¬çš„ã�ªãƒ©ã‚¤ãƒ–ãƒ©ãƒªã�®ã‚¤ãƒ³ãƒ�ãƒ¼ãƒˆ
import lightgbm as lgb#ãƒ¢ãƒ‡ãƒ«ã�¯lightgbmã‚’ä½¿ç”¨
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split



train_df = pd.read_csv("/kaggle/input/home-credit-default-risk/application_train.csv")#å­¦ç¿’ãƒ‡ãƒ¼ã‚¿
test_df = pd.read_csv("/kaggle/input/home-credit-default-risk/application_test.csv")#ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿




bureau = pd.read_csv("/kaggle/input/home-credit-default-risk/bureau.csv")#é¡§å®¢ã�®å¤šé‡‘è��æ©Ÿé–¢ã�§ã�®ãƒ­ãƒ¼ãƒ³æƒ…å ±â‘ ä½¿ç”¨â€¼
bureau_bal = pd.read_csv("/kaggle/input/home-credit-default-risk/bureau_balance.csv")#ä»–é‡‘è��æ©Ÿé–¢ã�§ã�®æœˆæ¬¡è¿”æ¸ˆå±¥æ­´

pos_cash = pd.read_csv("/kaggle/input/home-credit-default-risk/POS_CASH_balance.csv")#é��å�»ã�®posã‚„ç�¾é‡‘ãƒ­ãƒ¼ãƒ³ã�®æœˆæ¬¡æƒ…å ±ä½¿ç”¨â€¼
prev_app = pd.read_csv("/kaggle/input/home-credit-default-risk/previous_application.csv")#é��å�»ã�®homecreditã�§ã�®ç”³è«‹å±¥æ­´â‘ ä½¿ç”¨â€¼

credit_bal = pd.read_csv("/kaggle/input/home-credit-default-risk/credit_card_balance.csv")#ã‚¯ãƒ¬ã‚¸ãƒƒãƒˆã‚«ãƒ¼ãƒ‰ã�§ã�®æœˆæ¬¡åˆ©ç”¨æƒ…å ±ä½¿ç”¨â€¼
installments = pd.read_csv("/kaggle/input/home-credit-default-risk/installments_payments.csv")#åˆ†å‰²æ‰•ã�„ã�«é–¢ã�™ã‚‹è©³ç´°å±¥æ­´ä½¿ç”¨â€¼


bureau.columns


#ã‚¯ãƒ©ãƒƒã‚·ãƒ¥å›�é�¿ç”¨
import gc

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

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
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            # objectå�‹ã�¯ categoryå�‹ã�«å¤‰æ�›ã�™ã‚‹ã�¨ãƒ¡ãƒ¢ãƒªã�Œæ¸›ã‚‹å ´å�ˆã�Œå¤šã�„
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


credit_bal.info()


credit_bal.isnull().sum()



import pandas as pd
import numpy as np

def aggregate_credit_bal_safe(credit_bal):
    df = credit_bal.copy()

    # ========= 0åŸ‹ã‚�ã�—ã�¦æ„�å‘³ã�Œé€šã‚‹åˆ— =========
    fill_zero_cols = [
        'AMT_BALANCE',
        'AMT_CREDIT_LIMIT_ACTUAL',
        'AMT_DRAWINGS_ATM_CURRENT',
        'AMT_DRAWINGS_CURRENT',
        'AMT_DRAWINGS_OTHER_CURRENT',
        'AMT_DRAWINGS_POS_CURRENT',
        'AMT_INST_MIN_REGULARITY',
        'AMT_PAYMENT_CURRENT',
        'CNT_DRAWINGS_ATM_CURRENT',
        'CNT_DRAWINGS_OTHER_CURRENT',
        'CNT_DRAWINGS_POS_CURRENT',
        'CNT_INSTALMENT_MATURE_CUM'
    ]
    df[fill_zero_cols] = df[fill_zero_cols].fillna(0)

    # ========= inf / -inf ã‚’å…¨æ•°å€¤åˆ—ã�§é™¤å�» =========
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

    # ========= ç•°å¸¸å€¤ã�®æœ€ä½�é™�è£œæ­£ =========
    df.loc[df['AMT_PAYMENT_CURRENT'] < 0, 'AMT_PAYMENT_CURRENT'] = 0

    # ========= åˆ©ç”¨ç�‡ï¼ˆwarning å®Œå…¨å›�é�¿ï¼‰ =========
    credit_limit = df['AMT_CREDIT_LIMIT_ACTUAL']
    balance = df['AMT_BALANCE']

    utilization = np.where(
        credit_limit > 0,
        balance / credit_limit,
        0
    )

    df['UTILIZATION'] = (
        pd.Series(utilization, index=df.index)
        .clip(0, 2)
    )

    # ========= é�…å»¶é¡� =========
    df['DELAY'] = (
        df['AMT_INST_MIN_REGULARITY'] - df['AMT_PAYMENT_CURRENT']
    ).clip(-50000, 50000)

    # ========= ãƒ•ãƒ©ã‚°ï¼ˆå¿…ã�š fillna â†’ æ¯”è¼ƒï¼‰ =========
    df['HAS_CREDIT_LIMIT_FLAG'] = (credit_limit > 0).astype(int)
    df['DELAY_FLAG'] = (df['DELAY'] > 0).astype(int)
    df['DPD_NONZERO'] = (df['SK_DPD'].fillna(0) > 0).astype(int)

    # ========= é›†ç´„ =========
    numeric_cols = [
        'AMT_BALANCE',
        'AMT_CREDIT_LIMIT_ACTUAL',
        'AMT_DRAWINGS_CURRENT',
        'AMT_PAYMENT_CURRENT',
        'UTILIZATION',
        'DELAY'
    ]

    agg_funcs = ['sum', 'mean', 'max']

    customer_agg = (
        df.groupby('SK_ID_CURR')[numeric_cols]
        .agg(agg_funcs)
    )
    customer_agg.columns = [
        f'{col}_{func}' for col, func in customer_agg.columns
    ]
    customer_agg = customer_agg.reset_index()

    # ========= ãƒ•ãƒ©ã‚°ç³» =========
    flag_cols = ['HAS_CREDIT_LIMIT_FLAG', 'DELAY_FLAG', 'DPD_NONZERO']
    flag_agg = df.groupby('SK_ID_CURR')[flag_cols].mean().reset_index()

    customer_agg = customer_agg.merge(
        flag_agg, on='SK_ID_CURR', how='left'
    )
    # ========= æ™‚ç³»åˆ—ç”¨ã�®æº–å‚™ =========
    # MONTHS_BALANCE ã�¯é€šå¸¸ã€�ç›´è¿‘ã�Œ 0, é��å�»ã�«é�¡ã‚‹ã�»ã�©è² ã�®å€¤ (-1, -2...)
    # ã‚½ãƒ¼ãƒˆã�—ã�¦ã�Šã��ã�“ã�¨ã�§è¨ˆç®—ã‚’å®‰å®šã�•ã�›ã‚‹
    df = df.sort_values(['SK_ID_CURR', 'MONTHS_BALANCE'], ascending=False)

    # 1. ç›´è¿‘ã�®å€¤ã‚’æŠ½å‡º (MONTHS_BALANCE == -1 or æœ€ã‚‚æ–°ã�—ã�„ãƒ¬ã‚³ãƒ¼ãƒ‰)
    last_val = df.groupby('SK_ID_CURR').first().reset_index()
    last_val = last_val[['SK_ID_CURR', 'AMT_BALANCE', 'UTILIZATION', 'SK_DPD']]
    last_val.columns = [f'LAST_{col}' if col != 'SK_ID_CURR' else col for col in last_val.columns]

    # 2. ç›´è¿‘ N ãƒ¶æœˆã�®é›†ç´„ (ä¾‹: ç›´è¿‘6ãƒ¶æœˆ)
    recent_6m = df[df['MONTHS_BALANCE'] >= -6].groupby('SK_ID_CURR')[numeric_cols].agg(['mean', 'max'])
    recent_6m.columns = [f'RECENT6M_{col}_{func}' for col, func in recent_6m.columns]
    recent_6m = recent_6m.reset_index()

    # 3. ãƒˆãƒ¬ãƒ³ãƒ‰ï¼ˆå‚¾ã��ï¼‰ã�®ç°¡æ˜“è¨ˆç®—: ç›´è¿‘å€¤ - å…¨ä½“å¹³å�‡
    # æ—¢å­˜ã�® customer_agg (å…¨ä½“é›†ç´„) ã‚’åˆ©ç”¨
    customer_agg = df.groupby('SK_ID_CURR')[numeric_cols].agg(['sum', 'mean', 'max'])
    customer_agg.columns = [f'{col}_{func}' for col, func in customer_agg.columns]
    customer_agg = customer_agg.reset_index()

    # ãƒ�ãƒ¼ã‚¸
    customer_agg = customer_agg.merge(last_val, on='SK_ID_CURR', how='left')
    customer_agg = customer_agg.merge(recent_6m, on='SK_ID_CURR', how='left')

    # ãƒˆãƒ¬ãƒ³ãƒ‰ç‰¹å¾´é‡�ã�®ä½œæˆ� (ä¾‹: ç›´è¿‘æ®‹é«˜ã�Œå¹³å�‡ã‚ˆã‚Šé«˜ã�„ã�‹)
    customer_agg['TREND_BALANCE'] = customer_agg['LAST_AMT_BALANCE'] - customer_agg['AMT_BALANCE_mean']
    
    # 4. æŒ‡æ•°ç§»å‹•å¹³å�‡ (EWM) ã�®æœ€æ–°å€¤
    # ç›´è¿‘ã�®è¡Œå‹•ã‚’ã‚ˆã‚Šé‡�è¦–ã�™ã‚‹ã‚¹ã‚³ã‚¢
    df['EWM_UTILIZATION'] = df.groupby('SK_ID_CURR')['UTILIZATION'].transform(lambda x: x.ewm(alpha=0.5).mean())
    ewm_latest = df.groupby('SK_ID_CURR')['EWM_UTILIZATION'].first().reset_index()
    
    customer_agg = customer_agg.merge(ewm_latest, on='SK_ID_CURR', how='left')

    return customer_agg

    



# å®Ÿè¡Œ
credit_bal_agg = aggregate_credit_bal_safe(credit_bal)
print(credit_bal_agg.head())


#ãƒ¡ãƒ¢ãƒªåœ§ç¸®
credit_bal_agg = reduce_mem_usage(credit_bal_agg)


credit_bal_agg.isna().sum().sort_values(ascending=False).head(10)



# train ãƒ‡ãƒ¼ã‚¿ã�«çµ�å�ˆ
train_df = train_df.merge(credit_bal_agg, on='SK_ID_CURR', how='left')

# test ãƒ‡ãƒ¼ã‚¿ã�«çµ�å�ˆ
test_df = test_df.merge(credit_bal_agg, on='SK_ID_CURR', how='left')


del credit_bal
gc.collect() 


#ãƒ�ã‚¹ãƒˆã�§ç‰¹å¾´é‡�ã‚’ã�¤ã��ã‚‹


from sklearn.model_selection import KFold
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict



#ã‚¹ã‚¿ãƒƒã‚­ãƒ³ã‚°ç”¨ã�®ãƒ‡ãƒ¼ã‚¿ä½œæˆ�
#step1 äºˆæ¸¬ç”¨ãƒ‡ãƒ¼ã‚¿ã‚»ãƒƒãƒˆã�®ä½œæˆ�
# 1. ãƒ¡ã‚¤ãƒ³ã�®å­¦ç¿’ãƒ‡ãƒ¼ã‚¿ã�‹ã‚‰SK_ID_CURRã�¨Targetåˆ—ã�®ã�¿ã‚’æŠ½å‡º
train_target = train_df[['SK_ID_CURR', 'TARGET']]

# 2. POS_CASHãƒ‡ãƒ¼ã‚¿ã�¨Targetåˆ—ã‚’çµ�å�ˆ
# 'how="left"' ã‚’ä½¿ã�†ã�“ã�¨ã�§ã€�POS_CASHã�®ã�™ã�¹ã�¦ã�®å±¥æ­´ã�«Targetã‚’ç´�ã�¥ã�‘ã‚‹
credit_bal_with_target = pd.merge(credit_bal_agg, 
                                train_target, 
                                on='SK_ID_CURR', 
                                how='left')

# 3. Targetã�Œæ¬ æ��ã�—ã�¦ã�„ã‚‹è¡Œï¼ˆãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿ã�®é¡§å®¢IDï¼‰ã�¯é™¤å¤–/åˆ†é›¢
# ã�“ã‚Œã�§ã€�ã‚µãƒ–ãƒ¢ãƒ‡ãƒ«ã�®å­¦ç¿’ã�«ä½¿ã�ˆã‚‹ãƒ‡ãƒ¼ã‚¿ã�Œæº–å‚™å®Œäº†


#ãƒªãƒƒã‚¸å›�å¸°ã�§ã‚¹ã‚¿ãƒƒã‚­ãƒ³ã‚°

#ã‚¤ãƒ³ãƒ�ãƒ¼ãƒˆ
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

#ãƒªãƒƒã‚¸å›�å¸°ã‚¹ã‚¿ãƒƒã‚­ãƒ³ã‚°é–¢æ•°
def get_ridge_preds(input_df, target_col='TARGET', id_col='SK_ID_CURR'):
    """
    ã‚µãƒ–ãƒ‡ãƒ¼ã‚¿ã�‹ã‚‰Ridgeå›�å¸°ã�®äºˆæ¸¬å€¤ã‚’ç®—å‡ºã�™ã‚‹é–¢æ•°
    """
    # 1. ãƒ‡ãƒ¼ã‚¿ã�®æº–å‚™ï¼ˆTARGETã�Œã�‚ã‚‹ã‚‚ã�®ï¼�å­¦ç¿’ç”¨ã€�ã�ªã�„ã‚‚ã�®ï¼�ãƒ†ã‚¹ãƒˆç”¨ï¼‰
    # â€»input_dfã�¯ã�™ã�§ã�«1é¡§å®¢1è¡Œã�«é›†ç´„ã�•ã‚Œã�¦ã�„ã‚‹å‰�æ��
    train_idx = input_df[target_col].notnull()
    
    # ç‰¹å¾´é‡�ã�¨ã‚¿ãƒ¼ã‚²ãƒƒãƒˆã�®åˆ†é›¢
    # IDã�¨TARGETä»¥å¤–ã�®åˆ—ã‚’ã�™ã�¹ã�¦ç‰¹å¾´é‡�ã�¨ã�—ã�¦ä½¿ã�†
    features = [c for c in input_df.columns if c not in [id_col, target_col]]
    
    X = input_df[features]
    y = input_df[target_col]
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[~train_idx]
    
    # 2. ãƒ‘ã‚¤ãƒ—ãƒ©ã‚¤ãƒ³ã�®æ§‹ç¯‰ï¼ˆæ¬ æ��å€¤è£œå®Œ + æ¨™æº–åŒ– + Ridgeï¼‰
    # Ridgeã�¯ã‚¹ã‚±ãƒ¼ãƒªãƒ³ã‚°ã�Œå¿…é ˆã�§ã�™
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=10.0)) # alphaã�¯1.0ã€œ10.0ç¨‹åº¦ã�§OK
    ])
    
    # 3. å­¦ç¿’ãƒ‡ãƒ¼ã‚¿ã�«å¯¾ã�™ã‚‹ã€Œäºˆæ¸¬ã‚¹ã‚³ã‚¢ã€�ã�®ç®—å‡º (Out-of-Fold)
    # 5åˆ†å‰²äº¤å·®æ¤œè¨¼ã�§ã€�å�„ãƒ‡ãƒ¼ã‚¿ã�Œã€Œãƒ†ã‚¹ãƒˆå½¹ã€�ã�«ã�ªã�£ã�Ÿæ™‚ã�®äºˆæ¸¬å€¤ã‚’æºœã‚�ã‚‹#ãƒ‡ãƒ¼ã‚¿ãƒªãƒ¼ã‚¯ã�®é˜²æ­¢ç­–
    print(f"Generating OOF predictions for {len(X_train)} samples...")
    oof_preds = cross_val_predict(pipe, X_train, y_train, cv=5, method='predict')
    
    # 4. ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿ï¼ˆæœ¬ç•ªäºˆæ¸¬ç”¨ï¼‰ã�«å¯¾ã�™ã‚‹äºˆæ¸¬
    print(f"Generating Test predictions for {len(X_test)} samples...")
    pipe.fit(X_train, y_train)
    test_preds = pipe.predict(X_test)
    
    # 5. çµ�æ�œã‚’ä¸€ã�¤ã�®Seriesã�«ã�¾ã�¨ã‚�ã‚‹
    all_preds = pd.Series(index=input_df.index, dtype='float64')
    all_preds[train_idx] = oof_preds
    all_preds[~train_idx] = test_preds
    
    return all_preds





#extratreeã�§ã‚¹ã‚¿ãƒƒã‚­ãƒ³ã‚°
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_predict

def get_et_preds(input_df, target_col='TARGET', id_col='SK_ID_CURR'):
    """
    Extra Treesã‚’ç”¨ã�„ã�¦ãƒ¡ã‚¿ç‰¹å¾´é‡�ã‚’ç”Ÿæˆ�ã�™ã‚‹é–¢æ•°
    """
    train_idx = input_df[target_col].notnull()
    features = [c for c in input_df.columns if c not in [id_col, target_col]]
    
    X = input_df[features]
    y = input_df[target_col]
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[~train_idx]
    
    # ãƒ‘ã‚¤ãƒ—ãƒ©ã‚¤ãƒ³ï¼ˆETã�¯ä¸­å¤®å€¤è£œå®Œã� ã�‘ã�§å‹•ã��ã�¾ã�™ï¼‰
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('et', ExtraTreesClassifier(
            n_estimators=200,    # æœ¨ã�®æ•°ã€‚200ã€œ500ã��ã‚‰ã�„ã�‚ã‚‹ã�¨å®‰å®šã�—ã�¾ã�™
            max_depth=10,        # æ·±ã��ã�—ã�™ã��ã‚‹ã�¨LGBMã�¨å�Œã�˜ã‚ˆã�†ã�ªäºˆæ¸¬ã�«ã�ªã‚‹ã�®ã�§ã€�10å‰�å¾Œã�§ãƒ�ã‚¤ãƒ«ãƒ‰ã�«
            min_samples_leaf=20, # 1ã�¤ã�®è‘‰ã�«æœ€ä½�20ã‚µãƒ³ãƒ—ãƒ«ã€‚ã�“ã‚Œã‚‚é��å­¦ç¿’é˜²æ­¢
            n_jobs=-1,           # ä¸¦åˆ—å‡¦ç�†ã�§é«˜é€ŸåŒ–
            random_state=42
        ))
    ])
    
    # OOFäºˆæ¸¬
    print(f"Generating ET OOF predictions...")
    oof_preds = cross_val_predict(pipe, X_train, y_train, cv=5, method='predict_proba')[:, 1]
    
    # ãƒ†ã‚¹ãƒˆäºˆæ¸¬
    print(f"Generating ET Test predictions...")
    pipe.fit(X_train, y_train)
    test_preds = pipe.predict_proba(X_test)[:, 1]
    
    # çµ�æ�œã�®çµ±å�ˆ
    all_preds = pd.Series(index=input_df.index, dtype='float64')
    all_preds[train_idx] = oof_preds
    all_preds[~train_idx] = test_preds
    
    return all_preds


#extratreeã�§ã�®ã‚¹ã‚¿ãƒƒã‚­ãƒ³ã‚°å®Ÿè¡Œ
credit_bal_with_target['POS_CASH_et_score'] = get_et_preds(credit_bal_with_target)
# train_df ã�«æ�¥ç¶š
train_df = train_df.merge(
    credit_bal_with_target[['SK_ID_CURR', 'POS_CASH_et_score']], 
    on='SK_ID_CURR', 
    how='left'
)

# test_df ã�«æ�¥ç¶š
test_df = test_df.merge(
    credit_bal_with_target[['SK_ID_CURR', 'POS_CASH_et_score']], 
    on='SK_ID_CURR', 
    how='left'
)


credit_bal_with_target['POS_CASH_ridge_score'] = get_ridge_preds(credit_bal_with_target)

# å­¦ç¿’ãƒ‡ãƒ¼ã‚¿(train_df)ã�«ãƒ�ãƒ¼ã‚¸
train_df = train_df.merge(
    credit_bal_with_target[['SK_ID_CURR', 'POS_CASH_ridge_score']], 
    on='SK_ID_CURR', 
    how='left'
)

# ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿(test_df)ã�«ãƒ�ãƒ¼ã‚¸
test_df = test_df.merge(
    credit_bal_with_target[['SK_ID_CURR', 'POS_CASH_ridge_score']], 
    on='SK_ID_CURR', 
    how='left'
)


#LIGHTGBMã�§ã‚¹ã‚¿ãƒƒã‚­ãƒ³ã‚°
NFOLDS = 5
folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)

# 1. å­¦ç¿’ç”¨ã�¨ãƒ†ã‚¹ãƒˆç”¨ã�«åˆ†ã�‘ã‚‹
train_data = credit_bal_with_target[credit_bal_with_target['TARGET'].notnull()].reset_index(drop=True)
test_data = credit_bal_with_target[credit_bal_with_target['TARGET'].isnull()].reset_index(drop=True)

# 2. ç‰¹å¾´é‡�åˆ—ã‚’æŠ½å‡ºï¼ˆSK_ID_CURR ã�¨ TARGET ã�¯é™¤å¤–ï¼‰
feature_cols = [c for c in credit_bal_agg.columns if c != 'SK_ID_CURR']

# 3. objectåˆ—ã‚’ã‚«ãƒ†ã‚´ãƒªå�‹ã�«å¤‰æ�›
for col in feature_cols:
    if train_data[col].dtype == 'object':
        train_data[col] = train_data[col].astype('category')
        test_data[col] = test_data[col].astype('category')

# 4. OOF é…�åˆ—
oof_preds_pc = np.zeros(train_data.shape[0])
test_preds_pc = np.zeros(test_data.shape[0])

# 5. KFold å­¦ç¿’
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_data[feature_cols], train_data['TARGET'])):
    X_train_fold = train_data.iloc[train_idx][feature_cols]
    y_train_fold = train_data.iloc[train_idx]['TARGET']
    X_valid_fold = train_data.iloc[valid_idx][feature_cols]
    y_valid_fold = train_data.iloc[valid_idx]['TARGET']

    lgb_model = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=32,
        colsample_bytree=0.8,
        subsample=0.8,
        random_state=42,
        n_jobs=-1
    )
    lgb_model.fit(
        X_train_fold, y_train_fold,
        eval_metric='auc',
        eval_set=[(X_valid_fold, y_valid_fold)],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
    )

    # OOFäºˆæ¸¬
    oof_preds_pc[valid_idx] = lgb_model.predict_proba(X_valid_fold)[:, 1]
    # ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿äºˆæ¸¬
    test_preds_pc += lgb_model.predict_proba(test_data[feature_cols])[:, 1] / NFOLDS

# 6. OOFåˆ—ã‚’è¿½åŠ 
train_data['CB_OOF_LGBM'] = oof_preds_pc
test_data['CB_OOF_LGBM'] = test_preds_pc

# 7. SK_ID_CURR ã�¨ OOFåˆ—ã� ã�‘æŠ½å‡º
oof_credit_bal = train_data[['SK_ID_CURR','CB_OOF_LGBM']]
test_credit_bal = test_data[['SK_ID_CURR','CB_OOF_LGBM']]

# å†�å®Ÿè¡Œã�—ã�¦ã‚‚åˆ—ã�Œå¢—ã�ˆã�ªã�„ã‚ˆã�†ã�«ã�™ã‚‹ã�Ÿã‚�ã�®å®‰å…¨ç­–
if 'CB_OOF_LGBM' in train_df.columns:
    train_df = train_df.drop(columns=['CB_OOF_LGBM'])
if 'CB_OOF_LGBM' in test_df.columns:
    test_df = test_df.drop(columns=['CB_OOF_LGBM'])

# 8. å…ƒã�® train_df/test_df ã�« merge
train_df = train_df.merge(oof_credit_bal, on='SK_ID_CURR', how='left')
test_df = test_df.merge(test_credit_bal, on='SK_ID_CURR', how='left')



#è¦�ã‚‰ã�ªã�„ã‚‚ã�®ã�¯æ¶ˆã�™
del credit_bal_agg
gc.collect() 


#installments.info()


#installments.isnull().sum()


import pandas as pd
import numpy as np
import gc

# --- 1. Installments é›†è¨ˆé–¢æ•° ---
def aggregate_installments_prev(installments):
    df = installments.copy()
    
    # å‰�å‡¦ç�†
    df['PAYMENT_MISSING'] = df['AMT_PAYMENT'].isna().astype(int)
    df['DELAY'] = df['DAYS_ENTRY_PAYMENT'] - df['DAYS_INSTALMENT']
    df['DBD'] = (df['DAYS_INSTALMENT'] - df['DAYS_ENTRY_PAYMENT']).clip(lower=0) # æ—©ã��æ‰•ã�£ã�Ÿæ—¥æ•°

    df['DELAY_FLAG'] = (df['DELAY'] > 0).astype(int)
    df['DELAY_5D_FLAG'] = (df['DELAY'] > 5).astype(int)
    df['DELAY_30D_FLAG'] = (df['DELAY'] > 30).astype(int)

    recent_1y_mask = df['DAYS_INSTALMENT'] >= -365
    df['DELAY_RECENT_1Y_FLAG'] = np.where(recent_1y_mask, df['DELAY_FLAG'], np.nan)
    
    df['AMT_SHORTFALL'] = df['AMT_INSTALMENT'] - df['AMT_PAYMENT']
    df['PAYMENT_PERC'] = df['AMT_PAYMENT'] / df['AMT_INSTALMENT'].replace(0, np.nan)

    df['DPD'] = (df['DAYS_ENTRY_PAYMENT'] - df['DAYS_INSTALMENT']).clip(lower=0)
ã€€  #df['PAYMENT_DIFF'] = df['AMT_INSTALMENT'] - df['AMT_PAYMENT']
    
    # ãƒ�ãƒ¼ã‚¸ãƒ§ãƒ³æƒ…å ±ã�®ãƒ€ãƒŸãƒ¼åŒ–
    version_flag = pd.get_dummies(df['NUM_INSTALMENT_VERSION'], prefix='INST_VER')
    df = pd.concat([df, version_flag], axis=1)
    
    agg_ops = {
        'DELAY': ['max', 'mean', 'sum'],
        'DPD': ['max', 'mean', 'sum'], # DBDã‚’è¿½åŠ 
        'DBD': ['max', 'mean', 'sum'],
        'DELAY_FLAG': ['mean', 'sum'],
        'DELAY_5D_FLAG': ['sum'],
        'DELAY_30D_FLAG': ['sum'],
        'DELAY_RECENT_1Y_FLAG': ['mean'],
        'AMT_SHORTFALL': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean'], # è¿½åŠ 
        'PAYMENT_DIFF': ['max', 'mean', 'sum'], # ã�“ã�“ã‚’è¿½åŠ ï¼�
        'PAYMENT_MISSING': ['sum', 'mean'],      # ã�“ã�“ã‚’è¿½åŠ ï¼�
        'DELAY_FLAG': ['mean', 'sum'],           # POSã�®IS_DPDã�¨å�Œæ§˜ã�®é›†è¨ˆ
        'AMT_INSTALMENT': ['sum', 'mean'],
        'NUM_INSTALMENT_NUMBER': ['max', 'mean']
    }
    
    ver_cols = [c for c in df.columns if 'INST_VER_' in c]
    for c in ver_cols:
        agg_ops[c] = ['sum']

    prev_agg = df.groupby('SK_ID_PREV').agg(agg_ops)
    prev_agg.columns = ['INST_' + '_'.join(col).upper() for col in prev_agg.columns]
    return prev_agg.reset_index()


# 2. POS_CASH ã‚’ SK_ID_PREV å�˜ä½�ã�§é›†è¨ˆ
def aggregate_pos_cash_prev(df):
    df = df.copy()
    
    df['CNT_INSTALMENT'] = df['CNT_INSTALMENT'].fillna(0)
    df['CNT_INSTALMENT_FUTURE'] = df['CNT_INSTALMENT_FUTURE'].fillna(0)
    
    df['CNT_INSTALMENT_RATIO'] = df['CNT_INSTALMENT_FUTURE'] / df['CNT_INSTALMENT'].replace(0, np.nan)
    df['REPAYMENT_PROGRESS'] = 1 - df['CNT_INSTALMENT_RATIO']
    df['REPAYMENT_PROGRESS'] = df['REPAYMENT_PROGRESS'].fillna(0)
    
    df['DPD_NONZERO'] = (df['SK_DPD'] > 0).astype(int)
    df['IS_DPD'] = (df['SK_DPD'] > 0).astype(int)
    
    agg_funcs = {
        'CNT_INSTALMENT': ['max','mean','sum'],
        'CNT_INSTALMENT_FUTURE': ['max','mean','sum'],
        'CNT_INSTALMENT_RATIO': ['mean','max'],
        'REPAYMENT_PROGRESS': ['mean','max'],
        'SK_DPD': ['max','mean','sum','median'],
        'DPD_NONZERO':  ['sum']
    }
    
    df_agg = df.groupby('SK_ID_PREV').agg(agg_funcs)
    df_agg.columns = ['POS_' + '_'.join(col).upper() for col in df_agg.columns]
    
    # ã‚¹ãƒ†ãƒ¼ã‚¿ã‚¹ã‚«ã‚¦ãƒ³ãƒˆ
    df['NAME_CONTRACT_STATUS'] = 'POS_CASH_' + df['NAME_CONTRACT_STATUS'].astype(str)
    status_cnt = df.pivot_table(
        index='SK_ID_PREV',
        columns='NAME_CONTRACT_STATUS',
        values='MONTHS_BALANCE',
        aggfunc='size',
        fill_value=0
    )
    
    df_agg = df_agg.merge(status_cnt, on='SK_ID_PREV', how='left')
    return df_agg.reset_index()


def aggregate_prev_app_advanced(prev_app, pos_cash, installments):
    """
    åˆ†æ��çµ�æ�œã�«åŸºã�¥ã�„ã�Ÿã€Œé ­é‡‘æ¯”ç�‡ã€�ã‚„ã€Œé��å�»ã�®æ”¯æ‰•ãƒªã‚¹ã‚¯ã€�ã‚’
    é›†ç´„ã�—ã�¦è¿”ã�™é«˜åº¦ã�ªé›†è¨ˆé–¢æ•°
    """
    
    # --- 1. ä¸‹ä½�ãƒ†ãƒ¼ãƒ–ãƒ«ã�®é›†è¨ˆ ---
    # â€» aggregate_pos_cash_prev ã�¨ aggregate_installments_prev ã�Œå®šç¾©ã�•ã‚Œã�¦ã�„ã‚‹å‰�æ��
    pos_agg = aggregate_pos_cash_prev(pos_cash)
    inst_agg = aggregate_installments_prev(installments)
    
    # --- 2. Previous Application ã�®ç‰¹å¾´é‡�ä½œæˆ� ---
    df = prev_app.copy()
    
    # æ±ºå®šé¡�ã�¨ç”³è«‹é¡�ã�®æ¯”ç�‡ï¼ˆæ¸›é¡�ã�•ã‚Œã�Ÿã�‹ã�©ã�†ã�‹ã�®æŒ‡æ¨™ï¼‰
    df['APP_CREDIT_PERC'] = df['AMT_APPLICATION'] / df['AMT_CREDIT'].replace(0, np.nan)
    
    # ã€�æœ€é‡�è¦�ã€‘é ­é‡‘ã�®å‰²å�ˆï¼ˆç„¡ç�†ã�ªãƒ­ãƒ¼ãƒ³ã�®æ¤œçŸ¥ï¼‰
    # ã�‚ã�ªã�Ÿã�Œæ��æ¡ˆã�—ã�Ÿã€Œé ­é‡‘ / è��è³‡é¡�ã€�
    df['APP_CREDIT_ATAMARATE'] = df['AMT_DOWN_PAYMENT'] / df['AMT_CREDIT'].replace(0, np.nan)
    # å•†å“�ã�®å€¤æ®µã�«å¯¾ã�—ã�¦ã�©ã‚Œã��ã‚‰ã�„ã�®é ­é‡‘ã‚’ç©�ã‚“ã� ã�‹
    df['APP_GOODS_ATAMARATE'] = df['AMT_DOWN_PAYMENT'] / df['AMT_GOODS_PRICE'].replace(0, np.nan)
    
    # åˆ©ç�‡ã�®è¿‘ä¼¼ï¼ˆAMT_ANNUITY * æœŸé–“ / AMT_CREDITï¼‰
    # â€»æœŸé–“ã�Œä¸�æ˜�ã�ªå ´å�ˆã�Œå¤šã�„ã�Œã€�ç°¡æ˜“çš„ã�ªã‚³ã‚¹ãƒˆæŒ‡æ¨™ã�¨ã�—ã�¦æœ‰åŠ¹
    df['APP_TOTAL_COST_REAL'] = df['AMT_ANNUITY'] * df['CNT_PAYMENT'] / df['AMT_CREDIT'].replace(0, np.nan)

    # --- 3. ä¸‹ä½�ãƒ†ãƒ¼ãƒ–ãƒ«ã�®çµ�å�ˆ ---
    # SK_ID_PREVï¼ˆé��å�»ã�®å€‹åˆ¥ã�®ç”³ã�—è¾¼ã�¿IDï¼‰å�˜ä½�ã�§çµ�å�ˆ
    df = df.merge(pos_agg, on='SK_ID_PREV', how='left')
    df = df.merge(inst_agg, on='SK_ID_PREV', how='left')
    
    # --- 4. ã‚«ãƒ†ã‚´ãƒªå¤‰æ•°ã�®å‡¦ç�† ---
    # æ–‡å­—åˆ—ãƒ‡ãƒ¼ã‚¿ã‚’æ•°å€¤ï¼ˆãƒ©ãƒ™ãƒ«ï¼‰ã�«å¤‰æ�›
    cat_cols = df.select_dtypes(include=['object']).columns.drop(['SK_ID_PREV'], errors='ignore')
    for col in cat_cols:
        df[col], _ = pd.factorize(df[col])
    
    # --- 5. é›†è¨ˆãƒ«ãƒ¼ãƒ«ã�®å®šç¾© ---
    # å…¨ã�¦ã�®æ•°å€¤åˆ—ã�«å¯¾ã�—ã�¦çµ±è¨ˆé‡�ã‚’è¨ˆç®—ï¼ˆã�“ã�“ã�«è¿½åŠ ã�—ã�ŸATAMARATEã�ªã�©ã‚‚è‡ªå‹•ã�§å�«ã�¾ã‚Œã�¾ã�™ï¼‰
    num_cols = df.select_dtypes(include=['number']).columns.drop(['SK_ID_CURR', 'SK_ID_PREV'])
    
    agg_funcs = {}
    for col in num_cols:
        # åŸºæœ¬ã�®4çµ±è¨ˆé‡�
        agg_funcs[col] = ['min', 'max', 'mean', 'sum']
    
    # --- 6. å…¨ä½“ã�®çµ±è¨ˆé‡�ã‚’é›†è¨ˆ (SK_ID_CURR å�˜ä½�) ---
    df_agg = df.groupby('SK_ID_CURR').agg(agg_funcs)
    df_agg.columns = ['PREV_AGG_' + '_'.join(col).upper() for col in df_agg.columns]
    df_agg = df_agg.reset_index()
    
    # --- 7. ã€Œç›´è¿‘ã�®ç”³è«‹ã€�æƒ…å ±ã�®æŠ½å‡º ---
    # DAYS_DECISIONï¼ˆç”³è«‹æ—¥ï¼‰ã�Œæœ€å¤§ã�®ã‚‚ã�®ã�Œæœ€æ–°
    # æœ€æ–°ã�®ç”³è«‹ã�§ã�®é ­é‡‘æ¯”ç�‡ã�ªã�©ã�¯ã€�ç�¾åœ¨ã�®è¿”æ¸ˆèƒ½åŠ›ã‚’æœ€ã‚‚ã‚ˆã��è¡¨ã�™
    df_sorted = df.sort_values(['SK_ID_CURR', 'DAYS_DECISION'])
    last_app = df_sorted.groupby('SK_ID_CURR').last().reset_index()
    
    # ã‚«ãƒ©ãƒ å��ã�« LAST_ ã‚’ã�¤ã�‘ã�¦åŒºåˆ¥
    last_app_cols = ['LAST_' + col if col not in ['SK_ID_CURR', 'SK_ID_PREV'] else col for col in last_app.columns]
    last_app.columns = last_app_cols

    # --- 8. å…¨ä½“ã�®çµ±è¨ˆã�¨ç›´è¿‘ã�®æƒ…å ±ã‚’çµ�å�ˆã�—ã�¦å®Œæˆ� ---
    final_df = df_agg.merge(last_app.drop(columns=['SK_ID_PREV']), on='SK_ID_CURR', how='left')
    
    # ãƒ¡ãƒ¢ãƒªç¯€ç´„
    del df, df_agg, last_app, pos_agg, inst_agg
    gc.collect()
    
    return final_df


final_features = aggregate_prev_app_advanced(prev_app, pos_cash, installments)



train_df = train_df.merge(final_features, on='SK_ID_CURR', how='left')
test_df  = test_df .merge(final_features, on='SK_ID_CURR', how='left')


train_df['EXT_X_ATAMARATE'] = train_df['EXT_SOURCE_2'] * train_df['PREV_AGG_APP_CREDIT_ATAMARATE_MAX']
test_df['EXT_X_ATAMARATE'] = test_df['EXT_SOURCE_2'] * test_df['PREV_AGG_APP_CREDIT_ATAMARATE_MAX']


#final_featuresã�§ã�®äºˆæ¸¬
#step1 äºˆæ¸¬ç”¨ãƒ‡ãƒ¼ã‚¿ã‚»ãƒƒãƒˆã�®ä½œæˆ�
# 1. ãƒ¡ã‚¤ãƒ³ã�®å­¦ç¿’ãƒ‡ãƒ¼ã‚¿ã�‹ã‚‰SK_ID_CURRã�¨Targetåˆ—ã�®ã�¿ã‚’æŠ½å‡º
train_target = train_df[['SK_ID_CURR', 'TARGET']]

# 2. POS_CASHãƒ‡ãƒ¼ã‚¿ã�¨Targetåˆ—ã‚’çµ�å�ˆ
# 'how="left"' ã‚’ä½¿ã�†ã�“ã�¨ã�§ã€�POS_CASHã�®ã�™ã�¹ã�¦ã�®å±¥æ­´ã�«Targetã‚’ç´�ã�¥ã�‘ã‚‹
final_features_with_target = pd.merge(final_features, 
                                train_target, 
                                on='SK_ID_CURR', 
                                how='left')

# 3. Targetã�Œæ¬ æ��ã�—ã�¦ã�„ã‚‹è¡Œï¼ˆãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿ã�®é¡§å®¢IDï¼‰ã�¯é™¤å¤–/åˆ†é›¢
# ã�“ã‚Œã�§ã€�ã‚µãƒ–ãƒ¢ãƒ‡ãƒ«ã�®å­¦ç¿’ã�«ä½¿ã�ˆã‚‹ãƒ‡ãƒ¼ã‚¿ã�Œæº–å‚™å®Œäº†


#extratreeã�§ã�®ã‚¹ã‚¿ãƒƒã‚­ãƒ³ã‚°å®Ÿè¡Œ
final_features_with_target['final_ridge_et_score'] = get_et_preds(final_features_with_target)
# train_df ã�«æ�¥ç¶š
train_df = train_df.merge(
    final_features_with_target[['SK_ID_CURR', 'final_ridge_et_score']], 
    on='SK_ID_CURR', 
    how='left'
)

# test_df ã�«æ�¥ç¶š
test_df = test_df.merge(
    final_features_with_target[['SK_ID_CURR', 'final_ridge_et_score']], 
    on='SK_ID_CURR', 
    how='left'
)


#ãƒªãƒƒã‚¸å›�å¸°ã�§ã�®ã‚¹ã‚¿ãƒƒã‚­ãƒ³ã‚°
final_features_with_target['final_ridge_score'] = get_ridge_preds(final_features_with_target)

# å­¦ç¿’ãƒ‡ãƒ¼ã‚¿(train_df)ã�«ãƒ�ãƒ¼ã‚¸
train_df = train_df.merge(
    final_features_with_target[['SK_ID_CURR', 'final_ridge_score']], 
    on='SK_ID_CURR', 
    how='left'
)

# ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿(test_df)ã�«ãƒ�ãƒ¼ã‚¸
test_df = test_df.merge(
    final_features_with_target[['SK_ID_CURR', 'final_ridge_score']], 
    on='SK_ID_CURR', 
    how='left'
)


NFOLDS = 5
folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)

# 1. å­¦ç¿’ç”¨ã�¨ãƒ†ã‚¹ãƒˆç”¨ã�«åˆ†ã�‘ã‚‹
train_data = final_features_with_target[final_features_with_target['TARGET'].notnull()].reset_index(drop=True)
test_data = final_features_with_target[final_features_with_target['TARGET'].isnull()].reset_index(drop=True)

# 2. ç‰¹å¾´é‡�åˆ—ã‚’æŠ½å‡ºï¼ˆSK_ID_CURR ã�¨ TARGET ã�¯é™¤å¤–ï¼‰
feature_cols = [c for c in final_features.columns if c != 'SK_ID_CURR']

# 3. objectåˆ—ã‚’ã‚«ãƒ†ã‚´ãƒªå�‹ã�«å¤‰æ�›
for col in feature_cols:
    if train_data[col].dtype == 'object':
        train_data[col] = train_data[col].astype('category')
        test_data[col] = test_data[col].astype('category')

# 4. OOF é…�åˆ—
oof_preds_pc = np.zeros(train_data.shape[0])
test_preds_pc = np.zeros(test_data.shape[0])

# 5. KFold å­¦ç¿’
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_data[feature_cols], train_data['TARGET'])):
    X_train_fold = train_data.iloc[train_idx][feature_cols]
    y_train_fold = train_data.iloc[train_idx]['TARGET']
    X_valid_fold = train_data.iloc[valid_idx][feature_cols]
    y_valid_fold = train_data.iloc[valid_idx]['TARGET']

    lgb_model = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=32,
        colsample_bytree=0.8,
        subsample=0.8,
        random_state=42,
        n_jobs=-1
    )
    lgb_model.fit(
        X_train_fold, y_train_fold,
        eval_metric='auc',
        eval_set=[(X_valid_fold, y_valid_fold)],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
    )

    # OOFäºˆæ¸¬
    oof_preds_pc[valid_idx] = lgb_model.predict_proba(X_valid_fold)[:, 1]
    # ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿äºˆæ¸¬
    test_preds_pc += lgb_model.predict_proba(test_data[feature_cols])[:, 1] / NFOLDS

# 6. OOFåˆ—ã‚’è¿½åŠ 
train_data['FF_OOF_LGBM'] = oof_preds_pc
test_data['FF_OOF_LGBM'] = test_preds_pc




# 7. SK_ID_CURR ã�¨ OOFåˆ—ã� ã�‘æŠ½å‡º
oof_final_features = train_data[['SK_ID_CURR','FF_OOF_LGBM']]
test_final_features = test_data[['SK_ID_CURR','FF_OOF_LGBM']]

# å†�å®Ÿè¡Œã�—ã�¦ã‚‚åˆ—ã�Œå¢—ã�ˆã�ªã�„ã‚ˆã�†ã�«ã�™ã‚‹ã�Ÿã‚�ã�®å®‰å…¨ç­–
if 'FF_OOF_LGBM' in train_df.columns:
    train_df = train_df.drop(columns=['FF_OOF_LGBM'])
if 'FF_OOF_LGBM' in test_df.columns:
    test_df = test_df.drop(columns=['FF_OOF_LGBM'])

# 8. å…ƒã�® train_df/test_df ã�« merge
train_df = train_df.merge(oof_final_features, on='SK_ID_CURR', how='left')
test_df = test_df.merge(test_final_features, on='SK_ID_CURR', how='left')


#è¦�ã‚‰ã�ªã�„ã‚‚ã�®ã�¯æ¶ˆã�™
del final_features
gc.collect() 


# trainã�¨testã�®å½¢çŠ¶ï¼ˆè¡Œæ•°, åˆ—æ•°ï¼‰ã‚’è¡¨ç¤º
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# ç‰¹å¾´é‡�ã�®æ•°ï¼ˆIDã�¨TARGETã‚’é™¤ã�„ã�Ÿæ•°ï¼‰
features_count = len([c for c in train_df.columns if c not in ['SK_ID_CURR', 'TARGET']])
print(f"ç�¾åœ¨ã�®ç‰¹å¾´é‡�ã�®æ•°: {features_count}")


# #æ—§
# import pandas as pd
# import numpy as np
# #installmentã‚’ã�µã‚‹ã�„IDã�§é›†è¨ˆã�™ã‚‹
# def aggregate_installments_prev(installments):
#     df = installments.copy()
#     
#     # 1. å‰�å‡¦ç�†ã�¨åŸºæœ¬ãƒ•ãƒ©ã‚°
#     # df['PAYMENT_MISSING'] = df['AMT_PAYMENT'].isna().astype(int)
#     
#     # é�…å»¶æ—¥æ•°ï¼ˆDAYS_ENTRY_PAYMENTã�Œæ¬ æ��ã�—ã�¦ã�„ã‚‹å ´å�ˆã�¯ä¸€æ—¦0ã�§è£œå®Œã�™ã‚‹ã�‹NaNã�®ã�¾ã�¾ã�«ã�™ã‚‹ï¼‰
#     # ã�“ã�“ã�§ã�¯å®Ÿå‹™çš„ã�«ã€Œæœªæ‰•ã�„ã�¯æœ€å¤§é�…å»¶ã€�ã�¨ã�—ã�¦æ‰±ã�†ã�‹ã€�å�˜ã�ªã‚‹å·®åˆ†ã‚’ã�¨ã‚Šã�¾ã�™
#     # df['DELAY'] = df['DAYS_ENTRY_PAYMENT'] - df['DAYS_INSTALMENT']
#     # df['DELAY_FLAG'] = (df['DELAY'] > 0).astype(int)
# 
#     # 5æ—¥ä»¥ä¸Šã�®å®Ÿè³ªçš„ã�ªé�…å»¶ / 30æ—¥ä»¥ä¸Šã�®æ·±åˆ»ã�ªé�…å»¶
#     # df['DELAY_5D_FLAG'] = (df['DELAY'] > 5).astype(int)
#     # df['DELAY_30D_FLAG'] = (df['DELAY'] > 30).astype(int)
# 
#     # 2. ç›´è¿‘1å¹´ï¼ˆ-365æ—¥ä»¥å†…ï¼‰ã�®ãƒ‡ãƒ¼ã‚¿ã�«çµ�ã�£ã�Ÿãƒ•ãƒ©ã‚°
#     # recent_1y_mask = df['DAYS_INSTALMENT'] >= -365
#     # df['DELAY_RECENT_1Y_FLAG'] = np.where(recent_1y_mask, df['DELAY_FLAG'], np.nan)
#     
#     # 3. é‡‘é¡�å·®ï¼ˆä¸�è¶³ãƒ»é��æ‰•ã�„ï¼‰
#     # df['AMT_SHORTFALL'] = df['AMT_INSTALMENT'] - df['AMT_PAYMENT']
#     # å¤‰æ•°å��ã‚’ AMT_SHORTFALL ã�«çµ±ä¸€ã�—ã�¦ä¿®æ­£
#     # df['PAYMENT_SHORTFALL_1000_FLAG'] = (df['AMT_SHORTFALL'] > 1000).astype(int)
#     # df['SHORTFALL_FLAG'] = (df['AMT_SHORTFALL'] > 0).astype(int)
#     # df['OVERPAY_FLAG'] = (df['AMT_SHORTFALL'] < 0).astype(int)
#     
#     # 4. ãƒ�ãƒ¼ã‚¸ãƒ§ãƒ³æƒ…å ±ã�®ãƒ€ãƒŸãƒ¼åŒ–
#     # ãƒ¡ãƒ¢ãƒªç¯€ç´„ã�®ã�Ÿã‚�ä¸Šä½�ã�®ã‚‚ã�®ã� ã�‘ã�«çµ�ã‚‹ã�‹ã€�ã��ã�®ã�¾ã�¾é›†è¨ˆ
#     # version_flag = pd.get_dummies(df['NUM_INSTALMENT_VERSION'], prefix='INST_VER')
#     # df = pd.concat([df, version_flag], axis=1)
#     
#     # 5. é›†ç´„å‡¦ç�† (SK_ID_PREV å�˜ä½�)
#     # å¾Œã�®ãƒ�ãƒ¼ã‚¸ã�®ã�Ÿã‚�ã�« SK_ID_CURR ã‚‚æ®‹ã�™
#     # agg_ops = {
#     #     'DELAY': ['max', 'mean', 'sum'],
#     #     'DELAY_FLAG': ['mean', 'sum'],
#     #     'DELAY_5D_FLAG': ['sum'],
#     #     'DELAY_30D_FLAG': ['sum'],
#     #     'DELAY_RECENT_1Y_FLAG': ['mean'],
#     #     'AMT_SHORTFALL': ['max', 'mean', 'sum'],
#     #     'PAYMENT_SHORTFALL_1000_FLAG': ['sum'],
#     #     'SHORTFALL_FLAG': ['mean'],
#     #     'OVERPAY_FLAG': ['mean'],
#     #     'AMT_INSTALMENT': ['sum', 'mean'],
#     #     'NUM_INSTALMENT_NUMBER': ['max', 'mean']
#     # }
#     
#     # ãƒ�ãƒ¼ã‚¸ãƒ§ãƒ³ãƒ•ãƒ©ã‚°ã�®åˆ—å��ã‚’å�–å¾—ã�—ã�¦é›†è¨ˆå¯¾è±¡ã�«è¿½åŠ 
#     # ver_cols = [c for c in df.columns if 'INST_VER_' in c]
#     # for c in ver_cols:
#     #     agg_ops[c] = ['sum']
# 
#     # ãƒ­ãƒ¼ãƒ³å¥‘ç´„(SK_ID_PREV)ã�”ã�¨ã�®é›†è¨ˆ
#     # prev_agg = df.groupby(['SK_ID_CURR', 'SK_ID_PREV']).agg(agg_ops)
#     # prev_agg.columns = ['_'.join(col).upper() for col in prev_agg.columns]
#     # prev_agg = prev_agg.reset_index()
# 
#     # 6. æœ€çµ‚é›†ç´„ (SK_ID_CURR å�˜ä½�)
#     # é��å�»ã�®è¤‡æ•°ã�®ãƒ­ãƒ¼ãƒ³ã‚’ã�¾ã�¨ã‚�ã�¦1ãƒ¦ãƒ¼ã‚¶ãƒ¼ã�®ç‰¹å¾´é‡�ã�«ã�™ã‚‹
#     # final_agg = prev_agg.groupby('SK_ID_CURR').agg(['mean', 'max', 'sum'])
#     # final_agg.columns = ['INST_FIN_' + '_'.join(col).upper() for col in final_agg.columns]
#     # 
#     # return final_agg.reset_index()


pos_cash.info()


pos_cash.isnull().sum()


# #æ—§
# import pandas as pd
# import numpy as np
# 
# # =========================
# # POS_CASH ã‚’ SK_ID_PREV å�˜ä½�ã�§é›†ç´„(å�¤ã�„æ–¹ã�®IDï¼‰
# # =========================
# def aggregate_pos_cash_prev(df):
#     df = df.copy()
#     
#     # æ¬ æ��å€¤å‡¦ç�†
#     # df['CNT_INSTALMENT'] = df['CNT_INSTALMENT'].fillna(0)
#     # df['CNT_INSTALMENT_FUTURE'] = df['CNT_INSTALMENT_FUTURE'].fillna(0)
#     
#     # è¿”æ¸ˆé€²æ�—ãƒ»å‰²å�ˆ
#     # df['CNT_INSTALMENT_RATIO'] = df['CNT_INSTALMENT_FUTURE'] / df['CNT_INSTALMENT'].replace(0, np.nan)
#     # df['REPAYMENT_PROGRESS'] = 1 - df['CNT_INSTALMENT_RATIO']
#     # df['REPAYMENT_PROGRESS'] = df['REPAYMENT_PROGRESS'].fillna(0)
#     
#     # å»¶æ»�ãƒ•ãƒ©ã‚°
#     # df['DPD_NONZERO'] = (df['SK_DPD'] > 0).astype(int)
#     
#     # é›†ç´„
#     # agg_funcs = {
#     #     'CNT_INSTALMENT': ['max','mean','sum'],
#     #     'CNT_INSTALMENT_FUTURE': ['max','mean','sum'],
#     #     'CNT_INSTALMENT_RATIO': ['mean','max'],
#     #     'REPAYMENT_PROGRESS': ['mean','max'],
#     #     'SK_DPD': ['max','mean','sum','median'],
#     #     'DPD_NONZERO': 'sum'
#     # }
#     
#     # df_agg = df.groupby('SK_ID_PREV').agg(agg_funcs)
#     # df_agg.columns = ['_'.join(col).strip() for col in df_agg.columns.values]
#     # df_agg = df_agg.


# #æ—§
# import pandas as pd
# import numpy as np
# 
# # =========================
# # previous_application + POS_CASH + INSTALLMENTS é›†ç´„ï¼ˆçµ�å�ˆæ¸ˆã�¿ç‰ˆï¼‰
# # =========================
# def aggregate_prev_app_with_features(prev_app):
#     df = prev_app.copy()
#     
#     # =========================
#     # æ•°å€¤åˆ—ã�®æ¬ æ��å‡¦ç�†ï¼ˆä¸­å¤®å€¤è£œå®Œï¼‰ï¼‹æ¬ æ��ãƒ•ãƒ©ã‚°
#     # =========================
#     # num_cols = df.select_dtypes(include=['float64','int64']).columns.drop(['SK_ID_CURR','SK_ID_PREV'], errors='ignore')
#     # for col in num_cols:
#     #     df[col + '_MISSING'] = df[col].isna().astype(int)
#     #     df[col] = df[col].fillna(df[col].median())
#     
#     # =========================
#     # ã‚«ãƒ†ã‚´ãƒªåˆ—ã�®æ¬ æ��è£œå®Œ
#     # =========================
#     # cat_cols = df.select_dtypes(include=['object']).columns.drop(['SK_ID_CURR','SK_ID_PREV'], errors='ignore')
#     # df[cat_cols] = df[cat_cols].fillna('Missing')
#     
#     # =========================
#     # SK_ID_CURR å�˜ä½�ã�§æ•°å€¤åˆ—ã‚’é›†ç´„
#     # =========================
#     # agg_funcs = {}
#     # for col in num_cols:
#     #     agg_funcs[col] = ['max','mean','sum']
#     
#     # df_num_agg = df.groupby('SK_ID_CURR').agg(agg_funcs)
#     # df_num_agg.columns = ['_'.join(col).strip() for col in df_num_agg.columns.values]
#     # df_num_agg = df_num_agg.reset_index()
#     
#     # =========================
#     # ã‚«ãƒ†ã‚´ãƒªåˆ—ã‚’ãƒ¯ãƒ³ãƒ›ãƒƒãƒˆåŒ–ã�—ã�¦é›†ç´„
#     # =========================
#     # cat_ohe = pd.get_dummies(df[cat_cols].astype(str))
#     # cat_ohe['SK_ID_CURR'] = df['SK_ID_CURR']
#     # cat_ohe_agg = cat_ohe.groupby('SK_ID_CURR').sum().reset_index()
#     
#     # =========================
#     # æ•°å€¤åˆ—é›†ç´„ã�¨ã‚«ãƒ†ã‚´ãƒªåˆ—é›†ç´„ã‚’çµ�å�ˆ
#     # =========================
#     # prev_app_df = df_num_agg.merge(cat_ohe_agg, on='SK_ID_CURR', how='left')
#     
#     # return prev_app_df
# 
# # =====================================
# # ä½¿ç”¨ä¾‹
# # =====================================
# # prev_app ã�¯ installments ã�¨ pos_cash ã‚’ SK_ID_PREV å�˜ä½�ã�§çµ�å�ˆæ¸ˆã�¿ã�®ãƒ‡ãƒ¼ã‚¿
# # prev_app_df = aggregate_prev_app_with_features(prev_app)
# 
# # print(prev_app_df.head())
# # print(prev_app_df.shape)


#æ—§
#train_df = train_df.merge(prev_app_df, on='SK_ID_CURR', how='left')
#test_df  = test_df .merge(prev_app_df, on='SK_ID_CURR', how='left')



# ==========================================================
# STEP 1: bureau_aggã�®åˆ�æœŸåŒ–ã�¨ã€Œä»¶æ•°ãƒ»æ¯”ç�‡ã€�ç‰¹å¾´é‡�ã�®ä½œæˆ�
# ==========================================================

## 1-1. ãƒ­ãƒ¼ãƒ³ä»¶æ•°ã�¨ãƒ•ãƒ©ã‚°ã�®é›†è¨ˆï¼ˆbureau_aggã�®ãƒ™ãƒ¼ã‚¹ä½œæˆ�ï¼‰
# bureau_aggã‚’ã�“ã�“ã�§å®šç¾©ãƒ»åˆ�æœŸåŒ–â€¼
bureau_agg = bureau.groupby('SK_ID_CURR')['SK_ID_BUREAU'].count().reset_index()
bureau_agg.rename(columns={'SK_ID_BUREAU': 'BUREAU_LOAN_COUNT'}, inplace=True)
bureau_agg['HIGH_LOAN_FLAG'] = (bureau_agg['BUREAU_LOAN_COUNT'] >= 8).astype(int)
bureau_agg['VERY_HIGH_LOAN_FLAG'] = (bureau_agg['BUREAU_LOAN_COUNT'] >= 20).astype(int)

## 1-2. ACTIVE / CLOSED / SOLD ã�®é›†è¨ˆ
bureau_status_counts = bureau.pivot_table(
    index='SK_ID_CURR',
    columns='CREDIT_ACTIVE',
    values='SK_ID_BUREAU',
    aggfunc='count',
    fill_value=0 
).reset_index()
bureau_status_counts.columns.name = None
status_cols = ['SK_ID_CURR', 'Active', 'Closed', 'Sold']
bureau_status_counts = bureau_status_counts[[col for col in status_cols if col in bureau_status_counts.columns]]

## 1-3. ä¸�è‰¯ãƒ­ãƒ¼ãƒ³ä»¶æ•°ã�®é›†è¨ˆ
bureau_bad_count = bureau.groupby('SK_ID_CURR')['CREDIT_ACTIVE'].apply(
    lambda x: ((x == 'Bad debt') | (x == 'Delinquent')).sum()
).reset_index(name='BUREAU_BAD_LOAN_COUNT')

## 1-4. bureau_aggã�«ãƒ�ãƒ¼ã‚¸
bureau_agg = bureau_agg.merge(bureau_status_counts, on='SK_ID_CURR', how='left').fillna(0)
bureau_agg = bureau_agg.merge(bureau_bad_count, on='SK_ID_CURR', how='left').fillna(0)

## 1-5. ç·�ãƒ­ãƒ¼ãƒ³ä»¶æ•°ã�¨æ¯”ç�‡ç‰¹å¾´é‡�ã�®è¨ˆç®—
bureau_agg['TOTAL_LOAN'] = bureau_agg[['Active', 'Closed', 'Sold']].sum(axis=1)
denom = bureau_agg['TOTAL_LOAN'].replace(0, 1)

bureau_agg['ACTIVE_RATIO'] = bureau_agg['Active'] / denom
bureau_agg['CLOSED_RATIO'] = bureau_agg['Closed'] / denom
bureau_agg['SOLD_RATIO'] = bureau_agg['Sold'] / denom
bureau_agg['BAD_LOAN_RATIO'] = bureau_agg['BUREAU_BAD_LOAN_COUNT'] / denom

print("ã‚»ãƒ«1ã�®å‡¦ç�†å®Œäº†: bureau_aggã�Œåˆ�æœŸåŒ–ã�•ã‚Œã�¾ã�—ã�Ÿã€‚")


# =========================================
# bureau_balance ã�®é›†è¨ˆï¼ˆSK_ID_BUREAUå�˜ä½�ï¼‰
# =========================================
bureau_bal['DELAY_FLAG'] = bureau_bal['STATUS'].isin(['1','2','3','4','5']).astype(int)

# SK_ID_BUREAU ã�”ã�¨ã�®é›†è¨ˆ
bureau_bal_agg = bureau_bal.groupby('SK_ID_BUREAU').agg(
    DELAY_RATIO=('DELAY_FLAG','mean'),
    MONTHS_HISTORY=('MONTHS_BALANCE', lambda x: x.max()-x.min())
).reset_index()

# æœ€æ–°ã‚¹ãƒ†ãƒ¼ã‚¿ã‚¹ã‚’ãƒ¯ãƒ³ãƒ›ãƒƒãƒˆåŒ–
latest_status = bureau_bal.sort_values(['SK_ID_BUREAU','MONTHS_BALANCE']).groupby('SK_ID_BUREAU').tail(1)
status_dummies = pd.get_dummies(latest_status['STATUS'], prefix='STATUS')
status_dummies['SK_ID_BUREAU'] = latest_status['SK_ID_BUREAU'].values
status_agg = status_dummies.groupby('SK_ID_BUREAU').sum().reset_index()

# ãƒ�ãƒ¼ã‚¸ã�—ã�¦ bureau_balance ã�® SK_ID_BUREAU å�˜ä½�é›†ç´„å®Œæˆ�
bureau_bal_agg = bureau_bal_agg.merge(status_agg, on='SK_ID_BUREAU', how='left')

# =========================================
# bureau_balance ã‚’ bureau ã�«çµ�å�ˆ
# =========================================
bureau_full = bureau.merge(bureau_bal_agg, on='SK_ID_BUREAU', how='left')

# =========================================
# SK_ID_CURR å�˜ä½�ã�§æœ€çµ‚é›†ç´„
# =========================================
# æ•°å€¤åˆ—ã‚’è‡ªå‹•æ¤œå‡º
num_cols = bureau_full.select_dtypes(include=['int64','float64']).columns.drop(['SK_ID_CURR','SK_ID_BUREAU'])
agg_funcs = {col: ['mean','max','sum'] for col in num_cols}

bureau_bal_features = bureau_full.groupby('SK_ID_CURR').agg(agg_funcs)
bureau_bal_features.columns = ['_'.join(col).strip() for col in bureau_bal_features.columns.values]
bureau_bal_features = bureau_bal_features.reset_index()

# =========================================
# å…ƒã�® bureau_agg ã�«ãƒ�ãƒ¼ã‚¸
# =========================================
bureau_agg = bureau_agg.merge(bureau_bal_features, on='SK_ID_CURR', how='left')

print("bureau_balance å��æ˜ æ¸ˆã�¿ bureau_agg ã�Œå®Œæˆ�ã�—ã�¾ã�—ã�Ÿã€‚")
print(bureau_agg.head())



# ==========================================================
# STEP 2: ã€ŒæœŸé–“ãƒ»æ´»å‹•ã€�ç‰¹å¾´é‡�ã�®ä½œæˆ�ã�¨ bureau_agg ã�¸ã�®çµ�å�ˆ
# ==========================================================

## 2-1. å…¨ãƒ­ãƒ¼ãƒ³æœŸé–“ã�«é–¢ã�™ã‚‹é›†ç´„
days_features = bureau.groupby('SK_ID_CURR')['DAYS_CREDIT'].agg(
    YEARS_CREDIT_RECENT='max',
    YEARS_CREDIT_OLDEST='min',
    YEARS_CREDIT_MEAN='mean',
    CREDIT_STD='std'
).reset_index()

# æ—¥æ•°ã‚’å¹´ã�«æ�›ç®—ã�—ã€�è² ã�®å€¤ã‚’æ­£ã�«å¤‰æ�›
for col in ['YEARS_CREDIT_RECENT','YEARS_CREDIT_OLDEST','YEARS_CREDIT_MEAN']:
    days_features[col] = -days_features[col] / 365

# æ¨™æº–å��å·®ã‚’å¹´ã�«æ�›ç®—ã�—ã€�NaNã‚’0ã�§ç½®æ�›
days_features['YEARS_CREDIT_STD'] = (days_features['CREDIT_STD'] / 365).fillna(0)
days_features.drop(columns=['CREDIT_STD'], inplace=True)

# ãƒ­ãƒ¼ãƒ³æ´»å‹•ã�®ã‚¹ãƒ‘ãƒ³
days_features['YEARS_CREDIT_SPAN'] = days_features['YEARS_CREDIT_OLDEST'] - days_features['YEARS_CREDIT_RECENT']

# 1å¹´ä»¥å†…ã�«ãƒ­ãƒ¼ãƒ³å¥‘ç´„ã�—ã�¦ã�„ã‚‹ã�‹
days_features['RECENT_LOAN_FLAG'] = (days_features['YEARS_CREDIT_RECENT'] <= 1).astype(int)


## 2-2. ã‚¢ã‚¯ãƒ†ã‚£ãƒ–ãƒ­ãƒ¼ãƒ³æœŸé–“ã�«é–¢ã�™ã‚‹é›†ç´„
active_loans = bureau[bureau['CREDIT_ACTIVE'] == 'Active']
oldest_active = active_loans.groupby('SK_ID_CURR')['DAYS_CREDIT'].min().reset_index()
oldest_active['YEARS_CREDIT_OLDEST_ACTIVE'] = -oldest_active['DAYS_CREDIT'] / 365
oldest_active = oldest_active[['SK_ID_CURR', 'YEARS_CREDIT_OLDEST_ACTIVE']]


## 2-3. bureau_agg ã�¸ã�®ãƒ�ãƒ¼ã‚¸ï¼ˆå¼•ã��ç¶™ã��ï¼‰
# ã�“ã�“ã�§ã€Œã‚»ãƒ«1ã�§å®šç¾©ã�•ã‚Œã�Ÿbureau_aggã€�ã‚’èª­ã�¿è¾¼ã�¿ã€�æ–°ã�—ã�„ç‰¹å¾´é‡�ã‚’ãƒ�ãƒ¼ã‚¸ã�—ã�¦ã€�
# å†�ã�³bureau_aggã�¨ã�—ã�¦ä¸Šæ›¸ã��ä¿�å­˜ï¼ˆå¼•ã��ç¶™ã��ï¼‰ã�—ã�¾ã�™ã€‚
bureau_agg = bureau_agg.merge(days_features, on='SK_ID_CURR', how='left')
bureau_agg = bureau_agg.merge(oldest_active, on='SK_ID_CURR', how='left')

# ã‚¢ã‚¯ãƒ†ã‚£ãƒ–ãƒ­ãƒ¼ãƒ³ã�Œã�ªã�„é¡§å®¢ã�® 'YEARS_CREDIT_OLDEST_ACTIVE' ã�® NaN ã‚’ 0 ã�§åŸ‹ã‚�ã‚‹
bureau_agg['YEARS_CREDIT_OLDEST_ACTIVE'] = bureau_agg['YEARS_CREDIT_OLDEST_ACTIVE'].fillna(0)

print("ã‚»ãƒ«2ã�®å‡¦ç�†å®Œäº†: bureau_aggã�«æœŸé–“ãƒ»æ´»å‹•ã�®ç‰¹å¾´é‡�ã�Œè¿½åŠ ã�•ã‚Œã�¾ã�—ã�Ÿã€‚")


# ==========================================================
# â˜… STEP 3: ã€Œå»¶æ»�ã€�ç‰¹å¾´é‡�ã�®ä½œæˆ�ã�¨ bureau_agg ã�¸ã�®çµ�å�ˆ
# ==========================================================

# 1. å»¶æ»�ã�«é–¢ã�™ã‚‹ç‰¹å¾´é‡�ã�®é›†è¨ˆ

# å»¶æ»�çµŒé¨“ã�®æœ‰ç„¡ (max > 0)
bureau_has_overdue = (
    bureau.groupby('SK_ID_CURR')['CREDIT_DAY_OVERDUE'].max() > 0
).astype(int).reset_index(name='HAS_OVERDUE')

# æœ€å¤§å»¶æ»�æ—¥æ•°
bureau_max_overdue = bureau.groupby('SK_ID_CURR')['CREDIT_DAY_OVERDUE'].max().reset_index(name='MAX_OVERDUE')

# å¹³å�‡å»¶æ»�æ—¥æ•°
bureau_mean_overdue = bureau.groupby('SK_ID_CURR')['CREDIT_DAY_OVERDUE'].mean().reset_index(name='MEAN_OVERDUE')

# å»¶æ»�ãƒ­ãƒ¼ãƒ³ã�®å‰²å�ˆ (å»¶æ»�æ—¥æ•°>0ã�®å‰²å�ˆ)
overdue_ratio = bureau.groupby('SK_ID_CURR')['CREDIT_DAY_OVERDUE'].apply(
    lambda x: (x > 0).mean()
).reset_index(name='OVERDUE_RATIO')

# å»¶æ»�ãƒ­ãƒ¼ãƒ³ã�®ä»¶æ•°
overdue_count = bureau.groupby('SK_ID_CURR')['CREDIT_DAY_OVERDUE'].apply(
    lambda x: (x > 0).sum()
).reset_index(name='OVERDUE_COUNT')


# 2. é›†è¨ˆçµ�æ�œã‚’ bureau_agg ã�«ãƒ�ãƒ¼ã‚¸ï¼ˆçµ�å�ˆï¼‰

# ã�¾ã�šã€�å»¶æ»�ç‰¹å¾´é‡�ã‚’ä¸€æ™‚çš„ã�ªDFã�«ã�¾ã�¨ã‚�ã‚‹
overdue_features = bureau_has_overdue.copy()
overdue_features = overdue_features.merge(bureau_max_overdue, on='SK_ID_CURR', how='left')
overdue_features = overdue_features.merge(bureau_mean_overdue, on='SK_ID_CURR', how='left')
overdue_features = overdue_features.merge(overdue_ratio, on='SK_ID_CURR', how='left')
overdue_features = overdue_features.merge(overdue_count, on='SK_ID_CURR', how='left')

# bureau_agg ã�«ãƒ�ãƒ¼ã‚¸
# ã‚»ãƒ«2ã�®å®Ÿè¡Œçµ�æ�œã�§ã�‚ã‚‹ bureau_agg ã�«ã€�æ–°ã�—ã�„ç‰¹å¾´é‡�ã‚’ãƒ�ãƒ¼ã‚¸ã�—ã�¦å¼•ã��ç¶™ã��ã�¾ã�™
bureau_agg = bureau_agg.merge(overdue_features, on='SK_ID_CURR', how='left')

# â€» æ³¨æ„�: ã�“ã�“ã�¾ã�§ã�®é›†è¨ˆã�¯ã€�å…ƒã�®ã€ŒCREDIT_DAY_OVERDUEã€�åˆ—ï¼ˆç•°å¸¸å€¤ä¿®æ­£å‰�ï¼‰ã�«åŸºã�¥ã�„ã�¦ã�„ã�¾ã�™ã€‚

# 3. ç•°å¸¸å€¤ã�®ä¿®æ­£ï¼ˆå…ƒã�®ãƒ‡ãƒ¼ã‚¿ãƒ•ãƒ¬ãƒ¼ãƒ  bureau ã�«å¯¾ã�™ã‚‹å‡¦ç�†ï¼‰

# ç•°å¸¸å€¤ã�®ä¿®æ­£ï¼ˆä¸Šé™�ã‚«ãƒƒãƒˆå‡¦ç�†ã‚’ bureau ã�«é�©ç”¨ï¼‰
cap_value = 180

# ä¸Šé™�ã‚«ãƒƒãƒˆï¼ˆå…ƒã�®ãƒ‡ãƒ¼ã‚¿ãƒ•ãƒ¬ãƒ¼ãƒ  bureau ã‚’æ›´æ–°ï¼‰
bureau['CREDIT_DAY_OVERDUE_CAPPED'] = np.where(
    bureau['CREDIT_DAY_OVERDUE'] > cap_value,
    cap_value,
    bureau['CREDIT_DAY_OVERDUE']
)

# ä¿®æ­£ãƒ•ãƒ©ã‚°ï¼ˆå…ƒã�®ãƒ‡ãƒ¼ã‚¿ãƒ•ãƒ¬ãƒ¼ãƒ  bureau ã‚’æ›´æ–°ï¼‰
bureau['CREDIT_DAY_OVERDUE_CAPPED_FLAG'] = (bureau['CREDIT_DAY_OVERDUE'] > cap_value).astype(int)

# â˜…è£œè¶³:
# ç•°å¸¸å€¤ä¿®æ­£å¾Œã�® 'CREDIT_DAY_OVERDUE_CAPPED' ã‚’ä½¿ã�£ã�Ÿç‰¹å¾´é‡�ã�Œå¿…è¦�ã�ªå ´å�ˆã�¯ã€�
# ã�“ã�®ä¿®æ­£å¾Œã�«ã€�å†�åº¦ã‚¹ãƒ†ãƒƒãƒ—1ã�®ã‚ˆã�†ã�ªé›†è¨ˆå‡¦ç�†ã‚’è¡Œã�†å¿…è¦�ã�Œã�‚ã‚Šã�¾ã�™ã€‚
# ç�¾çŠ¶ã�®ã‚³ãƒ¼ãƒ‰ã�§ã�¯ã€�ä¿®æ­£ã�¯å…ƒã�®DF(bureau)ã�«å¯¾ã�—ã�¦è¡Œã‚�ã‚Œã€�bureau_agg ã�«ã�¯è¿½åŠ ã�•ã‚Œã�¦ã�„ã�¾ã�›ã‚“ã€‚

print("ã‚»ãƒ«3ã�®å‡¦ç�†å®Œäº†: bureau_aggã�«å»¶æ»�ã�®ç‰¹å¾´é‡�ã�Œè¿½åŠ ã�•ã‚Œã�¾ã�—ã�Ÿã€‚bureauã�«ã�¯ç•°å¸¸å€¤ä¿®æ­£ã�Œé�©ç”¨ã�•ã‚Œã�¾ã�—ã�Ÿã€‚")


# ==========================================================
# â˜… å‰�å‡¦ç�†: DAYS_CREDIT_ENDDATE ã�®ç•°å¸¸å€¤ä¿®æ­£ï¼ˆä¸Šé™�/ä¸‹é™�ã‚«ãƒƒãƒˆï¼‰
# ==========================================================

# ãƒ­ãƒ¼ãƒ³çµ‚äº†æ—¥/äºˆå®šæ—¥ã�®ç•°å¸¸å€¤ã‚«ãƒƒãƒˆç¯„å›²ã‚’è¨­å®š
cap_low = -3600    # é��å�»10å¹´ï¼ˆ-365 * 10 ç¨‹åº¦ï¼‰ã‚ˆã‚Šå�¤ã�„ãƒ¬ã‚³ãƒ¼ãƒ‰ã�¯ã‚«ãƒƒãƒˆ
cap_high = 3600    # æœªæ�¥10å¹´ï¼ˆ365 * 10 ç¨‹åº¦ï¼‰ã‚ˆã‚Šå…ˆã�®äºˆå®šæ—¥ã�¯ã‚«ãƒƒãƒˆ

# ç•°å¸¸å€¤ã‚«ãƒƒãƒˆã‚’é�©ç”¨ã�—ã�Ÿæ–°ã�—ã�„ç‰¹å¾´é‡�ã‚’ä½œæˆ�
bureau['DAYS_CREDIT_ENDDATE_CAPPED'] = bureau['DAYS_CREDIT_ENDDATE'].clip(lower=cap_low, upper=cap_high)

# ç•°å¸¸å€¤ãƒ•ãƒ©ã‚°ã‚’ä½œæˆ�ï¼ˆä¿®æ­£ã�Œå¿…è¦�ã� ã�£ã�Ÿãƒ¬ã‚³ãƒ¼ãƒ‰ã�«1ã‚’ç«‹ã�¦ã‚‹ï¼‰
bureau['DAYS_CREDIT_ENDDATE_CAPPED_FLAG'] = (
    (bureau['DAYS_CREDIT_ENDDATE'] < cap_low) |  
    (bureau['DAYS_CREDIT_ENDDATE'] > cap_high)
).astype(int)

# â˜… è£œè¶³:
# ã�“ã�®ä¿®æ­£ã�¯ 'bureau' ãƒ‡ãƒ¼ã‚¿ãƒ•ãƒ¬ãƒ¼ãƒ ã�«é�©ç”¨ã�•ã‚Œã�¾ã�—ã�Ÿã€‚
# ä»¥é™�ã€�å¹³å�‡ã‚„åˆ†æ•£ã‚’è¨ˆç®—ã�™ã‚‹éš›ã�¯ã€�åŸºæœ¬çš„ã�«ã�“ã�® 'DAYS_CREDIT_ENDDATE_CAPPED' ã‚’ä½¿ç”¨ã�—ã�¾ã�™ã€‚


import pandas as pd
import numpy as np

def make_enddate_features_fast(bureau: pd.DataFrame) -> pd.DataFrame:
    if 'DAYS_CREDIT_ENDDATE_CAPPED' not in bureau.columns:
        print("Error: 'DAYS_CREDIT_ENDDATE_CAPPED' åˆ—ã�Œè¦‹ã�¤ã�‹ã‚Šã�¾ã�›ã‚“ã€‚")
        return pd.DataFrame({'SK_ID_CURR': bureau['SK_ID_CURR'].unique()})

    GRP_COL = 'DAYS_CREDIT_ENDDATE_CAPPED'
    
    # --- 1. ä¸‹æº–å‚™ï¼šãƒ™ã‚¯ãƒˆãƒ«æ¼”ç®—ã�§ãƒ•ãƒ©ã‚°ã‚„å€¤ã‚’å…ˆã�«è¨ˆç®—ã�—ã�¦ã�Šã�� ---
    # ã�“ã�†ã�™ã‚‹ã�“ã�¨ã�§ã€�å¾Œã�§ .sum() ã‚„ .mean() ã‚’å‘¼ã�¶ã� ã�‘ã�§æ¸ˆã‚€ã‚ˆã�†ã�«ã�ªã‚Šã�¾ã�™
    bureau['IS_CLOSED'] = (bureau[GRP_COL] < 0).astype(int)
    bureau['IS_ONGOING'] = (bureau[GRP_COL] > 0).astype(int)
    
    # è¿”æ¸ˆä¸­ã�®æ—¥æ•°ï¼ˆè² ã�®å€¤ã‚„0ã�¯NaNã�«ã�—ã�¦é›†è¨ˆå¯¾è±¡ã�‹ã‚‰å¤–ã�™ï¼‰
    bureau['DAYS_REMAINING'] = bureau[GRP_COL].where(bureau[GRP_COL] > 0)
    
    # å®Œæ¸ˆæ¸ˆã�¿ã�®æ—¥æ•°ï¼ˆçµ¶å¯¾å€¤ã‚’ã�¨ã‚‹ã€‚æ­£ã�®å€¤ã�¯NaNã�«ã�™ã‚‹ï¼‰
    bureau['DAYS_PAST_CLOSED'] = bureau[GRP_COL].where(bureau[GRP_COL] < 0).abs()
    
    # å¤–ã‚Œå€¤ãƒ•ãƒ©ã‚°
    bureau['IS_OUTLIER'] = (bureau['DAYS_CREDIT_ENDDATE'].abs() > 5000).astype(int)

    # --- 2. ä¸€æ°—ã�«é›†è¨ˆ ---
    # .apply() ã�¯ä½¿ã‚�ã�šã€�Pandasç´”æ­£ã�®é›†è¨ˆé–¢æ•°ã�®ã�¿ã‚’ä½¿ç”¨
    agg_ops = {
        'IS_CLOSED': ['max', 'mean', 'sum'],  # HAS_ANY_CLOSED, COMPLETED_RATIO, NUM_CLOSED
        'IS_ONGOING': ['max', 'mean', 'sum'], # HAS_ANY_ONGOING, ONGOING_RATIO, NUM_ONGOING
        'DAYS_REMAINING': ['mean', 'max', 'sum'],
        'DAYS_PAST_CLOSED': ['min', 'mean'],  # RECENT_CLOSED_DAYS, PAST_CLOSED_DAYS_MEAN
        GRP_COL: ['var'],                     # ENDDATE_VAR
        'IS_OUTLIER': ['max']                 # ENDDATE_OUTLIER
    }
    
    bureau_feat = bureau.groupby('SK_ID_CURR').agg(agg_ops)
    
    # --- 3. ã‚«ãƒ©ãƒ å��ã�®æ•´ç�† ---
    bureau_feat.columns = [
        'HAS_ANY_CLOSED', 'COMPLETED_RATIO', 'NUM_CLOSED',
        'HAS_ANY_ONGOING', 'ONGOING_RATIO', 'NUM_ONGOING',
        'REMAINING_DAYS_MEAN', 'REMAINING_DAYS_MAX', 'REMAINING_DAYS_SUM',
        'RECENT_CLOSED_DAYS', 'PAST_CLOSED_DAYS_MEAN',
        'ENDDATE_VAR',
        'ENDDATE_OUTLIER'
    ]
    
    # æ¬ æ��å€¤åŸ‹ã‚�ï¼ˆä»¶æ•°ç³»ã�ªã�©ã�¯NaNã‚ˆã‚Š0ã�®æ–¹ã�Œæ‰±ã�„ã‚„ã�™ã�„ï¼‰
    fill_zero_cols = ['REMAINING_DAYS_MEAN', 'REMAINING_DAYS_MAX', 'REMAINING_DAYS_SUM']
    bureau_feat[fill_zero_cols] = bureau_feat[fill_zero_cols].fillna(0)
    
    # å¾Œå‡¦ç�†ï¼šbureauã�«è¿½åŠ ã�—ã�Ÿä¸€æ™‚çš„ã�ªåˆ—ã‚’å‰Šé™¤ï¼ˆãƒ¡ãƒ¢ãƒªç¯€ç´„ï¼‰
    bureau.drop(columns=['IS_CLOSED', 'IS_ONGOING', 'DAYS_REMAINING', 'DAYS_PAST_CLOSED', 'IS_OUTLIER'], inplace=True)
    
    return bureau_feat.reset_index()


# ğŸ’» ã‚»ãƒ« 4: ENDDATE ç‰¹å¾´é‡�ã�®ä½œæˆ�ã�¨ bureau_agg ã�¸ã�®çµ�å�ˆ#æ™‚é–“ã�‹ã�‹ã‚‹

# 1. é–¢æ•°ã‚’å®Ÿè¡Œã�—ã�¦ç‰¹å¾´é‡�ãƒ‡ãƒ¼ã‚¿ãƒ•ãƒ¬ãƒ¼ãƒ ã‚’å�–å¾—
enddate_features = make_enddate_features_fast(bureau)

# 2. bureau_agg ã�«ãƒ�ãƒ¼ã‚¸ï¼ˆå¼•ã��ç¶™ã��ï¼‰
# ã‚»ãƒ«3ã�®å®Ÿè¡Œçµ�æ�œã�§ã�‚ã‚‹ bureau_agg ã�«ã€�æ–°ã�—ã�„ç‰¹å¾´é‡�ã‚’ãƒ�ãƒ¼ã‚¸ã�—ã�¦å¼•ã��ç¶™ã��ã�¾ã�™
bureau_agg = bureau_agg.merge(enddate_features, on='SK_ID_CURR', how='left')

# 3. ãƒ�ãƒ¼ã‚¸å¾Œã�®å¾Œå‡¦ç�† (å®Œæ¸ˆæ¸ˆã�¿ãƒ­ãƒ¼ãƒ³ã�Œã�ªã�„é¡§å®¢ã�® NaN ã‚’å‡¦ç�†)
# å®Œæ¸ˆæ¸ˆã�¿ãƒ­ãƒ¼ãƒ³ã�Œã�ªã�„å ´å�ˆã€�RECENT_CLOSED_DAYS/PAST_CLOSED_DAYS_MEAN ã�¯ NaN ã�«ã�ªã‚‹ã�Ÿã‚�ã€�0ã�§åŸ‹ã‚�ã‚‹
bureau_agg['RECENT_CLOSED_DAYS'] = bureau_agg['RECENT_CLOSED_DAYS'].fillna(0)
bureau_agg['PAST_CLOSED_DAYS_MEAN'] = bureau_agg['PAST_CLOSED_DAYS_MEAN'].fillna(0)

print("ã‚»ãƒ«4ã�®å‡¦ç�†å®Œäº†: bureau_aggã�«DAYS_CREDIT_ENDDATEã�«é–¢ã�™ã‚‹ç‰¹å¾´é‡�ã�Œè¿½åŠ ã�•ã‚Œã�¾ã�—ã�Ÿã€‚")


# ==========================================================
# â˜… STEP 5: æœ€å¤§å»¶æ»�é¡�ç‰¹å¾´é‡�ã�®ä½œæˆ�ã�¨ bureau_agg ã�¸ã�®çµ�å�ˆ
# ==========================================================

# 1. ç•°å¸¸å€¤ã�®ä¸Šé™�ã‚’è¨­å®š
OVERDUE_CUT = 1_000_000

# 2. ç•°å¸¸å€¤ãƒ•ãƒ©ã‚°ä½œæˆ�ï¼ˆå…ƒã�®ãƒ‡ãƒ¼ã‚¿ bureau ã�«å¯¾ã�™ã‚‹å‡¦ç�†ï¼‰
bureau['EXTREME_OVERDUE_FLAG'] = (bureau['AMT_CREDIT_MAX_OVERDUE'] > OVERDUE_CUT).astype(int)

# 3. ä¸Šé™�ã�§ã‚«ãƒƒãƒˆï¼ˆå…ƒã�®ãƒ‡ãƒ¼ã‚¿ bureau ã�«å¯¾ã�™ã‚‹å‡¦ç�†ï¼‰
bureau['AMT_CREDIT_MAX_OVERDUE_CAPPED'] = bureau['AMT_CREDIT_MAX_OVERDUE'].clip(upper=OVERDUE_CUT)

# 4. é¡§å®¢å�˜ä½�ã�§é›†è¨ˆ
bureau_overdue_agg = bureau.groupby('SK_ID_CURR').agg(
    MAX_OVERDUE=('AMT_CREDIT_MAX_OVERDUE_CAPPED', 'max'),
    MEAN_OVERDUE=('AMT_CREDIT_MAX_OVERDUE_CAPPED', 'mean'),
    NUM_OVERDUE=('AMT_CREDIT_MAX_OVERDUE_CAPPED', lambda x: (x > 0).sum()),
    OVERDUE_RATIO=('AMT_CREDIT_MAX_OVERDUE_CAPPED', lambda x: (x > 0).sum() / len(x)),
).reset_index()


# 5. ç•°å¸¸å€¤ãƒ•ãƒ©ã‚°ï¼ˆEXTREME_OVERDUE_FLAGï¼‰ã‚’é¡§å®¢ãƒ¬ãƒ™ãƒ«ã�«é›†ç´„ã�—ã€�ãƒ�ãƒ¼ã‚¸
# â€» é¡§å®¢å�˜ä½�ã�§é›†ç´„ã�™ã‚‹å¿…è¦�ã�Œã�‚ã‚‹ã�Ÿã‚�ã€�maxã‚„sumã�ªã�©ã�§é›†ç´„ã�™ã‚‹ã�®ã�Œä¸€èˆ¬çš„ã�§ã�™ã€‚
# ä»Šå›�ã�¯ drop_duplicates() ã‚’ä½¿ã�£ã�¦ã�„ã�¾ã�™ã�Œã€�é¡§å®¢ã�”ã�¨ã�«EXTREME_OVERDUE_FLAGã�®æœ€å¤§å€¤(max)ã‚’ãƒ�ãƒ¼ã‚¸ã�™ã‚‹ã�®ã�Œã‚ˆã‚Šé�©åˆ‡ã�§ã�™ã€‚
extreme_flag_agg = bureau.groupby('SK_ID_CURR')['EXTREME_OVERDUE_FLAG'].max().reset_index(name='HAS_EXTREME_OVERDUE_FLAG')
bureau_overdue_agg = bureau_overdue_agg.merge(
    extreme_flag_agg,
    on='SK_ID_CURR',
    how='left'
)

# 6. bureau_agg ã�¸çµ�å�ˆï¼ˆå¼•ã��ç¶™ã��ï¼‰
# ã‚»ãƒ«4ã�®å®Ÿè¡Œçµ�æ�œã�§ã�‚ã‚‹ bureau_agg ã�«ã€�æ–°ã�—ã�„ç‰¹å¾´é‡�ã‚’ãƒ�ãƒ¼ã‚¸ã�—ã�¦å¼•ã��ç¶™ã��ã�¾ã�™
bureau_agg = bureau_agg.merge(bureau_overdue_agg, on='SK_ID_CURR', how='left')




# ã‚¿ãƒ¼ã‚²ãƒƒãƒˆã�¨ã�ªã‚‹åˆ—å��ã‚’ãƒªã‚¹ãƒˆã‚¢ãƒƒãƒ—
# NUM_OVERDUEã�¯KeyErrorã�«ã�¯å‡ºã�¦ã�„ã�¾ã�›ã‚“ã�§ã�—ã�Ÿã�Œã€�å�ˆã‚�ã�›ã�¦å‡¦ç�†ã�—ã�¾ã�™ã€‚
target_cols = ['MAX_OVERDUE', 'MEAN_OVERDUE', 'NUM_OVERDUE', 'OVERDUE_RATIO', 'HAS_EXTREME_OVERDUE_FLAG']

# bureau_agg ã�«å®Ÿéš›ã�«å­˜åœ¨ã�™ã‚‹åˆ—ã�®ã�¿ã‚’ãƒ•ã‚£ãƒ«ã‚¿ãƒªãƒ³ã‚°ã�—ã�¦æŠ½å‡º
cols_to_fill = [col for col in target_cols if col in bureau_agg.columns]

# ãƒ•ã‚£ãƒ«ã‚¿ãƒªãƒ³ã‚°ã�•ã‚Œã�Ÿåˆ—ã�«å¯¾ã�—ã�¦ã�®ã�¿ fillna(0) ã‚’å®Ÿè¡Œã�™ã‚‹ã�“ã�¨ã�§ KeyError ã‚’å›�é�¿
bureau_agg[cols_to_fill] = bureau_agg[cols_to_fill].fillna(0)

# HAS_EXTREME_OVERDUE_FLAG ã�¯ filtered_cols ã�«å�«ã�¾ã‚Œã‚‹ã�Ÿã‚�ã€�ã�“ã‚Œã�§å‡¦ç�†ã�•ã‚Œã�¾ã�™ã€‚

print("ã‚¹ãƒ†ãƒƒãƒ—5ã�®å‡¦ç�†å®Œäº†: bureau_aggã�«æœ€å¤§å»¶æ»�é¡�ã�«é–¢ã�™ã‚‹ç‰¹å¾´é‡�ã�Œè¿½åŠ ã�•ã‚Œã�¾ã�—ã�Ÿã€‚")


# ==========================================================
# â˜… STEP 6: ãƒ­ãƒ¼ãƒ³å»¶é•·ç‰¹å¾´é‡�ã�®ä½œæˆ�ã�¨ bureau_agg ã�¸ã�®çµ�å�ˆ
# ==========================================================

# 1. ãƒ­ãƒ¼ãƒ³ãƒ¬ã‚³ãƒ¼ãƒ‰ãƒ¬ãƒ™ãƒ«ã�§ã�®å»¶æ»�çµŒé¨“ãƒ•ãƒ©ã‚°ã‚’ä½œæˆ�ï¼ˆâ€»ã�“ã‚Œã�¯å¾Œã�®é›†è¨ˆã�§ã�¯ä½¿ã�£ã�¦ã�„ã�¾ã�›ã‚“ï¼‰
bureau['HAS_PROLONG_FLAG'] = (bureau['CNT_CREDIT_PROLONG'] > 0).astype(int)

# 2. é¡§å®¢å�˜ä½�ã�§é›†è¨ˆ
bureau_prolong_agg = bureau.groupby('SK_ID_CURR').agg(
    MAX_PROLONG=('CNT_CREDIT_PROLONG', 'max'),
    SUM_PROLONG=('CNT_CREDIT_PROLONG', 'sum'),
    HAS_PROLONG=('CNT_CREDIT_PROLONG', lambda x: int((x > 0).any()))
).reset_index()

# 3. bureau_agg ã�¸çµ�å�ˆï¼ˆå¼•ã��ç¶™ã��ï¼‰
# ã‚»ãƒ«5ã�¾ã�§ã�®å®Ÿè¡Œçµ�æ�œã�§ã�‚ã‚‹ bureau_agg ã�«ã€�æ–°ã�—ã�„ç‰¹å¾´é‡�ã‚’ãƒ�ãƒ¼ã‚¸ã�—ã�¦å¼•ã��ç¶™ã��ã�¾ã�™
bureau_agg = bureau_agg.merge(bureau_prolong_agg, on='SK_ID_CURR', how='left')

# 4. å¾Œå‡¦ç�†ï¼ˆå»¶é•·çµŒé¨“ã�Œã�ªã�„é¡§å®¢ã�® NaN ã‚’ 0 ã�§åŸ‹ã‚�ã‚‹ï¼‰
# ã‚‚ã�— bureau_agg ã�«å­˜åœ¨ã�™ã‚‹ã�Œ bureau ã�«å¯¾å¿œãƒ¬ã‚³ãƒ¼ãƒ‰ã�Œã�ªã�„é¡§å®¢ã�®å ´å�ˆã€�NaNã�Œç™ºç”Ÿã�™ã‚‹ã�Ÿã‚�ã€�0ã�§åŸ‹ã‚�ã�¾ã�™ã€‚
bureau_agg[['MAX_PROLONG', 'SUM_PROLONG', 'HAS_PROLONG']] = \
    bureau_agg[['MAX_PROLONG', 'SUM_PROLONG', 'HAS_PROLONG']].fillna(0)


print("ã‚»ãƒ«6ã�®å‡¦ç�†å®Œäº†: bureau_aggã�«ãƒ­ãƒ¼ãƒ³å»¶é•·ã�«é–¢ã�™ã‚‹ç‰¹å¾´é‡�ã�Œè¿½åŠ ã�•ã‚Œã�¾ã�—ã�Ÿã€‚")


# ==========================================================
# â˜… STEP 7: å€Ÿå…¥ç·�é¡�/æ� ç‰¹å¾´é‡�ã�®ä½œæˆ�ã�¨ bureau_agg ã�¸ã�®çµ�å�ˆ
# ==========================================================

# 1. ç•°å¸¸å€¤ã�®ä¸Šé™�ã‚’è¨­å®šã�¨ä¿®æ­£
CREDIT_SUM_CUT = 5_000_000

# ç•°å¸¸å€¤ãƒ•ãƒ©ã‚°ä½œæˆ�ï¼ˆå…ƒã�®ãƒ‡ãƒ¼ã‚¿ bureau ã�«å¯¾ã�™ã‚‹å‡¦ç�†ï¼‰
bureau['EXTREME_CREDIT_FLAG'] = (bureau['AMT_CREDIT_SUM'] > CREDIT_SUM_CUT).astype(int)

# ä¸Šé™�ã�§ã‚«ãƒƒãƒˆï¼ˆå…ƒã�®ãƒ‡ãƒ¼ã‚¿ bureau ã�«å¯¾ã�™ã‚‹å‡¦ç�†ï¼‰
bureau['AMT_CREDIT_SUM_CAPPED'] = bureau['AMT_CREDIT_SUM'].clip(upper=CREDIT_SUM_CUT)


# 2. é¡§å®¢å�˜ä½�ã�§é›†è¨ˆ
bureau_credit_agg = bureau.groupby('SK_ID_CURR').agg(
    MAX_CREDIT=('AMT_CREDIT_SUM_CAPPED', 'max'),
    MIN_CREDIT=('AMT_CREDIT_SUM_CAPPED', 'min'),
    SUM_CREDIT=('AMT_CREDIT_SUM_CAPPED', 'sum'),
    MEAN_CREDIT=('AMT_CREDIT_SUM_CAPPED', 'mean')
).reset_index()


# 3. æ´¾ç”Ÿç‰¹å¾´é‡�ï¼ˆãƒ©ãƒ³ã‚¯ã�¨é–¾å€¤ãƒ•ãƒ©ã‚°ï¼‰ã�®ä½œæˆ�

# SUM_CREDIT ã‚’åŸºæº–ã�«4æ®µéš�ãƒ©ãƒ³ã‚¯
# qcutã�§NaNã‚¨ãƒ©ãƒ¼ã‚’é�¿ã�‘ã‚‹ã�Ÿã‚�ã€�é›†è¨ˆçµ�æ�œã�«NaNã�Œã�ªã�„ã�‹ç¢ºèª�ã�™ã‚‹ã�‹ã€�qcutã�®å¼•æ•°ã‚’èª¿æ•´ã�—ã�¦ã��ã� ã�•ã�„ã€‚
# (ä¾‹: pd.qcut(..., duplicates='drop'))
bureau_credit_agg['SUM_CREDIT_RANK'] = pd.qcut(
    bureau_credit_agg['SUM_CREDIT'], 
    q=4, 
    labels=['low', 'medium', 'high', 'extreme'],
    duplicates='drop' # é‡�è¤‡ã�—ã�Ÿå¢ƒç•Œå€¤ã�Œã�‚ã‚‹å ´å�ˆã�«å¯¾å¿œ
)

# MAX_CREDIT ã‚’åŸºæº–ã�«é–¾å€¤ãƒ•ãƒ©ã‚°
bureau_credit_agg['MAX_CREDIT_MEDIUM_FLAG'] = (
    (bureau_credit_agg['MAX_CREDIT'] > 500_000) &
    (bureau_credit_agg['MAX_CREDIT'] <= 1_000_000)
).astype(int)
bureau_credit_agg['MAX_CREDIT_HIGH_FLAG'] = (bureau_credit_agg['MAX_CREDIT'] > 1_000_000).astype(int)


# 4. ç•°å¸¸å€¤ãƒ•ãƒ©ã‚°ã�®é›†ç´„ã�¨ãƒ�ãƒ¼ã‚¸
# é¡§å®¢å�˜ä½�ã�§ã€�ç•°å¸¸å€¤ãƒ¬ã‚³ãƒ¼ãƒ‰ã‚’æŒ�ã�¤ã�‹å�¦ã�‹ (max) ã‚’é›†ç´„
extreme_credit_flag_agg = bureau.groupby('SK_ID_CURR')['EXTREME_CREDIT_FLAG'].max().reset_index(name='HAS_EXTREME_CREDIT_FLAG')
bureau_credit_agg = bureau_credit_agg.merge(
    extreme_credit_flag_agg,
    on='SK_ID_CURR',
    how='left'
)

# 5. bureau_agg ã�¸çµ�å�ˆï¼ˆå¼•ã��ç¶™ã��ï¼‰
# ã‚»ãƒ«6ã�¾ã�§ã�®å®Ÿè¡Œçµ�æ�œã�§ã�‚ã‚‹ bureau_agg ã�«ã€�æ–°ã�—ã�„ç‰¹å¾´é‡�ã‚’ãƒ�ãƒ¼ã‚¸ã�—ã�¦å¼•ã��ç¶™ã��ã�¾ã�™
bureau_agg = bureau_agg.merge(bureau_credit_agg, on='SK_ID_CURR', how='left')

# 6. å¾Œå‡¦ç�†
# æ¬ æ��å€¤ï¼ˆNaNï¼‰ã‚’0ã�§åŸ‹ã‚�ã‚‹ï¼ˆå¯¾å¿œã�™ã‚‹ãƒ­ãƒ¼ãƒ³ã�Œã�ªã�„é¡§å®¢ã‚’æƒ³å®šï¼‰
bureau_agg[['MAX_CREDIT', 'MIN_CREDIT', 'SUM_CREDIT', 'MEAN_CREDIT', 'HAS_EXTREME_CREDIT_FLAG']] = \
    bureau_agg[['MAX_CREDIT', 'MIN_CREDIT', 'SUM_CREDIT', 'MEAN_CREDIT', 'HAS_EXTREME_CREDIT_FLAG']].fillna(0)

# ãƒ©ãƒ³ã‚¯åˆ—ã�¯ã‚«ãƒ†ã‚´ãƒªå�‹ã�®ã�Ÿã‚�ã€�NaNã�¯ 'missing' ã�ªã�©ã�§åŸ‹ã‚�ã‚‹ã�®ã�Œæœ›ã�¾ã�—ã�„
if 'SUM_CREDIT_RANK' in bureau_agg.columns:
    bureau_agg['SUM_CREDIT_RANK'] = bureau_agg['SUM_CREDIT_RANK'].cat.add_categories('missing').fillna('missing')


print("ã‚»ãƒ«7ã�®å‡¦ç�†å®Œäº†: bureau_aggã�«å€Ÿå…¥ç·�é¡�/æ� ã�«é–¢ã�™ã‚‹ç‰¹å¾´é‡�ã�Œè¿½åŠ ã�•ã‚Œã�¾ã�—ã�Ÿã€‚")


# ==========================================================
# â˜… STEP 8: å‚µå‹™æ®‹é«˜ç‰¹å¾´é‡�ã�®ä½œæˆ�ã�¨ bureau_agg ã�¸ã�®çµ�å�ˆ
# ==========================================================

# 1. ç•°å¸¸å€¤ã�®ä¸Šé™�ã‚’è¨­å®šã�¨ä¿®æ­£
MAX_DEBT_CUT = 100_000_000

# ä¸Šé™�ãƒ»ä¸‹é™�ã�§ã‚«ãƒƒãƒˆï¼ˆå…ƒã�®ãƒ‡ãƒ¼ã‚¿ bureau ã�«å¯¾ã�™ã‚‹å‡¦ç�†ï¼‰
# è² å‚µæ®‹é«˜ã�¯è² ã�«ã�ªã‚‹ã�“ã�¨ã�¯åŸºæœ¬çš„ã�«ã�ªã�„ã�Ÿã‚�ã€�lower=0ã�§ã‚¯ãƒªãƒƒãƒ”ãƒ³ã‚°
bureau['AMT_CREDIT_SUM_DEBT_CAPPED'] = bureau['AMT_CREDIT_SUM_DEBT'].clip(lower=0, upper=MAX_DEBT_CUT)


# 2. é¡§å®¢å�˜ä½�ã�§é›†è¨ˆ
bureau_debt_agg = bureau.groupby('SK_ID_CURR').agg(
    MAX_DEBT=('AMT_CREDIT_SUM_DEBT_CAPPED', 'max'),
    MIN_DEBT=('AMT_CREDIT_SUM_DEBT_CAPPED', 'min'),
    SUM_DEBT=('AMT_CREDIT_SUM_DEBT_CAPPED', 'sum'),
    MEAN_DEBT=('AMT_CREDIT_SUM_DEBT_CAPPED', 'mean')
).reset_index()


# 3. bureau_agg ã�¸çµ�å�ˆï¼ˆå¼•ã��ç¶™ã��ï¼‰
# ã‚»ãƒ«7ã�¾ã�§ã�®å®Ÿè¡Œçµ�æ�œã�§ã�‚ã‚‹ bureau_agg ã�«ã€�æ–°ã�—ã�„ç‰¹å¾´é‡�ã‚’ãƒ�ãƒ¼ã‚¸ã�—ã�¦å¼•ã��ç¶™ã��ã�¾ã�™
bureau_agg = bureau_agg.merge(bureau_debt_agg, on='SK_ID_CURR', how='left')

# 4. å¾Œå‡¦ç�†
# æ¬ æ��å€¤ï¼ˆNaNï¼‰ã‚’0ã�§åŸ‹ã‚�ã‚‹ï¼ˆå¯¾å¿œã�™ã‚‹ãƒ­ãƒ¼ãƒ³ã�Œã�ªã�„é¡§å®¢ã‚’æƒ³å®šã€‚ç‰¹ã�«å‚µå‹™æ®‹é«˜ã�¯0ã�Œè‡ªç„¶ï¼‰
bureau_agg[['MAX_DEBT', 'MIN_DEBT', 'SUM_DEBT', 'MEAN_DEBT']] = \
    bureau_agg[['MAX_DEBT', 'MIN_DEBT', 'SUM_DEBT', 'MEAN_DEBT']].fillna(0)


print("ã‚»ãƒ«8ã�®å‡¦ç�†å®Œäº†: bureau_aggã�«å‚µå‹™æ®‹é«˜ã�«é–¢ã�™ã‚‹ç‰¹å¾´é‡�ã�Œè¿½åŠ ã�•ã‚Œã�¾ã�—ã�Ÿã€‚")


# ==========================================================
# â˜… STEP 9: æœ€æ–°ã�®é™�åº¦é¡�ç‰¹å¾´é‡�ã�®ä½œæˆ�ã�¨ bureau_agg ã�¸ã�®çµ�å�ˆ
# ==========================================================

# 1. AMT_CREDIT_SUM_LIMIT ã�®å‰�å‡¦ç�†ï¼ˆé›†è¨ˆã�«ã�¯ä½¿ç”¨ã�—ã�ªã�„ã�Œã€�æœ€æ–°ãƒ¬ã‚³ãƒ¼ãƒ‰æŠ½å‡ºæ™‚ã�«ä½¿ç”¨ï¼‰
MAX_CREDIT_LIMIT_CUT = 4_500_000

# ä¸Šé™�ãƒ»ä¸‹é™�ã�§ã‚«ãƒƒãƒˆï¼ˆå…ƒã�®ãƒ‡ãƒ¼ã‚¿ bureau ã�«å¯¾ã�™ã‚‹å‡¦ç�†ï¼‰
bureau['AMT_CREDIT_SUM_LIMIT_CAPPED'] = bureau['AMT_CREDIT_SUM_LIMIT'].clip(lower=0, upper=MAX_CREDIT_LIMIT_CUT)


# 2. é¡§å®¢å�˜ä½�ã�§æœ€æ–°å±¥æ­´ã‚’å�–å¾—
# DAYS_CREDIT_UPDATEï¼ˆæ›´æ–°æ—¥ï¼‰ã�®çµ¶å¯¾å€¤ã�Œå°�ã�•ã�„ï¼ˆè² ã�®å€¤ã�Œ0ã�«è¿‘ã�„ï¼‰ã�»ã�©æ–°ã�—ã�„
bureau_latest = bureau.sort_values(
    ['SK_ID_CURR', 'DAYS_CREDIT_UPDATE'], 
    ascending=[True, False] # SK_ID_CURRã�§æ˜‡é †ã€�DAYS_CREDIT_UPDATEï¼ˆæœ€æ–°ï¼�æœ€å¤§è² æ•°ï¼‰ã�§é™�é †
)

# é¡§å®¢ã�”ã�¨ã�«æœ€æ–°ã�®ãƒ¬ã‚³ãƒ¼ãƒ‰ï¼ˆå…ˆé ­è¡Œï¼‰ã�®ã�¿ã‚’æŠ½å‡º
bureau_latest = bureau_latest.groupby('SK_ID_CURR').first().reset_index()


# 3. æœ€æ–°ã�®é™�åº¦é¡�ã‚’ç‰¹å¾´é‡�ã�¨ã�—ã�¦æ�¡ç”¨
# æœ€æ–°ã�®é™�åº¦é¡�ã‚’ã��ã�®ã�¾ã�¾ç‰¹å¾´é‡�ã�«ï¼ˆ0æœªæº€ã�®å€¤ã‚’0ã�«ã‚¯ãƒªãƒƒãƒ”ãƒ³ã‚°ï¼‰
bureau_latest['CURRENT_CREDIT_LIMIT'] = bureau_latest['AMT_CREDIT_SUM_LIMIT'].clip(lower=0)

# æœ€æ–°ã�®é™�åº¦é¡�ã�Œå­˜åœ¨ã�™ã‚‹ã�‹ã�©ã�†ã�‹ã�®ãƒ•ãƒ©ã‚°
bureau_latest['HAS_CREDIT_LIMIT'] = (bureau_latest['CURRENT_CREDIT_LIMIT'] > 0).astype(int)

# ãƒ�ãƒ¼ã‚¸ã�«å¿…è¦�ã�ªåˆ—ã�®ã�¿ã�«çµ�ã‚‹
latest_features = bureau_latest[['SK_ID_CURR', 'CURRENT_CREDIT_LIMIT', 'HAS_CREDIT_LIMIT']]


# 4. bureau_agg ã�¸çµ�å�ˆï¼ˆå¼•ã��ç¶™ã��ï¼‰
# ã‚»ãƒ«8ã�¾ã�§ã�®å®Ÿè¡Œçµ�æ�œã�§ã�‚ã‚‹ bureau_agg ã�«ã€�æ–°ã�—ã�„ç‰¹å¾´é‡�ã‚’ãƒ�ãƒ¼ã‚¸ã�—ã�¦å¼•ã��ç¶™ã��ã�¾ã�™
bureau_agg = bureau_agg.merge(latest_features, on='SK_ID_CURR', how='left')

# 5. å¾Œå‡¦ç�†
# è©²å½“ã�™ã‚‹ãƒ­ãƒ¼ãƒ³ãƒ¬ã‚³ãƒ¼ãƒ‰ã�Œã�ªã�„é¡§å®¢ã�®æ¬ æ��å€¤ï¼ˆNaNï¼‰ã‚’0ã�§åŸ‹ã‚�ã‚‹
bureau_agg['CURRENT_CREDIT_LIMIT'] = bureau_agg['CURRENT_CREDIT_LIMIT'].fillna(0)
bureau_agg['HAS_CREDIT_LIMIT'] = bureau_agg['HAS_CREDIT_LIMIT'].fillna(0)


print("ã‚»ãƒ«9ã�®å‡¦ç�†å®Œäº†: bureau_aggã�«æœ€æ–°ã�®é™�åº¦é¡�ã�«é–¢ã�™ã‚‹ç‰¹å¾´é‡�ã�Œè¿½åŠ ã�•ã‚Œã�¾ã�—ã�Ÿã€‚")


# ==========================================================
# â˜… STEP 10: æœ€æ–°ã�®å»¶æ»�é‡‘é¡�ç‰¹å¾´é‡�ã�®ä½œæˆ�ã�¨ bureau_agg ã�¸ã�®çµ�å�ˆ
# ==========================================================

# 1. é¡§å®¢å�˜ä½�ã�§æœ€æ–°å±¥æ­´ã‚’å�–å¾—
# DAYS_CREDIT_UPDATEï¼ˆæ›´æ–°æ—¥ï¼‰ã�®çµ¶å¯¾å€¤ã�Œå°�ã�•ã�„ï¼ˆè² ã�®å€¤ã�Œ0ã�«è¿‘ã�„ï¼‰ã�»ã�©æ–°ã�—ã�„
bureau_latest = bureau.sort_values(
    ['SK_ID_CURR', 'DAYS_CREDIT_UPDATE'], 
    ascending=[True, False] # SK_ID_CURRã�§æ˜‡é †ã€�DAYS_CREDIT_UPDATEï¼ˆæœ€æ–°ï¼�æœ€å¤§è² æ•°ï¼‰ã�§é™�é †
)

# é¡§å®¢ã�”ã�¨ã�«æœ€æ–°ã�®ãƒ¬ã‚³ãƒ¼ãƒ‰ï¼ˆå…ˆé ­è¡Œï¼‰ã�®ã�¿ã‚’æŠ½å‡º
bureau_latest = bureau_latest.groupby('SK_ID_CURR').first().reset_index()


# 2. æœ€æ–°ã�®å»¶æ»�é‡‘é¡�ã‚’ç‰¹å¾´é‡�ã�¨ã�—ã�¦æ�¡ç”¨
# æœ€æ–°ã�®å»¶æ»�é‡‘é¡�ã‚’ã��ã�®ã�¾ã�¾ç‰¹å¾´é‡�ã�«ï¼ˆ0æœªæº€ã�®å€¤ã‚’0ã�«ã‚¯ãƒªãƒƒãƒ”ãƒ³ã‚°ï¼‰
bureau_latest['CURRENT_CREDIT_OVERDUE'] = bureau_latest['AMT_CREDIT_SUM_OVERDUE'].clip(lower=0)

# æœ€æ–°ã�®å»¶æ»�ã�Œã�‚ã‚‹ã�‹ã�©ã�†ã�‹ã�®ãƒ•ãƒ©ã‚°
bureau_latest['HAS_OVERDUE_LATEST'] = (bureau_latest['CURRENT_CREDIT_OVERDUE'] > 0).astype(int)

# ãƒ�ãƒ¼ã‚¸ã�«å¿…è¦�ã�ªåˆ—ã�®ã�¿ã�«çµ�ã‚‹
latest_overdue_features = bureau_latest[['SK_ID_CURR', 'CURRENT_CREDIT_OVERDUE', 'HAS_OVERDUE_LATEST']]


# 3. bureau_agg ã�¸çµ�å�ˆï¼ˆå¼•ã��ç¶™ã��ï¼‰
# ã‚»ãƒ«9ã�¾ã�§ã�®å®Ÿè¡Œçµ�æ�œã�§ã�‚ã‚‹ bureau_agg ã�«ã€�æ–°ã�—ã�„ç‰¹å¾´é‡�ã‚’ãƒ�ãƒ¼ã‚¸ã�—ã�¦å¼•ã��ç¶™ã��ã�¾ã�™
bureau_agg = bureau_agg.merge(latest_overdue_features, on='SK_ID_CURR', how='left')

# 4. å¾Œå‡¦ç�†
# è©²å½“ã�™ã‚‹ãƒ­ãƒ¼ãƒ³ãƒ¬ã‚³ãƒ¼ãƒ‰ã�Œã�ªã�„é¡§å®¢ã�®æ¬ æ��å€¤ï¼ˆNaNï¼‰ã‚’0ã�§åŸ‹ã‚�ã‚‹
bureau_agg['CURRENT_CREDIT_OVERDUE'] = bureau_agg['CURRENT_CREDIT_OVERDUE'].fillna(0)
bureau_agg['HAS_OVERDUE_LATEST'] = bureau_agg['HAS_OVERDUE_LATEST'].fillna(0)


print("ã‚»ãƒ«10ã�®å‡¦ç�†å®Œäº†: bureau_aggã�«æœ€æ–°ã�®å»¶æ»�é‡‘é¡�ã�«é–¢ã�™ã‚‹ç‰¹å¾´é‡�ã�Œè¿½åŠ ã�•ã‚Œã�¾ã�—ã�Ÿã€‚")


# ==========================================================
# â˜… STEP 11: ãƒ­ãƒ¼ãƒ³ã‚¿ã‚¤ãƒ—åˆ¥ä»¶æ•°ç‰¹å¾´é‡�ã�®ä½œæˆ�ã�¨ bureau_agg ã�¸ã�®çµ�å�ˆ
# ==========================================================

# 1. ãƒ€ãƒŸãƒ¼ä½œæˆ�
# å…ƒã�®ãƒ‡ãƒ¼ã‚¿ bureau ã�® CREDIT_TYPE åˆ—ã‚’ One-Hot Encoding
bureau_dummies = pd.get_dummies(bureau['CREDIT_TYPE'], prefix='CREDIT_TYPE')

# 2. å…ƒãƒ‡ãƒ¼ã‚¿ã�«çµ�å�ˆï¼ˆé›†è¨ˆã�®ã�Ÿã‚�ã�«ä¸€æ™‚çš„ã�«è¿½åŠ ï¼‰
bureau = pd.concat([bureau, bureau_dummies], axis=1)

# 3. é¡§å®¢å�˜ä½�ã�§å�ˆè¨ˆï¼ˆå¥‘ç´„ä»¶æ•°ï¼‰ã‚’é›†è¨ˆ
# å�„ãƒ­ãƒ¼ãƒ³ã‚¿ã‚¤ãƒ—ã�®ãƒ€ãƒŸãƒ¼å¤‰æ•°ã�®å�ˆè¨ˆã‚’é¡§å®¢IDå�˜ä½�ã�§è¨ˆç®—
credit_type_count = bureau.groupby('SK_ID_CURR').agg(
    {col: 'sum' for col in bureau_dummies.columns}
).reset_index()

# 4. bureau_agg ã�¸çµ�å�ˆï¼ˆå¼•ã��ç¶™ã��ï¼‰
# ã‚»ãƒ«10ã�¾ã�§ã�®å®Ÿè¡Œçµ�æ�œã�§ã�‚ã‚‹ bureau_agg ã�«ã€�æ–°ã�—ã�„ç‰¹å¾´é‡�ã‚’ãƒ�ãƒ¼ã‚¸ã�—ã�¦å¼•ã��ç¶™ã��ã�¾ã�™
bureau_agg = bureau_agg.merge(credit_type_count, on='SK_ID_CURR', how='left')

# 5. å¾Œå‡¦ç�†
# bureau_aggã�«å­˜åœ¨ã�™ã‚‹ã�Œ bureau ã�«å¯¾å¿œãƒ¬ã‚³ãƒ¼ãƒ‰ã�Œã�ªã�„é¡§å®¢ã�¯ NaN ã�«ã�ªã‚‹ã�Ÿã‚�ã€�
# ãƒ­ãƒ¼ãƒ³ä»¶æ•°ã�Œ 0 ã�®é¡§å®¢ã�¨ã�—ã�¦æ‰±ã�†ã�Ÿã‚� 0 ã�§åŸ‹ã‚�ã‚‹
bureau_agg[bureau_dummies.columns] = bureau_agg[bureau_dummies.columns].fillna(0)


print("ã‚»ãƒ«11ã�®å‡¦ç�†å®Œäº†: bureau_aggã�«å…¨ãƒ­ãƒ¼ãƒ³ã‚¿ã‚¤ãƒ—åˆ¥ä»¶æ•°ç‰¹å¾´é‡�ã�Œè¿½åŠ ã�•ã‚Œã�¾ã�—ã�Ÿã€‚")


# ==========================================================
# â˜… STEP 12: æœ€çµ‚æ›´æ–°æ—¥ãƒ•ãƒ©ã‚°ã�®ä½œæˆ�ã�¨ bureau_agg ã�¸ã�®çµ�å�ˆ
# ==========================================================

# 1. æœ€æ–°æƒ…å ±æ›´æ–°ãƒ•ãƒ©ã‚°ã‚’ä½œæˆ�
# DAYS_CREDIT_UPDATE > -180 ã�¯ã€�æœ€çµ‚æ›´æ–°æ—¥ã�Œã€Œç�¾åœ¨ã�‹ã‚‰180æ—¥ä»¥å†…ã�§ã�‚ã‚‹ã€�ã�“ã�¨ã‚’æ„�å‘³ã�—ã�¾ã�™ã€‚
bureau_latest['IS_RECENT'] = (bureau_latest['DAYS_CREDIT_UPDATE'] > -180).astype(int)

# 2. ãƒ�ãƒ¼ã‚¸ã�«å¿…è¦�ã�ªåˆ—ã�®ã�¿ã�«çµ�ã‚‹
# latest_features_recent ã�« SK_ID_CURR ã�¨ IS_RECENT ã�®ã�¿å�«ã‚�ã‚‹
latest_features_recent = bureau_latest[['SK_ID_CURR', 'IS_RECENT']]

# 3. bureau_agg ã�¸çµ�å�ˆï¼ˆå¼•ã��ç¶™ã��ï¼‰
# ã‚»ãƒ«11ã�¾ã�§ã�®å®Ÿè¡Œçµ�æ�œã�§ã�‚ã‚‹ bureau_agg ã�«ã€�æ–°ã�—ã�„ç‰¹å¾´é‡�ã‚’ãƒ�ãƒ¼ã‚¸ã�—ã�¦å¼•ã��ç¶™ã��ã�¾ã�™
bureau_agg = bureau_agg.merge(latest_features_recent, on='SK_ID_CURR', how='left')

# 4. å¾Œå‡¦ç�†
# è©²å½“ã�™ã‚‹ãƒ­ãƒ¼ãƒ³ãƒ¬ã‚³ãƒ¼ãƒ‰ã�Œã�ªã�„é¡§å®¢ã�®æ¬ æ��å€¤ï¼ˆNaNï¼‰ã‚’0ã�§åŸ‹ã‚�ã‚‹
bureau_agg['IS_RECENT'] = bureau_agg['IS_RECENT'].fillna(0)


print("ã‚»ãƒ«12ã�®å‡¦ç�†å®Œäº†: bureau_aggã�«æœ€æ–°æƒ…å ±æ›´æ–°ãƒ•ãƒ©ã‚°ã�Œè¿½åŠ ã�•ã‚Œã�¾ã�—ã�Ÿã€‚")
print("ã�“ã‚Œã�§ã€�å…¨ã�¦ã�®ãƒ“ãƒ¥ãƒ¼ãƒ­ãƒ¼ç‰¹å¾´é‡�ä½œæˆ�ã�Œå®Œäº†ã�—ã�¾ã�—ã�Ÿã€‚ã�Šç–²ã‚Œæ§˜ã�§ã�—ã�Ÿï¼�âœ¨")


#ãƒ¡ãƒ¢ãƒªåœ§ç¸®ï¼ˆä¿�ç•™ï¼‰
#bureau_agg = reduce_mem_usage(bureau_agg)


train_df = train_df.merge(bureau_agg, on='SK_ID_CURR', how='left')
test_df = test_df.merge(bureau_agg, on='SK_ID_CURR', how='left')



#bureau_aggã�§ã�®äºˆæ¸¬
#step1 äºˆæ¸¬ç”¨ãƒ‡ãƒ¼ã‚¿ã‚»ãƒƒãƒˆã�®ä½œæˆ�
# 1. ãƒ¡ã‚¤ãƒ³ã�®å­¦ç¿’ãƒ‡ãƒ¼ã‚¿ã�‹ã‚‰SK_ID_CURRã�¨Targetåˆ—ã�®ã�¿ã‚’æŠ½å‡º
train_target = train_df[['SK_ID_CURR', 'TARGET']]

# 2. POS_CASHãƒ‡ãƒ¼ã‚¿ã�¨Targetåˆ—ã‚’çµ�å�ˆ
# 'how="left"' ã‚’ä½¿ã�†ã�“ã�¨ã�§ã€�POS_CASHã�®ã�™ã�¹ã�¦ã�®å±¥æ­´ã�«Targetã‚’ç´�ã�¥ã�‘ã‚‹
bureau_agg_with_target = pd.merge(bureau_agg, 
                                train_target, 
                                on='SK_ID_CURR', 
                                how='left')

# 3. Targetã�Œæ¬ æ��ã�—ã�¦ã�„ã‚‹è¡Œï¼ˆãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿ã�®é¡§å®¢IDï¼‰ã�¯é™¤å¤–/åˆ†é›¢
# ã�“ã‚Œã�§ã€�ã‚µãƒ–ãƒ¢ãƒ‡ãƒ«ã�®å­¦ç¿’ã�«ä½¿ã�ˆã‚‹ãƒ‡ãƒ¼ã‚¿ã�Œæº–å‚™å®Œäº†


#ãƒ‡ãƒ¼ã‚¿ã�®ã‚³ãƒ”ãƒ¼ã‚’ä½œæˆ�ã�—ã�¦ã‚«ãƒ†ã‚´ãƒªåˆ—ã‚’æŠ½å‡º
df_prepared = bureau_agg_with_target.copy()
cat_cols = df_prepared.select_dtypes(include=['object']).columns.tolist()
#ãƒ¯ãƒ³ãƒ›ãƒƒãƒˆã‚¨ãƒ³ã‚³ãƒ¼ãƒ‡ã‚£ãƒ³ã‚°ã�™ã‚‹
df_prepared = pd.get_dummies(df_prepared, columns=cat_cols, dummy_na=True)
df_prepared = df_prepared.apply(pd.to_numeric, errors='coerce')
#ãƒªãƒƒã‚¸å›�å¸°ã‚’å›�ã�™
bureau_agg_with_target['bb_ridge_score'] = get_ridge_preds(df_prepared)
train_df = train_df.merge(
    bureau_agg_with_target[['SK_ID_CURR', 'bb_ridge_score']],
    on='SK_ID_CURR',
    how='left'
)
# ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿(test_df)ã�«ãƒ�ãƒ¼ã‚¸
test_df = test_df.merge(
    bureau_agg_with_target[['SK_ID_CURR', 'bb_ridge_score']], 
    on='SK_ID_CURR', 
    how='left'
)


#ãƒ‡ãƒ¼ã‚¿ã�®ã‚³ãƒ”ãƒ¼ã‚’ä½œæˆ�ã�—ã�¦ã‚«ãƒ†ã‚´ãƒªåˆ—ã‚’æŠ½å‡º
df_prepared = bureau_agg_with_target.copy()
cat_cols = df_prepared.select_dtypes(include=['object']).columns.tolist()
#ãƒ¯ãƒ³ãƒ›ãƒƒãƒˆã‚¨ãƒ³ã‚³ãƒ¼ãƒ‡ã‚£ãƒ³ã‚°ã�™ã‚‹
df_prepared = pd.get_dummies(df_prepared, columns=cat_cols, dummy_na=True)
df_prepared = df_prepared.apply(pd.to_numeric, errors='coerce')



#extratreeã�§ã�®ã‚¹ã‚¿ãƒƒã‚­ãƒ³ã‚°å®Ÿè¡Œ
bureau_agg_with_target['bb_et_score'] = get_et_preds(df_prepared)
# train_df ã�«æ�¥ç¶š
train_df = train_df.merge(
    bureau_agg_with_target[['SK_ID_CURR', 'bb_et_score']], 
    on='SK_ID_CURR', 
    how='left'
)

# test_df ã�«æ�¥ç¶š
test_df = test_df.merge(
    bureau_agg_with_target[['SK_ID_CURR', 'bb_et_score']], 
    on='SK_ID_CURR', 
    how='left'
)


NFOLDS = 5
folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)

# 1. å­¦ç¿’ç”¨ã�¨ãƒ†ã‚¹ãƒˆç”¨ã�«åˆ†ã�‘ã‚‹
train_data = bureau_agg_with_target[bureau_agg_with_target['TARGET'].notnull()].reset_index(drop=True)
test_data = bureau_agg_with_target[bureau_agg_with_target['TARGET'].isnull()].reset_index(drop=True)

# 2. ç‰¹å¾´é‡�åˆ—ã‚’æŠ½å‡ºï¼ˆSK_ID_CURR ã�¨ TARGET ã�¯é™¤å¤–ï¼‰
feature_cols = [c for c in bureau_agg.columns if c != 'SK_ID_CURR']

# 3. objectåˆ—ã‚’ã‚«ãƒ†ã‚´ãƒªå�‹ã�«å¤‰æ�›
for col in feature_cols:
    if train_data[col].dtype == 'object':
        train_data[col] = train_data[col].astype('category')
        test_data[col] = test_data[col].astype('category')

# 4. OOF é…�åˆ—
oof_preds_pc = np.zeros(train_data.shape[0])
test_preds_pc = np.zeros(test_data.shape[0])

# 5. KFold å­¦ç¿’
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_data[feature_cols], train_data['TARGET'])):
    X_train_fold = train_data.iloc[train_idx][feature_cols]
    y_train_fold = train_data.iloc[train_idx]['TARGET']
    X_valid_fold = train_data.iloc[valid_idx][feature_cols]
    y_valid_fold = train_data.iloc[valid_idx]['TARGET']

    lgb_model = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=32,
        colsample_bytree=0.8,
        subsample=0.8,
        random_state=42,
        n_jobs=-1
    )
    lgb_model.fit(
        X_train_fold, y_train_fold,
        eval_metric='auc',
        eval_set=[(X_valid_fold, y_valid_fold)],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
    )

    # OOFäºˆæ¸¬
    oof_preds_pc[valid_idx] = lgb_model.predict_proba(X_valid_fold)[:,1]
    # ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿äºˆæ¸¬
    test_preds_pc += lgb_model.predict_proba(test_data[feature_cols])[:,1] / NFOLDS

# 6. OOFåˆ—ã‚’è¿½åŠ 
train_data['PC_OOF_LGBM'] = oof_preds_pc
test_data['PC_OOF_LGBM'] = test_preds_pc

# 7. SK_ID_CURR ã�¨ OOFåˆ—ã� ã�‘æŠ½å‡º
oof_bureau_agg = train_data[['SK_ID_CURR','PC_OOF_LGBM']]
test_bureau_agg = test_data[['SK_ID_CURR','PC_OOF_LGBM']]

# 8. å…ƒã�® train_df/test_df ã�« merge
train_merged = train_df.merge(oof_bureau_agg, on='SK_ID_CURR', how='left')
test_merged = test_df.merge(test_bureau_agg, on='SK_ID_CURR', how='left')





train_df.info()


# trainã�¨testã�®å½¢çŠ¶ï¼ˆè¡Œæ•°, åˆ—æ•°ï¼‰ã‚’è¡¨ç¤º
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# ç‰¹å¾´é‡�ã�®æ•°ï¼ˆIDã�¨TARGETã‚’é™¤ã�„ã�Ÿæ•°ï¼‰
features_count = len([c for c in train_df.columns if c not in ['SK_ID_CURR', 'TARGET']])
print(f"ç�¾åœ¨ã�®ç‰¹å¾´é‡�ã�®æ•°: {features_count}")


#è¦�ã‚‰ã�ªã�„ã‚‚ã�®ã�¯æ¶ˆã�™
del bureau_agg,bureau_agg_with_target
gc.collect() 


#trainã�®å¤‰ã�ªã�¨ã�“ã‚�ã‚’ä¿®æ­£
# å‡¦ç�†å¯¾è±¡ã�®DAYS_ç³»ã‚«ãƒ©ãƒ 
#æ—¥æ•°å�˜ä½�ã�®ã�Ÿã‚�ä¿®æ­£ã�Œå¿…è¦�
days_features = [
    'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 
    'DAYS_ID_PUBLISH', 'DAYS_LAST_PHONE_CHANGE'
]
#ãƒªã‚¹ãƒˆã‚’ä½¿ç”¨

def preprocess_days_features_and_drop_old(df):
    """
    DAYS_ç³»ã�®å‰�å‡¦ç�†ï¼ˆç•°å¸¸å€¤å‡¦ç�†ã€�å¹´æ•°å¤‰æ�›ã€�å…ƒã�®ç‰¹å¾´é‡�å‰Šé™¤ï¼‰ã‚’è¡Œã�†çµ±å�ˆé–¢æ•°
    """
    df_new = df.copy()
    #å®‰å…¨ã�®ã�Ÿã‚�ã�«dfã‚’ã‚³ãƒ”ãƒ¼ã�—ã�¦ä½¿ç”¨

    # 1. DAYS_EMPLOYED ã�®ç•°å¸¸å€¤å‡¦ç�†ï¼ˆé‡�è¦�ï¼‰
    # ç•°å¸¸å€¤ã�‹ã�©ã�†ã�‹ã‚’ç¤ºã�™ãƒ•ãƒ©ã‚°ã‚’ä½œæˆ�
    df_new['DAYS_EMPLOYED_ANOM'] = (df_new['DAYS_EMPLOYED'] == 365243)
    #ã�“ã�®ç•°å¸¸å€¤ã�¯ç„¡è�·ã‚„æ¬ æ��ã‚’è¡¨ã�™
    #ç•°å¸¸å€¤ã�«ãƒ•ãƒ©ã‚°ã‚’ã�Ÿã�¦ã‚‹
    
    # ç•°å¸¸å€¤ã‚’æ¬ æ��å€¤ (NaN) ã�«ç½®ã��æ�›ã�ˆ
    df_new['DAYS_EMPLOYED'] = df_new['DAYS_EMPLOYED'].replace({365243: np.nan})

    # 2. å¹´å�˜ä½�ã�«å¤‰æ�›ã�—ã�Ÿæ–°ã�—ã�„ç‰¹å¾´é‡�ã‚’è¿½åŠ 
    for feature in days_features:
        # æ–°ã�—ã�„ç‰¹å¾´é‡�ã�®å��å‰�
        new_feature_name = feature.replace('DAYS_', 'YEARS_')
        
        # ç¬¦å�·å��è»¢ã�—ã€�365ã�§å‰²ã�£ã�¦å¹´æ•°ã�«å¤‰æ�›
        # DAYS_EMPLOYED ã‚‚ã�“ã�“ã�§æ­£ã�—ã��å¹´æ•°ã�«å¤‰æ�›ã�•ã‚Œã‚‹
        df_new[new_feature_name] = -df_new[feature] / 365

    # 3. å…ƒã�®ç‰¹å¾´é‡�ã�®å‰Šé™¤ï¼ˆãƒ«ãƒ¼ãƒ—ã�®å¤–å�´ã�§è¡Œã�†ï¼‰
    df_new.drop(columns=days_features, inplace=True)
        
    return df_new
#ã�“ã�“ã�¾ã�§ã�§é–¢æ•°å®šç¾©


# ä¿®æ­£å¾Œã�®é–¢æ•°ã�®å‘¼ã�³å‡ºã�—ï¼ˆtrain_dfã�¨test_dfã�¯äº‹å‰�ã�«èª­ã�¿è¾¼ã�¾ã‚Œã�¦ã�„ã‚‹å¿…è¦�ã�Œã�‚ã‚Šã�¾ã�™ï¼‰
train_df_processed = preprocess_days_features_and_drop_old(train_df)
test_df_processed = preprocess_days_features_and_drop_old(test_df)
print("train_dfã�®å¤‰ã�ªã�¨ã�“ã‚�ã‚’ä¿®æ­£ã�—ã�¾ã�—ã�Ÿ")


# å�‹ã�®ç¢ºèª�
print(type(train_df_processed))
print(type(test_df_processed))

# DataFrame ã�§ã�‚ã‚‹å ´å�ˆã�®ã�¿æ–‡å­—åˆ—åˆ—ã‚’æŠ½å‡º
if isinstance(train_df_processed, pd.DataFrame):
    cat_cols_train = train_df_processed.select_dtypes(include='object').columns
    print("train_df_processed ã�®ã‚«ãƒ†ã‚´ãƒªåˆ—:", list(cat_cols_train))
else:
    print("train_df_processed ã�¯ DataFrame ã�§ã�¯ã�‚ã‚Šã�¾ã�›ã‚“")

if isinstance(test_df_processed, pd.DataFrame):
    cat_cols_test = test_df_processed.select_dtypes(include='object').columns
    print("test_df_processed ã�®ã‚«ãƒ†ã‚´ãƒªåˆ—:", list(cat_cols_test))
else:
    print("test_df_processed ã�¯ DataFrame ã�§ã�¯ã�‚ã‚Šã�¾ã�›ã‚“")



ä¸€å›�ã‚¹ãƒˆãƒƒãƒ—


from sklearn.preprocessing import LabelEncoder

# train, test ã�Œ DataFrame ã�§ã�‚ã‚‹å ´å�ˆã�®ã�¿æ–‡å­—åˆ—å�‹åˆ—ã‚’æŠ½å‡º
cat_cols_train = train_df_processed.select_dtypes(include='object').columns.tolist()
cat_cols_test = test_df_processed.select_dtypes(include='object').columns.tolist()

# ä¸¡æ–¹ã�«å­˜åœ¨ã�™ã‚‹åˆ—ã�®ã�¿ã‚’å¯¾è±¡ã�«ã�™ã‚‹
cat_cols = list(set(cat_cols_train) & set(cat_cols_test))

# ãƒ©ãƒ™ãƒ«ã‚¨ãƒ³ã‚³ãƒ¼ãƒ‡ã‚£ãƒ³ã‚°
for col in cat_cols:
    le = LabelEncoder()
    # train ã�«å�ˆã‚�ã�›ã�¦ fit
    train_df_processed[col] = le.fit_transform(train_df_processed[col].astype(str))
    # test ã�¯ train ã�®ã‚«ãƒ†ã‚´ãƒªã�«å�ˆã‚�ã�›ã‚‹
    test_df_processed[col] = le.transform(test_df_processed[col].astype(str))



# å�‹ã�®ç¢ºèª�
print(type(train_df_processed))
print(type(test_df_processed))

# DataFrame ã�§ã�‚ã‚‹å ´å�ˆã�®ã�¿æ–‡å­—åˆ—åˆ—ã‚’æŠ½å‡º
if isinstance(train_df_processed, pd.DataFrame):
    cat_cols_train = train_df_processed.select_dtypes(include='object').columns
    print("train_df_processed ã�®ã‚«ãƒ†ã‚´ãƒªåˆ—:", list(cat_cols_train))
else:
    print("train_df_processed ã�¯ DataFrame ã�§ã�¯ã�‚ã‚Šã�¾ã�›ã‚“")

if isinstance(test_df_processed, pd.DataFrame):
    cat_cols_test = test_df_processed.select_dtypes(include='object').columns
    print("test_df_processed ã�®ã‚«ãƒ†ã‚´ãƒªåˆ—:", list(cat_cols_test))
else:
    print("test_df_processed ã�¯ DataFrame ã�§ã�¯ã�‚ã‚Šã�¾ã�›ã‚“")



remaining_obj_cols = train_df_processed.select_dtypes(include=['object']).columns.tolist()
print(remaining_obj_cols)



# æ•°å€¤å�‹åˆ—ã�®æŠ½å‡ºï¼ˆTARGETã�¯å­˜åœ¨ã�™ã‚‹å ´å�ˆã� ã�‘é™¤å¤–ï¼‰
numeric_cols_train = [c for c in train_df_processed.select_dtypes(include=['number']).columns if c != 'TARGET']
numeric_cols_test = [c for c in test_df_processed.select_dtypes(include=['number']).columns if c in numeric_cols_train]

# æ•°å€¤å�‹åˆ—ã�®æ¬ æ��å€¤ã‚’0ã�§åŸ‹ã‚�ã‚‹
train_df_processed[numeric_cols_train] = train_df_processed[numeric_cols_train].fillna(0)
test_df_processed[numeric_cols_test] = test_df_processed[numeric_cols_test].fillna(0)

print("æ•°å€¤ãƒ»ã‚«ãƒ†ã‚´ãƒªå�‹ã�®æ¬ æ��å€¤å‡¦ç�†ã�Œå®Œäº†ã�—ã�¾ã�—ã�Ÿã€‚")



# 1. train_dfå…¨ä½“ã�‹ã‚‰ã‚«ãƒ†ã‚´ãƒªã‚«ãƒ«å¤‰æ•°ã‚’æŠ½å‡º
# ã‚¤ãƒ³ãƒ‡ãƒ³ãƒˆã�¯ä¸�è¦�ï¼ˆä¸€ç•ªå·¦ã�‹ã‚‰æ›¸ã��ï¼‰
categorical_cols_all = train_df_processed.select_dtypes(
    include=['object', 'category']
).columns.tolist()

# 2. é™¤å¤–ã�™ã�¹ã��åˆ—ã‚’å‰Šé™¤
exclude_cols = ['SK_ID_CURR']  # IDã�ªã�©ã�¯ã‚¨ãƒ³ã‚³ãƒ¼ãƒ‡ã‚£ãƒ³ã‚°ã�‹ã‚‰é™¤å¤–
categorical_cols_all = [
    col for col in categorical_cols_all 
    if col not in exclude_cols
]

print(f"Target Encodingå¯¾è±¡ã�®ã‚«ãƒ†ã‚´ãƒªã‚«ãƒ«å¤‰æ•°: {len(categorical_cols_all)}å€‹")
print(categorical_cols_all[:10])  # æœ€åˆ�ã�®10å€‹ã‚’ç¢ºèª�


train_dateã‚’5åˆ†å‰²ã�™ã‚‹


from sklearn.model_selection import KFold

#KFoldã�®è¨­å®š
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# åˆ†å‰²ã�®ç¢ºèª�
print(f"trainãƒ‡ãƒ¼ã‚¿ã‚’{n_splits}åˆ†å‰²ã�—ã�¾ã�™")
print(f"train_dfã�®è¡Œæ•°: {len(train_df_processed)}")

# å�„Foldã�®ã‚µã‚¤ã‚ºã‚’ç¢ºèª�
for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_df_processed), 1):
    print(f"Fold {fold_idx}: å­¦ç¿’ç”¨={len(train_idx)}è¡Œ, æ¤œè¨¼ç”¨={len(val_idx)}è¡Œ")



# ã‚¹ãƒ ãƒ¼ã‚¸ãƒ³ã‚°ã�®å¼·åº¦ï¼ˆã‚«ãƒ†ã‚´ãƒªä»¶æ•°ã�Œå°‘ã�ªã�„ã�¨ã��ã�«ã�©ã‚Œã��ã‚‰ã�„å…¨ä½“å¹³å�‡ã�«å¯„ã�›ã‚‹ã�‹ï¼‰
min_samples_leaf = 10
smoothing = 1

def get_smoothed_te(train_subset, col, target_col, global_mean):
    # å�„ã‚«ãƒ†ã‚´ãƒªã�®å¹³å�‡ã�¨ä»¶æ•°ã‚’è¨ˆç®—
    agg = train_subset.groupby(col)[target_col].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']
    
    # ã‚¹ãƒ ãƒ¼ã‚¸ãƒ³ã‚°ä¿‚æ•°ã�®è¨ˆç®—
    # 1 / (1 + exp(-(counts - min_samples_leaf) / smoothing))
    smooth = 1 / (1 + np.exp(-(counts - min_samples_leaf) / smoothing))
    
    # å…¨ä½“å¹³å�‡ã�¨ã‚«ãƒ†ã‚´ãƒªå¹³å�‡ã‚’é‡�ã�¿ä»˜ã�‘
    # (ã‚«ãƒ†ã‚´ãƒªå¹³å�‡ * smooth) + (å…¨ä½“å¹³å�‡ * (1 - smooth))
    return (means * smooth + global_mean * (1 - smooth)).to_dict()

# --- Target Encodingã�®å®Ÿè¡Œ ---
print("Target Encodingã‚’é–‹å§‹ã�—ã�¾ã�™...")
global_mean = train_df_processed['TARGET'].mean()

for col in categorical_cols_all:
    new_col = f'{col}_te'
    train_df_processed[new_col] = global_mean
    
    # ã‚¤ãƒ³ãƒ‡ãƒƒã‚¯ã‚¹ã‚’ãƒªã‚»ãƒƒãƒˆã�—ã�¦ã�Šã��ã�¨å®‰å…¨
    # train_df_processed = train_df_processed.reset_index(drop=True)

    for train_idx, val_idx in kf.split(train_df_processed):
        # å­¦ç¿’ãƒ‡ãƒ¼ã‚¿ã�®çµ±è¨ˆé‡�ã�‹ã‚‰ãƒ�ãƒƒãƒ—ã‚’ä½œæˆ�ï¼ˆã‚¹ãƒ ãƒ¼ã‚¸ãƒ³ã‚°é�©ç”¨ï¼‰
        encoding_map = get_smoothed_te(train_df_processed.iloc[train_idx], col, 'TARGET', global_mean)
        
        # æ¤œè¨¼ãƒ‡ãƒ¼ã‚¿ã�«é�©ç”¨
        train_df_processed.iloc[val_idx, train_df_processed.columns.get_loc(new_col)] = \
            train_df_processed.iloc[val_idx][col].map(encoding_map).fillna(global_mean)
            
    # testç”¨: trainå…¨ä½“ã�®çµ±è¨ˆé‡�ã‚’è¨ˆç®—ã�—ã�¦é�©ç”¨
    encoding_map_test = get_smoothed_te(train_df_processed, col, 'TARGET', global_mean)
    test_df_processed[new_col] = test_df_processed[col].map(encoding_map_test).fillna(global_mean)
    
    print(f"  â†’ {new_col} ä½œæˆ�å®Œäº†ï¼ˆSmoothingé�©ç”¨æ¸ˆï¼‰")


train_df_processed.columns = train_df_processed.columns.str.replace(r'[^A-Za-z0-9_]', '_', regex=True)
test_df_processed.columns  = test_df_processed.columns.str.replace(r'[^A-Za-z0-9_]', '_', regex=True)



# å…ƒã�®ã‚«ãƒ†ã‚´ãƒªã‚«ãƒ«å¤‰æ•°ã‚’å‰Šé™¤ã�™ã‚‹ã‚³ãƒ¼ãƒ‰ä¾‹
# train_df_processed ã�‹ã‚‰ categorical_cols_all ã‚’å‰Šé™¤
train_df_final = train_df_processed.drop(columns=categorical_cols_all)
test_df_final = test_df_processed.drop(columns=categorical_cols_all)

print(f"å‰Šé™¤å‰�ã�®åˆ—æ•°: {len(train_df_processed.columns)}")
print(f"å‰Šé™¤å¾Œã�®åˆ—æ•°: {len(train_df_final.columns)}")


# 1. ç›®çš„å¤‰æ•°(y)ã�¨ç‰¹å¾´é‡�(X)ã‚’åˆ†ã�‘ã‚‹
# æ­£è§£ãƒ©ãƒ™ãƒ«ã�¯ train_df ã�‹ã‚‰å�–å¾—ã�—ã�¾ã�™
y = train_df_final['TARGET']
X = train_df_final.drop(columns=['TARGET', 'SK_ID_CURR'])

# 2. ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿ã‚‚å�Œæ§˜ã�« ID åˆ—ã�ªã�©ã‚’é™¤å¤–ã�—ã�¦åˆ—ã‚’æ�ƒã�ˆã‚‹
# X ã�¨ test_X ã�®åˆ—ã�®é †ç•ªã�¨ç¨®é¡�ã�Œå®Œå…¨ã�«ä¸€è‡´ã�—ã�¦ã�„ã‚‹å¿…è¦�ã�Œã�‚ã‚Šã�¾ã�™
test_X = test_df_final.drop(columns=['SK_ID_CURR'])

# 3. ã‚«ãƒ†ã‚´ãƒªå¤‰æ•°ã�®å†�ç¢ºèª�
# Target Encoding ã�Œçµ‚ã‚�ã�£ã�¦ã�„ã‚Œã�°ã€�ã�“ã�“ã�¯ç©ºã�®ãƒªã‚¹ãƒˆ [] ã�«ã�ªã‚‹ã�¯ã�šã�§ã�™
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"å­¦ç¿’ãƒ‡ãƒ¼ã‚¿ã�®ç‰¹å¾´é‡�æ•°: {X.shape[1]}")
print(f"ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿ã�®ç‰¹å¾´é‡�æ•°: {test_X.shape[1]}")
print(f"ã‚«ãƒ†ã‚´ãƒªå¤‰æ•°ã�¨ã�—ã�¦æ®‹ã�£ã�¦ã�„ã‚‹åˆ—: {categorical_features}")

# åˆ—ã�®ä¸�ä¸€è‡´ã�Œã�ªã�„ã�‹æœ€çµ‚ãƒ�ã‚§ãƒƒã‚¯
if list(X.columns) == list(test_X.columns):
    print("âœ… å­¦ç¿’ãƒ‡ãƒ¼ã‚¿ã�¨ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿ã�®åˆ—ã�Œä¸€è‡´ã�—ã�¾ã�—ã�Ÿï¼�")
else:
    print("âš ï¸� åˆ—ã�Œä¸€è‡´ã�—ã�¦ã�„ã�¾ã�›ã‚“ã€‚ç¢ºèª�ã�Œå¿…è¦�ã�§ã�™ã€‚")


# def objective(trial):
#     param = {
#         "objective": "binary",
#         "metric": "auc",
#         "boosting_type": "gbdt",
#         "num_leaves": trial.suggest_int("num_leaves", 50, 80),
#         "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.01),
#         "feature_fraction": trial.suggest_float("feature_fraction", 0.3, 0.5),
#         "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 0.7),
#         "min_child_samples": trial.suggest_int("min_child_samples", 80, 120),
#         "reg_lambda": 0.2,
#         "random_state": 42,
#         "verbose": -1
#     }
# 
#     gbm = lgb.train(
#         param,
#         train_data,
#         num_boost_round=3000,
#         valid_sets=[valid_data],
#         callbacks=[
#             lgb.early_stopping(stopping_rounds=100),
#             lgb.log_evaluation(period=0)  # verbose_eval=False ã�®ä»£ã‚�ã‚Š
#         ]
#     )
# 
#     y_pred = gbm.predict(X_valid)
#     auc = roc_auc_score(y_valid, y_pred)
# 
#     del gbm
#     gc.collect()
# 
#     return auc


import optuna
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import gc
import numpy as np

# ==================================================
# 1. train / valid åˆ†å‰²
# ==================================================
# yã�®æ¯”ç�‡ã‚’ç¶­æŒ�ã�—ã�¦åˆ†å‰² (stratify=y)
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_valid, label=y_valid)

# ==================================================
# 2. Optuna ç›®çš„é–¢æ•°
# ==================================================
def objective(trial):
    param = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "feature_pre_filter": False,
        # æ�¢ç´¢ç¯„å›²ã�®è¨­å®š
        "num_leaves": trial.suggest_int("num_leaves", 60, 128),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.01, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 0.8),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 80, 150),
        "reg_alpha": trial.suggest_float("reg_alpha", 10, 50.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 10, 50.0, log=True),
        "random_state": 42,
        "verbose": -1,
    }

    # å­¦ç¿’
    gbm = lgb.train(
        param,
        train_data,
        num_boost_round=10000,
        valid_sets=[valid_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=0) # ãƒ­ã‚°ã‚’å‡ºã�•ã�ªã�„è¨­å®šï¼ˆæ�¢ç´¢ä¸­ã�¯é�™ã�‹ã�«ã�™ã‚‹ã�Ÿã‚�ï¼‰
        ]
    )

    # äºˆæ¸¬ã�¨è©•ä¾¡
    y_pred = gbm.predict(X_valid)
    auc = roc_auc_score(y_valid, y_pred)

    del gbm
    gc.collect()

    return auc

# ==================================================
# 3. Optuna æ�¢ç´¢
# ==================================================
# sqliteã‚’ä½¿ã�£ã�¦é€²æ�—ã‚’ä¿�å­˜ã�™ã‚‹ã‚ˆã�†ã�«è¨­å®š
study = optuna.create_study(
    direction="maximize",
    study_name="my_lgbm_optuna",
    storage="sqlite:///optuna_study.db",
    load_if_exists=True
)

# æœ€é�©åŒ–å®Ÿè¡Œ (n_trialsã�¯æ™‚é–“ã�«å�ˆã‚�ã�›ã�¦èª¿æ•´ã�—ã�¦ã��ã� ã�•ã�„)
study.optimize(objective, n_trials=30)

# ==================================================
# 4. çµ�æ�œè¡¨ç¤º
# ==================================================
trial = study.best_trial
print("Best trial:")
print(f"  AUC: {trial.value}")
print("Params:")
for key, value in trial.params.items():
    print(f"    {key}: {value}")


# import optuna
# import lightgbm as lgb
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_auc_score
# import gc
# 
# # ==================================================
# # 1. train / valid åˆ†å‰²
# # ==================================================
# # X_train, X_valid, y_train, y_valid = train_test_split(
# #     X, y,
# #     test_size=0.2,
# #     random_state=42,
# #     stratify=y
# # )
# 
# # train_data = lgb.Dataset(X_train, label=y_train)
# # valid_data = lgb.Dataset(X_valid, label=y_valid)
# 
# # ==================================================
# # 2. Optuna ç›®çš„é–¢æ•°
# # ==================================================
# # def objective(trial):
# #     param = {
# #         "objective": "binary",
# #         "metric": "auc",
# #         "boosting_type": "gbdt",
# #         "num_leaves": trial.suggest_int("num_leaves", 70, 128),
# #         "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.01),
# #         "feature_fraction": trial.suggest_float("feature_fraction", 0.3, 0.5),
# #         "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 0.7),
# #         "min_child_samples": trial.suggest_int("min_child_samples", 100, 150),
# #         "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 50.0, log=True), 
# #         "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 50.0, log=True),# L1æ­£å‰‡åŒ–ã‚‚ã‚»ãƒƒãƒˆã�§
# #         "random_state": 42,
# #         "verbose": -1,
# #         "feature_pre_filter": False
# #     }
# 
# #     gbm = lgb.train(
# #         param,
# #         train_data,
# #         num_boost_round=10000,
# #         valid_sets=[valid_data],
# #         callbacks=[
# #             lgb.early_stopping(stopping_rounds=100),
# #             lgb.log_evaluation(period=0)
# #         ]
# #     )
# 
# #     y_pred = gbm.predict(X_valid)
# #     auc = roc_auc_score(y_valid, y_pred)
# 
# #     del gbm
# #     gc.collect()
# 
# #     return auc
# 
# # ==================================================
# # 3. Optuna æ�¢ç´¢
# # ==================================================
# # study = optuna.create_study(
# #     direction="maximize",
# #     study_name="my_lgbm_optuna",
# #     storage="sqlite:///optuna_study.db",
# #     load_if_exists=True
# # )
# 
# # æœ€é�©åŒ–å®Ÿè¡Œ
# # study.optimize(objective, n_trials=30)
# 
# # ==================================================
# # 4. çµ�æ�œè¡¨ç¤º
# # ==================================================
# # trial = study.best_trial
# # print("Best trial:")
# # print(f"AUC: {trial.value}")
# # print("Params:")
# # for key, value in trial.params.items():
# #     print(f"    {key}: {value}")


#study.optimize(objective, n_trials=30)



from sklearn.model_selection import train_test_split
import lightgbm as lgb

# ==================================================
# 1. train / valid ã�«åˆ†å‰²
# ==================================================
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==================================================
# 2. ã‚«ãƒ†ã‚´ãƒªå¤‰æ•°ã�¯ã�™ã�§ã�«æ•°å€¤åŒ–æ¸ˆã�¿ã�ªã�®ã�§ä¸�è¦�
# ==================================================
# categorical_features = X.select_dtypes(include='object').columns.tolist()
# for col in categorical_features:
#     X_train[col] = X_train[col].astype('category')
#     X_valid[col] = X_valid[col].astype('category')

train_data = lgb.Dataset(X_train, label=y_train)  # ã‚«ãƒ†ã‚´ãƒªã�¯æ•°å€¤åŒ–æ¸ˆã�¿
valid_data = lgb.Dataset(X_valid, label=y_valid)

# ==================================================
# 3. LightGBM ãƒ‘ãƒ©ãƒ¡ãƒ¼ã‚¿è¨­å®š
# ==================================================



# After (ä¿®æ­£ç‰ˆ)
#params = {
#    "objective": "binary",
#    "metric": "auc",
#    "boosting_type": "gbdt",
#    "reg_lambda": 0.2,
#   "num_leaves": 70,#before70
#    "learning_rate": 0.008,#before0,01
    #"num_boost_round": 3000,
#    "feature_fraction": 0.5,      # å°�ã�•ã�„æ–¹ã�Œé��å­¦ç¿’ã�—ã�«ã��ã�„ã�Œæ€§èƒ½ä½�ä¸‹ã�®å�¯èƒ½æ€§ã‚‚before0.5
#    "min_data_in_leaf": 80,       # å°�ã�•ã�„æ–¹ã�Œç´°ã�‹ã��å­¦ã�¶before100
#    "bagging_fraction": 0.7,#å…ƒã�¯0.7
#    "bagging_freq": 5,             
#    "verbose": -1,
#    "random_state": 42
#}

#params = {
#    "objective": "binary",
#    "metric": "auc",
#    "boosting_type": "gbdt",
#    "reg_lambda": 0.2,#5
#    "num_leaves": 79,#before70
 #   "learning_rate": 0.009032560591570094,#before0,01
    #"num_boost_round": 3000,
#    "feature_fraction":  0.32358299713321215,      # å°�ã�•ã�„æ–¹ã�Œé��å­¦ç¿’ã�—ã�«ã��ã�„ã�Œæ€§èƒ½ä½�ä¸‹ã�®å�¯èƒ½æ€§ã‚‚before0.5
#    "min_data_in_leaf": 110,       # å°�ã�•ã�„æ–¹ã�Œç´°ã�‹ã��å­¦ã�¶before100
#    "bagging_fraction": 0.6245750554341316,#å…ƒã�¯0.7
    #"bagging_freq": 5,             
#    "verbose": -1,
 #   "random_state": 42
#}

params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "reg_lambda": 10.253,#14.819778
    "reg_alpha": 8.496,#14.819778
    "num_leaves": 63,#before70
    "learning_rate": 0.00406,#before0.005117
    #"num_boost_round": 3000,
    "feature_fraction":  0.559,      # å°�ã�•ã�„æ–¹ã�Œé��å­¦ç¿’ã�—ã�«ã��ã�„ã�Œæ€§èƒ½ä½�ä¸‹ã�®å�¯èƒ½æ€§ã‚‚before0.30134
    "min_data_in_leaf": 89,       # å°�ã�•ã�„æ–¹ã�Œç´°ã�‹ã��å­¦ã�¶before102
    "bagging_fraction": 0.775,#å…ƒã�¯0.650379
    "bagging_freq": 7,             
    "verbose": -1,
    "random_state": 42
}


# ==================================================
# 4. ãƒ¢ãƒ‡ãƒ«å­¦ç¿’
# ==================================================
model = lgb.train(
    params,
    train_data,
    num_boost_round=5000,  # å­¦ç¿’å›�æ•°
    valid_sets=[train_data, valid_data],
    callbacks=[
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(period=100)
    ]
)



#é‡�è¦�åº¦ã�®æ¸¬å®š
import pandas as pd

split_importance_df = pd.DataFrame({
    "feature": model.feature_name(),
    "importance": model.feature_importance(importance_type="split")
}).sort_values("importance", ascending=False)



split_importance_df


# ä¸Šä½�30ã€œ50ç‰¹å¾´é‡�ã‚’ç¢ºèª�
#split_importance_df.sort_values(by='importance', ascending=False).head(50)



# ä¸‹ä½�30%ã‚’å‰Šé™¤ã�™ã‚‹ä¾‹
#threshold = np.percentile(split_importance_df['importance'], 10)
#low_importance_features = split_importance_df[split_importance_df['importance'] <= threshold]['feature'].tolist()

#X_train_reduced = X_train.drop(columns=low_importance_features)
#X_valid_reduced = X_valid.drop(columns=low_importance_features)






# train_data = lgb.Dataset(X_train_reduced, label=y_train)  # ã‚«ãƒ†ã‚´ãƒªã�¯æ•°å€¤åŒ–æ¸ˆã�¿
# valid_data = lgb.Dataset(X_valid_reduced , label=y_valid)

# ==================================================
# 3. LightGBM ãƒ‘ãƒ©ãƒ¡ãƒ¼ã‚¿è¨­å®š
# ==================================================

# Before (å…ƒã€…ã�®è¨­å®š)
# params_before = {
#      "objective": "binary",
#      "metric": "auc",
#      "boosting_type": "gbdt",
#      "reg_lambda": 0.2,
#      "num_leaves": 16,
#      "learning_rate": 0.01,
#      "num_boost_round": 5000,
#      "feature_fraction": 0.7,
#      "min_data_in_leaf": 80,
#      "bagging_fraction": 0.7,
#      "bagging_freq": 5,
#      "verbose": -1,
#      "random_state": 42
# }

# After (ä¿®æ­£ç‰ˆ)
# params_reduced = {
#     "objective": "binary",
#     "metric": "auc",
#     "boosting_type": "gbdt",
#     "reg_lambda": 0.2,
#     "num_leaves": 70,#before70
#     "learning_rate": 0.008,#before0,01
#     #"num_boost_round": 3000,
#     "feature_fraction": 0.5,      # å°�ã�•ã�„æ–¹ã�Œé��å­¦ç¿’ã�—ã�«ã��ã�„ã�Œæ€§èƒ½ä½�ä¸‹ã�®å�¯èƒ½æ€§ã‚‚before0.5
#     "min_data_in_leaf": 80,       # å°�ã�•ã�„æ–¹ã�Œç´°ã�‹ã��å­¦ã�¶before100
#     "bagging_fraction": 0.7,#å…ƒã�¯0.7
#     "bagging_freq": 5,              
#     "verbose": -1,
#     "random_state": 42
# }

# ==================================================
# 4. ãƒ¢ãƒ‡ãƒ«å­¦ç¿’
# ==================================================
# model_reduced = lgb.train(
#     params_reduced,
#     train_data,
#     num_boost_round=3000,  # å­¦ç¿’å›�æ•°
#     valid_sets=[train_data, valid_data],
#     callbacks=[
#         lgb.early_stopping(stopping_rounds=100),
#         lgb.log_evaluation(period=100)
#     ]
# )


[100]	training's auc: 0.786945	valid_1's auc: 0.769492
[200]	training's auc: 0.798266	valid_1's auc: 0.774899


#split_importance_df


#æ¤œè¨¼ãƒ‡ãƒ¼ã‚¿ã�§AUCç¢ºèª�
from sklearn.metrics import roc_auc_score

y_pred_valid = model.predict(X_valid, num_iteration=model.best_iteration)
auc = roc_auc_score(y_valid, y_pred_valid)
print(f"Validation AUC: {auc:.4f}")



0.7790
0.7804
0.7865#å…¨ãƒ‡ãƒ¼ã‚¿çµ�å�ˆ
0.7859ã€€#æ­£è¦�åŒ–ç‰¹å¾´é‡�è¿½åŠ 
0.7818#ç‰¹å¾´é‡�è¦‹ç›´ã�—pos_cash
0.7875#ç‰¹å¾´é‡�è¿½åŠ +ã�™ã�¹ã�¦ãƒ©ãƒ™ãƒ«ã‚¨ãƒ³ã‚³ãƒ¼ãƒ‡ã‚£ãƒ³ã‚°
0.7894
0.7899#ãƒ�ã‚¹ãƒˆç‰¹å¾´é‡�
0.790569#ãƒ�ã‚¹ãƒˆç‰¹å¾´é‡�+ãƒ‘ãƒ©ãƒ¡ãƒ¼ã‚¿ã�“ã‚Œã� ã�¨0.78899ã�¨ã�‹ã�«ã�ªã‚‹
0.789709#ç‰¹å¾´é‡�å‰Šé™¤
0.790576#ãƒªãƒƒã‚¸å›�å¸°è¿½åŠ 
0.790702#ãƒªãƒƒã‚¸+optuna
0.791182
0.791819
0.791939#ã‚¿ãƒ¼ã‚²ãƒƒãƒˆã‚¨ãƒ³ã‚³ãƒ¼ãƒ‡ã‚£ãƒ³ã‚°



# X å…¨ä½“ã€�y å…¨ä½“ã‚’ä½¿ã�†
X_full = X  # train_df_processed ã�‹ã‚‰ä½œã�£ã�Ÿ X
y_full = y  # train_df_processed['TARGET']



print("å­¦ç¿’ãƒ‡ãƒ¼ã‚¿ã‚µã‚¤ã‚º:", X_full.shape)
print("ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿ã‚µã‚¤ã‚º:", test_df_processed.shape)
train_df_fina


import gc
import pandas as pd
import numpy as np

# --- 1. ã‚«ãƒ†ã‚´ãƒªå¤‰æ�› ---
# â€» X_full ã�¨ X_test ã�®å�‹ã‚’å®Œå…¨ã�«ä¸€è‡´ã�•ã�›ã�¦ã�Šã��
X_test = test_df_processed[X_full.columns].copy()

for col in categorical_features:
    X_full[col] = X_full[col].astype('category')
    X_test[col] = X_test[col].astype('category')



import gc
import pandas as pd
import numpy as np

# --- 1. ã‚«ãƒ†ã‚´ãƒªå¤‰æ�› ---
# â€» X_full ã�¨ X_test ã�®å�‹ã‚’å®Œå…¨ã�«ä¸€è‡´ã�•ã�›ã�¦ã�Šã��
X_test = test_df_processed[X_full.columns].copy()

for col in categorical_features:
    X_full[col] = X_full[col].astype('category')
    X_test[col] = X_test[col].astype('category')

# --- 2. ãƒ¢ãƒ‡ãƒ«å®šç¾© (å¼•æ•°ã�®æ›¸ã��æ–¹ã‚’ä¿®æ­£) ---
model_full = lgb.LGBMClassifier(
    objective="binary",
    metric="auc",
    boosting_type="gbdt",
    reg_lambda=10.253,
    reg_alpha=8.496,
    num_leaves=63,
    learning_rate=0.00406,
    n_estimators=5000,      # å…ˆã�»ã�©ã�®Early Stoppingã�§ã�®ãƒ™ã‚¹ãƒˆå›�æ•°ã‚’å…¥ã‚Œã‚‹ã�®ã�Œãƒ™ã‚¹ãƒˆ
    feature_fraction= 0.559,   # 1300åˆ—ã�ªã‚‰0.1~0.2ã�Œç²¾åº¦ã�Œå‡ºã‚„ã�™ã�„ã�§ã�™
    min_child_samples=89,   # min_data_in_leafã�®ä»£ã‚�ã‚Š
    bagging_fraction=0.775,
    #bagging_freq=5,
    verbose=-1,
    random_state=42
)

# --- 3. å­¦ç¿’ ---
model_full.fit(X_full, y_full)

# å­¦ç¿’ã�Œçµ‚ã‚�ã�£ã�Ÿã‚‰ã�™ã��ã�«ãƒ¡ãƒ¢ãƒªè§£æ”¾
del X_full, y_full
gc.collect()

# --- 4. äºˆæ¸¬ (predict_probaã‚’ä½¿ç”¨ï¼�) ---
# predict_probaã�®çµ�æ�œã�¯ [0ã�§ã�‚ã‚‹ç¢ºç�‡, 1ã�§ã�‚ã‚‹ç¢ºç�‡] ã�®é †ã�§è¿”ã‚‹ã�®ã�§ã€�[:, 1] ã‚’æŒ‡å®šã�—ã�¾ã�™
y_pred_test = model_full.predict_proba(X_test)[:, 1]

print("æœ¬ç•ªäºˆæ¸¬å®Œäº†")

# --- 5. æ��å‡ºãƒ•ã‚¡ã‚¤ãƒ«ä½œæˆ� ---
submission = pd.DataFrame({
    "SK_ID_CURR": test_df_processed["SK_ID_CURR"],
    "TARGET": y_pred_test
})

submission.to_csv("ã‚¿ãƒ¼ã‚²ãƒƒãƒˆã‚¨ãƒ³ã‚³ãƒ¼ãƒ‡ã‚£ãƒ³ã‚°è¿½åŠ ç‰ˆ.csv", index=False)


import numpy as np
import pandas as pd
import lightgbm as lgb
import gc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# ==================================================
# 3. LightGBM ãƒ‘ãƒ©ãƒ¡ãƒ¼ã‚¿è¨­å®š (ã�‚ã�ªã�Ÿã�®æœ€å¼·ãƒ‘ãƒ©ãƒ¡ãƒ¼ã‚¿ã‚’ç¶™æ‰¿)
# ==================================================
params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "reg_lambda": 10.253,
    "reg_alpha": 8.496,
    "num_leaves": 63,
    "learning_rate": 0.00406,
    "feature_fraction": 0.559,
    "min_data_in_leaf": 89,
    "bagging_fraction": 0.775,
    "bagging_freq": 7,  # ã�“ã‚Œã�Œé‡�è¦�ï¼�
    "verbose": -1,
    "random_state": 42
}

# ==================================================
# 4. ãƒ¢ãƒ‡ãƒ«å­¦ç¿’ (ã�“ã�“ã‚’ K-Fold ãƒ«ãƒ¼ãƒ—ã�«ç½®ã��æ�›ã�ˆ)
# ==================================================
# 5åˆ†å‰²ã�®è¨­å®š
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# ã‚¹ã‚³ã‚¢ã‚„äºˆæ¸¬å€¤ã‚’ä¿�å­˜ã�™ã‚‹ç®±
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
cv_scores = []

print(f"--- {n_splits}åˆ†å‰²äº¤å·®æ¤œè¨¼ (K-Fold) ã‚’é–‹å§‹ã�—ã�¾ã�™ ---")

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    print(f"\n[Fold {fold + 1}/{n_splits}]")
    
    # ãƒ‡ãƒ¼ã‚¿ã�®åˆ†å‰²
    X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
    y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]
    
    # Datasetã�®ä½œæˆ�
    dtrain = lgb.Dataset(X_train_fold, label=y_train_fold)
    dvalid = lgb.Dataset(X_valid_fold, label=y_valid_fold)
    
    # å­¦ç¿’ (early_stoppingã�§å�„Foldã�®æœ€é�©å›�æ•°ã‚’è¦‹æ¥µã‚�ã‚‹)
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=10000,
        valid_sets=[dtrain, dvalid],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=200)
        ]
    )
    
    # ã�“ã�®Foldã�§ã�®æ¤œè¨¼ãƒ‡ãƒ¼ã‚¿ã�¸ã�®äºˆæ¸¬
    oof_preds[valid_idx] = model.predict(X_valid_fold)
    
    # ã�“ã�®Foldã�§ã�®ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿ã�¸ã�®äºˆæ¸¬ (5å›�åˆ†ã‚’å¹³å�‡ã�™ã‚‹ã�Ÿã‚�ã�«åŠ ç®—)
    test_preds += model.predict(X_test) / n_splits
    
    # ã‚¹ã‚³ã‚¢ã�®è¨ˆç®—
    auc_score = roc_auc_score(y_valid_fold, oof_preds[valid_idx])
    cv_scores.append(auc_score)
    print(f"Fold {fold + 1} AUC: {auc_score:.6f}")
    
    # ãƒ¡ãƒ¢ãƒªè§£æ”¾
    del X_train_fold, X_valid_fold, dtrain, dvalid, model
    gc.collect()

# ==================================================
# 5. æœ€çµ‚çµ�æ�œã�®è¡¨ç¤ºã�¨ä¿�å­˜
# ==================================================
mean_auc = np.mean(cv_scores)
print(f"\nå¹³å�‡ CV AUC: {mean_auc:.6f}")




import numpy as np
import pandas as pd
import lightgbm as lgb
import gc
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# ==================================================
# 1. ãƒ‘ãƒ©ãƒ¡ãƒ¼ã‚¿è¨­å®š (ã�‚ã�ªã�Ÿã�®æœ€å¼·ãƒ‘ãƒ©ãƒ¡ãƒ¼ã‚¿)
# ==================================================
params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "reg_lambda": 10.253,
    "reg_alpha": 8.496,
    "num_leaves": 63,
    "learning_rate": 0.00406,
    "feature_fraction": 0.559,
    "min_data_in_leaf": 89,
    "bagging_fraction": 0.775,
    "bagging_freq": 7,
    "verbose": -1,
    "random_state": 42
}

# ==================================================
# 2. æº–å‚™ (X, y, X_test ã�Œå­˜åœ¨ã�™ã‚‹ã�“ã�¨ã‚’ç¢ºèª�)
# ==================================================
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# çµ�æ�œæ ¼ç´�ç”¨ã�®ç®±
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
fold_indices = np.zeros(len(X)) # å¾Œã�§ Fold 1 ã�®åˆ†æ��ã‚’ã�™ã‚‹ã�Ÿã‚�ã�«å¿…è¦�
cv_scores = []

print(f"--- {n_splits}åˆ†å‰²äº¤å·®æ¤œè¨¼ (K-Fold) ã‚’é–‹å§‹ã�—ã�¾ã�™ ---")
print("â€»å­¦ç¿’ã�«ã�¯æ•°æ™‚é–“ã�‹ã�‹ã‚Šã�¾ã�™ã€‚ãƒ–ãƒ©ã‚¦ã‚¶ã‚’é–‰ã�˜ã�šã�«å¾…æ©Ÿã�—ã�¦ã��ã� ã�•ã�„ã€‚")

# ==================================================
# 3. K-Fold ãƒ«ãƒ¼ãƒ—
# ==================================================
for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    print(f"\n[Fold {fold + 1}/{n_splits} ã�®å­¦ç¿’é–‹å§‹]")
    
    # ã�©ã�®ãƒ‡ãƒ¼ã‚¿ã�Œã�©ã�®ãƒ•ã‚©ãƒ¼ãƒ«ãƒ‰ã�‹è¨˜éŒ²
    fold_indices[valid_idx] = fold + 1
    
    # ãƒ‡ãƒ¼ã‚¿ã�®åˆ†å‰²
    X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
    y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]
    
    # Datasetã�®ä½œæˆ�
    dtrain = lgb.Dataset(X_train_fold, label=y_train_fold)
    dvalid = lgb.Dataset(X_valid_fold, label=y_valid_fold)
    
    # å­¦ç¿’
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=10000,
        valid_sets=[dtrain, dvalid],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=200)
        ]
    )
    
    # æ¤œè¨¼ãƒ‡ãƒ¼ã‚¿ã�¸ã�®äºˆæ¸¬
    oof_preds[valid_idx] = model.predict(X_valid_fold)
    
    # ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿ã�¸ã�®äºˆæ¸¬ (å¹³å�‡åŒ–)
    test_preds += model.predict(X_test) / n_splits
    
    # ã�“ã�®ãƒ•ã‚©ãƒ¼ãƒ«ãƒ‰ã�®ã‚¹ã‚³ã‚¢ã‚’è¨˜éŒ²
    auc_score = roc_auc_score(y_valid_fold, oof_preds[valid_idx])
    cv_scores.append(auc_score)
    print(f"Fold {fold + 1} AUC: {auc_score:.6f}")
    
    # ãƒ¡ãƒ¢ãƒªè§£æ”¾
    del X_train_fold, X_valid_fold, dtrain, dvalid, model
    gc.collect()

# ==================================================
# 4. æœ€çµ‚ã‚¹ã‚³ã‚¢ã�®è¨ˆç®—ã�¨è¡¨ç¤º
# ==================================================
mean_auc = np.mean(cv_scores)
print("\n" + "="*30)
print(f"å¹³å�‡ CV AUC: {mean_auc:.6f}")
print("="*30)

# ==================================================
# 5. æ��å‡ºç”¨ãƒ•ã‚¡ã‚¤ãƒ«ã�®ä½œæˆ� (ã‚¨ãƒ©ãƒ¼å›�é�¿ç‰ˆ)
# ==================================================
# test_df_processed ã�Œå­˜åœ¨ã�™ã‚‹ã�“ã�¨ã‚’ç¢ºèª�
filename = f"submission_KFold_final_{mean_auc:.5f}.csv"

submission = pd.DataFrame({
    "SK_ID_CURR": test_df_processed["SK_ID_CURR"],
    "TARGET": test_preds
})

submission.to_csv(filename, index=False)
print(f"\nğŸ�‰ å®Œäº†ã�—ã�¾ã�—ã�Ÿï¼�ãƒ•ã‚¡ã‚¤ãƒ« '{filename}' ã‚’æ��å‡ºã�—ã�¦ã��ã� ã�•ã�„ã€‚")

# ==================================================
# 6. (ã‚ªãƒ—ã‚·ãƒ§ãƒ³) ã‚¨ãƒ©ãƒ¼åˆ†æ��ç”¨ã�®ãƒ‡ãƒ¼ã‚¿ã‚’ä¿�å­˜
# ==================================================
# ã�“ã‚Œã‚’ã‚„ã�£ã�¦ã�Šã��ã�¨ã€�å¾Œã�§ã€Œã�ªã�œFold 1ã�¯ä½�ã�‹ã�£ã�Ÿã�®ã�‹ã€�ã‚’åˆ†æ��ã�§ã��ã�¾ã�™
analysis_results = pd.DataFrame({
    'SK_ID_CURR': X.index, # IDã�Œã‚¤ãƒ³ãƒ‡ãƒƒã‚¯ã‚¹ã�«ã�‚ã‚‹ã�¨ä»®å®š
    'target': y,
    'oof_pred': oof_preds,
    'fold': fold_indices
})
analysis_results.to_csv("kfold_oof_analysis.csv", index=False)


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

# ==================================================
# 1. ãƒ‡ãƒ¼ã‚¿ã�®èª­ã�¿è¾¼ã�¿ã�¨ã€Œèª¤å·®ã€�ã�®è¨ˆç®—
# ==================================================
file_path = "kfold_oof_analysis.csv"

if not os.path.exists(file_path):
    print(f"ã€�ã‚¨ãƒ©ãƒ¼ã€‘: '{file_path}' ã�Œè¦‹ã�¤ã�‹ã‚Šã�¾ã�›ã‚“ã€‚")
else:
    df_analysis = pd.read_csv(file_path)
    
    # abs_error ã�Œã�ªã�‘ã‚Œã�°ã�“ã�“ã�§ä½œæˆ�ï¼ˆé‡�è¦�ï¼�ï¼‰
    if 'abs_error' not in df_analysis.columns:
        df_analysis['abs_error'] = abs(df_analysis['target'] - df_analysis['oof_pred'])
        print("abs_error ã‚’è¨ˆç®—ã�—ã�¦è¿½åŠ ã�—ã�¾ã�—ã�Ÿã€‚")

    # ==================================================
    # 2. Fold 1 ã�Œæœ¬å½“ã�«ã€Œé›£ã�—ã�‹ã�£ã�Ÿã€�ã�®ã�‹ã‚’ç¢ºèª�
    # ==================================================
    fold_summary = df_analysis.groupby('fold')['abs_error'].mean()
    print("\n--- Foldåˆ¥ã�®å¹³å�‡èª¤å·® (é«˜ã�„ã�»ã�©é›£ã�—ã�„) ---")
    print(fold_summary)

    # Fold 1 ã�¨ã��ã‚Œä»¥å¤–ã�®èª¤å·®åˆ†å¸ƒã‚’æ¯”è¼ƒ
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df_analysis[df_analysis['fold'] == 1]['abs_error'], label='Fold 1 (Target)', fill=True)
    sns.kdeplot(df_analysis[df_analysis['fold'] != 1]['abs_error'], label='Other Folds', fill=True)
    plt.title('Error Distribution: Fold 1 vs Others')
    plt.xlabel('Absolute Error')
    plt.legend()
    plt.show()

    # ==================================================
    # 3. Fold 1 ã�®ä¸­ã�®ã€Œè¶…é›£å•� (Worst 10)ã€�ã‚’ç‰¹å®š
    # ==================================================
    fold1_worst = df_analysis[df_analysis['fold'] == 1].sort_values('abs_error', ascending=False).head(10)
    print("\n--- Fold 1 ã�®ä¸­ã�§ãƒ¢ãƒ‡ãƒ«ã�Œå¤§ã��ã��å¤–ã�—ã�Ÿé¡§å®¢ãƒˆãƒƒãƒ—10 ---")
    print(fold1_worst[['SK_ID_CURR', 'target', 'oof_pred', 'abs_error']])

    # ã�“ã�® ID ã‚’ä½¿ã�£ã�¦ã€�å…ƒã�® X ã�®ãƒ‡ãƒ¼ã‚¿ã‚’ç¢ºèª�ã�™ã‚‹ã�®ã�Œæ¬¡ã�®ã‚¹ãƒ†ãƒƒãƒ—ã�§ã�™
    worst_ids = fold1_worst['SK_ID_CURR'].tolist()

# ==================================================
# 4. å…ƒãƒ‡ãƒ¼ã‚¿ X ã�¨ç´�ä»˜ã�‘ã�¦ã€Œã�ªã�œï¼Ÿã€�ã‚’è€ƒã�ˆã‚‹
# ==================================================
def analyze_feature_in_fold1(feature_name):
    """
    Fold 1 ã�®ã€Œé›£å•�ã€�ã�Ÿã�¡ã�®ç‰¹å¾´é‡�ã�Œã€�å…¨ä½“ã�¨ã�©ã�†é�•ã�†ã�‹ã‚’è¦‹ã‚‹
    """
    if 'X' not in globals():
        print("ç‰¹å¾´é‡�ãƒ‡ãƒ¼ã‚¿ X ã�Œãƒ¡ãƒ¢ãƒªã�«ã�‚ã‚Šã�¾ã�›ã‚“ã€‚")
        return

    plt.figure(figsize=(12, 6))
    # å…¨ä½“ã�®åˆ†å¸ƒ
    sns.kdeplot(X[feature_name], label='All Data', fill=True, alpha=0.3)
    # Fold 1 ã�®é›£å•�ã�Ÿã�¡ã�®å€¤
    worst_values = X.loc[worst_ids, feature_name] if all(idx in X.index for idx in worst_ids) else []
    
    for val in worst_values:
        plt.axvline(val, color='red', linestyle='--', alpha=0.6)
    
    plt.title(f'Feature Analysis: {feature_name} (Red lines = Fold 1 Hard Samples)')
    plt.legend()
    plt.show()

# å®Ÿè¡Œä¾‹: 
analyze_feature_in_fold1('EXT_SOURCE_2')


def analyze_worst_10_features(features_to_check):
    """
    ãƒ¯ãƒ¼ã‚¹ãƒˆ10äººã�®ç‰¹å¾´é‡�ã�¨ã€�å…¨ä½“ã�®å¹³å�‡å€¤ã‚’æ¯”è¼ƒã�™ã‚‹
    """
    if 'X' not in globals():
        print("ç‰¹å¾´é‡�ãƒ‡ãƒ¼ã‚¿ X ã�Œãƒ¡ãƒ¢ãƒªã�«ã�‚ã‚Šã�¾ã�›ã‚“ã€‚")
        return

    # å…¨ä½“ã�®å¹³å�‡
    overall_mean = X[features_to_check].mean()
    
    # ãƒ¯ãƒ¼ã‚¹ãƒˆ10äººã�®ãƒ‡ãƒ¼ã‚¿
    worst_data = X.loc[worst_ids, features_to_check]
    worst_mean = worst_data.mean()
    
    # æ¯”è¼ƒç”¨ã�®è¡¨ã‚’ä½œæˆ�
    comparison_df = pd.DataFrame({
        'Overall Mean': overall_mean,
        'Worst 10 Mean': worst_mean,
        'Difference (%)': ((worst_mean - overall_mean) / overall_mean) * 100
    })
    
    print("\n--- å…¨ä½“å¹³å�‡ vs è‹¦æ‰‹ã�ª10äººã�®å¹³å�‡ ---")
    print(comparison_df)
    
    # ã‚°ãƒ©ãƒ•ã�§å�¯è¦–åŒ–
    for feature in features_to_check:
        plt.figure(figsize=(10, 5))
        sns.kdeplot(X[feature], label='All Data', fill=True, alpha=0.3)
        for val in worst_data[feature]:
            plt.axvline(val, color='red', linestyle='--', alpha=0.6)
        plt.title(f'Focus Analysis: {feature} (Red = Worst 10 Samples)')
        plt.legend()
        plt.show()


# --- æ¬¡ã�«å®Ÿè¡Œã�™ã‚‹ã�¹ã��åˆ†æ��ä¾‹ ---
# ãƒ¢ãƒ‡ãƒ«ã�Œé‡�è¦–ã�—ã�¦ã�„ã‚‹ä¸»è¦�ã�ªç‰¹å¾´é‡�ã‚’ãƒªã‚¹ãƒˆã‚¢ãƒƒãƒ—ã�—ã�¦å®Ÿè¡Œã�—ã�¦ã��ã� ã�•ã�„
check_features = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']#'DAYS_BIRTH', 'AMT_INCOME_TOTAL'
analyze_worst_10_features(check_features)


def deep_dive_high_ext_outliers():
    """
    EXT_SOURCEã�Œé«˜ã�„ã�®ã�«ãƒ‡ãƒ•ã‚©ãƒ«ãƒˆã�—ã�Ÿ10äºº (False Negative) ã�¨ã€�
    EXT_SOURCEã�Œé«˜ã��ã�¦å®Œæ¸ˆã�—ã�Ÿäºº (True Negative) ã‚’æ¯”è¼ƒã�—ã�¦ã€�
    ãƒ‡ãƒ•ã‚©ãƒ«ãƒˆã�®ã€�éš ã‚Œã�Ÿè¦�å› ã€�ã‚’ç‰¹å®šã�™ã‚‹ã€‚
    """
    if 'X' not in globals(): return

    # 1. æ¯”è¼ƒå¯¾è±¡ï¼ˆæœ¬ç‰©ã�®å„ªè‰¯å®¢ï¼šEXT1 > 0.5 ä¸”ã�¤ target=0ï¼‰ã‚’æŠ½å‡º
    # â€» æ•°å€¤ã�¯ä»Šå›�ã�®åˆ†æ��çµ�æ�œ (0.56) ã�«å�ˆã‚�ã�›ã�¦èª¿æ•´
    true_safe_idx = df_analysis[(df_analysis['target'] == 0) & (df_analysis['oof_pred'] < 0.05)].index
    
    # 2. èª¿æŸ»ã�—ã�Ÿã�„ã€Œéš ã‚Œã�Ÿãƒªã‚¹ã‚¯ã€�ã�«ã�ªã‚Šã��ã�†ã�ªç‰¹å¾´é‡�ãƒªã‚¹ãƒˆ
    # å��å…¥ã�«å¯¾ã�™ã‚‹å€Ÿå…¥é¡�ã€�æ”¯æ‰•é�…å»¶(DPD)ã€�å‹¤ç¶šå¹´æ•°ã€�å±…ä½�å½¢æ…‹ã�ªã�©
    risk_features = [c for c in X.columns if any(k in c for k in ['DPD', 'DBD', 'AMT_CREDIT', 'DAYS_EMPLOYED', 'PAYMENT'])]
    
    # 3. 10äººã�¨æ¯”è¼ƒå¯¾è±¡ã�®å¹³å�‡ã‚’æ¯”è¼ƒ
    false_neg_mean = X.loc[worst_ids, risk_features].mean()
    true_neg_mean = X.loc[true_safe_idx, risk_features].mean()
    
    comparison = pd.DataFrame({
        'Fake Safe (Worst 10)': false_neg_mean,
        'True Safe (Average)': true_neg_mean,
        'Ratio (Fake/True)': false_neg_mean / true_neg_mean
    }).sort_values('Ratio (Fake/True)', ascending=False)
    
    print("\n--- æ“¬ä¼¼å„ªè‰¯å®¢ vs æœ¬ç‰©ã�®å„ªè‰¯å®¢ï¼šéš ã‚Œã�Ÿãƒªã‚¹ã‚¯è¦�å›  ---")
    print("Ratioã�Œ1ã‚ˆã‚Šæ¥µç«¯ã�«å¤§ã��ã�„/å°�ã�•ã�„é …ç›®ã�«æ³¨ç›®ã�—ã�¦ã��ã� ã�•ã�„")
    print(comparison.head(20)) # å·®ã�Œå¤§ã��ã�„é …ç›®ã‚’è¡¨ç¤º

# --- æ¬¡ã�«å®Ÿè¡Œã�™ã‚‹ã�¹ã��åˆ†æ�� ---
deep_dive_high_ext_outliers()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

# ==================================================
# 1. æ��å‡ºãƒ•ã‚¡ã‚¤ãƒ«ã�®ä½œæˆ� (ã‚¨ãƒ©ãƒ¼å›�é�¿ç‰ˆ)
# ==================================================
def save_final_submission(test_preds, test_df_processed, mean_auc):
    """
    sample_submission.csv ã‚’ä½¿ã‚�ã�šã€�åŠ å·¥æ¸ˆã�¿ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿ã�®
    SK_ID_CURR åˆ—ã‚’ç›´æ�¥ä½¿ã�£ã�¦æ��å‡ºãƒ•ã‚¡ã‚¤ãƒ«ã‚’ä½œæˆ�ã�™ã‚‹
    """
    filename = f"submission_KFold_final_{mean_auc:.5f}.csv"
    
    # ä»¥å‰�æˆ�åŠŸã�—ã�Ÿãƒ‘ã‚¿ãƒ¼ãƒ³ã‚’å†�ç�¾
    submission = pd.DataFrame({
        "SK_ID_CURR": test_df_processed["SK_ID_CURR"],
        "TARGET": test_preds
    })
    
    submission.to_csv(filename, index=False)
    print(f"--- æ��å‡ºãƒ•ã‚¡ã‚¤ãƒ«ä½œæˆ�å®Œäº† ---")
    print(f"ãƒ•ã‚¡ã‚¤ãƒ«å��: {filename}")
    print(f"å¹³å�‡ CV AUC: {mean_auc:.6f}")

# ==================================================
# 2. Foldã�”ã�¨ã�®ã‚¨ãƒ©ãƒ¼åˆ†æ�� (ã�ªã�œFold 1ã�¯é›£ã�—ã�‹ã�£ã�Ÿã�®ã�‹ï¼Ÿ)
# ==================================================
def run_error_analysis(y_true, oof_preds, fold_indices):
    """
    oof_preds: å­¦ç¿’ãƒ‡ãƒ¼ã‚¿å…¨ä»¶ã�«å¯¾ã�™ã‚‹ã€�ã��ã‚Œã��ã‚Œã�®æ¤œè¨¼æ™‚ã�®äºˆæ¸¬å€¤
    fold_indices: å�„ãƒ‡ãƒ¼ã‚¿ã�Œã�©ã�®Foldã�«å±�ã�—ã�¦ã�„ã�Ÿã�‹ã�®ãƒªã‚¹ãƒˆ
    """
    results = pd.DataFrame({
        'target': y_true,
        'pred': oof_preds,
        'abs_error': abs(y_true - oof_preds),
        'fold': fold_indices
    })

    # Foldã�”ã�¨ã�®å¹³å�‡ã‚¨ãƒ©ãƒ¼ã�¨ãƒ‡ãƒ•ã‚©ãƒ«ãƒˆç�‡ã‚’ç¢ºèª�
    fold_summary = results.groupby('fold').agg({
        'target': 'mean',
        'abs_error': 'mean',
        'pred': 'count'
    }).rename(columns={'target': 'Default_Rate', 'pred': 'Sample_Count'})
    
    print("\n--- Foldåˆ¥ã‚µãƒ�ãƒªãƒ¼ ---")
    print(fold_summary)
    
    # ç‰¹ã�«äºˆæ¸¬ã‚’å¤–ã�—ã�Ÿã€Œé›£å•�ãƒ‡ãƒ¼ã‚¿ã€�ã�Œ Fold 1 ã�«ã�©ã‚Œã��ã‚‰ã�„ã�‚ã‚‹ã�‹
    hard_samples = results[results['abs_error'] > 0.8]
    print(f"\nå…¨ãƒ‡ãƒ¼ã‚¿ä¸­ã�®å¤§å¤–ã‚Œæ•°: {len(hard_samples)}")
    print(f"Fold 1 ã�®å¤§å¤–ã‚Œæ•°: {len(hard_samples[hard_samples['fold'] == 1])}")
    
    return results

# ==================================================
# å®Ÿè¡Œã‚³ãƒ�ãƒ³ãƒ‰
# ==================================================
# å­¦ç¿’ã�Œçµ‚ã‚�ã�£ã�Ÿç›´å¾Œã�®ã‚»ãƒ«ã�§ã€�ä»¥ä¸‹ã�®1è¡Œã‚’å®Ÿè¡Œã�—ã�¦ã��ã� ã�•ã�„ã€‚
# â€» mean_auc ã�¯ãƒ«ãƒ¼ãƒ—ã�®æœ€å¾Œã�§è¨ˆç®—ã�—ã�Ÿå¹³å�‡å€¤ã‚’ä½¿ã�„ã�¾ã�™ã€‚

save_final_submission(test_preds, test_df_processed, mean_auc)

# åˆ†æ��ã‚‚ã�—ã�Ÿã�„å ´å�ˆã�¯ã�“ã�¡ã‚‰ï¼ˆoof_predsã�¨fold_indicesã�Œå¿…è¦�ã�§ã�™ï¼‰
# analysis_df = run_error_analysis(y, oof_preds, fold_indices)

