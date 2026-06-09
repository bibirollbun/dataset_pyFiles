import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score,  GridSearchCV
from sklearn.metrics import roc_auc_score
from scipy.stats import uniform, randint
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans

from warnings import filterwarnings
filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


train.head()


test.head()


train.describe().T


test.describe().T


train.isnull().sum()


test.isnull().sum()


train["rainfall"].value_counts()


def get_season(day):
    if 80 <= day <= 171:
        return "spring"
    elif 172 <= day <= 263:
        return "summer"
    elif 264 <= day <= 354:
        return "fall"
    else:
        return "winter"
        
train["season"] = train["day"].apply(get_season)
test["season"] = test["day"].apply(get_season)

train["temp_range"] = train["maxtemp"] - train["mintemp"]
test["temp_range"] = test["maxtemp"] - test["mintemp"]

train["dew_humidity_ratio"] = train["dewpoint"] / (train["humidity"] + 1e-5)
test["dew_humidity_ratio"] = test["dewpoint"] / (test["humidity"] + 1e-5)

train["temp_dew_diff"] = train["temparature"] - train["dewpoint"]
test["temp_dew_diff"] = test["temparature"] - test["dewpoint"]

train["cloud_sun_ratio"] = train["cloud"] / (train["sunshine"] + 1e-5)
test["cloud_sun_ratio"] = test["cloud"] / (test["sunshine"] + 1e-5)

train["low_sun"] = (train["sunshine"] < 1).astype(int)
test["low_sun"] = (test["sunshine"] < 1).astype(int)

train["cloud_humidity"] = train["humidity"] * train["cloud"]
test["cloud_humidity"] = test["humidity"] * test["cloud"]

train["temp_humidity"] = train["humidity"] * train["temp_dew_diff"]
test["temp_humidity"] = test["humidity"] * test["temp_dew_diff"]

season_map = {"winter": 0, "spring": 1, "summer": 2, "fall": 3}

train["season_num"] = train["season"].map(season_map)
test["season_num"] = test["season"].map(season_map)

train["cloud_sun_season"] = train["cloud_sun_ratio"] * train["season_num"]
test["cloud_sun_season"] = test["cloud_sun_ratio"] * test["season_num"]

train["cloud_sun_intersect"] = train["cloud"] * train["sunshine"]
test["cloud_sun_intersect"] = test["cloud"] * test["sunshine"]

train["cloud_humidity_intersect"] = train["cloud"] * train["humidity"]
test["cloud_humidity_intersect"] = test["cloud"] * test["humidity"]

train["cloud_sun_intersect"] = train["cloud"] / (train["sunshine"] + 1e-3)
test["cloud_sun_intersect"] = test["cloud"] / (test["sunshine"] + 1e-3)

train["humidity_dewpoint_intersect"] = train["humidity"] * train["dewpoint"]
test["humidity_dewpoint_intersect"] = test["humidity"] * test["dewpoint"]

train["sun_wind_intersect"] = train["sunshine"] / (train["windspeed"] + 1e-3)
test["sun_wind_intersect"] = test["sunshine"] / (test["windspeed"] + 1e-3)

train["cloud_low_sun_intersect"] = train["cloud"] * train["low_sun"]
test["cloud_low_sun_intersect"] = test["cloud"] * test["low_sun"]

bool_cols = train.select_dtypes(include='bool').columns

for col in bool_cols:
    train[col] = train[col].astype(int)
    test[col] = test[col].astype(int)
    
train = train.drop(["season"],axis = 1)
test = test.drop(["season"], axis = 1)


test["winddirection"].fillna(test["winddirection"].mean(), inplace=True)


X = train.drop(["id", "rainfall"], axis=1)
y = train["rainfall"]

X_test = test.drop(["id"],axis = 1)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)


y_train=y
X=X_scaled


import lightgbm as lgb
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization
import numpy as np

class_0 = y_train.sum()
class_1 = len(y_train) - class_0
scale_pos_weight = class_1 / class_0
print(scale_pos_weight)

def lgb_cv_score(
    num_leaves,
    learning_rate,
    n_estimators,
    max_depth,
    min_child_samples,
    subsample,
    colsample_bytree,
    reg_alpha,
    reg_lambda
):
    """
    è´�å�¶æ–¯ä¼˜åŒ–çš„ç›®æ ‡å‡½æ•°ã€‚
    è¾“å…¥ï¼šè¶…å�‚æ•°
    è¾“å‡ºï¼šäº¤å�‰éªŒè¯�çš„F1-scoreå�‡å€¼
    """
    # å°†è¶…å�‚æ•°è½¬æ�¢ä¸ºæ•´æ•°æˆ–æµ®ç‚¹æ•°
    params = {
        'objective': 'binary',
        'metric': 'f1',
        'num_leaves': int(num_leaves),
        'learning_rate': learning_rate,
        'n_estimators': int(n_estimators),
        'max_depth': int(max_depth),
        'min_child_samples': int(min_child_samples),
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,
        "verbose": -1,  # éš�è—�ä¸�å¿…è¦�çš„è­¦å‘Š
        'n_jobs': -1,
        'seed': 42
    }
    
    # ä½¿ç”¨äº¤å�‰éªŒè¯�è¯„ä¼°æ¨¡å�‹æ€§èƒ½
    model = lgb.LGBMClassifier(**params,class_weight={0: scale_pos_weight, 1: 1})
    scores = cross_val_score(model, X, y, cv=5, scoring='f1', error_score='raise')
    
    # è´�å�¶æ–¯ä¼˜åŒ–ä¼šå°�è¯•æœ€å¤§åŒ–è¿™ä¸ªè¿”å›�å€¼
    return np.mean(scores)

# å®šä¹‰æ¯�ä¸ªè¶…å�‚æ•°çš„æ�œç´¢èŒƒå›´ï¼ˆæµ®ç‚¹æ•°æˆ–æ•´æ•°ï¼‰
pbounds = {
    'num_leaves': (20, 100),
    'learning_rate': (0.01, 0.2),
    'n_estimators': (100, 1000),
    'max_depth': (5, 20),
    'min_child_samples': (10,100),
    'subsample': (0.5, 1.0),
    'colsample_bytree': (0.5, 1.0),
    'reg_alpha': (0.01, 1.0),
    'reg_lambda': (0.01, 1.0)
}


# åˆ›å»ºè´�å�¶æ–¯ä¼˜åŒ–å™¨å®�ä¾‹
optimizer = BayesianOptimization(
    f=lgb_cv_score,  # æˆ‘ä»¬çš„ç›®æ ‡å‡½æ•°
    pbounds=pbounds, # è¶…å�‚æ•°æ�œç´¢èŒƒå›´
    random_state=42,
    verbose=2 # è®¾ç½®ä¸º1æˆ–0å�¯ä»¥å‡�å°‘è¾“å‡º
)

# è¿�è¡Œä¼˜åŒ–è¿‡ç¨‹
# n_iter: è¿­ä»£æ¬¡æ•°ï¼Œå�³è°ƒç”¨ç›®æ ‡å‡½æ•°çš„æ¬¡æ•°
# init_points: åˆ�å§‹éš�æœºæ�¢ç´¢çš„æ¬¡æ•°
optimizer.maximize(init_points=5, n_iter=20)

# æ‰“å�°æœ€ä½³ç»“æ�œ
print("æœ€ä½³å�‚æ•°ç»„å�ˆï¼š", optimizer.max['params'])
print("æœ€ä½³F1-scoreï¼š", optimizer.max['target'])


import lightgbm as lgb
from sklearn.datasets import make_classification
import pandas as pd


test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
test_ID = test['id']
# ä¹‹å‰�å®šä¹‰çš„æ¨¡å�‹å�‚æ•°
params = {
    'objective': 'binary',
    'metric': 'f1',
    'num_leaves': 76, 
    'learning_rate': 0.013911053916202464, 
    'n_estimators': 972, 
    'max_depth': 17, 
    'min_child_samples':29,
    'subsample': 0.5909124836035503, 
    'colsample_bytree': 0.5917022549267169, 
    'reg_alpha': 0.31119982052994233, 
    'reg_lambda': 0.5295088673159155,
    "verbose": -1,  # éš�è—�ä¸�å¿…è¦�çš„è­¦å‘Š
    'n_jobs': -1,
    'seed': 42
}

# å®�ä¾‹åŒ–LightGBMåˆ†ç±»å™¨
model = lgb.LGBMClassifier(**params)

# --- æ¨¡å�‹è®­ç»ƒ ---
# ä½¿ç”¨ .fit() æ–¹æ³•åœ¨è®­ç»ƒæ•°æ�®ä¸Šè®­ç»ƒæ¨¡å�‹
print("å¼€å§‹è®­ç»ƒæ¨¡å�‹...")
model.fit(X, y)
print("æ¨¡å�‹è®­ç»ƒå®Œæˆ�ã€‚")

# --- è¿›è¡Œé¢„æµ‹ ---
# ä½¿ç”¨è®­ç»ƒå¥½çš„æ¨¡å�‹å¯¹ X_test è¿›è¡Œé¢„æµ‹
# .predict() æ–¹æ³•è¿”å›�ç±»åˆ«é¢„æµ‹ (0 æˆ– 1)
print("å¼€å§‹å¯¹æµ‹è¯•é›†è¿›è¡Œé¢„æµ‹...")
predictions = model.predict_proba(X_test_scaled)[:, 1]
print("é¢„æµ‹å®Œæˆ�ã€‚")

# --- ä¿�å­˜åˆ° submission.csv ---
# å°†é¢„æµ‹ç»“æ�œä¿�å­˜ä¸º .csv æ–‡ä»¶
# åˆ›å»ºä¸€ä¸ªåŒ…å�«é¢„æµ‹ç»“æ�œçš„ Pandas DataFrame
submission_df = pd.DataFrame({
    'id': test_ID,  # å�‡è®¾æµ‹è¯•æ•°æ�®æœ‰IDï¼Œè¿™é‡Œç”¨ç´¢å¼•ä½œä¸ºç¤ºä¾‹
    'rainfall': predictions
})

submission_df.to_csv('submission.csv', index=False)

print("é¢„æµ‹ç»“æ�œå·²æˆ�åŠŸä¿�å­˜åˆ° submission.csv æ–‡ä»¶ä¸­ã€‚")

