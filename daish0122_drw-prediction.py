# é¡¹ç›®ï¼šDRW Crypto Market Prediction
# ç¼–å†™è€…ï¼šå‘¨èµ‹ç‘¶

# 1. å¯¼å…¥æ ¸å¿ƒå·¥å…·åŒ…
import pandas as pd  # æ•°æ�®å¤„ç�†ä¸»åŠ›åº“
import numpy as np   # æ•°å€¼è®¡ç®—æ”¯æŒ�
import matplotlib.pyplot as plt  # å�¯è§†åŒ–
import seaborn as sns  # é«˜çº§å�¯è§†åŒ–è¾…åŠ©
from sklearn.model_selection import train_test_split  # ä»�scikit-learnå¯¼å…¥æ•°æ�®é›†åˆ†å‰²å·¥å…·ï¼Œç”¨äº�åˆ’åˆ†è®­ç»ƒé›†å’Œæµ‹è¯•é›†
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # å¯¼å…¥å›�å½’è¯„ä¼°æŒ‡æ ‡ï¼šå�‡æ–¹è¯¯å·®ã€�å¹³å�‡ç»�å¯¹è¯¯å·®ã€�RÂ²åˆ†æ•°
from sklearn.feature_selection import SelectKBest, f_regression  # å¯¼å…¥ç‰¹å¾�é€‰æ‹©å·¥å…·ï¼šSelectKBestå’Œf_regression
import optuna  # è¶…å�‚æ•°ä¼˜åŒ–åº“
import warnings  # å¯¼å…¥Pythonè­¦å‘Šæ¨¡å�—ï¼Œç”¨äº�æ�§åˆ¶è­¦å‘Šä¿¡æ�¯çš„å¤„ç�†
warnings.filterwarnings("ignore")  # è®¾ç½®å¿½ç•¥æ‰€æœ‰è­¦å‘Šä¿¡æ�¯ï¼Œé�¿å…�å¹²æ‰°è¾“å‡ºç»“æ�œ


# ç¼–å†™è€…ï¼šå‘¨èµ‹ç‘¶
# 2. è¯»å�–è®­ç»ƒé›†
# æŒ‡å®šè®­ç»ƒæ•°æ�®æ–‡ä»¶è·¯å¾„
data_path = '/kaggle/input/drw-crypto-market-prediction/train.parquet'
df = pd.read_parquet(data_path)  # è¯»å�– parquet æ–‡ä»¶æ ¼å¼�

# è¾“å‡ºæ•°æ�®é›†ç»´åº¦ä¿¡æ�¯
print("æ•°æ�®ç»´åº¦:", df.shape) 
# è¾“å‡ºå‰�10åˆ—çš„å��ç§°
# columnså±�æ€§è¿”å›�æ‰€æœ‰åˆ—å��ï¼Œ[:10]åˆ‡ç‰‡å�–å‰�10ä¸ªï¼Œtolist()è½¬æ�¢ä¸ºåˆ—è¡¨æ ¼å¼�
print("å‰�å‡ åˆ—å��ç§°:", df.columns[:10].tolist())

# dtypesè¿”å›�æ¯�åˆ—çš„æ•°æ�®ç±»å�‹ï¼Œhead()é™�åˆ¶è¾“å‡ºè¡Œæ•°é�¿å…�æ�§åˆ¶å�°ä¿¡æ�¯è¿‡é•¿
print(df.dtypes.head())  # æ˜¾ç¤ºDataFrameå‰�5åˆ—çš„æ•°æ�®ç±»å�‹ä¿¡æ�¯


# ç¼–å†™è€…ï¼šå‘¨èµ‹ç‘¶
# 3. ç¼ºå¤±å€¼ / é‡�å¤�å€¼ æ£€æŸ¥
print("\n ç¼ºå¤±å€¼ç»Ÿè®¡ï¼ˆä»…æ˜¾ç¤ºå­˜åœ¨ç¼ºå¤±çš„åˆ—ï¼‰:")
print(df.isna().sum()[df.isna().sum() > 0])
print(" é‡�å¤�è¡Œæ•°é‡�:", df.duplicated().sum())


# ç¼–å†™è€…ï¼šå‘¨èµ‹ç‘¶
# 4. å¼‚å¸¸å€¼å¤„ç�†ï¼ˆinf / -infï¼‰
inf_cols = df.columns[np.isinf(df).any()]
df[inf_cols] = df[inf_cols].replace([np.inf, -np.inf], 0)  # æ›¿æ�¢ä¸º 0
print("æ›¿æ�¢ inf çš„åˆ—æ•°:", len(inf_cols))


# ç¼–å†™è€…ï¼šå‘¨èµ‹ç‘¶
# 5. å�¯è§†åŒ–åˆ†æ��

import matplotlib.pyplot as plt
from matplotlib import font_manager
from statsmodels.tsa.stattools import adfuller

# åŠ è½½å¹¶æ³¨å†Œ SimHei å­—ä½“
simhei_path = "/kaggle/input/simhei/SIMHEI.TTF"
font_prop = font_manager.FontProperties(fname=simhei_path)

# è®¾ç½® matplotlib æ­£ç¡®æ˜¾ç¤ºè´Ÿå�·
plt.rcParams['axes.unicode_minus'] = False

# ç”»ç›®æ ‡æ”¶ç›Šçš„æ—¶åº�å›¾
plt.figure(figsize=(10, 3))
plt.plot(df['label'], color='blue', label='ç›®æ ‡æ”¶ç›Š')

# ä¸­æ–‡æ ‡é¢˜ä½¿ç”¨ fontproperties æŒ‡å®šå­—ä½“
plt.title("Label æ—¶åº�èµ°åŠ¿", fontproperties=font_prop)
plt.legend(prop=font_prop)  # å›¾ä¾‹ä¹Ÿè¦�æŒ‡å®šå­—ä½“
plt.tight_layout()
plt.show()

# æ»šåŠ¨å�‡å€¼å’Œæ ‡å‡†å·®
roll_mean = df['label'].rolling(window=500).mean()
roll_std = df['label'].rolling(window=500).std()

plt.figure(figsize=(10, 3))
plt.plot(df['label'], label='å�Ÿå§‹', alpha=0.5)
plt.plot(roll_mean, label='æ»‘åŠ¨å�‡å€¼', color='green')
plt.plot(roll_std, label='æ»‘åŠ¨æ ‡å‡†å·®', color='orange')
plt.title("Label æ»šåŠ¨ç»Ÿè®¡åˆ†æ��",fontproperties=font_prop)
plt.legend(prop=font_prop)
plt.tight_layout()
plt.show()



# ç¼–å†™è€…ï¼šå‘¨èµ‹ç‘¶
# 6. ç‰¹å¾�å·¥ç¨‹ï¼ˆLag, Rolling, å¸‚åœºè¡Œä¸ºï¼‰
df['label_lag_1'] = df['label'].shift(1)
df['label_lag_2'] = df['label'].shift(2)
df['label_lag_3'] = df['label'].shift(3)
df['label_lag_5'] = df['label'].shift(5)

df['label_roll_mean_3'] = df['label'].rolling(3).mean()
df['label_roll_mean_5'] = df['label'].rolling(5).mean()
df['label_roll_std_3'] = df['label'].rolling(3).std()
df['label_roll_std_5'] = df['label'].rolling(5).std()

# å¾®ç»“æ�„ç‰¹å¾�
df['volume_diff'] = df['volume'].diff()
df['buy_sell_diff'] = df['buy_qty'] - df['sell_qty']
df['buy_sell_ratio'] = df['buy_qty'] / (df['sell_qty'] + 1e-6)
df['buy_volume_ratio'] = df['buy_qty'] / (df['volume'] + 1e-6)
df['sell_volume_ratio'] = df['sell_qty'] / (df['volume'] + 1e-6)

# åˆ é™¤å›  shift å’Œ rolling äº§ç”Ÿçš„ç¼ºå¤±å€¼
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)


# ç¼–å†™è€…ï¼šå‘¨èµ‹ç‘¶
# 7. ç‰¹å¾�é€‰æ‹©ï¼ˆä¿�ç•™æœ€æœ‰ç”¨çš„100ä¸ªç‰¹å¾�ï¼‰
y = df['label']  # ç›®æ ‡å�˜é‡�
X = df.drop(columns=['label'])  # æ‰€æœ‰å…¶ä»–å�˜é‡�ä½œä¸ºè¾“å…¥ç‰¹å¾�

selector = SelectKBest(score_func=f_regression, k=100)  # é€‰æ‹©å‰�100ä¸ªä¸� y ç›¸å…³æ€§æœ€é«˜çš„ç‰¹å¾�
X_selected = selector.fit_transform(X, y)
selected_columns = X.columns[selector.get_support()]  # æ��å�–è¢«é€‰ä¸­çš„åˆ—å��
X = X[selected_columns]


# ç¼–å†™è€…ï¼šå‘¨èµ‹ç‘¶
# 8. æ‹†åˆ†è®­ç»ƒ/éªŒè¯�é›†
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)


# ç¼–å†™è€…ï¼šå‘¨èµ‹ç‘¶
# 9. å®šä¹‰æ¨¡å�‹è¯„ä¼°å‡½æ•°
def evaluate_model(name, y_true, y_pred):
    print(f"\nğŸ“Š æ¨¡å�‹è¯„ä¼°: {name}")
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
    print("R2 Score:", r2_score(y_true, y_pred))
    print("-" * 40)


# ç¼–å†™è€…ï¼šå‘¨èµ‹ç‘¶
# 10. è®­ç»ƒå¤šä¸ªæ¨¡å�‹å¯¹æ¯”è¡¨ç�°
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# çº¿æ€§å›�å½’
lr = LinearRegression().fit(X_train, y_train)
evaluate_model("çº¿æ€§å›�å½’", y_val, lr.predict(X_val))

# å†³ç­–æ ‘
tree = DecisionTreeRegressor(max_depth=10, random_state=0).fit(X_train, y_train)
evaluate_model("å†³ç­–æ ‘", y_val, tree.predict(X_val))

# XGBoost
xgb = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=0).fit(X_train, y_train)
evaluate_model("XGBoost", y_val, xgb.predict(X_val))

# LightGBM
lgb = LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=0).fit(X_train, y_train)
evaluate_model("LightGBM", y_val, lgb.predict(X_val))

# CatBoost é»˜è®¤å�‚æ•°
cat = CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6, verbose=0, random_seed=0).fit(X_train, y_train)
evaluate_model("CatBoost", y_val, cat.predict(X_val))


import pandas as pd
import numpy as np
import optuna
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import time
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#  å®šä¹‰Optunaä¼˜åŒ–ç›®æ ‡å‡½æ•°
def xgb_objective(trial):
    params = {
        # æ ¸å¿ƒå�‚æ•°ï¼šä½¿ç”¨å¤šæ£µæ ‘å’Œå�ˆç�†çš„å­¦ä¹ ç�‡
        'n_estimators': trial.suggest_categorical('n_estimators', [100, 200, 300]),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        
        # æ ‘ç»“æ�„å�‚æ•°
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-5, 10),
        'gamma': trial.suggest_float('gamma', 0, 1),
        
        # æ­£åˆ™åŒ–å�‚æ•°
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 5),
        
        # é‡‡æ ·å�‚æ•°
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        
        # GPUä¼˜åŒ–
        'tree_method': 'gpu_hist',
        'gpu_id': 0,
        'random_state': 42,
        'verbosity': 0,
        'eval_metric': 'rmse'
    }
    
    # è‡ªåŠ¨æ—©å�œ
    early_stop = trial.suggest_int('early_stopping_rounds', 10, 50)
    
    model = XGBRegressor(**params)
    
    start_time = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=early_stop,
        verbose=False
    )
    train_time = time.time() - start_time
    
    # ä½¿ç”¨æœ€ä½³è¿­ä»£é¢„æµ‹
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    
    trial.set_user_attr('train_time', train_time)
    trial.set_user_attr('best_iteration', model.best_iteration)
    return rmse

# 4. åˆ›å»ºOptunaç ”ç©¶å¹¶ä¼˜åŒ–
study = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.TPESampler(
        n_startup_trials=10,
        seed=42
    ),
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=10
    )
)

# ä¼˜åŒ–å�‚æ•° - å¢�åŠ è¯•éªŒæ¬¡æ•°ä»¥è�·å¾—æ›´å¥½çš„ç»“æ�œ
study.optimize(
    xgb_objective, 
    n_trials=30,
    timeout=7200,
    show_progress_bar=True
)

# è¾“å‡ºæœ€ä¼˜å�‚æ•°
print("="*50)
print("æœ€ä½³å�‚æ•°:", study.best_params)
print("æœ€å°� RMSE:", study.best_value)
print(f"è®­ç»ƒæ—¶é—´: {study.best_trial.user_attrs['train_time']:.2f}ç§’")
print(f"æœ€ä½³è¿­ä»£æ¬¡æ•°: {study.best_trial.user_attrs['best_iteration']}")


# 5. ç”¨æœ€ä½³å�‚æ•°è®­ç»ƒæœ€ç»ˆæ¨¡å�‹
best_params = study.best_params.copy()
# ç§»é™¤æ—©å�œå�‚æ•°ï¼Œå› ä¸ºå®ƒä¸�æ˜¯æ¨¡å�‹æœ¬èº«çš„å�‚æ•°
best_params.pop('early_stopping_rounds', None)

# ä½¿ç”¨æœ€ä½³è¿­ä»£æ¬¡æ•°ä½œä¸ºn_estimators
if 'best_iteration' in study.best_trial.user_attrs:
    best_iteration = study.best_trial.user_attrs['best_iteration']
    best_params['n_estimators'] = best_iteration

final_model = XGBRegressor(
    **best_params,
    tree_method='gpu_hist',
    random_state=42
)

# å�ˆå¹¶è®­ç»ƒé›†å’ŒéªŒè¯�é›†è¿›è¡Œæœ€ç»ˆè®­ç»ƒ
X_full = np.vstack((X_train, X_val))
y_full = np.concatenate((y_train, y_val))

final_model.fit(X_full, y_full)

# è¯„ä¼°æœ€ç»ˆæ¨¡å�‹
val_preds = final_model.predict(X_val)
evaluate_model("æœ€ç»ˆæ¨¡å�‹éªŒè¯�é›†", y_val, val_preds)

# ä¿�å­˜æ¨¡å�‹
final_model.save_model('optimized_xgb_model.json')
joblib.dump(final_model, 'optimized_xgb_model.pkl')


# ç¼–å†™è€…ï¼šå‘¨èµ‹ç‘¶
# 12. é¢„æµ‹æµ‹è¯•é›†å¹¶ç”Ÿæˆ�æ��äº¤æ–‡ä»¶
test = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')
test.reset_index(drop=True, inplace=True)

# åº”ç”¨ä¸�è®­ç»ƒä¸€è‡´çš„ç‰¹å¾�æ�„é€ é€»è¾‘ï¼ˆæ�’é™¤ä¸�èƒ½ç”¨ label å�šæ»�å��çš„éƒ¨åˆ†ï¼‰
test['volume_diff'] = test['volume'].diff()
test['buy_sell_diff'] = test['buy_qty'] - test['sell_qty']
test['buy_sell_ratio'] = test['buy_qty'] / (test['sell_qty'] + 1e-6)
test['buy_volume_ratio'] = test['buy_qty'] / (test['volume'] + 1e-6)
test['sell_volume_ratio'] = test['sell_qty'] / (test['volume'] + 1e-6)

# æŒ‰ç…§ç‰¹å¾�é€‰æ‹©å™¨è¾“å‡ºçš„é¡ºåº�é€‰æ‹©åˆ—
X_kaggle = test.reindex(columns=selected_columns).fillna(0)
kaggle_preds = final_model.predict(X_kaggle)

# æ��äº¤æ–‡ä»¶
submission = pd.DataFrame({
    "ID": (test["row_id"] if "row_id" in test.columns else test.index)+1,
    "prediction": kaggle_preds
})
submission.to_csv("submission.csv", index=False)

