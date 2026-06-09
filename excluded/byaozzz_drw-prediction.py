# é¡¹ç›®ï¼šDRW Crypto Market Prediction
# ç¼–å†™è€…ï¼šå‘¨èµ‹ç‘¶

# 1. å¯¼å…¥æ ¸å¿ƒå·¥å…·åŒ…
import pandas as pd  # æ•°æ�®å¤„ç�†ä¸»åŠ›åº“
import numpy as np   # æ•°å€¼è®¡ç®—æ”¯æŒ�
import matplotlib.pyplot as plt  # å�¯è§†åŒ–
import seaborn as sns  # é«˜çº§å�¯è§†åŒ–è¾…åŠ©
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import optuna  # è¶…å�‚æ•°ä¼˜åŒ–åº“
import warnings
warnings.filterwarnings("ignore")


# ç¼–å†™è€…ï¼šå‘¨èµ‹ç‘¶
# 2. è¯»å�–è®­ç»ƒé›†
data_path = '/kaggle/input/drw-crypto-market-prediction/train.parquet'
df = pd.read_parquet(data_path)  # è¯»å�– parquet æ–‡ä»¶æ ¼å¼�
print("æ•°æ�®ç»´åº¦:", df.shape)
print("å‰�å‡ åˆ—å��ç§°:", df.columns[:10].tolist())
print(df.dtypes.head())  # æ˜¾ç¤ºå‰�å‡ åˆ—æ•°æ�®ç±»å�‹


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


# ç¼–å†™è€…ï¼šå‘¨èµ‹ç‘¶
# 11. ä½¿ç”¨ Optuna è°ƒä¼˜ CatBoost
def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 150, 300),
        'depth': trial.suggest_int('depth', 5, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.3),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 9),
        'random_strength': trial.suggest_float('random_strength', 0.0, 1.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'verbose': 0,
        'random_seed': 0
    }
    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return mean_squared_error(y_val, preds, squared=False)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

# è¾“å‡ºæœ€ä¼˜å�‚æ•°
print("æœ€ä½³å�‚æ•°:", study.best_params)
print("æœ€å°� RMSE:", study.best_value)

# ç”¨æœ€ä½³å�‚æ•°é‡�æ–°è®­ç»ƒ CatBoost
best_cat = CatBoostRegressor(**study.best_params, verbose=0).fit(X_train, y_train)
evaluate_model("CatBoost + Optuna", y_val, best_cat.predict(X_val))


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
kaggle_preds = best_cat.predict(X_kaggle)

# æ��äº¤æ–‡ä»¶
submission = pd.DataFrame({
    "row_id": test["row_id"] if "row_id" in test.columns else test.index,
    "target": kaggle_preds
})
submission.to_csv("submission.csv", index=False)

