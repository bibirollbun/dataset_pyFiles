pip install --upgrade xgboost scikit-learn


import pandas as pd
import numpy as np
import lightgbm as lgb # æ›´æ”¹ç‚¹1: å¯¼å…¥ lightgbm
from sklearn.preprocessing import StandardScaler
from lightgbm import callback
from sklearn.model_selection import train_test_split
# æ³¨æ„�ï¼šæˆ‘ä»¬é€šå¸¸å�¯ä»¥ç›´æ�¥ä½¿ç”¨ lgb.LGBMRegressor æˆ– lgb.trainï¼Œæ— éœ€å�•ç‹¬å¯¼å…¥ XGBRegressor çš„å¯¹åº”ç‰©
# ğŸŒŸ å¼•å…¥ç¥�ç»�ç½‘ç»œå’Œæ¨¡å�‹è��å�ˆæ‰€éœ€çš„åº“
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.linear_model import Ridge # ç”¨äº�ç¬¬äºŒå±‚ Meta Model



df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
target = df.columns.tolist()[-1]
print(df.shape)
df.head()


def create_frequency_features(train_df, test_df, cols, num, cat):
    """
    Add frequency and binning features to the dataset.
    
    - For each column, create <col>_freq = how often each value appears in train data.
    - For numeric columns, split values into 5 and 10 quantile bins (groups) to show rank or range.
    """
    train, test = train_df.copy(), test_df.copy()

    for col in cols:
        # Frequency encoding: how common each value is
        freq = train[col].value_counts(normalize=True)
        train[f"{col}_freq"] = train[col].map(freq)
        test[f"{col}_freq"] = test[col].map(freq).fillna(train[f"{col}_freq"].mean())

        # Binning: group numeric values into quantiles
        if col in num:
            for q in [5, 10, 15]:
                try:
                    train[f"{col}_bin{q}"], bins = pd.qcut(train[col], q=q, labels=False, retbins=True, duplicates="drop")
                    test[f"{col}_bin{q}"] = pd.cut(test[col], bins=bins, labels=False, include_lowest=True)
                except Exception:
                    train[f"{col}_bin{q}"] = test[f"{col}_bin{q}"] = 0

    new_num = train.drop(columns=cat+[target]).columns.tolist()
    return train, test, new_num


# Identify feature
cols = df.drop(columns=target).columns.tolist()

# Categorical features
cat = [col for col in cols if df[col].dtype in ["object","category"] and col != target]

# Numerical features
num = [col for col in cols if df[col].dtype not in ["object","category","bool"] and col not in ["id", target]]

# Creating new features based on the frequency of numerical features
df, df_test, new_num = create_frequency_features(df, df_test.copy(), cols, num, cat)

# Preparing categorical features
df[cat], df_test[cat] = df[cat].astype("category"), df_test[cat].astype("category")

# Mapping a column
map_col = "num_reported_accidents"
map_num_reported = {0:0, 1:0, 2:0, 3:2, 4:4, 5:3, 6:1, 7:0}
df[map_col] = df[map_col].map(map_num_reported)
df_test[map_col] = df_test[map_col].map(map_num_reported)

# Dropping unnecessary columns
# è§„åˆ™1ï¼šæ°¸è¿œä¸�è¦�ä¸¢å¼ƒå�Ÿå§‹ç‰¹å¾�
# æˆ‘ä»¬å�¯ä»¥ä¿�ç•™è‡ªåˆ›çš„ 'id_freq'ï¼Œçœ‹çœ‹æ¨¡å�‹æ˜¯å�¦ä¼šç”¨å®ƒ
# æˆ–è€…ä¸¢å¼ƒå®ƒï¼Œå› ä¸ºå®ƒæ˜¯ä¸€ä¸ªè‡ªåˆ›ç‰¹å¾�
remove = ["id_freq"] # <-- å�ªä¸¢å¼ƒè‡ªåˆ›çš„ç‰¹å¾� (æˆ–è€…ç•™ç©º: remove = [])
df = df.drop(columns=remove)
df_test = df_test.drop(columns=remove)

# Dropping ID and duplicates
df.drop(columns="id", inplace=True)
df.drop_duplicates(inplace=True)


# æ­¥éª¤ 2ï¼šåˆ›å»º Pairwise (æˆ�å¯¹) äº¤äº’ç‰¹å¾�
# æ‰¾å‡ºæ‰€æœ‰åˆ†ç±»ç‰¹å¾�ï¼ŒåŒ…æ‹¬ä½ åˆšæ‰�â€œæ•‘â€�å›�æ�¥çš„é‚£äº›
cat_cols = [col for col in df.columns if df[col].dtype.name in ['category', 'object', 'bool']]

# å�ªé€‰æ‹©å‡ ä¸ªé«˜åŸºæ•°ï¼ˆhigh-cardinalityï¼‰çš„åˆ†ç±»ç‰¹å¾�æ�¥äº¤äº’ï¼Œé˜²æ­¢ç»„å�ˆçˆ†ç‚¸
interaction_cols = ['road_type', 'lighting', 'weather', 'time_of_day', 'num_lanes']

from itertools import combinations

# åˆ›é€  Pairwise äº¤äº’
for col1, col2 in combinations(interaction_cols, 2):
    new_col_name = f'{col1}_{col2}'
    print(f"Creating feature: {new_col_name}")
    
    # å°†ä¸¤åˆ—è½¬ä¸ºå­—ç¬¦ä¸²ï¼Œç„¶å��å�ˆå¹¶
    df[new_col_name] = df[col1].astype(str) + '_' + df[col2].astype(str)
    df_test[new_col_name] = df_test[col1].astype(str) + '_' + df_test[col2].astype(str)
    
    # å°†æ–°åˆ—è½¬æ�¢ä¸º 'category' ç±»å�‹ï¼Œä»¥ä¾¿ LightGBM å¤„ç�†
    df[new_col_name] = df[new_col_name].astype('category')
    df_test[new_col_name] = df_test[new_col_name].astype('category')

# é‡�æ–°è¯†åˆ«æ‰€æœ‰ç‰¹å¾�ï¼Œä¸ºLGBMå’ŒNNå�šå‡†å¤‡
cat_features = [col for col in df.columns if df[col].dtype.name in ['category', 'object', 'bool'] and col != target]
num_features = [col for col in df.columns if col not in cat_features and col != target]

print(f"ç�°åœ¨æ€»å…±æœ‰ {len(df.columns)} ä¸ªç‰¹å¾�")


print(df.columns.tolist())


df.head()




# ğŸŒŸ Cell 8 (ä¿®æ”¹å��): å�ªå®šä¹‰æ¨¡å�‹

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. å®šä¹‰ç¥�ç»�ç½‘ç»œæ¨¡å�‹ (ä¿�æŒ�ä¸�å�˜)
def create_mlp_model(input_shape):
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_shape,)),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(1) 
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse') 
    return model

# 2. (åˆ é™¤å�Ÿæœ‰çš„æ‰€æœ‰æ•°æ�®å‡†å¤‡ã€�æ ‡å‡†åŒ–ã€�è®­ç»ƒå’Œé¢„æµ‹ä»£ç �)
#    (æˆ‘ä»¬å°†åœ¨ä¸‹é�¢çš„ OOF å¾ªç�¯ä¸­å¤„ç�†å®ƒä»¬)

print("NN model function 'create_mlp_model' is defined.")


# ğŸŒŸ Cell 9 (ä¿®æ”¹å��): å�ªå®šä¹‰ LightGBM å�‚æ•°

import lightgbm as lgb
from lightgbm import callback
import pandas as pd

# LightGBM å�‚æ•° (æ‰€æœ‰ç©ºæ ¼å’Œç¼©è¿›éƒ½å·²æ¸…ç�†)
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'n_estimators': 2000, # ğŸŒŸ è®¾ç½®ä¸€ä¸ªè¾ƒå¤§çš„å€¼, æˆ‘ä»¬å°†åœ¨CVä¸­ä½¿ç”¨æ—©å�œ
    'max_depth': 11,
    'learning_rate': 0.030,
    'subsample': 0.82,
    'colsample_bytree': 0.81,
    'min_child_weight': 3,
    'reg_alpha': 0.05,
    'reg_lambda': 0.1,
    'num_leaves': 200,
    'n_jobs': -1,
    'seed': 42
}

# (å�Ÿæœ‰çš„ dtrain, lgb.cv, best_round ç­‰ä»£ç �å…¨éƒ¨åˆ é™¤)
# (æˆ‘ä»¬å°†åœ¨ä¸‹é�¢çš„ OOF å¾ªç�¯ä¸­å¤„ç�†å®ƒä»¬)

print("LGBM parameters are defined.")


# ğŸŒŸ æ–°ä»£ç � (æ›¿æ�¢ Cell 10 å’Œ Cell 11):
# æ‰§è¡Œ Out-of-Fold (OOF) é¢„æµ‹å’Œ Stacking

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import lightgbm as lgb
import numpy as np

# --- 1. å‡†å¤‡æ•°æ�® ---

# å®Œæ•´è®­ç»ƒæ•°æ�® (LGBM ä½¿ç”¨)
X = df.drop(columns=target)
y = df[target]

# æµ‹è¯•æ•°æ�® (LGBM ä½¿ç”¨)
X_test_lgbm = df_test.drop(columns='id')

# å‡†å¤‡ NN ä½¿ç”¨çš„æ•°æ�® (éœ€è¦�ç¼©æ”¾)
X_nn = df.drop(columns=target).copy()
y_nn = df[target].copy()
X_test_nn = df_test.drop(columns='id').copy()

# è¯†åˆ«åˆ†ç±»ç‰¹å¾�å’Œæ•°å€¼ç‰¹å¾�
categorical_features = [col for col in X.columns if X[col].dtype.name in ['category', 'object', 'bool']]
numerical_features = [col for col in X.columns if col not in categorical_features]

# ğŸŒŸ ä¿®å¤�: ä¸º NN å‡†å¤‡æ•°æ�® (å°† category è½¬æ�¢ä¸º codes)
# è¿™åœ¨ä½ çš„å�Ÿå§‹ Cell 8 ä¸­å·²ç»�å�šäº†ï¼Œæˆ‘ä»¬åœ¨è¿™é‡Œé‡�å�šä¸€é��ä»¥ç¡®ä¿�
for col in categorical_features:
    X_nn[col] = X_nn[col].astype('category').cat.codes
    X_test_nn[col] = X_test_nn[col].astype('category').cat.codes

# å°† X_nn å’Œ X_test_nn è½¬æ�¢ä¸º float, å‡†å¤‡ç¼©æ”¾
X_nn = X_nn.astype(float)
X_test_nn = X_test_nn.astype(float)


# --- 2. è®¾ç½® K-Fold å’Œ OOF å­˜å‚¨ ---

N_SPLITS = 5 # ä½ å�¯ä»¥æ ¹æ�®éœ€è¦�è°ƒæ•´æŠ˜æ•° (ä¾‹å¦‚ 5 æˆ– 10)
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# å­˜å‚¨ OOF é¢„æµ‹ (ç”¨äº�è®­ç»ƒ Meta-Model)
oof_preds_lgbm = np.zeros(len(X))
oof_preds_nn = np.zeros(len(X))

# å­˜å‚¨æµ‹è¯•é›†é¢„æµ‹ (ç”¨äº�æœ€ç»ˆæ��äº¤)
test_preds_lgbm = np.zeros(len(X_test_lgbm))
test_preds_nn = np.zeros(len(X_test_nn))


# --- 3. æ‰§è¡Œ K-Fold å¾ªç�¯ ---

for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
    print(f"--- Fold {fold+1}/{N_SPLITS} ---")
    
    # --- A. ä¸º LGBM åˆ’åˆ†æ•°æ�® ---
    X_train_lgbm, X_val_lgbm = X.iloc[train_index], X.iloc[val_index]
    y_train_lgbm, y_val_lgbm = y.iloc[train_index], y.iloc[val_index]
    
    # --- B. ä¸º NN åˆ’åˆ†æ•°æ�® ---
    X_train_nn_fold, X_val_nn_fold = X_nn.iloc[train_index], X_nn.iloc[val_index]
    y_train_nn_fold, y_val_nn_fold = y_nn.iloc[train_index], y_nn.iloc[val_index]
    
    # ğŸŒŸ å…³é”®: åœ¨ Fold å†…éƒ¨è¿›è¡Œç¼©æ”¾ï¼Œé˜²æ­¢æ³„éœ²
    # æ³¨æ„�: æˆ‘ä»¬å�ªåœ¨æ•°å€¼ç‰¹å¾�ä¸Šè¿›è¡Œç¼©æ”¾ (åˆ†ç±»ç‰¹å¾�å·²ç»�æ˜¯ codes äº†)
    scaler = StandardScaler()
    X_train_nn_fold[numerical_features] = scaler.fit_transform(X_train_nn_fold[numerical_features])
    X_val_nn_fold[numerical_features] = scaler.transform(X_val_nn_fold[numerical_features])
    
    # å�Œæ ·ç¼©æ”¾æµ‹è¯•é›† (ä½¿ç”¨å½“å‰� fold çš„ scaler)
    X_test_nn_scaled = X_test_nn.copy() # å¤�åˆ¶ä»¥é˜²ä¿®æ”¹å�Ÿæ•°æ�®
    X_test_nn_scaled[numerical_features] = scaler.transform(X_test_nn_scaled[numerical_features])

    
    # --- C. è®­ç»ƒå’Œé¢„æµ‹ LGBM ---
    print("Training LGBM...")
    lgbm_model = lgb.LGBMRegressor(**lgb_params, random_state=42)
    lgbm_model.fit(
        X_train_lgbm, y_train_lgbm,
        eval_set=[(X_val_lgbm, y_val_lgbm)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)],
        categorical_feature=categorical_features
    )
    
    # ğŸŒŸ å­˜å‚¨ OOF é¢„æµ‹: å�ªåœ¨éªŒè¯�é›†ä¸Šé¢„æµ‹
    oof_preds_lgbm[val_index] = lgbm_model.predict(X_val_lgbm)
    
    # ç´¯åŠ æµ‹è¯•é›†é¢„æµ‹ (ç¨�å��å�–å¹³å�‡)
    test_preds_lgbm += lgbm_model.predict(X_test_lgbm) / N_SPLITS
    
    
    # --- D. è®­ç»ƒå’Œé¢„æµ‹ NN ---
    print("Training NN...")
    nn_model = create_mlp_model(X_train_nn_fold.shape[1]) # æ¯�æ¬¡éƒ½åˆ›å»ºæ–°æ¨¡å�‹
    nn_model.fit(
        X_train_nn_fold, y_train_nn_fold,
        validation_data=(X_val_nn_fold, y_val_nn_fold),
        epochs=10, # ä½ çš„å�Ÿå§‹è®¾ç½® (å�¯ä»¥å¢�åŠ )
        batch_size=32,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
        verbose=0 # è®¾ç½®ä¸º 0, é�¿å…�è¿‡å¤šæ—¥å¿—
    )
    
    # ğŸŒŸ å­˜å‚¨ OOF é¢„æµ‹: å�ªåœ¨éªŒè¯�é›†ä¸Šé¢„æµ‹
    oof_preds_nn[val_index] = nn_model.predict(X_val_nn_fold).flatten()
    
    # ç´¯åŠ æµ‹è¯•é›†é¢„æµ‹ (ç¨�å��å�–å¹³å�‡)
    test_preds_nn += nn_model.predict(X_test_nn_scaled).flatten() / N_SPLITS

print("--- OOF Loop Finished ---")


# --- 4. å‡†å¤‡ Meta-Model æ•°æ�® ---

# OOF é¢„æµ‹æ�„æˆ�äº† Meta-Model çš„è®­ç»ƒç‰¹å¾� (æ— æ³„éœ²)
X_meta_train = pd.DataFrame({
    'lgbm_pred': oof_preds_lgbm,
    'nn_pred': oof_preds_nn
})

# å¹³å�‡å��çš„æµ‹è¯•é›†é¢„æµ‹æ�„æˆ�äº† Meta-Model çš„æµ‹è¯•ç‰¹å¾�
X_meta_test = pd.DataFrame({
    'lgbm_pred': test_preds_lgbm,
    'nn_pred': test_preds_nn
})

# Meta-Model çš„ç›®æ ‡å�˜é‡�
y_meta_train = y


# --- 5. è®­ç»ƒ Meta-Model (Ridge) ---
print("Training Meta-Model (Ridge)...")
meta_model = Ridge(alpha=1.0) # ä½ å�¯ä»¥è°ƒæ•´ alpha
meta_model.fit(X_meta_train, y_meta_train)


# --- 6. è�·å¾—æœ€ç»ˆé¢„æµ‹å¹¶ä¿�å­˜ ---
print("Generating final predictions...")
final_pred = meta_model.predict(X_meta_test)

# å‡†å¤‡æ��äº¤æ–‡ä»¶
sub = pd.DataFrame({
    "id": df_test["id"],
    target: final_pred
})

# ä¿�å­˜æ��äº¤æ–‡ä»¶
sub.to_csv("submission_stacked_lgbm_nn_OOF.csv", index=False)
print("Submission file saved as 'submission_stacked_lgbm_nn_OOF.csv'")

