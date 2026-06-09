# é€™å€‹ Python 3 ç’°å¢ƒé �è£�äº†è¨±å¤šå¯¦ç”¨çš„åˆ†æ��å‡½å¼�åº«
# å®ƒæ˜¯ä»¥ kaggle/python Docker æ˜ åƒ�æª”ç‚ºåŸºç¤�æ‰€å®šç¾©ï¼š[https://github.com/kaggle/docker-python](https://github.com/kaggle/docker-python)
# ä¾‹å¦‚ï¼Œä»¥ä¸‹æ˜¯ä¸€äº›è¼‰å…¥çš„å¯¦ç”¨å¥—ä»¶

import numpy as np # ç·šæ€§ä»£æ•¸
import pandas as pd # è³‡æ–™è™•ç�†ã€�CSV æª”æ¡ˆè¼¸å…¥/è¼¸å‡º (ä¾‹å¦‚ pd.read_csv)

# è¼¸å…¥è³‡æ–™æª”æ¡ˆä½�æ–¼å”¯è®€çš„ "../input/" ç›®éŒ„ä¸‹
# ä¾‹å¦‚ï¼ŒåŸ·è¡Œæ­¤è™• (é»�æ“Š "run" æˆ–æŒ‰ä¸‹ Shift+Enter) å°‡æœƒåˆ—å‡ºè¼¸å…¥ç›®éŒ„ä¸‹çš„æ‰€æœ‰æª”æ¡ˆ

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# ä½ å�¯ä»¥å°‡æœ€å¤š 20GB çš„è³‡æ–™å¯«å…¥ç›®å‰�ç›®éŒ„ (/kaggle/working/)ï¼Œç•¶ä½ ä½¿ç”¨ "Save & Run All" å»ºç«‹ç‰ˆæœ¬æ™‚ï¼Œé€™äº›è³‡æ–™æœƒè¢«å„²å­˜ä¸‹ä¾†
# ä½ ä¹Ÿå�¯ä»¥å°‡æš«å­˜æª”æ¡ˆå¯«å…¥ /kaggle/temp/ï¼Œä½†é€™äº›æª”æ¡ˆåœ¨ç›®å‰�å·¥ä½œéš�æ®µçµ�æ�Ÿå¾Œä¸�æœƒè¢«å„²å­˜


import pandas as pd
import numpy as np
import seaborn as sns
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
print("æ‰€æœ‰å¥—ä»¶è¼‰å…¥å®Œæˆ�ï¼�")


print("\n=== é–‹å§‹è®€å�–è³‡æ–™ ===")
# è®€å�–è¨“ç·´è³‡æ–™å’Œæ¸¬è©¦è³‡æ–™
df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
print(f"è¨“ç·´è³‡æ–™å½¢ç‹€: {df.shape}")
print(f"æ¸¬è©¦è³‡æ–™å½¢ç‹€: {df_test.shape}")


print("\n=== å®šç¾©é‡�è¦�è®Šæ•¸ ===")
target = "Personality"  # è¦�é �æ¸¬çš„ç›®æ¨™æ¬„ä½�
id_col = "id"          # ID æ¬„ä½�ï¼Œä¸�ç”¨ä¾†è¨“ç·´æ¨¡å�‹

print(f"ç›®æ¨™è®Šæ•¸: {target}")
print(f"IDæ¬„ä½�: {id_col}")
print(f"ç›®æ¨™è®Šæ•¸çš„é¡�åˆ¥: {df[target].unique()}")


print("\n=== è™•ç�†é¡�åˆ¥å�‹è®Šæ•¸ ===")
# æ‰¾å‡ºæ‰€æœ‰æ–‡å­—é¡�å�‹çš„æ¬„ä½�ï¼ˆé¡�åˆ¥å�‹è®Šæ•¸ï¼‰
cat_cols = df.select_dtypes(include="object").columns.tolist()
# ç§»é™¤ç›®æ¨™è®Šæ•¸å’ŒIDæ¬„ä½�ï¼Œå› ç‚ºå®ƒå€‘ä¸�éœ€è¦�ç·¨ç¢¼
cat_cols = [col for col in cat_cols if col not in [target, id_col]]

print(f"é¡�åˆ¥å�‹è®Šæ•¸: {cat_cols}")

# å¦‚æ�œæœ‰é¡�åˆ¥å�‹è®Šæ•¸ï¼Œå°±é€²è¡Œ One-Hot ç·¨ç¢¼
if cat_cols:
    print("æ­£åœ¨é€²è¡Œ One-Hot ç·¨ç¢¼...")
    # One-Hot ç·¨ç¢¼ï¼šæŠŠé¡�åˆ¥è®Šæ•¸è½‰æ�›æˆ�æ•¸å­—ï¼ˆ0æˆ–1ï¼‰
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ohe.fit(df[cat_cols])  # å­¸ç¿’å¦‚ä½•ç·¨ç¢¼
    
    # å°�è¨“ç·´è³‡æ–™å’Œæ¸¬è©¦è³‡æ–™é€²è¡Œç·¨ç¢¼
    X_cat_train = ohe.transform(df[cat_cols])
    X_cat_test = ohe.transform(df_test[cat_cols])
    
    print(f"ç·¨ç¢¼å¾Œçš„é¡�åˆ¥ç‰¹å¾µæ•¸é‡�: {X_cat_train.shape[1]}")
else:
    print("æ²’æœ‰é¡�åˆ¥å�‹è®Šæ•¸éœ€è¦�ç·¨ç¢¼")
    X_cat_train = np.array([]).reshape(len(df), 0)
    X_cat_test = np.array([]).reshape(len(df_test), 0)


print("\n=== è™•ç�†æ•¸å€¼å�‹è®Šæ•¸ ===")
# æ‰¾å‡ºæ‰€æœ‰æ•¸å€¼å�‹æ¬„ä½�ï¼ˆæ�’é™¤é¡�åˆ¥å�‹ã€�IDã€�ç›®æ¨™è®Šæ•¸ï¼‰
num_cols = [col for col in df.columns if col not in cat_cols + [id_col, target]]

print(f"æ•¸å€¼å�‹è®Šæ•¸: {num_cols}")
print(f"æ•¸å€¼å�‹è®Šæ•¸æ•¸é‡�: {len(num_cols)}")

# å�–å‡ºæ•¸å€¼å�‹è³‡æ–™
X_num_train = df[num_cols]
X_num_test = df_test[num_cols]


print("\n=== å�ˆä½µæ‰€æœ‰ç‰¹å¾µ ===")
# å°‡æ•¸å€¼å�‹å’Œé¡�åˆ¥å�‹ç‰¹å¾µå�ˆä½µåœ¨ä¸€èµ·
if cat_cols:
    # å¦‚æ�œæœ‰é¡�åˆ¥å�‹è®Šæ•¸
    X_train = pd.concat([
        pd.DataFrame(X_num_train.reset_index(drop=True)),
        pd.DataFrame(X_cat_train, columns=ohe.get_feature_names_out(cat_cols))
    ], axis=1)
    
    X_test = pd.concat([
        pd.DataFrame(X_num_test.reset_index(drop=True)),
        pd.DataFrame(X_cat_test, columns=ohe.get_feature_names_out(cat_cols))
    ], axis=1)
else:
    # å¦‚æ�œæ²’æœ‰é¡�åˆ¥å�‹è®Šæ•¸ï¼Œå°±å�ªç”¨æ•¸å€¼å�‹è®Šæ•¸
    X_train = X_num_train.copy()
    X_test = X_num_test.copy()

print(f"æœ€çµ‚ç‰¹å¾µæ•¸é‡�: {X_train.shape[1]}")
print(f"è¨“ç·´è³‡æ–™å½¢ç‹€: {X_train.shape}")
print(f"æ¸¬è©¦è³‡æ–™å½¢ç‹€: {X_test.shape}")


print("\n=== è™•ç�†ç›®æ¨™è®Šæ•¸ ===")
# å¦‚æ�œç›®æ¨™è®Šæ•¸æ˜¯æ–‡å­—é¡�å�‹ï¼Œéœ€è¦�è½‰æ�›æˆ�æ•¸å­—
le = LabelEncoder()
if df[target].dtype == "object":
    print("ç›®æ¨™è®Šæ•¸æ˜¯æ–‡å­—é¡�å�‹ï¼Œæ­£åœ¨è½‰æ�›æˆ�æ•¸å­—...")
    df[target] = le.fit_transform(df[target])
    print(f"è½‰æ�›å¾Œçš„ç›®æ¨™è®Šæ•¸: {df[target].unique()}")
else:
    print("ç›®æ¨™è®Šæ•¸å·²ç¶“æ˜¯æ•¸å­—é¡�å�‹")

y_train = df[target]
print(f"ç›®æ¨™è®Šæ•¸åˆ†å¸ƒ:\n{y_train.value_counts()}")


print("\n=== åˆ†å‰²è¨“ç·´å’Œé©—è­‰è³‡æ–™ ===")
# å°‡è³‡æ–™åˆ†æˆ�è¨“ç·´é›†å’Œé©—è­‰é›†ï¼ˆ90% è¨“ç·´ï¼Œ10% é©—è­‰ï¼‰
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, 
    test_size=0.1,          # 10% ä½œç‚ºé©—è­‰é›†
    stratify=y_train,       # ä¿�æŒ�å�„é¡�åˆ¥æ¯”ä¾‹ç›¸å�Œ
    random_state=42         # è®“çµ�æ�œå�¯é‡�ç�¾
)

print(f"è¨“ç·´é›†å¤§å°�: {X_train_split.shape[0]}")
print(f"é©—è­‰é›†å¤§å°�: {X_val_split.shape[0]}")


print("\n=== å»ºç«‹ LightGBM æ¨¡å�‹ ===")

# LightGBM æ¨¡å�‹å�ƒæ•¸è¨­å®šï¼ˆé�©å�ˆåˆ�å­¸è€…çš„ä¿�å®ˆè¨­å®šï¼‰
lgb_params = {
    'objective': 'multiclass',      # å¤šåˆ†é¡�å•�é¡Œ
    'num_class': len(y_train.unique()),  # é¡�åˆ¥æ•¸é‡�
    'metric': 'multi_logloss',      # è©•ä¼°æŒ‡æ¨™
    'boosting_type': 'gbdt',        # æ¢¯åº¦æ��å�‡æ±ºç­–æ¨¹
    'learning_rate': 0.1,           # å­¸ç¿’é€Ÿåº¦ï¼ˆä¸�è¦�å¤ªå¿«é�¿å…�é��æ“¬å�ˆï¼‰
    'max_depth': 6,                 # æ¨¹çš„æœ€å¤§æ·±åº¦
    'num_leaves': 31,               # æ¯�æ£µæ¨¹çš„è‘‰å­�æ•¸é‡�
    'feature_fraction': 0.8,        # æ¯�æ£µæ¨¹éš¨æ©Ÿé�¸æ“‡ 80% çš„ç‰¹å¾µ
    'bagging_fraction': 0.8,        # æ¯�æ¬¡è¨“ç·´éš¨æ©Ÿé�¸æ“‡ 80% çš„è³‡æ–™
    'bagging_freq': 5,              # æ¯� 5 æ¬¡è¿­ä»£é€²è¡Œä¸€æ¬¡ bagging
    'min_data_in_leaf': 20,         # æ¯�å€‹è‘‰å­�è‡³å°‘è¦�æœ‰ 20 å€‹æ¨£æœ¬
    'random_state': 42,             # è®“çµ�æ�œå�¯é‡�ç�¾
    'verbose': -1                   # é—œé–‰é��å¤šçš„è¼¸å‡ºè¨Šæ�¯
}

print("æ¨¡å�‹å�ƒæ•¸è¨­å®šå®Œæˆ�ï¼�")
print("å�ƒæ•¸èªªæ˜�ï¼š")
print("- learning_rate: æ�§åˆ¶å­¸ç¿’é€Ÿåº¦ï¼Œå¤ªå¤§å®¹æ˜“é��æ“¬å�ˆ")
print("- max_depth: æ�§åˆ¶æ¨¹çš„æ·±åº¦ï¼Œé�¿å…�æ¨¡å�‹å¤ªè¤‡é›œ")
print("- feature_fraction: æ¯�æ£µæ¨¹å�ªçœ‹éƒ¨åˆ†ç‰¹å¾µï¼Œå¢�åŠ æ¨¡å�‹å¤šæ¨£æ€§")
print("- bagging_fraction: æ¯�æ¬¡è¨“ç·´å�ªç”¨éƒ¨åˆ†è³‡æ–™ï¼Œæ¸›å°‘é��æ“¬å�ˆ")


print("\n=== é–‹å§‹è¨“ç·´æ¨¡å�‹ ===")

# å°‡è³‡æ–™è½‰æ�›æˆ� LightGBM çš„å°ˆç”¨æ ¼å¼�ï¼ˆDatasetï¼‰
print("æ­£åœ¨æº–å‚™ LightGBM è³‡æ–™é›†...")
train_data = lgb.Dataset(X_train_split, label=y_train_split)
val_data = lgb.Dataset(X_val_split, label=y_val_split, reference=train_data)

# é–‹å§‹è¨“ç·´ï¼�
print("é–‹å§‹è¨“ç·´æ¨¡å�‹ï¼Œè«‹ç¨�ç­‰...")
model = lgb.train(
    params=lgb_params,              # ä½¿ç”¨æˆ‘å€‘è¨­å®šçš„å�ƒæ•¸
    train_set=train_data,           # è¨“ç·´è³‡æ–™
    valid_sets=[val_data],          # é©—è­‰è³‡æ–™
    num_boost_round=500,            # æœ€å¤šè¨“ç·´ 500 è¼ª
    callbacks=[
        lgb.early_stopping(50),     # å¦‚æ�œ 50 è¼ªæ²’æ”¹å–„å°±å�œæ­¢
        lgb.log_evaluation(50)      # æ¯� 50 è¼ªé¡¯ç¤ºä¸€æ¬¡çµ�æ�œ
    ]
)

print("æ¨¡å�‹è¨“ç·´å®Œæˆ�ï¼�")


print("\n=== è©•ä¼°æ¨¡å�‹æ•ˆæ�œ ===")

# åœ¨é©—è­‰é›†ä¸Šå�šé �æ¸¬
val_pred = model.predict(X_val_split)
val_pred_class = np.argmax(val_pred, axis=1)  # å�–æ©Ÿç�‡æœ€é«˜çš„é¡�åˆ¥

# è¨ˆç®—æº–ç¢ºç�‡
accuracy = accuracy_score(y_val_split, val_pred_class)
print(f"é©—è­‰é›†æº–ç¢ºç�‡: {accuracy:.4f}")

# é¡¯ç¤ºè©³ç´°çš„åˆ†é¡�å ±å‘Š
print("\nè©³ç´°åˆ†é¡�å ±å‘Š:")
if df[target].dtype != "object" and 'le' in locals():
    # å¦‚æ�œæœ‰å�šé��æ¨™ç±¤ç·¨ç¢¼ï¼Œé¡¯ç¤ºå�Ÿå§‹æ¨™ç±¤å��ç¨±
    target_names = le.classes_ if hasattr(le, 'classes_') else None
    print(classification_report(y_val_split, val_pred_class, target_names=target_names))
else:
    print(classification_report(y_val_split, val_pred_class))


print("\n=== ç‰¹å¾µé‡�è¦�æ€§åˆ†æ�� ===")
# å�–å¾— LightGBM æ¨¡å�‹ä¸­çš„ç‰¹å¾µé‡�è¦�æ€§
importance = pd.Series(model.feature_importance(importance_type='gain'), index=X_train.columns)

# å°‡çµ�æ�œè½‰æ�›ç‚º DataFrameï¼Œä¸¦ç›´æ�¥æ�’åº� + å�ªå�–å‰� 10 å��
importance_df = pd.DataFrame(importance.sort_values(ascending=False).head(10)).reset_index()
importance_df.columns = ['feature', 'importance']

print("Feature Importance (Top 10):")
print(importance_df.head(10))


# å�–å¾—å‰�5å€‹æœ€é‡�è¦�çš„ç‰¹å¾µ
top_5_features = importance_df.head(5).sort_values(by='importance', ascending=True)
# å�¦å¤–ç¹ªè£½æ°´å¹³é•·æ¢�åœ–ï¼ˆæ›´é�©å�ˆé•·ç‰¹å¾µå��ç¨±ï¼‰
plt.figure(figsize=(10, 6))
bars = plt.barh(range(len(top_5_features)), top_5_features['importance'], 
                color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#DDA0DD'])

# è¨­å®šåœ–è¡¨æ¨™é¡Œå’Œæ¨™ç±¤
plt.title('Top 5 Most Important Features (Horizontal Bar Chart)', fontsize=16, fontweight='bold')
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature Names', fontsize=12)

# è¨­å®š y è»¸æ¨™ç±¤ï¼ˆå€’åº�é¡¯ç¤ºï¼Œè®“æœ€é‡�è¦�çš„åœ¨ä¸Šé�¢ï¼‰
plt.yticks(range(len(top_5_features)), top_5_features['feature'][::-1])

# åœ¨æ¯�å€‹é•·æ¢�ä¸Šé¡¯ç¤ºæ•¸å€¼
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width + width*0.01, bar.get_y() + bar.get_height()/2.,
             f'{width:.0f}', ha='left', va='center', fontweight='bold')

# èª¿æ•´ç‰ˆé�¢
plt.tight_layout()
plt.grid(axis='x', alpha=0.3)  # åŠ ä¸Šæ ¼ç·š
plt.show()

print("\nç‰¹å¾µé‡�è¦�æ€§èªªæ˜�:")
print("- é‡�è¦�æ€§åˆ†æ•¸è¶Šé«˜ï¼Œè¡¨ç¤ºè©²ç‰¹å¾µå°�é �æ¸¬çµ�æ�œçš„å½±éŸ¿è¶Šå¤§")
print("- LightGBM ä½¿ç”¨ 'gain' æ–¹å¼�è¨ˆç®—é‡�è¦�æ€§")
print("- é€™äº›ç‰¹å¾µåœ¨å�šé �æ¸¬æ™‚æ‰®æ¼”é—œé�µè§’è‰²")


print("\n=== å°�æ¸¬è©¦è³‡æ–™å�šé �æ¸¬ ===")

# å°�æ¸¬è©¦è³‡æ–™å�šé �æ¸¬
test_pred = model.predict(X_test)
test_pred_class = np.argmax(test_pred, axis=1)  # å�–æ©Ÿç�‡æœ€é«˜çš„é¡�åˆ¥

# å¦‚æ�œç›®æ¨™è®Šæ•¸æœ‰å�šé��æ¨™ç±¤ç·¨ç¢¼ï¼Œè¦�è½‰æ�›å›�å�Ÿå§‹æ¨™ç±¤
if df[target].dtype != "object" and 'le' in locals() and hasattr(le, 'classes_'):
    test_pred_original = le.inverse_transform(test_pred_class)
    print("å·²å°‡é �æ¸¬çµ�æ�œè½‰æ�›å›�å�Ÿå§‹æ¨™ç±¤")
else:
    test_pred_original = test_pred_class

print(f"æ¸¬è©¦è³‡æ–™é �æ¸¬å®Œæˆ�ï¼�å…± {len(test_pred_original)} ç­†é �æ¸¬çµ�æ�œ")
print(f"é �æ¸¬çµ�æ�œå‰�10ç­†: {test_pred_original[:10]}")


print("\n=== å„²å­˜é �æ¸¬çµ�æ�œ ===")

# å»ºç«‹æ��äº¤æª”æ¡ˆ
submission = pd.DataFrame({
    id_col: df_test[id_col],           # ID æ¬„ä½�
    target: test_pred_original         # é �æ¸¬çµ�æ�œ
})

print("æ��äº¤æª”æ¡ˆé �è¦½:")
display(submission.head(10))

# å„²å­˜æª”æ¡ˆ
submission.to_csv('submission.csv', index=False)
print("é �æ¸¬çµ�æ�œå·²å„²å­˜ç‚º 'submission.csv'")

print("\n=== ç¨‹å¼�åŸ·è¡Œå®Œæˆ�ï¼�===")
print("æ�­å–œä½ å®Œæˆ�äº† LightGBM æ©Ÿå™¨å­¸ç¿’å°ˆæ¡ˆï¼�")
print("\næ¨¡å�‹è¨“ç·´é‡�é»�å›�é¡§ï¼š")
print("1. è³‡æ–™å‰�è™•ç�†ï¼šé¡�åˆ¥ç·¨ç¢¼ã€�æ•¸å€¼è™•ç�†")
print("2. ç‰¹å¾µå·¥ç¨‹ï¼šå�ˆä½µä¸�å�Œé¡�å�‹çš„ç‰¹å¾µ")
print("3. æ¨¡å�‹è¨“ç·´ï¼šä½¿ç”¨é�©ç•¶çš„å�ƒæ•¸è¨­å®š")
print("4. æ¨¡å�‹è©•ä¼°ï¼šæª¢æŸ¥åœ¨é©—è­‰é›†ä¸Šçš„è¡¨ç�¾")
print("5. é �æ¸¬æ��äº¤ï¼šå°�æ¸¬è©¦è³‡æ–™å�šé �æ¸¬ä¸¦å„²å­˜çµ�æ�œ")

