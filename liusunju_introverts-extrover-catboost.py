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
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from catboost import CatBoostClassifier


# è¼‰å…¥è¨“ç·´èˆ‡æ¸¬è©¦è³‡æ–™
df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")

# æŒ‡å®šç›®æ¨™è®Šæ•¸èˆ‡ ID æ¬„ä½�
target = "Personality"
id_col = "id"


# æŸ¥çœ‹å‰�å¹¾ç­†è³‡æ–™
display(df.head())

# æª¢æŸ¥ç¼ºå¤±å€¼
print("\nMissing values in training data:")
print(df.isnull().sum())

# --- Label Encode ç›®æ¨™è®Šæ•¸ ---
le = LabelEncoder()
if df[target].dtype == "object":
    df[target + '_encoded'] = le.fit_transform(df[target])
else:
    df[target + '_encoded'] = df[target]


# ç›®æ¨™è®Šæ•¸åˆ†å¸ƒ
plt.figure(figsize=(8, 4))
sns.countplot(data=df, x=target)
plt.title('Distribution of Target Variable')
plt.xlabel(target)
plt.ylabel('Count')
plt.show()


# æ•¸å€¼æ¬„ä½�ç›¸é—œä¿‚æ•¸ç†±åŠ›åœ–
num_cols_for_corr = df.select_dtypes(include=np.number).columns.tolist()
if target in num_cols_for_corr:
    num_cols_for_corr.remove(target)

# ä½¿ç”¨ç·¨ç¢¼å¾Œçš„ç›®æ¨™è®Šæ•¸ä¾†è¨ˆç®—ç›¸é—œæ€§
corr = df[num_cols_for_corr + [target + '_encoded']].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
plt.show()


# é¡�åˆ¥æ¬„ä½�åˆ†å¸ƒ
cat_cols = df.select_dtypes(include="object").columns.tolist()
cat_cols = [col for col in cat_cols if col not in [target, id_col]]

for col in cat_cols:
    plt.figure(figsize=(8, 2))
    sns.countplot(data=df, x=col, hue=target)
    plt.title(f'Distribution of {col} by {target}')
    # plt.xticks(rotation=45)
    plt.legend(title=target)
    plt.show()


# é�¸æ“‡é¡�åˆ¥æ¬„ä½�ä¸¦æ�’é™¤ ID å’Œç›®æ¨™æ¬„ä½�
cat_cols = df.select_dtypes(include="object").columns.tolist()
cat_cols = [col for col in cat_cols if col not in [target, id_col]]

# ä½¿ç”¨ OneHotEncoder å°�é¡�åˆ¥æ¬„ä½�é€²è¡Œç·¨ç¢¼
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
ohe.fit(df[cat_cols])

# å°�è¨“ç·´èˆ‡æ¸¬è©¦è³‡æ–™é€²è¡Œè½‰æ�›
X_cat_train = ohe.transform(df[cat_cols])
X_cat_test = ohe.transform(df_test[cat_cols])

# å�–å¾— One-Hot ç·¨ç¢¼å¾Œçš„æ¬„ä½�å��ç¨±
ohe_feature_names = ohe.get_feature_names_out(cat_cols)


# æ•¸å€¼ç‰¹å¾µå��ç¨±ï¼ˆå�ªé�¸è¨“ç·´è³‡æ–™ä¸­å­˜åœ¨çš„æ•¸å€¼æ¬„ä½�ï¼Œä¸�åŒ…å�« ID å’Œ targetï¼‰
num_cols = [col for col in df.select_dtypes(include=np.number).columns
            if col not in [id_col, target]]

# ç¢ºä¿�é€™äº›æ¬„ä½�åœ¨ test ä¸­ä¹Ÿå­˜åœ¨
num_cols = [col for col in num_cols if col in df_test.columns]

# æ��å�–æ•¸å€¼æ¬„ä½�
X_num_train = df[num_cols]
X_num_test = df_test[num_cols]


# å°‡æ•¸å€¼å’Œé¡�åˆ¥ç‰¹å¾µå�ˆä½µæˆ�ä¸€å€‹ DataFrame
X_train = pd.concat([
    pd.DataFrame(X_num_train.reset_index(drop=True)),
    pd.DataFrame(X_cat_train, columns=ohe_feature_names)
], axis=1)

X_test = pd.concat([
    pd.DataFrame(X_num_test.reset_index(drop=True)),
    pd.DataFrame(X_cat_test, columns=ohe_feature_names)
], axis=1)

# å¦‚æ�œç›®æ¨™è®Šæ•¸æ˜¯æ–‡å­—ï¼Œå‰‡ä½¿ç”¨ LabelEncoder è½‰æ�›
le = LabelEncoder()
if df[target].dtype == "object":
    df[target] = le.fit_transform(df[target])

y_train = df[target]


# å°‡è¨“ç·´è³‡æ–™åˆ‡åˆ†ç‚ºè¨“ç·´é›†èˆ‡é©—è­‰é›†
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train,
    y_train,
    test_size=0.1,
    stratify=y_train,
    random_state=42   )


# åˆ�å§‹åŒ– CatBoost åˆ†é¡�å™¨
model = CatBoostClassifier(
    learning_rate=0.1,        # å­¸ç¿’ç�‡
    depth=3,                  # æ¨¹çš„æ·±åº¦
    n_estimators=200,         # æ¨¹çš„æ•¸é‡�
    subsample=0.8,            # æ¯�æ¬¡è¨“ç·´ä½¿ç”¨çš„è³‡æ–™æ¯”ä¾‹
    colsample_bylevel=0.7,    # æ¯�å±¤é�¸æ“‡çš„ç‰¹å¾µæ¯”ä¾‹
    eval_metric='Accuracy',   # é©—è­‰æŒ‡æ¨™
    verbose=50,               # æ¯� 50 è¼ªé¡¯ç¤ºä¸€æ¬¡è¨“ç·´é��ç¨‹
    random_state=42           # å›ºå®šçµ�æ�œ
)

# é–‹å§‹è¨“ç·´
model.fit(X_train_split, y_train_split)


# åœ¨é©—è­‰é›†ä¸Šè©•ä¼°æº–ç¢ºç�‡
val_score = model.score(X_val_split, y_val_split)
print(f"\nValidation Accuracy: {val_score:.4f}")


print("\n=== ç‰¹å¾µé‡�è¦�æ€§åˆ†æ�� ===")
# å�–å¾— CatBoost æ¨¡å�‹ä¸­çš„ç‰¹å¾µé‡�è¦�æ€§
importance = pd.Series(model.get_feature_importance(), index=X_train.columns)

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

# è¨­å®š y è»¸æ¨™ç±¤
plt.yticks(range(len(top_5_features)), top_5_features['feature'][::1])

# åœ¨æ¯�å€‹é•·æ¢�ä¸Šé¡¯ç¤ºæ•¸å€¼
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width + width*0.01, bar.get_y() + bar.get_height()/2.,
             f'{width:.2f}', ha='left', va='center', fontweight='bold')

# èª¿æ•´ç‰ˆé�¢
plt.tight_layout()
plt.grid(axis='x', alpha=0.3)  # åŠ ä¸Šæ ¼ç·š
plt.show()

print("\nç‰¹å¾µé‡�è¦�æ€§èªªæ˜�:")
print("- é‡�è¦�æ€§åˆ†æ•¸è¶Šé«˜ï¼Œè¡¨ç¤ºè©²ç‰¹å¾µå°�é �æ¸¬çµ�æ�œçš„å½±éŸ¿è¶Šå¤§")
print("- CatBoost ä½¿ç”¨ 'PredictionValuesChange' æ–¹å¼�è¨ˆç®—é‡�è¦�æ€§")
print("- é€™äº›ç‰¹å¾µåœ¨å�šé �æ¸¬æ™‚æ‰®æ¼”é—œé�µè§’è‰²")


# é€²è¡Œé �æ¸¬
predictions = model.predict(X_test)

# è‹¥ç›®æ¨™è®Šæ•¸æœ‰ç¶“é�� LabelEncoder è½‰æ�›ï¼Œå‰‡é‚„å�Ÿå›�å�»
if le:
    predictions = le.inverse_transform(predictions.astype(int))

# å»ºç«‹æ��äº¤æª”æ¡ˆ
submission = pd.DataFrame({
    'id': df_test[id_col],
    target: predictions
})

submission.to_csv('submission.csv', index=False)

print("é �æ¸¬çµ�æ�œå·²å„²å­˜ç‚º 'submission.csv'")

print("æ��äº¤æª”æ¡ˆé �è¦½:")
display(submission.head(10))

