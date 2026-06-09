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
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score


df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


target = "Personality"
id_col = "id"


cat_cols = df.select_dtypes(include="object").columns.tolist()
cat_cols = [col for col in cat_cols if col not in [target, id_col]]


ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)


ohe.fit(df[cat_cols])
X_cat_train = ohe.transform(df[cat_cols])
X_cat_test = ohe.transform(df_test[cat_cols])


num_cols = [col for col in df.columns if col not in cat_cols + [id_col, target]]
X_num_train = df[num_cols]
X_num_test = df_test[num_cols]


X_train = pd.concat([
    pd.DataFrame(X_num_train.reset_index(drop=True)),
    pd.DataFrame(X_cat_train, columns=ohe.get_feature_names_out(cat_cols))
], axis=1)

X_test = pd.concat([
    pd.DataFrame(X_num_test.reset_index(drop=True)),
    pd.DataFrame(X_cat_test, columns=ohe.get_feature_names_out(cat_cols))
], axis=1)


le = LabelEncoder()
if df[target].dtype == "object":
    df[target] = le.fit_transform(df[target])
y_train = df[target]


X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.1, stratify=y_train, random_state=42)


model = XGBClassifier(
    learning_rate=0.1,      # å­¸ç¿’é€Ÿåº¦ä¸�è¦�å¤ªå¿«
    max_depth=3,            # æ�§åˆ¶æ¨¹çš„æ·±åº¦ï¼Œé�¿å…�é��æ“¬å�ˆ
    n_estimators=200,       # è¨“ç·´ 200 æ£µæ¨¹
    subsample=0.8,          # æ¯�æ¬¡è¨“ç·´ç”¨ 80% çš„è³‡æ–™
    colsample_bytree=0.7,   # æ¯�æ£µæ¨¹å�ªçœ‹ 70% çš„ç‰¹å¾µï¼ˆé¼“å‹µä½¿ç”¨ä¸�å�Œç‰¹å¾µï¼‰
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42         # è®“çµ�æ�œå�¯é‡�ç�¾
)

model.fit(X_train_split, y_train_split)


# å�–å¾—ç‰¹å¾µå��ç¨±ï¼ˆç¢ºä¿� X_train æ˜¯ DataFrameï¼‰
feature_names = X_train.columns.tolist()

# å�–å¾—ç‰¹å¾µé‡�è¦�æ€§
importances = model.feature_importances_

# ç•«åœ–é¡¯ç¤ºå‰� 5 å€‹æœ€é‡�è¦�çš„ç‰¹å¾µ
top_n = 5
indices = importances.argsort()[-top_n:][::1]  # æ�’åº�ä¸¦å�–å‰� 5 å��

plt.figure(figsize=(10, 6))
plt.barh(range(top_n), importances[indices], align='center')
plt.yticks(range(top_n), [feature_names[i] for i in indices])
plt.title('Top 5 Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()


val_pred = model.predict(X_val_split)
print("Validation Accuracy:", accuracy_score(y_val_split, val_pred))


test_pred = model.predict(X_test)
test_pred_labels = le.inverse_transform(test_pred)


submission = pd.DataFrame({
    'id': df_test['id'],
    'Personality': test_pred_labels
})
submission.to_csv('submission.csv', index=False)


submission.head()

