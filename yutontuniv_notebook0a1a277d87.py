# ãƒ©ã‚¤ãƒ–ãƒ©ãƒªã�®ã‚¤ãƒ³ãƒ�ãƒ¼ãƒˆ
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ãƒ‡ãƒ¼ã‚¿èª­ã�¿è¾¼ã�¿
df = pd.read_csv('/kaggle/input/playground-series-s4e8/sample_submission.csv')

# æœ€åˆ�ã�®5è¡Œã‚’è¡¨ç¤º
print(df.head())


# ãƒ©ã‚¤ãƒ–ãƒ©ãƒªã�®ã‚¤ãƒ³ãƒ�ãƒ¼ãƒˆ
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ãƒ‡ãƒ¼ã‚¿èª­ã�¿è¾¼ã�¿
df = pd.read_csv('/kaggle/input/playground-series-s4e8/sample_submission.csv')

# æœ€åˆ�ã�®5è¡Œã‚’è¡¨ç¤º
print(df.head())


# ãƒ‡ãƒ¼ã‚¿èª­ã�¿è¾¼ã�¿
df = pd.read_csv('/kaggle/input/playground-series-s4e8/train.csv')

# æœ€åˆ�ã�®5è¡Œã‚’è¡¨ç¤º
print(df.head())


# ãƒ‡ãƒ¼ã‚¿ã‚»ãƒƒãƒˆã�®è¡Œæ•°ã�¨åˆ—æ•°ã‚’ç¢ºèª�
print(f"{df.shape[0]}è¡Œ, {df.shape[1]}åˆ—")


# ãƒ‡ãƒ¼ã‚¿å…¨ä½“ã�®æƒ…å ±ç¢ºèª�
print(df.info())


# å�„åˆ—ã�«å�«ã�¾ã‚Œã‚‹æ¬ æ��å€¤ã�®æ•°ã‚’å�ˆè¨ˆã�—ã�¦è¡¨ç¤º
print(df.isnull().sum())


# æ•°å€¤ãƒ‡ãƒ¼ã‚¿ã�®åŸºæœ¬çµ±è¨ˆé‡�
print(df.describe())


# æ–‡å­—ãƒ‡ãƒ¼ã‚¿ã�®åŸºæœ¬çµ±è¨ˆé‡�
print(df.describe(include='object'))


!pip install japanize_matplotlib
import japanize_matplotlib
# ã�‹ã�•ã�®ç›´å¾„ï¼ˆcap-diameterï¼‰ã�¨æ¯’æ€§ã�®é–¢ä¿‚ã‚’å�¯è¦–åŒ–
sns.histplot(data=df, x='cap-diameter', hue='class', kde=True, bins=50)
plt.title('ã�‹ã�•ã�®ç›´å¾„ã�¨æ¯’æ€§ã�®åˆ†å¸ƒ')
plt.show()


# ç”Ÿæ�¯åœ°ã�¨æ¯’æ€§ã�®é–¢ä¿‚
sns.countplot(data=df, x='habitat', hue='class')


df['habitat'].value_counts()


df['habitat'].unique()


import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef


test_df = pd.read_csv('/kaggle/input/playground-series-s4e8/test.csv')


# æ��å‡ºç”¨ã�«ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿ã�®IDã‚’ä¿�æŒ�
test_id = test_df['id']

# IDåˆ—ã‚’å‰Šé™¤
df = df.drop('id', axis=1)
test_df = test_df.drop('id', axis=1)


# ãƒ©ãƒ™ãƒ«ã‚¨ãƒ³ã‚³ãƒ¼ãƒ‡ã‚£ãƒ³ã‚°
print("ãƒ©ãƒ™ãƒ«ã‚¨ãƒ³ã‚³ãƒ¼ãƒ‡ã‚£ãƒ³ã‚°ã‚’é–‹å§‹ã�—ã�¾ã�™...")
# 'class'åˆ—ã‚’é™¤ã�„ã�Ÿã€�ã�™ã�¹ã�¦ã�®æ–‡å­—ãƒ‡ãƒ¼ã‚¿åˆ—ã‚’å¯¾è±¡ã�«ã�™ã‚‹
feature_cols = [col for col in df.select_dtypes(include=['object']).columns if col != 'class']

for col in feature_cols:
    # è¨“ç·´ãƒ‡ãƒ¼ã‚¿ã�¨ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿ã‚’çµ�å�ˆã�—ã€�ã�™ã�¹ã�¦ã�®ã‚«ãƒ†ã‚´ãƒªã‚’ã‚¨ãƒ³ã‚³ãƒ¼ãƒ€ãƒ¼ã�«å­¦ç¿’ã�•ã�›ã‚‹
    combined_series = pd.concat([df[col], test_df[col]]).astype(str)
    
    le = LabelEncoder()
    le.fit(combined_series)
    
    # è¨“ç·´ãƒ‡ãƒ¼ã‚¿ã�¨ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿ã‚’ã��ã‚Œã��ã‚Œå¤‰æ�›
    df[col] = le.transform(df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))

# classåˆ—ã‚‚æ•°å€¤ã�«å¤‰æ�› (e:0, p:1)
df['class'] = df['class'].map({'e': 0, 'p': 1})

print("ã‚¨ãƒ³ã‚³ãƒ¼ãƒ‡ã‚£ãƒ³ã‚°ã�Œå®Œäº†ã�—ã�¾ã�—ã�Ÿã€‚")


# ç‰¹å¾´é‡�Xã�¨ç›®çš„å¤‰æ•°yã‚’å®šç¾©
X = df.drop('class', axis=1)
y = df['class']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

lgbm_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'n_estimators': 784,
    'learning_rate': 0.046652989029122655,
    'num_leaves': 920,
    'max_depth': 12,
    'min_child_samples': 69,
    'feature_fraction': 0.527119104693057,
    'bagging_fraction': 0.7890844374994465,
    'bagging_freq': 4,
    'lambda_l1': 3.7474364409690137,
    'lambda_l2': 5.819299073494867e-05
}


# ãƒ¢ãƒ‡ãƒ«ã‚’ä½œæˆ�
model = lgb.LGBMClassifier(**lgbm_params)
# è¨“ç·´ãƒ‡ãƒ¼ã‚¿ã�®ä¸€éƒ¨(X_train)ã�§å­¦ç¿’
print("ãƒ¢ãƒ‡ãƒ«ã�®å­¦ç¿’ã‚’é–‹å§‹ã�—ã�¾ã�™...")
model.fit(X_train, y_train)
print("å­¦ç¿’ã�Œå®Œäº†ã�—ã�¾ã�—ã�Ÿã€‚")


# æ¤œè¨¼ç”¨ãƒ‡ãƒ¼ã‚¿(X_val)ã�§æ€§èƒ½ã‚’è©•ä¾¡
print("\n--- ãƒ­ãƒ¼ã‚«ãƒ«ã�§ã�®æ€§èƒ½è©•ä¾¡ ---")
predictions_local = model.predict(X_val)
mcc_local = matthews_corrcoef(y_val, predictions_local)
print(f"ãƒ­ãƒ¼ã‚«ãƒ«ã�§ã�®MCCã‚¹ã‚³ã‚¢: {mcc_local:.5f}")


# æ��å‡ºç”¨ãƒ¢ãƒ‡ãƒ«ã�®ä½œæˆ�ã�¨äºˆæ¸¬
# å…¨ã�¦ã�®è¨“ç·´ãƒ‡ãƒ¼ã‚¿ã‚’ä½¿ã�£ã�¦å†�å­¦ç¿’ã�—ã€�ãƒ¢ãƒ‡ãƒ«ã�®æ€§èƒ½ã‚’æœ€å¤§åŒ–
print("\nå…¨ãƒ‡ãƒ¼ã‚¿ã�§æœ€çµ‚ãƒ¢ãƒ‡ãƒ«ã‚’å†�å­¦ç¿’ã�—ã�¾ã�™...")
# verboseã‚’1ã�«ã�™ã‚‹ã�¨å­¦ç¿’é��ç¨‹ã�Œè¡¨ç¤ºã�•ã‚Œã‚‹
lgbm_params['verbose'] = 1 
final_model = lgb.LGBMClassifier(**lgbm_params)
final_model.fit(X, y)
print("å†�å­¦ç¿’ã�Œå®Œäº†ã�—ã�¾ã�—ã�Ÿã€‚")


# ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿ã�§äºˆæ¸¬
predictions_submission = final_model.predict(test_df)

# ç›®çš„å¤‰æ•°ã�®ãƒ�ãƒƒãƒ”ãƒ³ã‚°ã‚’å…ƒã�«æˆ»ã�™
class_map_reverse = {0: 'e', 1: 'p'}
predictions_labels = [class_map_reverse[pred] for pred in predictions_submission]

# æ��å‡ºç”¨ãƒ•ã‚¡ã‚¤ãƒ«ã�®ä½œæˆ�
submission_df = pd.DataFrame({'id': test_id, 'class': predictions_labels})
submission_df.to_csv('submission.csv', index=False)

print("\n'submission.csv' ã�Œä½œæˆ�ã�•ã‚Œã�¾ã�—ã�Ÿã€‚")
print(submission_df.head())

