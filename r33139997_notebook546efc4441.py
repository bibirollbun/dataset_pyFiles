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


# ãƒ©ã‚¤ãƒ–ãƒ©ãƒªã�®ã‚¤ãƒ³ãƒ�ãƒ¼ãƒˆ
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ãƒ‡ãƒ¼ã‚¿èª­ã�¿è¾¼ã�¿
df = pd.read_csv('/kaggle/input/playground-series-s4e8/train.csv')

# æœ€åˆ�ã�®5è¡Œã‚’è¡¨ç¤º
print(df.head())

# ãƒ‡ãƒ¼ã‚¿ã‚»ãƒƒãƒˆã�®è¡Œæ•°ã�¨åˆ—æ•°ã‚’ç¢ºèª�
print(f"{df.shape[0]}è¡Œ, {df.shape[1]}åˆ—")

# ãƒ‡ãƒ¼ã‚¿å…¨ä½“ã�®æƒ…å ±ç¢ºèª�
print(df.info())


# æ¬ æ��å€¤ã�®è¡¨ç¤º
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


drop = ['veil-type','spore-print-color','stem-root','veil-color','stem-surface','gill-spacing']
df = df.drop(drop, axis=1)


print(df.columns)


print(df.isnull().sum())


df_backup = df.copy()
print(df_backup.shape)


valid_habitats = [
    'd', 'g', 'l', 'm', 'h', 'w', 'p', 'u', 'e', 's', 
    'n', 't', 'r', 'y', 'a', 'k', 'c', 'b', 'o', 'f', 
    'i', 'x', 'z'
]
df = df[df['habitat'].isin(valid_habitats)]


df['habitat'].value_counts()


df['cap-surface'].value_counts()


# æ–‡å­—åˆ—ã�§ã€�ã�‹ã�¤é•·ã�•ã�Œ1ã‚ˆã‚Šå¤§ã��ã�„è¡Œã‚’ç‰¹å®š
# æ–‡å­—åˆ—ã�«ã�™ã�¹ã�¦å¤‰æ›´ã�—ã�¦1æ–‡å­—
condition = df['cap-surface'].astype(str).str.len() > 1
df = df[~condition]
df['cap-surface'].value_counts()


print(df.isnull().sum())


df['cap-color'].value_counts()


# æ–‡å­—åˆ—ã�§ã€�ã�‹ã�¤é•·ã�•ã�Œ1ã‚ˆã‚Šå¤§ã��ã�„è¡Œã‚’ç‰¹å®š
# æ–‡å­—åˆ—ã�«ã�™ã�¹ã�¦å¤‰æ›´ã�—ã�¦1æ–‡å­—
condition = df['cap-color'].astype(str).str.len() > 1
df = df[~condition]
df['cap-color'].value_counts()


print(df.isnull().sum())


df['cap-shape'].value_counts()


# æ–‡å­—åˆ—ã�§ã€�ã�‹ã�¤é•·ã�•ã�Œ1ã‚ˆã‚Šå¤§ã��ã�„è¡Œã‚’ç‰¹å®š
# æ–‡å­—åˆ—ã�«ã�™ã�¹ã�¦å¤‰æ›´ã�—ã�¦1æ–‡å­—
condition = df['cap-shape'].astype(str).str.len() > 1
df = df[~condition]
df['cap-shape'].value_counts()


print(df.isnull().sum())


df['does-bruise-or-bleed'].value_counts()


condition = df['does-bruise-or-bleed'].astype(str).str.len() > 1
df = df[~condition]
df['does-bruise-or-bleed'].value_counts()


print(df.isnull().sum())


df['gill-attachment'].value_counts()


condition = df['gill-attachment'].astype(str).str.len() > 1
df = df[~condition]
df['gill-attachment'].value_counts()


print(df.isnull().sum())


df['gill-color'].value_counts()


condition = df['gill-color'].astype(str).str.len() > 1
df = df[~condition]
df['gill-color'].value_counts()


values_to_delete = ['4', '5']
condition = df['gill-color'].isin(values_to_delete)
df = df[~condition]
df['gill-color'].value_counts()


print(df.isnull().sum())


df['stem-color'].value_counts()


condition = df['stem-color'].astype(str).str.len() > 1
df = df[~condition]
df['stem-color'].value_counts()


print(df.isnull().sum())


df['has-ring'].value_counts()


df['ring-type'].value_counts()


values_to_delete = ['4', '1']
condition = df['ring-type'].isin(values_to_delete)
df = df[~condition]
df['ring-type'].value_counts()


print(df.isnull().sum())


df['season'].value_counts()


print(df.info())


# è¡¨ç¤ºã�™ã‚‹ã‚°ãƒ©ãƒ•ã�®ç‰¹å¾´é‡�ã�¾ã�¨ã‚�
features_to_plot = [
    'cap-diameter', 'cap-shape', 'cap-surface', 'cap-color', 
    'does-bruise-or-bleed', 'gill-attachment', 'gill-color', 
    'stem-height', 'stem-width', 'stem-color', 'has-ring', 
    'ring-type', 'habitat', 'season'
]

# forãƒ«ãƒ¼ãƒ—ã�§å�„ç‰¹å¾´é‡�ã�®ã‚°ãƒ©ãƒ•ã‚’é †ç•ªã�«ä½œæˆ�
for feature in features_to_plot:
    plt.figure(figsize=(10, 6))
    
    if df[feature].dtype == 'object':
        sns.countplot(data=df, x=feature, hue='class', palette='viridis')
        plt.title(f'{feature} ã�¨æ¯’æ€§ã�®é–¢ä¿‚')
    else:
        # æ•°å€¤ãƒ‡ãƒ¼ã‚¿ã�®å ´å�ˆ: ãƒ’ã‚¹ãƒˆã‚°ãƒ©ãƒ 
        sns.histplot(data=df, x=feature, hue='class', kde=True, palette='plasma')
        plt.title(f'{feature} ã�¨æ¯’æ€§ã�®é–¢ä¿‚')
        
    plt.show() # ã‚°ãƒ©ãƒ•ã‚’1ã�¤ã�šã�¤è¡¨ç¤º


print(df.isnull().sum())


print(df.describe())


print(df.describe(include='object'))


import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef

# ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿ã‚’èª­ã�¿è¾¼ã‚€
test_df = pd.read_csv('/kaggle/input/playground-series-s4e8/test.csv')

# æ��å‡ºã�«å¿…è¦�ã�ªidã‚’åˆ¥ã�®å¤‰æ•°ã�«ä¿�å­˜ã�—ã�¦ã�Šã��
test_ids = test_df['id']

# äºˆæ¸¬ã�«ä½¿ã�†ç‰¹å¾´é‡�ï¼ˆidåˆ—ä»¥å¤–ï¼‰ã‚’æº–å‚™
X_to_submit = test_df.drop('id', axis=1)


# æ��å‡ºç”¨ã�«ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿ã�®IDã‚’ä¿�æŒ�
test_id = test_df['id']

# IDåˆ—ã‚’å‰Šé™¤
df = df.drop('id', axis=1)
test_df = test_df.drop('id', axis=1)

# ===== ã‚¹ãƒ†ãƒƒãƒ—3: ãƒ©ãƒ™ãƒ«ã‚¨ãƒ³ã‚³ãƒ¼ãƒ‡ã‚£ãƒ³ã‚° =====
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


# æ¬ æ��ã�Œå¤šã��ã�¦é™¤å¤–ã�—ã�Ÿã�„ç‰¹å¾´é‡�
drop_cols = [
    'stem-root', 'stem-surface', 'veil-type', 'veil-color', 'spore-print-color',
    'gill-spacing', 'cap-surface', 'gill-attachment'
]

# å­¦ç¿’ãƒ»äºˆæ¸¬ã�«ä½¿ã�†ç‰¹å¾´é‡�ã‚’å®šç¾©
feature_cols = [col for col in df.columns if col not in drop_cols + ['class']]

# ç‰¹å¾´é‡�ã‚’åˆ¶é™�ã�—ã�¦å†�ä»£å…¥
X = df[feature_cols]
test_X = test_df[feature_cols]  # â†� ã�“ã‚Œã�Œ X_test ç›¸å½“ã�«ã�ªã‚Šã�¾ã�™
y = df['class']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

lgbm_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'learning_rate': 0.01,
    'n_estimators': 1097,
    'num_leaves': 956,
    'min_child_samples': 112,
    'reg_alpha': 0.19812645495932385,
    'reg_lambda': 0.34036226127746694,
    'colsample_bytree': 0.43199251589983806,
    'random_state': 42,
    'verbose': -1
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


# å†�åº¦ä½¿ã�£ã�Ÿç‰¹å¾´é‡�ã‚’æ˜�ç¤ºçš„ã�«å®šç¾©
drop_cols = ['stem-root', 'stem-surface', 'veil-type', 'veil-color',
             'spore-print-color', 'gill-spacing', 'cap-surface', 'gill-attachment']
feature_cols = [col for col in df.columns if col not in drop_cols + ['class']]

# å­¦ç¿’æ™‚ã�®ç‰¹å¾´é‡�ã�¨å�Œã�˜ã‚‚ã�®ã�«æ�ƒã�ˆã‚‹
X_test = test_df[feature_cols]

# ===== ã‚¹ãƒ†ãƒƒãƒ—3: æ��å‡ºç”¨ãƒ¢ãƒ‡ãƒ«ã�®ä½œæˆ�ã�¨äºˆæ¸¬ =====
# å…¨ã�¦ã�®è¨“ç·´ãƒ‡ãƒ¼ã‚¿ã‚’ä½¿ã�£ã�¦å†�å­¦ç¿’ã�—ã€�ãƒ¢ãƒ‡ãƒ«ã�®æ€§èƒ½ã‚’æœ€å¤§åŒ–
print("\nå…¨ãƒ‡ãƒ¼ã‚¿ã�§æœ€çµ‚ãƒ¢ãƒ‡ãƒ«ã‚’å†�å­¦ç¿’ã�—ã�¾ã�™...")
# verboseã‚’1ã�«ã�™ã‚‹ã�¨å­¦ç¿’é��ç¨‹ã�Œè¡¨ç¤ºã�•ã‚Œã‚‹
lgbm_params['verbose'] = 1 
final_model = lgb.LGBMClassifier(**lgbm_params)
final_model.fit(X, y)
print("å†�å­¦ç¿’ã�Œå®Œäº†ã�—ã�¾ã�—ã�Ÿã€‚")


# ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿ã�§äºˆæ¸¬
predictions_submission = final_model.predict(test_X)

# ç›®çš„å¤‰æ•°ã�®ãƒ�ãƒƒãƒ”ãƒ³ã‚°ã‚’å…ƒã�«æˆ»ã�™
class_map_reverse = {0: 'e', 1: 'p'}
predictions_labels = [class_map_reverse[pred] for pred in predictions_submission]

# æ��å‡ºç”¨ãƒ•ã‚¡ã‚¤ãƒ«ã�®ä½œæˆ�
submission_df = pd.DataFrame({'id': test_id, 'class': predictions_labels})
submission_df.to_csv('submission.csv', index=False)

print("\n'submission.csv' ã�Œä½œæˆ�ã�•ã‚Œã�¾ã�—ã�Ÿã€‚")
print(submission_df.head())

