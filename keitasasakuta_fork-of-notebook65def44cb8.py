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
print(df.head(3))


# ãƒ‡ãƒ¼ã‚¿ã‚»ãƒƒãƒˆã�®è¡Œæ•°ã�¨åˆ—æ•°ã‚’ç¢ºèª�
print(f"{df.shape[0]}è¡Œ, {df.shape[1]}åˆ—")

# ãƒ‡ãƒ¼ã‚¿å…¨ä½“ã�®æƒ…å ±ç¢ºèª�
print(df.info())


print(df['cap-shape'].unique())


import numpy as np

def to_nan_if_number(val):
    try:
        float(val)
        return np.nan
    except:
        return val

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].apply(to_nan_if_number)


print(df['cap-shape'].unique())


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ---------- 1. æ¬ æ��å€¤ã‚’ "00" ã�§è£œå®Œ ----------
df_filled = df.copy()

for col in df_filled.columns:
    if df_filled[col].dtype == 'object':
        df_filled[col] = df_filled[col].fillna("00")
    else:
        df_filled[col] = df_filled[col].fillna(00)
# ---------- 2. ç›®çš„å¤‰æ•°yã�¨ç‰¹å¾´é‡�Xã�«åˆ†å‰² ----------
X = df_filled.drop('class', axis=1)
y = df_filled['class'].map({'e': 0, 'p': 1})


# ---------- 3. ç‰¹å¾´é‡�Xã‚’ãƒ¯ãƒ³ãƒ›ãƒƒãƒˆã‚¨ãƒ³ã‚³ãƒ¼ãƒ‡ã‚£ãƒ³ã‚° ----------
X_encoded = pd.get_dummies(X)

# True/False â†’ 1/0 ã�«å¤‰æ�›ï¼ˆæ˜�ç¤ºçš„ã�«å�‹å¤‰æ�›ï¼‰
X_encoded = X_encoded.astype(int)


print(f"ãƒ¯ãƒ³ãƒ›ãƒƒãƒˆå¾Œã�®åˆ—æ•°: {X_encoded.shape[1]}")


columns_with_00 = [col for col in X_encoded.columns if '00' in col]
print(columns_with_00)  # ã�¾ã�šã�“ã‚Œã�§è©²å½“åˆ—å��ã‚’ç¢ºèª�ã�™ã‚‹

X_encoded = X_encoded.drop(columns=columns_with_00)  # ç¢ºèª�ã�—ã�Ÿã�‚ã�¨ã�«å‰Šé™¤


print(f"ãƒ¯ãƒ³ãƒ›ãƒƒãƒˆå¾Œã�®åˆ—æ•°: {X_encoded.shape[1]}")


##ãƒ‡ãƒ¼ã‚¿ã�Œå¤šã�™ã��ã�¦ã‚¯ãƒ©ãƒƒã‚·ãƒ¥ã�—ã�¾ã�—ã�Ÿã€‚ã�ªã�®ã�§æ¶ˆã�—ã�¾ã�™
X_small = X_encoded.sample(n=150_000, random_state=42)
y_small = y.loc[X_small.index]
print(len(X_small))  # = 150000
print(len(y_small))  # = 150000


# ---------- 4. å­¦ç¿’ãƒ‡ãƒ¼ã‚¿ã�¨ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿ã�«åˆ†å‰² ----------
X_train, X_test, y_train, y_test = train_test_split(X_small, y_small, test_size=0.3, random_state=42)


# ---------- 5. ãƒ©ãƒ³ãƒ€ãƒ ãƒ•ã‚©ãƒ¬ã‚¹ãƒˆã�§å­¦ç¿’ ----------
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)


# ---------- 6. äºˆæ¸¬ã�¨è©•ä¾¡ ----------
y_pred = clf.predict(X_test)

print("\nğŸ�¯ ãƒ¢ãƒ‡ãƒ«è©•ä¾¡")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ---------- 7. ç‰¹å¾´é‡�é‡�è¦�åº¦ã�®ä¸Šä½�è¡¨ç¤ºï¼ˆä»»æ„�ï¼‰ ----------
importances = pd.Series(clf.feature_importances_, index=X_small.columns)
print("\nğŸ”¥ é‡�è¦�ã�ªç‰¹å¾´é‡�ãƒˆãƒƒãƒ—10:")
print(importances.sort_values(ascending=False).head(10))

