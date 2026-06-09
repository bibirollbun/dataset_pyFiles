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

