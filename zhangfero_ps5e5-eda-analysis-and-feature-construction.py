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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# è®¾ç½®æ–‡ä»¶è·¯å¾„
train_path = "/kaggle/input/playground-series-s5e5/train.csv"
test_path = "/kaggle/input/playground-series-s5e5/test.csv"

# è¯»å�–æ•°æ�®
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# æ•°æ�®æ¦‚å†µ
print("ğŸ”¹ Train shape:", train.shape)
print("ğŸ”¹ Test shape:", test.shape)
print("\nğŸ”¹ Train head:\n", train.head())
print("\nğŸ”¹ Train info:")
train.info()
print("\nğŸ”¹ Train describe:\n", train.describe())

# ç¼ºå¤±å€¼æ£€æµ‹
print("\nğŸ”¹ Missing values in train:\n", train.isnull().sum())

# åˆ é™¤IDæˆ–æ— å…³åˆ—ï¼ˆå¦‚æœ‰ï¼‰
drop_cols = [col for col in ['id', 'ID'] if col in train.columns]
if drop_cols:
    train.drop(columns=drop_cols, inplace=True)
    test.drop(columns=drop_cols, inplace=True)

# æ•°æ�®ç±»å�‹åˆ†ç¦»
numerical_features = train.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = train.select_dtypes(include=['object', 'category']).columns.tolist()


# ç›®æ ‡å�˜é‡�
target_col = 'calories_burned'
if target_col in train.columns:
    print("\nğŸ”¹ Target variable distribution:")
    sns.histplot(train[target_col], bins=50, kde=True)
    plt.title("Target Variable Distribution")
    plt.show()

# æ•°å€¼ç‰¹å¾�åˆ†å¸ƒ
for col in numerical_features:
    if col == target_col:
        continue
    plt.figure(figsize=(6, 3))
    sns.histplot(train[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.show()

# æ•°å€¼ç‰¹å¾�å’Œç›®æ ‡å�˜é‡�çš„ç›¸å…³æ€§
if target_col in train.columns:
    corr = train.corr()[target_col].sort_values(ascending=False)
    print("\nğŸ”¹ Correlation with target:\n", corr)

    plt.figure(figsize=(8, 5))
    sns.barplot(x=corr.index, y=corr.values)
    plt.title("Feature Correlation with Target")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# çƒ­åŠ›å›¾æŸ¥çœ‹æ•°å€¼ç‰¹å¾�ç›¸å…³æ€§
plt.figure(figsize=(12, 8))
sns.heatmap(train[numerical_features].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Numerical Feature Correlation Matrix")
plt.show()

# ç±»åˆ«ç‰¹å¾�é¢‘æ•°ï¼ˆå¦‚æ�œæœ‰ï¼‰
for col in categorical_features:
    plt.figure(figsize=(6, 3))
    train[col].value_counts().plot(kind='bar')
    plt.title(f"Value Counts of {col}")
    plt.tight_layout()
    plt.show()



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# è¯»å�–æ•°æ�®
train_path = "/kaggle/input/playground-series-s5e5/train.csv"
train = pd.read_csv(train_path)

# æ�„å»º BMI ç‰¹å¾�
train['BMI'] = train['Weight'] / ((train['Height'] / 100) ** 2)

# å¯¹å¹´é¾„è¿›è¡Œåˆ†ç»„
train['Age_Group'] = pd.cut(train['Age'], bins=[19, 30, 45, 60, 80], labels=["20-30", "31-45", "46-60", "61-80"])

# å�¯è§†åŒ–ï¼šä¸�å�Œå¹´é¾„æ®µçš„æ ·æœ¬æ•°
plt.figure(figsize=(6, 4))
sns.countplot(x='Age_Group', data=train, order=["20-30", "31-45", "46-60", "61-80"])
plt.title("Sample Count per Age Group")
plt.show()

# å�¯è§†åŒ–ï¼šä¸�å�Œå¹´é¾„æ®µçš„å¹³å�‡ BMI
plt.figure(figsize=(6, 4))
sns.barplot(x='Age_Group', y='BMI', data=train, order=["20-30", "31-45", "46-60", "61-80"])
plt.title("Average BMI per Age Group")
plt.show()

# å�¯è§†åŒ–ï¼šä¸�å�Œå¹´é¾„æ®µçš„å¹³å�‡çƒ­é‡�æ¶ˆè€—
plt.figure(figsize=(6, 4))
sns.barplot(x='Age_Group', y='Calories', data=train, order=["20-30", "31-45", "46-60", "61-80"])
plt.title("Average Calories Burned per Age Group")
plt.show()

# å�¯è§†åŒ–ï¼šBMI vs Calories æ•£ç‚¹å›¾ï¼ˆé€�æ˜�åº¦é™�ä½�ä¾¿äº�è§‚å¯Ÿå¯†åº¦ï¼‰
plt.figure(figsize=(6, 4))
sns.scatterplot(x='BMI', y='Calories', data=train, alpha=0.1)
plt.title("BMI vs Calories Burned")
plt.show()



# æ�„é€  Heart_Work = Heart_Rate Ã— Durationï¼ˆè¡¨ç¤ºå¿ƒè„�æ€»å·¥ä½œé‡�ï¼‰
train['Heart_Work'] = train['Heart_Rate'] * train['Duration']

# æ�„é€  BMI Ã— Duration äº¤äº’ç‰¹å¾�
train['BMI_Duration'] = train['BMI'] * train['Duration']

# æ•£ç‚¹å›¾ï¼šBMI Ã— Duration vs Calories
plt.figure(figsize=(6, 4))
sns.scatterplot(x='BMI_Duration', y='Calories', data=train, alpha=0.1)
plt.title("BMI Ã— Duration vs Calories Burned")
plt.show()

# æ•£ç‚¹å›¾ï¼šHeart_Work vs Calories
plt.figure(figsize=(6, 4))
sns.scatterplot(x='Heart_Work', y='Calories', data=train, alpha=0.1)
plt.title("Heart_Work vs Calories Burned")
plt.show()

# æ•£ç‚¹å›¾ï¼šDuration vs Caloriesï¼ŒæŒ‰ Age_Group ç�€è‰²
plt.figure(figsize=(6, 4))
sns.scatterplot(x='Duration', y='Calories', hue='Age_Group', data=train, alpha=0.1)
plt.title("Calories vs Duration by Age Group")
plt.legend(title="Age Group")
plt.show()

