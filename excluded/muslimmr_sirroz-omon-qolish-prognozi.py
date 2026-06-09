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

# Datasetlarni oâ€˜qib olish
train = pd.read_csv("/kaggle/input/multiclassificationtask/train.csv")
test = pd.read_csv("/kaggle/input/multiclassificationtask/test.csv")

# Birinchi 5 ta satrni koâ€˜ramiz
print("Train dataset:")
display(train.head())

print("Test dataset:")
display(test.head())


# shape orqali qator va ustunlar sonini koâ€˜ramiz.

print("Train shape:", train.shape)
print("Test shape:", test.shape)


print("TRAIN INFO:")
print(train.info())

print("\nTEST INFO:")
print(test.info())


# NaN qiymatlarni sanash
print("\nTrain NaN qiymatlar soni:")
print(train.isnull().sum())

print("\nTest NaN qiymatlar soni:")
print(test.isnull().sum())


# Kategoriyali ustunlar uchun NaN toâ€˜ldirish

cat_cols = ["Drug", "Sex", "Ascites", "Hepatomegaly", "Spiders", "Edema"]

for col in cat_cols:
    train[col] = train[col].fillna("missing")
    test[col] = test[col].fillna("missing")


# Sonli ustunlar uchun NaN toâ€˜ldirish
num_cols = ["Cholesterol", "Copper", "Alk_Phos", "SGOT", 
            "Tryglicerides", "Platelets", "Prothrombin"]

for col in num_cols:
    median_val = train[col].median()   # faqat trainâ€™dan median olish
    train[col] = train[col].fillna(median_val)
    test[col] = test[col].fillna(median_val)


print(train.isnull().sum().sum())  # Trainâ€™dagi jami NaN soni
print(test.isnull().sum().sum())   # Testâ€™dagi jami NaN soni


train.describe()


train.describe(include="object")


# Age ustunini toâ€˜gâ€˜ridan-toâ€˜gâ€˜ri yillarga aylantirish
train["Age"] = train["Age"] / 365
test["Age"] = test["Age"] / 365

# Natijani tekshirish
print(train["Age"].head())
print(test["Age"].head())


# Sonli ustunlar uchun describe
train.describe()


test.describe()


# Train va Testdagi Age ustunini 1â€“100 yil oralig'ida kesish
train["Age"] = train["Age"].clip(lower=1, upper=100)
test["Age"] = test["Age"].clip(lower=1, upper=100)

# Tekshirish
print(train["Age"].describe())
print(test["Age"].describe())


train.shape


test.shape


print("TRAIN DESCRIBE:")
print(train.describe())

print("\nTEST DESCRIBE:")
print(test.describe())



# Clip qilish uchun sonli ustunlar ro'yxati
numeric_cols = ["Platelets", "Alk_Phos", "SGOT"]

for col in numeric_cols:
    # Train 99% kvantil
    q99 = train[col].quantile(0.99)
    # Train va Testda clipping
    train[col] = train[col].clip(upper=q99)
    test[col] = test[col].clip(upper=q99)

# Tekshirish
print("TRAIN DESCRIBE:")
print(train[numeric_cols].describe())
print("\nTEST DESCRIBE:")
print(test[numeric_cols].describe())


# Kategoriyali ustunlar ro'yxati
categorical_cols = ["Drug", "Sex", "Ascites", "Hepatomegaly", "Spiders", "Edema"]

# Train va Testni one-hot encoding qilish
train_encoded = pd.get_dummies(train, columns=categorical_cols, drop_first=True)
test_encoded = pd.get_dummies(test, columns=categorical_cols, drop_first=True)

# Test ustunlarini train bilan moslashtirish (Status ustunini tashlab)
test_encoded = test_encoded.reindex(columns=train_encoded.columns.drop('Status'), fill_value=0)

# Tekshirish
print("Train shape:", train_encoded.shape)
print("Test shape:", test_encoded.shape)


import matplotlib.pyplot as plt
import seaborn as sns



plt.figure(figsize=(6,4))
sns.countplot(data=train, x='Status')
plt.title("Train dataset Status taqsimoti")
plt.show()


plt.figure(figsize=(8,4))
sns.histplot(train['Age'], bins=30, kde=True)
plt.title("Age taqsimoti (yillarda)")
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

cat_cols = ["Drug", "Sex", "Ascites", "Hepatomegaly", "Spiders", "Edema", "Status"]

for col in cat_cols:
    plt.figure(figsize=(8,4))
    
    sns.countplot(
        x=col, 
        data=train,
        order=train[col].value_counts().index,  # tartibni koâ€˜proqdan kamga qarab joylashtiradi
        palette="Set2"
    )
    
    plt.title(f"Countplot of {col} (Train)")
    plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# faqat train va testda umumiy mavjud boâ€˜lgan kategorik ustunlar
cat_cols = ["Drug", "Sex", "Ascites", "Hepatomegaly", "Spiders", "Edema"]

for col in cat_cols:
    fig, axes = plt.subplots(1, 2, figsize=(12,4), sharey=True)

    # Train
    sns.countplot(
        x=col, 
        data=train,
        order=train[col].value_counts().index,
        palette="Set2",
        ax=axes[0]
    )
    axes[0].set_title(f"Train - {col}")

    # Test
    sns.countplot(
        x=col, 
        data=test,
        order=test[col].value_counts().index,
        palette="Set2",
        ax=axes[1]
    )
    axes[1].set_title(f"Test - {col}")

    plt.tight_layout()
    plt.show()


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


from sklearn.preprocessing import LabelEncoder

#  Maqsad (target) ustun = "Status"
y = train["Status"]

# ID larni saqlab qolamiz
train_id = train["id"]
test_id = test["id"]

#  Status ni raqamli koâ€˜rinishga oâ€˜tkazamiz
le = LabelEncoder()
y = le.fit_transform(y)  # masalan: C->0, CL->1, D->2

#  X = qolgan ustunlar
X = train.drop(["Status", "id"], axis=1)
X_test = test.drop(["id"], axis=1)

#  Kategorik ustunlarni One-hot encoding qilamiz
X = pd.get_dummies(X)
X_test = pd.get_dummies(X_test)

#  Train va Testni ustunlar boâ€˜yicha moslashtiramiz
X, X_test = X.align(X_test, join="left", axis=1, fill_value=0)

#  Train setni boâ€˜lamiz (validatsiya uchun)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Modelni yaratamiz va oâ€˜qitamiz
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# Test uchun ehtimolliklarni olamiz
preds_proba = rf.predict_proba(X_test)

#  Submission faylni tayyorlaymiz
submission = pd.DataFrame(
    preds_proba,
    columns=["Status_C", "Status_CL", "Status_D"],  
)
submission.insert(0, "id", test_id)  # ID ustunni qoâ€˜shamiz


# 12. CSV faylga saqlaymiz
submission.to_csv("submission.csv", index=False)

print("âœ… submission.csv tayyor!")
print(submission.head())

