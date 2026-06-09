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

# Fayllarni o'qish
train = pd.read_csv("/kaggle/input/binaryclassificationwithabankchurndataset/train.csv")
test = pd.read_csv("/kaggle/input/binaryclassificationwithabankchurndataset/test.csv")

# Train faylni ko'rish
print("Train faylining dastlabki 5 qatori:")
print(train.head())

# Test faylni ko'rish
print("\nTest faylining dastlabki 5 qatori:")
print(test.head())

# Ustunlar va null qiymatlar
print("\nTrain ustunlar:", train.columns.tolist())
print("Test ustunlar:", test.columns.tolist())

print("\nTrain null qiymatlar:\n", train.isnull().sum())
print("\nTest null qiymatlar:\n", test.isnull().sum())



# Keraksiz ustunlar: id, CustomerId, Surname
drop_cols = ["id", "CustomerId", "Surname"]

train = train.drop(drop_cols, axis=1)
test = test.drop(drop_cols, axis=1)

# Gender ustunini 0/1 kodlash
train["Gender"] = train["Gender"].map({"Male": 0, "Female": 1})
test["Gender"] = test["Gender"].map({"Male": 0, "Female": 1})

# Geography ustunini One-Hot Encoding (train va test bir xil ustunlar bo'lishi uchun drop_first=True)
train = pd.get_dummies(train, columns=["Geography"], drop_first=True)
test = pd.get_dummies(test, columns=["Geography"], drop_first=True)

# Natijani tekshirish
print("Train ustunlari:", train.columns.tolist())
print("Test ustunlari:", test.columns.tolist())


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Targetni ajratish
X = train.drop("Exited", axis=1)
y = train["Exited"]

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standartlashtirish
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
test_scaled = scaler.transform(test)  # test faylni ham scale qilamiz

# Logistic Regression modelini yaratish va o'qitish
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Prognoz qilish
y_pred = model.predict(X_val_scaled)

# Natijalarni baholash
print("Accuracy:", accuracy_score(y_val, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_pred))
print("\nClassification Report:\n", classification_report(y_val, y_pred))


from sklearn.ensemble import RandomForestClassifier

# Random Forest modeli (imbalanced class uchun class_weight='balanced')
rf_model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
rf_model.fit(X_train_scaled, y_train)

# Validation set bo‘yicha prognoz
y_val_pred = rf_model.predict(X_val_scaled)

# Natijalarni baholash
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_val_pred))
print("\nClassification Report:\n", classification_report(y_val, y_val_pred))


import matplotlib.pyplot as plt
import seaborn as sns

# Modeldan har bir ustunning ahamiyatini (importance) olish
importances = rf_model.feature_importances_
feature_names = X.columns  # Ustunlarning nomlari

# Ahamiyatlarni DataFrame ga joylashtirish va kamayish tartibida saralash
feat_imp = pd.DataFrame({
    "Ustun": feature_names,
    "Ahamiyat": importances
}).sort_values(by="Ahamiyat", ascending=False)

# Natijani chop etish
print(feat_imp)

# Vizualizatsiya qilish
plt.figure(figsize=(10,6))
sns.barplot(x="Ahamiyat", y="Ustun", data=feat_imp, palette="viridis")
plt.title("Random Forest Modelidagi Ustunlarning Ahamiyati")
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Feature importance va ustun nomlarini DataFrame ga joylashtiramiz
feat_imp = pd.DataFrame({
    "Ustun": ["CreditScore", "Age", "Balance", "EstimatedSalary", "IsActiveMember", 
              "NumOfProducts", "Tenure", "Geography_Spain", "Geography_Germany", 
              "HasCrCard", "Gender"],
    "Ahamiyat": [0.18, 0.17, 0.16, 0.14, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02]
})

# Ustunlar ahamiyatini kamayish tartibida saralash
feat_imp = feat_imp.sort_values(by="Ahamiyat", ascending=False)

# Grafik chizish
plt.figure(figsize=(10,6))
sns.barplot(x="Ahamiyat", y="Ustun", data=feat_imp, palette="viridis")
plt.title("Random Forest Modelidagi Ustunlarning Ahamiyati")
plt.xlabel("Ahamiyat")
plt.ylabel("Ustunlar")
plt.show()


# Test fayl bo‘yicha prognoz (chiqish ehtimoli)
test_predictions = rf_model.predict(test_scaled)

# Sample submission tayyorlash
submission = pd.DataFrame({
    "id": range(len(test_predictions)),  # agar original sample_submission bo‘lsa, id ustunini almashtiring
    "Exited": test_predictions
})

# CSV ga saqlash
submission.to_csv("submission.csv", index=False)
print("submission.csv tayyor!")

