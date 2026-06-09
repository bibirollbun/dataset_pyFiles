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


# Kerakli kutubxonalarni import qilamiz
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
import lightgbm as lgb


# 1. Ma'lumotlarni yuklab olish
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


# 2. Ma'lumotlarni tahlil qilish
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Train null values:\n", train.isnull().sum())
print("Test null values:\n", test.isnull().sum())


# 3. Kategorik ustunlarni LabelEncoder bilan raqamlash
label_cols = ['Stage_fear', 'Drained_after_socializing']
le = LabelEncoder()
for col in label_cols:
    train[col] = le.fit_transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))


# Target (maqsad) ustunini ham raqamlash
target_encoder = LabelEncoder()
train['Personality'] = target_encoder.fit_transform(train['Personality'])


# 4. Bo'sh qiymatlarni to'ldirish
num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
            'Friends_circle_size', 'Post_frequency']
imputer = SimpleImputer(strategy='median')
train[num_cols] = imputer.fit_transform(train[num_cols])
test[num_cols] = imputer.transform(test[num_cols])


# 5. Model uchun X va y ajratish
X = train.drop(['id', 'Personality'], axis=1)
y = train['Personality']


# 6. Train-test split qilish (modelni baholash uchun)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# 7. LightGBM modelini yaratish va o'qitish

model = lgb.LGBMClassifier(random_state=42)
model.fit(X_train, y_train)


# 8. Validatsiya natijalarini ko'rish
y_pred = model.predict(X_val)
print("Accuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:\n", classification_report(y_val, y_pred))


# 9. Test ma'lumotlari uchun bashorat qilish
test_data = test.drop(['id'], axis=1)
test_predictions = model.predict(test_data)


# 10. Bashorat natijalarini asl nomlarga aylantirish
test['Personality'] = target_encoder.inverse_transform(test_predictions)


# 11. Submission fayl yaratish
submission = test[['id', 'Personality']]
submission.to_csv('submission.csv', index=False)

print("Yakunlandi. submission.csv fayli tayyor.")

