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


ship = pd.read_csv('/kaggle/input/2024_SMARCLE_KS_1/train.csv')
ship.head()


ship.info()


ship.describe()


# 문자형 컬럼 전처리 - 최빈값으로
for col in ['HomePlanet', 'Destination']:
    ship[col] = ship[col].fillna(ship[col].mode()[0])

# bool형 컬럼 전처리 - 최빈값으로
for col in ['CryoSleep', 'VIP']:
    ship[col] = ship[col].fillna(ship[col].mode()[0])

# 나이 전처리 - 중앙값으로
ship['Age'] = ship['Age'].fillna(ship['Age'].median())

# 숫자형 컬럼 전처리 - 0으로
num_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
ship[num_cols] = ship[num_cols].fillna(0)

# Cabin 컬럼 전처리
ship[['Deck', 'CabinNum', 'Side']] = ship['Cabin'].str.split('/', expand=True)
ship['Deck'] = ship['Deck'].fillna('U')
ship['CabinNum'] = ship['CabinNum'].fillna('0').astype(int)
ship['Side'] = ship['Side'].fillna('U')
ship = ship.drop('Cabin', axis=1)

ship['PassengerId'] = ship['PassengerId'].str.split('_').str[0].astype(int)

# 이름은 필요없어
ship = ship.drop('Name', axis=1)


ship.head()


ship.info()


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
obj = ['HomePlanet', 'Destination', 'Deck', 'Side']
for col in obj:
    ship[col] = le.fit_transform(ship[col])


ship.info()


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
nums=['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
ship[nums] = ss.fit_transform(ship[nums])
ship.describe()


X_data=ship[['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Deck', 'CabinNum', 'Side']]
y_data=ship['Transported']
X_data.head()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=512)
X_train


print(X_train.shape, X_test.shape)


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=512)
dt.fit(X_train, y_train)
print(dt.score(X_train, y_train))
print(dt.score(X_test, y_test))


import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plt.figure(figsize=(10, 7))
plot_tree(dt)
plt.show()


plt.figure(figsize=(10, 7))
plot_tree(dt, max_depth=1, filled=True, feature_names=['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Deck', 'CabinNum', 'Side'])
plt.show()


dt=DecisionTreeClassifier(max_depth=4, random_state=512)
dt.fit(X_train, y_train)
print(dt.score(X_train, y_train))
print(dt.score(X_test, y_test))


plt.figure(figsize=(20, 7))
plot_tree(dt, max_depth=4, filled=True, feature_names=['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Deck', 'CabinNum', 'Side'])
plt.show()

