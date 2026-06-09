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


# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from lightgbm import early_stopping

import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


train


train.dtypes


train.isnull().sum()


test.isnull().sum()


train.dtypes


numerical_features = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']

plt.figure(figsize=(15,20))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(4, 4, i)
    sns.boxplot(x='Personality', y=feature, data=train)
    plt.title(f'Personality vs {feature}')
plt.tight_layout()
plt.show()


#Categorical features
sns.countplot(x='Stage_fear', hue='Personality', data=train)
plt.title('Stage Fear | Personality')
plt.show()

sns.countplot(x='Drained_after_socializing', hue='Personality', data=train)
plt.title('Drained After Socializing | Personality')
plt.show()


for col in numerical_features:
    train[col] = train[col].fillna(train[col].median())
    test[col] = test[col].fillna(train[col].median())

for col in numerical_features:
    train[col] = train.groupby('Personality')[col].transform(lambda x: x.fillna(x.median()))


#Создаем для категориальных признаков
categorical_features = ['Drained_after_socializing', 'Stage_fear']

for col in categorical_features:
    train_mode = train[col].mode()[0] # Берем этот col тем же модой(самое частое) 
    train[col] = train[col].fillna(train_mode) # Заполняем     
    test[col] = test[col].fillna(train_mode)

    train[col] = train[col].map({'Yes': 1, 'No': 0})
    test[col] = test[col].map({'Yes': 1, 'No': 0})
    


df = train.copy()
df['Target'] = df['Personality'].map({'Extrovert' : 1, 'Introvert' : 0})
df.drop('id', axis=1, inplace=True)

#binary_columns = ['Stage_fear', 'Drained_after_socializing']
#for col in binary_columns:
#    df[col] = df[col].map({'Yes' : 1, 'No' : 0})

# Introvert(L) -> 0, Extrovert(O) -> 1

plt.figure(figsize=(10,5))
sns.heatmap(data=df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation heatmap')
plt.show()


#Drained_after_socializing and Stage_Fear имеют 0.78 вместе, много дублей надо удалить одно из них
df.drop('Stage_fear', axis=1, inplace=True)
test.drop('Stage_fear', axis=1, inplace=True)


# Friends_circle_size(F) and Post_Frequency(P) имеют 0.63 и 0.65 коллеряции у target, однако с собой вдвоем имеют всего лишь 0.48, что значит они не сильно похожи
df['Social_activity_score'] = df['Friends_circle_size'] * df['Post_frequency']
test['Social_activity_score'] = test['Friends_circle_size'] * test['Post_frequency']
# Еще помогают признаки антагонисты
# Если Balance > 0 то значит экстроверт, если < 0 то интроверт
df['Balance'] = df['Going_outside'] - df['Time_spent_Alone']
test['Balance'] = test['Going_outside'] - test['Time_spent_Alone']


print(f"train columns:\n {df.columns.tolist()}")
print(f"test columns:\n {test.columns.tolist()}")


X_train = df.drop(['Personality', 'Target'], axis=1).values
y_train = df['Target'].values

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state=42)
scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
f1_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='f1')


print(f"Accuracy Scores: {scores}")
print(f"F1 Scores: {scores}")


clf.fit(X_train, y_train)
X_test = test.drop('id', axis=1)

preds = clf.predict(X_test)

sub = pd.DataFrame({
    'id' : test['id'],
    'Personality' : preds
})

sub['Personality'] = sub['Personality'].map({1 : 'Extrovert', 0 : 'Introvert'})
sub.to_csv('/kaggle/working/submission.csv', index=False)




