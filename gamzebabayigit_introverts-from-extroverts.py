import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB 
warnings.filterwarnings("ignore")


train=pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
submission=pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


submission.head()


train.head()


train.info()


train.shape


train.isnull().sum()


train = train.dropna()


test.info()


num_cols = test.select_dtypes(include=['float64', 'int64']).columns
test[num_cols] = test[num_cols].fillna(test[num_cols].median())
 
cat_cols = test.select_dtypes(include=['object']).columns
for col in cat_cols:
 
    if len(test[col].mode()) > 0:
        test[col] = test[col].fillna(test[col].mode()[0])
    else:
        test[col] = test[col].fillna("Unknown")

print("Test seti satır sayısı:", len(test))


train['Stage_fear'].value_counts()


train['Drained_after_socializing'].value_counts()


train['Personality'].value_counts()


train.head()


test_ids = test['id'].copy()


x_train=train.drop(['Personality'],axis=1)
y_train=train[['Personality']]

x_test = test.copy()


x_train = x_train.drop(['id'], axis=1)
x_test = x_test.drop(['id'], axis=1)


x_train = pd.get_dummies(x_train, drop_first=True)
x_test = pd.get_dummies(x_test, drop_first=True)


x_train, x_test, y_train, y_test=train_test_split(x_train, y_train, test_size=0.20, random_state=42)


g=GaussianNB()
b=BernoulliNB()


g.fit(x_train,y_train)


b.fit(x_train,y_train)


gtahmin=g.predict(x_test)


accuracy_score(y_test, gtahmin)


btahmin=b.predict(x_test)


accuracy_score(y_test, btahmin)


confusion_matrix(y_test, btahmin)


confusion_matrix(y_test, gtahmin)


sns.heatmap(confusion_matrix(y_test, gtahmin), annot=True)


print(classification_report(y_test,gtahmin))
print(classification_report(y_test,btahmin))


# Hedef ve Özellikleri Ayırma
X = train.drop('Personality', axis=1)  # Özellikler
y = train['Personality']               # Hedef (Introvert/Extrovert)


from sklearn.preprocessing import LabelEncoder



le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)


X_encoded = pd.get_dummies(X, drop_first=True)
test_encoded = pd.get_dummies(test, drop_first=True)


X_encoded, test_encoded = X_encoded.align(test_encoded, join='left', axis=1, fill_value=0)



from xgboost import XGBClassifier



model = XGBClassifier(
    n_estimators=1000,      # Ağaç sayısı
    learning_rate=0.05,     # Öğrenme hızı
    max_depth=6,            # Ağaç derinliği
    random_state=42,        # Tekrarlanabilirlik için
    n_jobs=-1               # Tüm işlemci çekirdeklerini kullan
)
model.fit(X_encoded, y_encoded)


predictions = model.predict(test_encoded)


# 0 ve 1'leri tekrar 'Introvert' ve 'Extrovert' metinlerine çeviriyoruz.
predictions_labels = le_target.inverse_transform(predictions)


submission = pd.DataFrame({
    'id': test_ids,
    'Personality': predictions_labels
})


submission.to_csv('submission.csv', index=False)


print(submission.head())




