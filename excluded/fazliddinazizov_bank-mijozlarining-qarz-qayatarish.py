import pandas as pd
import numpy as np


test = pd.read_csv('/kaggle/input/binaryclassificationwithabankchurndataset/test.csv', index_col = 'id')
train = pd.read_csv('/kaggle/input/binaryclassificationwithabankchurndataset/train.csv', index_col = 'id')


train_test_data = [train, test] # train va test dataset'larini birlashtirish
sex_mapping = {"Male": 1, "Female": 0}
for dataset in train_test_data:
    dataset['Gender'] = dataset['Gender'].map(sex_mapping)


# delete unnecessary feature from dataset
train.drop('Surname', axis=1, inplace=True)
test.drop('Surname', axis=1, inplace=True)

train.drop('CustomerId', axis=1, inplace=True)
test.drop('CustomerId', axis=1, inplace=True)


train_test_data


test_cl = pd.get_dummies(test, columns=['Geography']).astype(float)
test_cl


from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()


train_cl = pd.get_dummies(train, columns=['Geography']).astype(float)
train_cl


train_cl.describe()


train_cl.info()


train_cl.select_dtypes(include=['number']).corrwith(train_cl['Exited'])


avoid=train_cl['Exited'].value_counts()/len(train_cl)*100
avoid


import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline


plt.figure(figsize=(5,5))
plt.pie(x=avoid, labels=['Qolgan','Ketgan'])
plt.show()


%matplotlib inline
train_cl.hist(bins=50, figsize=(20,15))
plt.show()


cols = train_cl.columns
for col in cols:
    plt.figure(figsize=(8,5))
    sns.histplot(data=train_cl, x=col, hue='Exited', bins=50, multiple='stack')
    plt.title(f'{col} by Exited')
    plt.show()


from sklearn.cluster import KMeans
X_train_fe = train_cl[['Age', 'NumOfProducts']]
X_test_fe  = test_cl[['Age', 'NumOfProducts']]

kmeans = KMeans(n_clusters=3, random_state=42)

# Fit clustering ONLY on train
train_cl['Cluster'] = kmeans.fit_predict(X_train_fe)

# Predict for test
test_cl['Cluster'] = kmeans.predict(X_test_fe)


    plt.figure(figsize=(8,5))
    sns.histplot(data=train_cl, x='Cluster', hue='Exited', bins=50, multiple='stack')
    plt.title(f'{col} by Exited')
    plt.show()


target = train_cl['Exited']
train_data = train_cl.drop('Exited', axis=1)
target.shape,  train_data.shape


test_data = test_cl


# Yuklash Classifier Modullarini

from sklearn.ensemble import RandomForestClassifier


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)


clf = RandomForestClassifier(n_estimators=13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# Random Forest score
round(np.mean(score)*100, 2)


clf.fit(train_data, target)


prediction = clf.predict(test_data)


submission = pd.DataFrame({
        'id': test_data.index,
        "Exited": prediction
    })

submission.to_csv('submission.csv', index=False)


submission = pd.read_csv('submission.csv')
submission.head(10)

