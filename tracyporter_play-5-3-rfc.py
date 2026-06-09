import pandas as pd
import numpy as np
import os

from scipy.stats import ks_2samp

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")


train


train.info()


train.isna().sum()


test


test.info()


test.isna().sum()


for col in test:
    test[col] = test[col].fillna(test[col].mode()[0])

test.isna().sum().sum()


submission


train.drop('id', axis=1,inplace=True)
test.drop('id', axis=1,inplace=True)

train.shape, test.shape


to_drop = []

for col in test:
    stat, pv = ks_2samp(train[col], test[col])
    if pv < 0.05:
        to_drop.append(col)
print(to_drop)

train.drop(to_drop,axis=1,inplace=True)
test.drop(to_drop,axis=1,inplace=True)

print(train.shape, test.shape)


target = train.pop('rainfall')


plt.hist(target)
plt.show()


target.value_counts()


corr = train.corr()
sns.heatmap(corr)


# Create correlation matrix
corr_matrix = train.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find features with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
print(to_drop)

# Drop features 
#train.drop(to_drop, axis=1, inplace=True)
#test.drop(to_drop, axis=1, inplace=True)

train.shape, test.shape


y = target
X = train
X_test = test



model = RandomForestClassifier(class_weight='balanced', random_state=42).fit(X, y)
model.score(X, y)


feature_lst = []

for name, importance in zip(X.columns, model.feature_importances_):
    features = name, importance
    feature_lst.append(features)

print(*feature_lst, sep='\n')


feature_lst.sort(key=lambda x:x[1])
print(*feature_lst, sep='\n')


to_drop = []

for name, importance in feature_lst:
    if importance < 0.001:
        features = name, importance
        to_drop.append(features[0])

print(to_drop)
X.drop(to_drop, axis=1, inplace=True)
X_test.drop(to_drop, axis=1, inplace=True)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, random_state=42)
X_train.shape, y_train.shape, X_val.shape, y_val.shape


model = RandomForestClassifier(class_weight='balanced', random_state=42).fit(X_train, y_train)
model.score(X_train, y_train)


features = X.columns
importance = model.feature_importances_
indices = np.argsort(importance)


plt.title('Feature Importances')
plt.barh(range(len(indices)), importance[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()



y_pred = model.predict(X_val)
y_pred


acc = accuracy_score(y_val, y_pred)
acc


df = pd.DataFrame({'actual':y_val, 'predicted':y_pred})
df


pred = model.predict_proba(X_test)
pred = pred[:,1]
pred


submission['rainfall'] = pred
submission.to_csv('submission.csv', index=False)
submission = pd.read_csv('submission.csv')
submission


