import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


train_data.head()


# All columns are in numeric format. Day and ID is not needed.
train_data.info()


# Checking for any missing values

train_data.isnull().sum()


# First we will drop 'id' and 'day' columns

train_data.drop(labels=['id','day'],axis=1,inplace=True)
test_data.drop(labels=['id','day'],axis=1, inplace=True)
train_data


train_data.corr()


plt.figure(figsize=(20,10))
sns.heatmap(train_data.corr(), annot=True)
plt.show()


correlations = train_data.corr()
exceeding_corrs = {}
for i in correlations.columns:
    for j in correlations.columns:
        if abs(correlations[i][j]) > 0.5 and i != j:
            key_exists = i + ' ' + j
            key_reverse_exists = j + ' ' + i
            if key_exists not in exceeding_corrs and key_reverse_exists not in exceeding_corrs:
                exceeding_corrs[i+' '+j] = correlations[i][j]

exceeding_corrs


print(train_data['mintemp'].mean())
print(train_data['maxtemp'].mean())
print(train_data['temparature'].mean())


train_data[['temparature','mintemp','maxtemp']]


train_data['avg_temp'] = test_data.apply(lambda row: np.mean([row['temparature'], row['mintemp'], row['maxtemp']]), axis=1)
test_data['avg_temp'] = train_data.apply(lambda row: np.mean([row['temparature'], row['mintemp'], row['maxtemp']]), axis=1)


train_data.drop(labels=['temparature','mintemp','maxtemp'], axis=1, inplace=True)
test_data.drop(labels=['temparature','mintemp','maxtemp'], axis=1, inplace=True)


train_data


plt.figure(figsize=(20,10))
sns.heatmap(train_data.corr(), annot=True)
plt.show()


correlations = train_data.corr()
exceeding_corrs = {}
for i in correlations.columns:
    for j in correlations.columns:
        if abs(correlations[i][j]) > 0.5 and i != j:
            key_exists = i + ' ' + j
            key_reverse_exists = j + ' ' + i
            if key_exists not in exceeding_corrs and key_reverse_exists not in exceeding_corrs:
                exceeding_corrs[i+' '+j] = correlations[i][j]

exceeding_corrs


def correlation(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr

correlation(train_data, 0.80)


cols = ['pressure', 'avg_temp', 'winddirection', 'sunshine', 'humidity']
plt.figure(figsize=(20,10))
len(train_data.columns)
for i, el in enumerate(train_data.columns):
    plt.subplot(3,3,i+1)
    sns.histplot(train_data[el], kde=True)


from sklearn.mixture import GaussianMixture


gmm = GaussianMixture(n_components=7, random_state=42)
df = train_data.copy()

df['Cluster'] = gmm.fit_predict(df[['winddirection']])


n_components = range(1, 10)
bics = []
aics = []

for n in n_components:
    gmm = GaussianMixture(n_components=n, random_state=42)
    gmm.fit(df[['winddirection']])
    bics.append(gmm.bic(df[['winddirection']]))
    aics.append(gmm.aic(df[['winddirection']]))

# Plot
plt.figure(figsize=(8, 4))
plt.plot(n_components, bics, label='BIC', marker='o')
plt.plot(n_components, aics, label='AIC', marker='s')
plt.xlabel('n_components')
plt.ylabel('Score')
plt.title('BIC & AIC vs Number of Components')
plt.legend()
plt.grid(True)
plt.show()


sns.histplot(df['Cluster'])


from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion='entropy', random_state=42)


train_data.columns


X = train_data.drop(labels=['rainfall'],axis=1)


y = train_data['rainfall']


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)


classifier.fit(X_train, y_train)


importances = classifier.feature_importances_
importances


y_pred = classifier.predict(X_test)

from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(acc)
print(cm)
print(f1)


y_pred = classifier.predict(test_data)
y_pred


test_ids = pd.read_csv('test.csv')
submission_df = pd.DataFrame({
    'id': test_ids['id'],
    'rainfall': y_pred
})
submission_df.to_csv('submission.csv', index=False)




