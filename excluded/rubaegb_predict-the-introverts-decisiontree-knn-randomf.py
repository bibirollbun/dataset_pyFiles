import pandas as pd

data = pd.read_csv("/kaggle/input/train-and-test-data/train.csv")
data


data.isnull().sum()


data.info()


data.describe().T.style.background_gradient(cmap='Reds')


data['Stage_fear'] = data['Stage_fear'].fillna('None')
data['Drained_after_socializing'] = data['Drained_after_socializing'].fillna('None')


data = data.dropna(subset=['id'])


data['Time_spent_Alone'] = data['Time_spent_Alone'].fillna(data['Time_spent_Alone'].mean())
data['Social_event_attendance'] = data['Social_event_attendance'].fillna(data['Social_event_attendance'].mean())
data['Going_outside'] = data['Going_outside'].fillna(data['Going_outside'].mean())
data['Friends_circle_size'] = data['Friends_circle_size'].fillna(data['Friends_circle_size'].mean())
data['Post_frequency'] = data['Post_frequency'].fillna(data['Post_frequency'].mean())


data.isnull().sum()


data.describe().T.style.background_gradient(cmap='Reds')


print('Count of people that are introvert and Stage_fear and Drained_after_socializing = true' ,((data['Personality'] == 'Introvert') & (data['Stage_fear'] == 'Yes') & (data['Drained_after_socializing'] == 'Yes')).sum())
print('Count of introverts: ', (data['Personality'] == 'Introvert').sum())


data['Stage_fear'] = data['Stage_fear'].replace({'Yes': 1, 'No': 0, 'None': -1})
data['Drained_after_socializing'] = data['Drained_after_socializing'].replace({'Yes': 1, 'No': 0, 'None': -1})


import seaborn as sns
import matplotlib.pyplot as plt

df = pd.DataFrame(data = data, columns = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency','Drained_after_socializing','Stage_fear'])

sns.heatmap(df.corr(), annot=True)
plt.title('Correlation Matrix')
plt.show()


# pie plot Stage_fear
plt.pie(data['Stage_fear'].value_counts(), labels=data['Stage_fear'].value_counts().index, autopct='%1.1f%%', colors=['#cc8b86ff','#f9eae1ff','#aa998fff'])
plt.title('Stage_fear')
plt.show()


# pie plot Drained_after_socializing
plt.pie(data['Drained_after_socializing'].value_counts(), labels=data['Drained_after_socializing'].value_counts().index, autopct='%1.1f%%', colors=['#cc8b86ff','#f9eae1ff','#aa998fff'])
plt.title('Drained_after_socializing')
plt.show()


# bar plot Time_spent_Alone
counts = data['Time_spent_Alone'].value_counts().sort_index()
plt.figure(figsize=(10, 6))
positions = range(len(counts))
plt.bar(positions, counts.values, color=['#cc8b86ff', '#f9eae1ff', '#aa998fff'])
plt.xticks(positions, counts.index, rotation=45)
plt.title('Time_spent_Alone')
plt.tight_layout()
plt.show()


# bar plot Social_event_attendance
counts = data['Social_event_attendance'].value_counts().sort_index()
plt.figure(figsize=(10, 6))
positions = range(len(counts))
plt.bar(positions, counts.values, color=['#cc8b86ff', '#f9eae1ff', '#aa998fff'])
plt.xticks(positions, counts.index, rotation=45)
plt.title('Social_event_attendance')
plt.tight_layout()
plt.show()


# bar plot Friends_circle_size
counts = data['Friends_circle_size'].value_counts().sort_index()
plt.figure(figsize=(10, 6))
positions = range(len(counts))
plt.bar(positions, counts.values, color=['#cc8b86ff', '#f9eae1ff', '#aa998fff'])
plt.xticks(positions, counts.index, rotation=45)
plt.title('Friends_circle_size')
plt.tight_layout()
plt.show()


# bar plot Going_outside
counts = data['Going_outside'].value_counts().sort_index()
plt.figure(figsize=(10, 6))
positions = range(len(counts))
plt.bar(positions, counts.values, color=['#cc8b86ff', '#f9eae1ff', '#aa998fff'])
plt.xticks(positions, counts.index, rotation=45)
plt.title('Going_outside')
plt.tight_layout()
plt.show()


# bar plot Post_frequency
counts = data['Post_frequency'].value_counts().sort_index()
plt.figure(figsize=(10, 6))
positions = range(len(counts))
plt.bar(positions, counts.values, color=['#cc8b86ff', '#f9eae1ff', '#aa998fff'])
plt.xticks(positions, counts.index, rotation=45)
plt.title('Post_frequency')
plt.tight_layout()
plt.show()


testData = pd.read_csv("/kaggle/input/train-and-test-data/test.csv")
testData['Drained_after_socializing'] = testData['Drained_after_socializing'].fillna('None')
testData['Stage_fear'] = testData['Stage_fear'].fillna('None')
testData['Drained_after_socializing'] = testData['Drained_after_socializing'].replace({'Yes': 1, 'No': 0, 'None': -1})
testData['Stage_fear'] = testData['Stage_fear'].replace({'Yes': 1, 'No': 0, 'None': -1})
testData['Time_spent_Alone'] = testData['Time_spent_Alone'].fillna(testData['Time_spent_Alone'].mean())
testData['Social_event_attendance'] = testData['Social_event_attendance'].fillna(testData['Social_event_attendance'].mean())
testData['Friends_circle_size'] = testData['Friends_circle_size'].fillna(testData['Friends_circle_size'].mean())
testData['Going_outside'] = testData['Going_outside'].fillna(testData['Going_outside'].mean())
testData['Post_frequency'] = testData['Post_frequency'].fillna(testData['Post_frequency'].mean())
testData


# split the data train, test
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

xTrain = data.drop(['Personality'], axis=1)
yTrain = data['Personality']

xTrainSplit, xVal, yTrainSplit, yVal = train_test_split(xTrain, yTrain, test_size=0.2, random_state=42)

xTest = testData


from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# tuning the algorithm for the best hyperparameters

# param_grid = {
#     'max_depth': [None, 5, 10, 20],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'criterion': ['gini', 'entropy']
# }

# tree = DecisionTreeClassifier()
# grid_search = GridSearchCV(tree, param_grid, cv=5)
# grid_search.fit(xTrainSplit, yTrainSplit)

# print("Best Parameters:", grid_search.best_params_)
# print("Best Score:", grid_search.best_score_)

# output
# Best Parameters: {'criterion': 'gini', 'max_depth': 5, 'min_samples_leaf': 2, 'min_samples_split': 10}
# Best Score: 0.9681494163509191


tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=2, min_samples_split=10, criterion='gini')
clf = tree.fit(xTrainSplit, yTrainSplit)
yPredDT = clf.predict(xTest)


from sklearn.metrics import accuracy_score, classification_report
yValPred = clf.predict(xVal)

print("Validation Accuracy:", accuracy_score(yVal, yValPred))
print("Validation Report:\n", classification_report(yVal, yValPred))


print("hii")


# from sklearn.svm import SVC

# # tuning the algorithm for the best hyperparameters

# param_grid = {
#     'C': [1, 10],
#     'kernel': ['linear', 'rbf'],
# }

# svc = SVC()
# random_search = GridSearchCV(svc, param_grid, cv=5)
# random_search.fit(xTrainSplit, yTrainSplit)

# print("Best Parameters:", random_search.best_params_)
# print("Best Score:", random_search.best_score_)

# tuning is taking so much time > 45mins, so we just skipped this algorithm.


# # SVC
# svc = SVC()
# svc.fit(xTrainSplit, yTrainSplit)
# yPredSVC = svc.predict(xTest)

# yValPred = svc.predict(xVal)

# print("Validation Accuracy:", accuracy_score(yVal, yValPred))
# print("Validation Report:\n", classification_report(yVal, yValPred))


# KNN
# tuning the algorithm for the best hyperparameters
from sklearn.neighbors import KNeighborsClassifier

# param_grid = {
#     'n_neighbors': [3, 5, 7],
#     'weights': ['uniform', 'distance'],
#     'p': [1, 2]
# }

# knn = KNeighborsClassifier()
# grid_search = GridSearchCV(knn, param_grid, cv=5)
# grid_search.fit(xTrainSplit, yTrainSplit)

# print("Best Parameters:", grid_search.best_params_)
# print("Best Score:", grid_search.best_score_)

# output
# Best Parameters: {'n_neighbors': 3, 'p': 1, 'weights': 'uniform'}
# Best Score: 0.944058525685433


# KNN
knn = KNeighborsClassifier(n_neighbors=3, p=1, weights='uniform')
knn.fit(xTrainSplit, yTrainSplit)
yPredKNN = knn.predict(xTest)

yValPred = knn.predict(xVal)

print("Validation Accuracy:", accuracy_score(yVal, yValPred))
print("Validation Report:\n", classification_report(yVal, yValPred))


from sklearn.ensemble import RandomForestClassifier

# rfc = RandomForestClassifier()
# params = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}
# grid = GridSearchCV(rfc, params, cv=5)
# yPredRFC = grid.fit(xTrainSplit, yTrainSplit)

# print("Best Parameters:", grid.best_params_)
# print("Best Score:", grid.best_score_)

# output
# Best Parameters: {'max_depth': 10, 'n_estimators': 200}
# Best Score: 0.96902663210637


rfc = RandomForestClassifier(max_depth=10, n_estimators=200)
rfc.fit(xTrainSplit, yTrainSplit)
yPredRFC = rfc.predict(xTest)

yValPred = rfc.predict(xVal)

print("Validation Accuracy:", accuracy_score(yVal, yValPred))
print("Validation Report:\n", classification_report(yVal, yValPred))


predictions = pd.DataFrame({'id': testData['id'], 'Personality': yPredRFC})
predictions.to_csv('predictions.csv', index=False)

