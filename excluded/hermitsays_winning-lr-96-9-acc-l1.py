import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression, RidgeClassifier

import warnings
warnings.filterwarnings('ignore')


traindata = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
testdata = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

dftrain = pd.DataFrame(traindata)
dftest = pd.DataFrame(testdata)


dftrain.head()


dftest.head()


dftrain.isnull().sum()


imputeInt = SimpleImputer(strategy = 'median')
costoimpute = ['Time_spent_Alone', 'Social_event_attendance', 
              'Going_outside', 'Friends_circle_size', 'Post_frequency']

for i in costoimpute:
    dftrain[i] = imputeInt.fit_transform(dftrain[[i]])

dftrain.isnull().sum()


dftrain['Stage_fear'] = dftrain['Stage_fear'].fillna(dftrain['Stage_fear'].mode()[0])
dftrain['Drained_after_socializing'] = dftrain['Drained_after_socializing'].fillna(dftrain['Drained_after_socializing'].mode()[0])


dftrain.isnull().sum()


costoimpute = ['Time_spent_Alone', 'Social_event_attendance', 
              'Going_outside', 'Friends_circle_size', 'Post_frequency']

for i in costoimpute:
    dftest[i] = imputeInt.fit_transform(dftest[[i]])

dftest['Stage_fear'] = dftest['Stage_fear'].fillna(dftest['Stage_fear'].mode()[0])
dftest['Drained_after_socializing'] = dftest['Drained_after_socializing'].fillna(dftest['Drained_after_socializing'].mode()[0])


dftest.isnull().sum()


encoder = LabelEncoder()

cols_to_encode = ['Stage_fear', 'Drained_after_socializing', 'Personality']
for i in cols_to_encode:
    dftrain[i] = encoder.fit_transform(dftrain[[i]])


dftrain.head()


cols_to_encode = ['Stage_fear', 'Drained_after_socializing']
for i in cols_to_encode:
    dftest[i] = encoder.fit_transform(dftest[[i]])


dftest.head()


scaler = StandardScaler()

cols_to_scale = ['Time_spent_Alone', 'Social_event_attendance',
                'Going_outside', 'Friends_circle_size', 'Post_frequency'
                ]

for i in cols_to_scale:
    dftrain[i] = scaler.fit_transform(dftrain[[i]])


cols_to_scale = ['Time_spent_Alone', 'Social_event_attendance',
                'Going_outside', 'Friends_circle_size', 'Post_frequency'
                ]

for i in cols_to_scale:
    dftest[i] = scaler.fit_transform(dftest[[i]])


dftrain.head()


dftest.head()


correlation_matrix = dftrain.corr()

plt.figure(figsize=(12,8))
sns.heatmap(correlation_matrix,
            annot = True,
            cmap = 'coolwarm',
            fmt = '.2f',
            center = 0
            
           )

plt.title("Feature Correlation Heatmap")
plt.show()


X = dftrain.drop(['id', 'Personality'], axis=1)
y = dftrain['Personality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y )


logistic = LogisticRegression(class_weight='balanced',
                              C=0.1, 
                              penalty='l1', 
                              solver='saga',
                              max_iter=500
                             )
logistic.fit(X_train, y_train)


params = {
    'C':[0.01, 0.1, 0.5,1,5,10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga', 'lbfgs'],
    'max_iter': [500, 1000]
}

grid = GridSearchCV(LogisticRegression(max_iter=1000), params, cv=5)
grid.fit(X,y)
print('best parameters', grid.best_params_)
print('best score', grid.best_score_)


train_preds = logistic.predict(X_train)
train_acc = accuracy_score(y_train, train_preds)
print(f'âœ… Training Accuracy = {train_acc:.5f}')


test_preds = logistic.predict(X_test)
test_acc = accuracy_score(y_test, test_preds)
print(f'âœ… Testing Accuracy = {test_acc:.5f}')


print('classification report \n', classification_report(y_test, test_preds))
print('confusion matrix \n', confusion_matrix(y_test, test_preds))


cv_score = cross_val_score(logistic, X, y, cv=5)
print(f'cross validation score: {cv_score}')
print(f'cross validation mean: {cv_score.mean()}')


ids = dftest['id']

test_processed = dftest.drop(columns=['id'])

y_test_pred = logistic.predict(test_processed)

label_map = {0: 'Extrovert', 1: 'Introvert'}
y_test_pred = [label_map[i] for i in y_test_pred]

submission = pd.DataFrame({
    'id': ids,
    'Personality': y_test_pred
})


submission.to_csv('/kaggle/working/submission.csv', index=False)


print(submission.head())


