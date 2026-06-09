import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import warnings
warnings.filterwarnings('ignore')


traindata = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
testdata = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")

dftrain = pd.DataFrame(traindata)
dftest = pd.DataFrame(testdata)

dftrain.head()


dftrain.isnull().sum()


dftest.isnull().sum()


dftrain.drop(['id', 'job', 'day', 'month'], axis=1, inplace = True)
dftest.drop(['job', 'day', 'month'], axis=1, inplace = True)


dftrain.head()


dftest.head()


dftrain.info()


dftrain['poutcome'].value_counts()


cols_to_encode = ['marital', 'marital', 'education', 'default', 'housing', 
                  'loan', 'contact', 'poutcome']

le = LabelEncoder()

for i in cols_to_encode:
    dftrain[i] = le.fit_transform(dftrain[i])
    dftest[i] = le.fit_transform(dftest[i])

dftrain.head()


cm = dftrain.corr()

plt.figure(figsize=(12,6))
sns.heatmap(
    cm,
    annot = True,
    fmt = '.2g',
    cmap = 'coolwarm'
)
plt.title("Feature correlation matrix")
plt.tight_layout()
plt.show()



X = dftrain.drop(['y'], axis=1)
y = dftrain['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sc = StandardScaler()

X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.fit_transform(X_test)


logistic = LogisticRegression(
    class_weight = 'balanced',
    n_jobs=-1,
    penalty = 'l2',
    solver = 'saga',
    C = 10
)
logistic.fit(X_train_sc, y_train)


log_train_preds = logistic.predict(X_train_sc)
acc = accuracy_score(y_train, log_train_preds)

print("Training accuracy of logistic : ", acc)

log_test_preds = logistic.predict(X_test_sc)
tacc = accuracy_score(y_test, log_test_preds)

print("Testing accuracy of logistic : ", tacc)


hgb = HistGradientBoostingClassifier(
    learning_rate = 0.01,
    max_iter = 500,
    class_weight = 'balanced',
    random_state=42,
    max_depth = 7,
    l2_regularization = 1
)
hgb.fit(X_train, y_train)


gbm_train_preds = hgb.predict(X_train)
acc = accuracy_score(y_train, gbm_train_preds)

print("Training accuracy of HistGradient : ", acc)

gbm_test_preds = hgb.predict(X_test)
tacc = accuracy_score(y_test, gbm_test_preds)

print("Testing accuracy of HistGradient : ", tacc)


print("classification report \n", classification_report(y_test, gbm_test_preds))
print("confusion matrix \n", confusion_matrix(y_test, gbm_test_preds))


cv_score = cross_val_score(hgb, X, y, cv=5)
print(f'cross validation score: {cv_score}')
print(f'cross validation mean: {cv_score.mean()}')


ids = dftest['id']
test_processed = dftest.drop(columns=['id'])
y_test_pred = hgb.predict(test_processed)

submission = pd.DataFrame({
    'id': ids,
    'y': y_test_pred
})


submission.to_csv('/kaggle/working/submission.csv', index=False)


print(submission.head())




