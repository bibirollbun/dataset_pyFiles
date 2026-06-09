import pandas as pd
import datetime as dt
import seaborn as sb

d = dt.date.today()
Ymd = d.strftime('%Y%m%d')
Ymd


submission = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")
train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


print(train.shape)
print(test.shape)
submission.head(2)


display(train.head())
display(test.head())


display(train.info())
display(train.describe().T)


display(test.info())
display(test.describe().T)


missing_values = train.isnull().sum().reset_index()
missing_values.columns = ['Column', 'MissingCount']
missing_values['MissingPercentage'] = (missing_values['MissingCount'] / train.shape[0]) * 10
missing_values


missing_values = test.isnull().sum().reset_index()
missing_values.columns = ['Column', 'MissingCount']
missing_values['MissingPercentage'] = (missing_values['MissingCount'] / test.shape[0]) * 10
missing_values


test['winddirection'] = test['winddirection'].fillna(test['winddirection'].mean())
test.head()


import matplotlib.pyplot as plt
corr_matrix = train.corr()
dataplot = sb.heatmap(corr_matrix, cmap="coolwarm", annot=True, annot_kws={"size":8.5}, fmt=".2f", linewidths=0.5)
plt.figure(figsize =(15, 8))
plt.show()


train['temp_range'] = train['maxtemp'] - train['mintemp']
test['temp_range'] = test['maxtemp'] - test['mintemp']

train['temp_dewpoint'] = train['temparature'] - train['dewpoint']
test['temp_dewpoint'] = train['temparature'] - train['dewpoint']


train=train.drop(['maxtemp','mintemp','dewpoint'],axis=1)
test=test.drop(['maxtemp','mintemp','dewpoint'],axis=1)
train.head(2)


X = train.drop(['rainfall', 'id'], axis=1)
y = train['rainfall']


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

cv = KFold(n_splits=5, random_state=42, shuffle=True)


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc,roc_auc_score

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

LR_model = LogisticRegression() # logistic regression
KNN_model = KNeighborsClassifier() # KNN
RF_model = RandomForestClassifier(n_estimators=100, random_state=42)
SVM_model = SVC()

# fit model
LR_model.fit(X_train, y_train)
KNN_model.fit(X_train, y_train)  
RF_model.fit(X_train, y_train)
SVM_model.fit(X_train, y_train)


classifier = [LR_model,KNN_model,RF_model,SVM_model]
for m in classifier:
    score = cross_val_score(m, X_train, y_train, cv=5, scoring='accuracy')  
    print (f"score_{m} : {score}")
    for i, result in enumerate(score, 1):
        print(f"  Fold {i}: {result * 100:.2f}%")  
    print(f'Mean Accuracy: {score.mean()* 100:.2f}%')


# predict probabilities
LR_predict = LR_model.predict_proba(X_test)[:,1]
RF_predict = RF_model.predict_proba(X_test)[:,1]


import matplotlib.pyplot as plt

def plot_roc_curve(true_y, y_prob):
    """
    plots the roc curve based of the probabilities
    """

    fpr, tpr, thresholds = roc_curve(true_y, y_prob)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

plot_roc_curve(y_test, LR_predict)
print(f'LR AUC score: {roc_auc_score(y_test, LR_predict)}')

plot_roc_curve(y_test, RF_predict)
print(f'RF AUC score: {roc_auc_score(y_test, RF_predict)}')


F_test = test.drop(['id'],axis=1)
display(F_test.head(2))
F_test_transform = scaler.fit_transform(F_test)


LR_pr_test = LR_model.predict_proba(F_test_transform)[:,1]
RF_pr_test = RF_model.predict_proba(F_test_transform)[:,1]


submission_df = pd.DataFrame({'id': test.id, 'rainfall': LR_pr_test})
submission_df.head()




