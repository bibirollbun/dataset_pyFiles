!pip install --no-cache-dir -U scikit-learn==1.5.2 imbalanced-learn==0.12.3 xgboost==2.1.1


import sklearn, xgboost, numpy as np
print("scikit-learn:", sklearn.__version__)
print("xgboost     :", xgboost.__version__)
print("numpy       :", np.__version__)



import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import seaborn as sns

pd.options.display.max_columns = None

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import *

from sklearn.ensemble import *
from xgboost import *


from sklearn.model_selection import train_test_split

df = pd.read_csv("/kaggle/input/basic-datasets/heart.csv")

X = df.drop(["output"], axis=1)
y = df["output"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split (X,y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)
y_hat = model.predict(X_test)

print('Accuracy :', accuracy_score(y_test, y_hat)) 
print('Matrice de confusion ;')
print(confusion_matrix(y_test, y_hat))
print(classification_report(y_test, y_hat))


from sklearn.model_selection import *

df = pd.read_csv("/kaggle/input/basic-datasets/heart.csv")

X = df.drop(["output"], axis=1)
y = df["output"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split (X,y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)
scores = cross_val_score(model, X, y, cv=100)
print('Accuracy moyenne ;',scores.mean())
print('Ecart type ;',scores.std())

RocCurveDisplay.from_estimator(model, X_test, y_test)


model.predict_proba(X_test)


from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("/kaggle/input/basic-datasets/heart.csv")

X = df.drop(["output"], axis=1)
y = df["output"].values


#arbre de décision
print('Arbre de décision')
model = DecisionTreeClassifier()
scores = cross_val_score(model, X, y, cv=100)
print('Accuracy moyenne ;',scores.mean())
print('Ecart type ;',scores.std())

#Regression logistique
print('Regression logistique')
model = DecisionTreeClassifier()
scores = cross_val_score(model, X, y, cv=100)
print('Accuracy moyenne ;',scores.mean())
print('Ecart type ;',scores.std())


from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("/kaggle/input/basic-datasets/heart.csv")

X = df.drop(["output"], axis=1)
y = df["output"].values


#arbre de décision
print('Arbre de décision niveau 3')

model = DecisionTreeClassifier(max_depth=3)
scores = cross_val_score(model, X, y, cv=100)
print('Accuracy moyenne ;',scores.mean())
print('Ecart type ;',scores.std())

#arbre de décision
print('Arbre de décision niveau 5')

model = DecisionTreeClassifier(max_depth=5)
scores = cross_val_score(model, X, y, cv=100)
print('Accuracy moyenne ;',scores.mean())
print('Ecart type ;',scores.std())

#arbre de décision
print('Arbre de décision niveau 10')

model = DecisionTreeClassifier(max_depth=10)
scores = cross_val_score(model, X, y, cv=100)
print('Accuracy moyenne ;',scores.mean())
print('Ecart type ;',scores.std())



df = pd.read_csv("/kaggle/input/basic-datasets/cancer.csv")
df.head()


df = pd.read_csv("/kaggle/input/basic-datasets/cancer.csv")

#preparation des donnees
df = df.drop(['id','Unnamed: 32'],axis=1)
df['diagnosis'] = df['diagnosis'].map({'B':0, 'M':1})

X = df.drop(["diagnosis"], axis=1)
y = df["diagnosis"].values

X_train, X_test, y_train, y_test = train_test_split (X,y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)
scores = cross_val_score(model, X, y, cv=100)
print('Accuracy moyenne ;',scores.mean())
print('Ecart type ;',scores.std())

RocCurveDisplay.from_estimator(model, X_test, y_test)


df = pd.read_csv("/kaggle/input/basic-datasets/penguins.csv")
df.head()


sns.pairplot(df[df.island=='Biscoe'], hue='species')


sns.pairplot(df, hue='species')


df.info


df = df.dropna()


df['sex'] = df['sex'].map({'male':0,'female':1})


df.head()


df = pd.get_dummies(df, columns=['island','species'])


df.head()


df = pd.read_csv("/kaggle/input/basic-datasets/penguins.csv")

#preparation des donnees
df = df.dropna()
df['sex'] = df['sex'].astype(str).str.lower().map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['island','species'])

#train/test
X = df.drop(["sex"], axis=1)
y = df["sex"].values

X_train, X_test, y_train, y_test = train_test_split (X,y, test_size=0.2)

#Arbre de decision
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_hat = model.predict(X_test)

# Matrique d'evaluation
print('Accuracy moyenne ;',accuracy_score(y_test, y_hat))
print('Matrique de confusion ;')
print(confusion_matrix(y_test,y_hat))
print(classification_report(y_test, y_hat))

RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.title("ROC curve (DecisionTree)")
plt.show()


df = pd.read_csv("/kaggle/input/basic-datasets/penguins.csv")

#preparation des donnees
df = df.dropna()
df['sex'] = df['sex'].map({'male':0,'female':1})
df = pd.get_dummies(df, columns=['island','species'])

#train/test
X = df.drop(["sex"], axis=1)
y = df["sex"].values

X_train, X_test, y_train, y_test = train_test_split (X,y, test_size=0.2)

#Arbre de decision
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_hat = model.predict(X_test)

# Matrique d'evaluation
print('RMSE : ', np.sqrt(mean_squared_error(y_test,y_hat)))
print('MAE : ', mean_absolute_error(y_test,y_hat))
print('MAPE : ', mean_absolute_percentage_error(y_test,y_hat))
print('Score R2 : ', r2_score(y_test,y_hat))

plt.scatter(y_test, y_hat)
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()], c='red')


df = pd.read_csv("/kaggle/input/basic-datasets/titanic.csv")
df.head()


df.info()


df['Age'] = df['Age'].fillna(df['Age'].mean())


df['Embarked'].value_counts()


df.columns


df = pd.read_csv("/kaggle/input/basic-datasets/titanic.csv")

#preparation des donnees
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Sex'] = df['Sex'].map({'male':0,'female':1})
df = pd.get_dummies(df, columns=['Embarked'])
df = df.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)

#train/test
X = df.drop(["Survived"], axis=1)
y = df["Survived"].values

X_train, X_test, y_train, y_test = train_test_split (X,y, test_size=0.2)

#Modèle Arbre de décision
model = XGBClassifier()
model.fit(X_train,y_train)
y_hat = model.predict(X_test)


# Métrique d'évaluation
print('Accuracy moyenne ;',accuracy_score(y_test,y_test))
print('Métrique de confusion')
print(confusion_matrix(y_test,y_hat))
print(classification_report(y_test, y_hat))

RocCurveDisplay.from_estimator(model, X_test, y_test)


df = pd.read_csv("/kaggle/input/basic-datasets/churn-small.csv")
df.head()


df.shape


df = pd.read_csv("/kaggle/input/basic-datasets/churn-big.csv")

#preparation des donnees

df['International plan'] = df['International plan'].map({'No':0,'Yes':1})
df['Voice mail plan'] = df['Voice mail plan'].map({'No':0,'Yes':1})
df = pd.get_dummies(df, columns=['State'])

#train/test
X = df.drop(["Churn"], axis=1)
y = df["Churn"].values

X_train, X_test, y_train, y_test = train_test_split (X,y, test_size=0.2)

#Modèle Arbre de décision
model = XGBClassifier()
model.fit(X_train,y_train)
y_hat = model.predict(X_test)


# Métrique d'évaluation
print('Accuracy;',accuracy_score(y_test,y_test))
print('Métrique de confusion')
print(confusion_matrix(y_test,y_hat))
print(classification_report(y_test, y_hat))

RocCurveDisplay.from_estimator(model, X_test, y_test)


df = pd.read_csv("/kaggle/input/credit-card-fraud-prediction/train.csv")
df.head()


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, RocCurveDisplay)
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt

df = pd.read_csv("/kaggle/input/credit-card-fraud-prediction/train.csv")

cols_to_drop = [c for c in ['id', 'Time'] if c in df.columns]
df = df.drop(columns=cols_to_drop)

X = df.drop(columns=["IsFraud"])
y = df["IsFraud"].values

#sampler = RandomUnderSampler()
#X, y = sampler.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_hat = model.predict(X_test)

print('Accuracy:', accuracy_score(y_test, y_hat))
print('Métrique de confusion:')
print(confusion_matrix(y_test, y_hat))
print(classification_report(y_test, y_hat))

RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.show()


from imblearn.under_sampling import RandomUnderSampler

sampler = RandomUnderSampler()
X,y = sampler.fit_resample(X,y)


from imblearn.over_sampling import RandomOverSampler

X,y = sampler.fit_resample(X,y)
df = pd.read_csv("/kaggle/input/credit-card-fraud-prediction/train.csv")

#preparation des donnees
df = df.drop(['id','Time'],axis=1)

#train/test
X = df.drop(["IsFraud"], axis=1)
y = df["IsFraud"].values

sampler = RandomOverSampler()
X,y = sampler.fit_resample(X,y)

X_train, X_test, y_train, y_test = train_test_split (X,y, test_size=0.2)


#Modèle Arbre de décision
model = RandomForestClassifier()
model.fit(X_train,y_train)
y_hat = model.predict(X_test)

# Métrique d'évaluation
print('Accuracy:',accuracy_score(y_test,y_hat))
print('Métrique de confusion:')
print(confusion_matrix(y_test,y_hat))
print(classification_report(y_test, y_hat))

RocCurveDisplay.from_estimator(model, X_test, y_test)

