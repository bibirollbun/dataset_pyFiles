import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("train.csv")
df.sample(3)


df.info()


df['Cover_Type'].value_counts()


X = df.drop('Cover_Type', axis=1)
y = df['Cover_Type']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.linear_model import LogisticRegression
lor = LogisticRegression()
lor.fit(X_train, y_train)


y_pred = lor.predict(X_test)


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)


y_pred = dtc.predict(X_test)


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)


y_pred = rfc.predict(X_test)


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


df.iloc[10].values


user_input = np.array([11, 2612,  201,    4,  180,   51,  735,  218,  243,  161, 6222,
          1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    1,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0])


rfc.predict([user_input]).reshape(1, -1)


import pickle
pickle.dump(rfc, open('rfc.pkl', 'wb'))

