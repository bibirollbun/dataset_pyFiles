import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
df = pd.DataFrame(df)
df.tail()


df.isnull().sum()


df.describe()


df.sample(5)


X_train = df.drop(['id','rainfall'],axis=1)
X_train.head()


y_train = df['rainfall']
y_train


test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
test_df = pd.DataFrame(test_df)
test_df.head()


test_df.isnull().sum()


test_df['winddirection'].describe()


round(test_df['winddirection'].mean(),0)


test_df['winddirection'].fillna(round(test_df['winddirection'].mean(),0), inplace=True)


test_df.isna().sum()


X_test = test_df.drop(['id'],axis=1)
X_test.head()


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train = mms.fit_transform(X_train)


X_train.shape


LogisticRegression_model = LogisticRegression(C=1, solver='lbfgs')
LogisticRegression_model.fit(X_train, y_train)
LogisticRegression_model.score(X_train, y_train)


X_test.shape


# Get probabilities of rainfall (class 1)
y_predicted = LogisticRegression_model.predict_proba(X_test)[:, 1]  # Extract only column 1

# Convert to DataFrame with correct column name
y_predicted = pd.DataFrame(y_predicted, columns=['rainfall'])


y_predicted.head()


y_predicted['id'] = test_df['id']
y_predicted = y_predicted[['id', 'rainfall']]  # Swap columns


y_predicted.head(5)


y_predicted.to_csv('/kaggle/working/submission.csv', index=False)


svm_model = SVC(C=10)
svm_model.fit(X_train, y_train)
svm_model.score(X_train, y_train)


y_predicted = LogisticRegression_model.predict(X_test)
y_predicted = pd.DataFrame(y_predicted, columns=['rainfall'])


y_predicted['id'] = test_df['id']
y_predicted = y_predicted[['id', 'rainfall']]  # Swap columns
y_predicted.head(5)










