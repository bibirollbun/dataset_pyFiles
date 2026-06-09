import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')


df.info()


df.describe().T


df = df.drop(['id'], axis = 1)


df.head


#sns.pairplot(df)


#plt.show()


#plt.figure(figsize=(12,8))
#sns.scatterplot(x='annual_income',y='credit_score',data=df,hue='loan_paid_back')


#plt.show


X = df.drop(['loan_paid_back'], axis = 1)


X = pd.get_dummies(X, drop_first = True)


y = df['loan_paid_back']


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score, auc
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV


#create train/test split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.40, random_state=101)  


param_grid = {
    "objective": ["binary"],
    "boosting_type":["gbdt"],
    "random_state": [42],
    'learning_rate': [0.05],
    'n_estimators': [1000],
    'num_leaves':[21],
    'max_depth': [10]
    }
lgbm_model = LGBMClassifier()
grid = GridSearchCV(lgbm_model,param_grid,cv=5)
grid.fit(X_train,y_train)


# predict target values for our features test dataset and print the classification report
grid_pred = grid.predict(X_test)
print(classification_report(y_test,grid_pred))


test_pred=grid.predict_proba(X_test)


test_pred


# prepare the test dataset to predict target values based on the fitted model and create a dataframe for submission
test_df=pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
test_df1=test_df.drop('id', axis=1)
test_df1=pd.get_dummies(test_df1, drop_first=True)
sub_pred=grid.predict_proba(test_df1)
sub_pred


submission = pd.DataFrame()
submission['id'] = test_df['id']
submission['loan_paid_back'] = grid.predict_proba(test_df1)[:,1]
file_name = 'submission.csv'
submission.to_csv(file_name, index=False)
submission




