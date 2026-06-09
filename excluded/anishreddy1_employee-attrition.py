


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


df_train=pd.read_csv('/kaggle/input/playground-series-s3e3/train.csv')
df_test=pd.read_csv('/kaggle/input/playground-series-s3e3/test.csv')


df_train.shape


df_train.head()


df_train.info()


df_train.describe()


from sklearn.preprocessing import OrdinalEncoder,StandardScaler
encoder=OrdinalEncoder()

cat_cols = df_train.select_dtypes(include=['object']).columns.to_list()
df_train[cat_cols]=encoder.fit_transform(df_train[cat_cols])
df_test[cat_cols]=encoder.fit_transform(df_test[cat_cols])


stds=df_train.describe().loc[['std']]
drop_cols= [a for a in stds.columns if stds[a]['std']==0]


df_train.head()


df_train.drop(columns=drop_cols,inplace=True)
df_test.drop(columns=drop_cols,inplace=True)


X=df_train.drop('Attrition',axis=1)
y=df_train['Attrition']


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.fit_transform(X_test)


model1=LogisticRegression(class_weight='balanced')
model1.fit(X_train_scaled,y_train)


param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy'],
    'max_features': [None, 'sqrt', 'log2']
}


from sklearn.model_selection import GridSearchCV
model2=DecisionTreeClassifier(class_weight='balanced')
grid_search = GridSearchCV(model2, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)


grid_search.best_params_


from sklearn.metrics import classification_report


print(classification_report(y_test,model1.predict(X_test_scaled)))


print(classification_report(y_test,grid_search.predict(X_test)))


df_test_scaled=scaler.fit_transform(df_test)
y_pred=model1.predict_proba(df_test_scaled)[:, 1]
submission = pd.DataFrame({"id": df_test['id'], "Attrition": y_pred})
submission.to_csv("submission.csv", index=False)




