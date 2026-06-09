import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier



df_train=pd.read_csv('/kaggle/input/playground-series-s3e5/train.csv')
df_test=pd.read_csv('/kaggle/input/playground-series-s3e5/test.csv')


df_train.shape


df_train.head()


df_train.isnull().sum()


df_train.duplicated().sum()


df_train.info()


df_train.quality.value_counts()


sns.countplot(x='quality',data=df_train) #highly imbalanced data


df_train.describe()


sns.histplot(df_train['total sulfur dioxide'])
plt.show()


sns.boxplot(df_train['total sulfur dioxide'])
plt.show()


df_train['total sulfur dioxide']=np.where(df_train['total sulfur dioxide']>200,df_train['total sulfur dioxide'].median(),df_train['total sulfur dioxide'])
df_test['total sulfur dioxide']=np.where(df_test['total sulfur dioxide']>200,df_test['total sulfur dioxide'].median(),df_test['total sulfur dioxide'])



sns.histplot(df_train['residual sugar'],bins=20)
plt.show()


sns.boxplot(df_train['residual sugar'])
plt.show()


df_train['residual sugar']=np.where(df_train['residual sugar']>7,df_train['residual sugar'].median(),df_train['residual sugar'])
df_test['residual sugar']=np.where(df_test['residual sugar']>7,df_test['residual sugar'].median(),df_test['residual sugar'])


df_train.quality.corr(df_train.density)


sns.boxplot(df_train['chlorides'])
plt.show()


df_train['chlorides']=np.where(df_train['chlorides']>0.3,df_train['chlorides'].median(),df_train['chlorides'])
df_test['chlorides']=np.where(df_test['chlorides']>0.3,df_test['chlorides'].median(),df_test['chlorides'])


df_train['so2_ratio'] = df_train['free sulfur dioxide'] / df_train['total sulfur dioxide']
df_test['so2_ratio'] = df_test['free sulfur dioxide'] / df_test['total sulfur dioxide']


df_train.corr()['quality'].sort_values(ascending=False)


plt.figure(figsize=(10,10))
sns.heatmap(df_train.corr(),annot=True,fmt='.2f',cmap='Blues')
plt.show()


X=df_train.drop(['Id','quality'],axis=1)
y=df_train['quality']


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


scaler=StandardScaler() #for linear models SVC and Logistic Regression

X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)


models = {
    "Logistic Regression": LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(class_weight='balanced', random_state=42),
    "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42),
    "SVM": SVC(class_weight='balanced', probability=True, random_state=42)
}
for name, model in models.items():
    print(f"\n  Model: {name}")
    
    # Use scaled data only for models that need it
    if name in ["Logistic Regression", "SVM"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    print(accuracy_score(y_test, y_pred),classification_report(y_test, y_pred,digits=3, zero_division=0))
final_model=RandomForestClassifier(class_weight='balanced', random_state=42)


final_model.fit(X_train, y_train)


y=df_test.drop('Id',axis=1)
y_pred=final_model.predict(y)


submission_df = pd.DataFrame({'Id': df_test['Id'], 'quality': y_pred})

# Сохранение submission в формате CSV
submission_df.to_csv('submission.csv', index=False)


submission_df.shape




