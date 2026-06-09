path="/content/train.csv"


import pandas as pd
from matplotlib import pyplot as plt


df=pd.read_csv(path)
df.head()


df.info()


df.shape


df.describe().T


df.isna().any(axis=1).value_counts()


df.duplicated().value_counts()


col=df.columns
col


for i in col[1:]:
  plt.hist(df[i])
  plt.title(i)
  plt.xlabel(i)
  plt.ylabel("Frequency")
  plt.show()


plt.scatter(df['y'],df['age'],color='g',alpha=0.6)
plt.show()


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
cols=['job','marital','education','default','housing','loan', 'contact','month','poutcome']
def label_encoder(df,cols):
  df[cols]=df[cols].apply(le.fit_transform)
  return df
label_encoder(df,cols)


from sklearn.model_selection import train_test_split
X=df.drop(['id','y'],axis=1)
y=df['y']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)


from sklearn.ensemble import GradientBoostingClassifier
gbc=GradientBoostingClassifier()
gbc.fit(X_train,y_train)


y_pred=gbc.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


import xgboost as xgb
xgbc=xgb.XGBClassifier()
xgbc.fit(X_train,y_train)


xgbc_pred=xgbc.predict(X_test)
print(classification_report(y_test,xgbc_pred))


neg, pos = (y == 0).sum(), (y == 1).sum()
scale = neg / pos

model = xgb.XGBClassifier(
    scale_pos_weight=scale,
    eval_metric='logloss'
)
model.fit(X_train, y_train)


xgbc_pred=xgbc.predict(X_test)
print(classification_report(y_test,xgbc_pred))


# No change in model performace


df_test=pd.read_csv("/content/test.csv")
df_test.head()


label_encoder(df_test,cols)


pred=xgbc.predict_proba(df_test.drop(['id'],axis=1))
pred[:,1]


submission=pd.DataFrame({'id':df_test['id'],'y':pred[:,1]})
submission.to_csv('submission.csv',index=False)

