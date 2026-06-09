train="/content/train.csv"
test="/content/test.csv"
sample_submission="/content/sample_submission.csv"


import pandas as pd


df=pd.read_csv(train)
df.head()


df.info()


df.isna().sum()


# From above data, it is clear that there are missing values, check for duplicate also
df.duplicated().sum()


df.describe().T


print(df['Time_spent_Alone'].isna().sum()/df['Personality'].count()) # sum -> count  nan values
print(df['Time_spent_Alone'].isna().count()/df['Personality'].count()) # count -> count non-nan values


df['Stage_fear'].unique()


group_col='Personality'


# Time_spent_Alone
# df.groupby('Personality')['Time_spent_Alone'].count() # Shows groupby result
# df['Time_spent_Alone'] = df['Time_spent_Alone'].fillna(df.groupby('Personality')['Time_spent_Alone'].transform('mean'))

target_cols=['Time_spent_Alone','Social_event_attendance','Going_outside','Friends_circle_size','Post_frequency']
def fill_missing_by_group_mean(df, group_col, target_cols):
    for col in target_cols:
        mean_values = df.groupby(group_col)[col].transform('mean').round()
        df[col] = df[col].fillna(mean_values)
    return df
fill_missing_by_group_mean(df, group_col, target_cols)


# Stage_fear
# df.groupby('Personality')['Stage_fear'].count() # Shows groupby result
# mode_map = df.groupby('Personality')['Stage_fear'].agg(lambda x: x.mode().iloc[0])
# df['Stage_fear'] = df['Stage_fear'].fillna(df['Personality'].map(mode_map))

target_cols1=['Stage_fear','Drained_after_socializing']
def fill_missing_with_mode_mapping(df, group_col, target_cols):
    for col in target_cols:
        df[col] = df[col].map({"No": 0, "Yes": 1})
        mode_map = df.groupby(group_col)[col].agg(lambda x: x.mode().iloc[0])
        df[col] = df[col].fillna(df[group_col].map(mode_map))
    return df

fill_missing_with_mode_mapping(df,group_col,target_cols1)


df['Stage_fear']=df['Stage_fear'].astype(int)
df['Drained_after_socializing']=df['Drained_after_socializing'].astype(int)


df.describe().T


import matplotlib.pyplot as plt


for col in target_cols:
  plt.hist(df[df['Personality']=="Extrovert"][col],color='dodgerblue',alpha=0.6)
  plt.hist(df[df['Personality']=="Introvert"][col],color='sandybrown',alpha=0.6)
  plt.xlabel(col)
  plt.ylabel("Frequency")
  plt.legend(["Extrovert","Introvert"],loc="upper left")
  plt.show()


# Change Personality value into integer
df['Personality'] = df['Personality'].map({'Introvert': 0, 'Extrovert': 1})


# Standard Scaler


df['Time_spent_Alone'].mean()


df.head()


X=df.drop(['id','Personality'],axis=1)
y=df['Personality']


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


lr=LogisticRegression(class_weight='balanced')
lr.fit(X_train,y_train)


from sklearn.tree import DecisionTreeClassifier


dt=DecisionTreeClassifier(class_weight='balanced')
dt.fit(X_train,y_train)


from sklearn.ensemble import RandomForestClassifier


rf=RandomForestClassifier(class_weight='balanced')
rf.fit(X_train,y_train)


y_pred_lr=lr.predict(X_test)
# accuracy_score(y_test,y_pred_lr)
print(classification_report(y_test,y_pred_lr))


y_pred_dt=dt.predict(X_test)
# accuracy_score(y_test,y_pred_dt)
print(classification_report(y_test,y_pred_dt))


y_pred_rf=rf.predict(X_test)
# accuracy_score(y_test,y_pred_rf)
print(classification_report(y_test,y_pred_rf))


# Here as seen in Classification report, Random Forest has better result


df_test=pd.read_csv('test.csv')


df_test['Stage_fear']=df_test['Stage_fear'].map({'No':0,'Yes':1})
df_test['Drained_after_socializing']=df_test['Drained_after_socializing'].map({'No':0,'Yes':1})


df_test['Time_spent_Alone']=df_test['Time_spent_Alone'].fillna(df_test['Time_spent_Alone'].mean().round())
df_test['Stage_fear']=df_test['Stage_fear'].fillna(df_test['Stage_fear'].mode())
df_test['Social_event_attendance']=df_test['Social_event_attendance'].fillna(df_test['Social_event_attendance'].mean())
df_test['Going_outside']=df_test['Going_outside'].fillna(df_test['Going_outside'].mean())
df_test['Drained_after_socializing']=df_test['Drained_after_socializing'].fillna(df_test['Drained_after_socializing'].mode())
df_test['Friends_circle_size']=df_test['Friends_circle_size'].fillna(df_test['Friends_circle_size'].mean())
df_test['Post_frequency']=df_test['Post_frequency'].fillna(df_test['Post_frequency'].mean().round())


df_test.info()


X_test_pred=df_test.drop(['id'],axis=1)


y_pred=rf.predict(X_test_pred)


submission=pd.DataFrame({'id':df_test['id'],'Personality':y_pred})


submission['Personality']=submission['Personality'].map({0:'Introvert',1:'Extrovert'})


submission.to_csv('submission.csv',index=False)










