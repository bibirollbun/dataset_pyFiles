
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


train_df=pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')


train_df.head()


train_df.shape


train_df.info()


train_df.describe()


train_df.isna().sum()


sns.countplot(x='Stage_fear', data=train_df)
plt.title('Value Counts of Stage Fear')
plt.ylabel('Count')
plt.show()


sns.countplot(x='Drained_after_socializing', data=train_df)
plt.title('Value Counsts of Drained After Socializing')
plt.ylabel('Count')
plt.show()


sns.countplot(x='Personality', data=train_df)
plt.title('Value Counts of Personality')
plt.ylabel('Count')
plt.show()


train_df.head()


numerical_cols_with_na=['Time_spent_Alone',
    'Social_event_attendance',
    'Going_outside',
    'Friends_circle_size',
    'Post_frequency']




plt.figure(figsize=(12, 8))

for ind, col in enumerate(numerical_cols_with_na):
    plt.subplot(2, 3, ind+1)
    sns.kdeplot(train_df[col].dropna(), fill=True)
    plt.title(f'KDE plot of {col}')

plt.tight_layout()
plt.show()


for col in numerical_cols_with_na:
    train_df[col].fillna(train_df[col].median(), inplace=True)


cat_cols_with_na=['Stage_fear', 'Drained_after_socializing']
for col in cat_cols_with_na:
    train_df[col].fillna(train_df[col].mode()[0], inplace=True)


label_encoder=LabelEncoder()
train_df['Personality']=label_encoder.fit_transform(train_df['Personality']) # Extrovert = 0 and Introvert =1


for col in cat_cols_with_na:
    train_df[col]=np.where(train_df[col]=='Yes', 1,0) # Yes = 1, No = 0


train_df.head()


train_df.isna().sum()


plt.figure(figsize=(10, 8))
sns.heatmap(train_df.corr(), annot=True)
plt.title("Correlation")


X=train_df.drop(columns=['Personality'])
y=train_df['Personality']


X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=2)


st=StandardScaler()
X_train_scaled=st.fit_transform(X_train)
X_test_scaled=st.transform(X_test)


model1=LogisticRegression()
model1.fit(X_train_scaled, y_train)
y_pred1=model1.predict(X_test_scaled)
acc=accuracy_score(y_test,y_pred1)
print("accuracy score on testing data: ",acc )


from sklearn.tree import DecisionTreeClassifier
model2=DecisionTreeClassifier(max_depth=5)
model2.fit(X_train,y_train)
y_pred2=model2.predict(X_test)
print("accuracy score on testing data: ",accuracy_score(y_test,y_pred2) )
print("accuracy score on training data: ",accuracy_score(y_train,model2.predict(X_train)))


from catboost import CatBoostClassifier
model3=CatBoostClassifier(verbose=0)
model3.fit(X_train,y_train)
y_pred3=model3.predict(X_test)
accuracy_score(y_test,y_pred3)
print("accuracy score on testing data: ",accuracy_score(y_test,y_pred3) )
print("accuracy score on training data: ",accuracy_score(y_train,model3.predict(X_train)))


model2.fit(X,y)


test_df=pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


test_df.shape


test_df.isna().sum()


for col in cat_cols_with_na:
    test_df[col].fillna(test_df[col].mode()[0], inplace=True)
for col in numerical_cols_with_na:
    test_df[col].fillna(test_df[col].median(), inplace=True)



for col in cat_cols_with_na:
    test_df[col] = np.where(test_df[col]=='Yes',1,0)


test_df.isna().sum()


X_test = test_df
y_pred_final=model2.predict(X_test)


submission=pd.DataFrame({
    'id': test_df['id'],
    'Personality':y_pred_final
})

submission.head()


submission['Personality']=np.where(submission['Personality']==1,'Introvert', 'Extrovert' )


submission.to_csv('submission.csv', index=False)

