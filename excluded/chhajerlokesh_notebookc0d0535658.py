import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


X=pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')



X.head()


X.isnull().sum()
#ALL VAlues are non null


X.info()


X[['annual_income', 'debt_to_income_ratio', 'credit_score',
       'loan_amount', 'interest_rate']].describe()


sns.set()



plt.figure(figsize=(6,6))
plt.scatter(X['annual_income'],X['loan_amount'])
plt.xlabel('annual_income')
plt.ylabel('loan_amount')


X.columns





X.shape


plt.figure(figsize=(6,6))
sns.kdeplot(X['annual_income'])


sns.kdeplot(X['loan_amount'])


sns.distplot(X['credit_score'])


sns.countplot(x='gender',data=X)


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
sns.countplot(ax=axes[0, 0], x='marital_status', data=X)
axes[0, 0].set_title('Marital status count')
axes[0,0].tick_params(axis='x', rotation=45) 
sns.countplot(ax=axes[0, 1], x='education_level', data=X)
axes[0, 1].set_title('education_level count')
axes[0,1].tick_params(axis='x', rotation=45) 
sns.countplot(ax=axes[1, 0], x='loan_purpose', data=X)
axes[1, 0].set_title('loan_purpose count')
axes[1,0].tick_params(axis='x', rotation=45) 
sns.countplot(ax=axes[1, 1], x='grade_subgrade', data=X)
axes[1, 1].set_title('grade_subgrade count')
axes[1,1].tick_params(axis='x', rotation=45) 


Index(['id', 'annual_income', 'debt_to_income_ratio', 'credit_score',
       'loan_amount', 'interest_rate', 'gender', 'marital_status',
       'education_level', 'education_level', 'loan_purpose',
       'grade_subgrade', 'loan_paid_back'],
      dtype='object')


sns.set()
df=X[['annual_income', 'debt_to_income_ratio', 'credit_score',
       'loan_amount', 'interest_rate','loan_paid_back']]
corr_matrix=df.corr()
plt.figure(figsize=(8, 6)) # Adjust figure size
sns.heatmap(corr_matrix,
            annot=True,       # Show values in cells
            cmap='coolwarm',  # Choose a color map
            fmt=".2f",        # Format annotations to two decimal places
            linewidths=.5,    # Add lines between cells
            linecolor='black' # Set line color
           )
plt.title('Correlation Heatmap of Features') # Add a title
plt.show()


X.shape


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
X['gender']=encoder.fit_transform(X['gender'])
X['marital_status']=encoder.fit_transform(X['marital_status'])
X['education_level']=encoder.fit_transform(X['education_level'])
X['loan_purpose']=encoder.fit_transform(X['loan_purpose'])
X['grade_subgrade']=encoder.fit_transform(X['grade_subgrade'])

X['employment_status']=encoder.fit_transform(X['employment_status'])


X.head()


X['grade_subgrade'].describe()
X=X.drop(columns='grade_subgrade')


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)



lr=LogisticRegression()
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)


from sklearn.metrics import r2_score
r2score=r2_score(y_test,y_pred)


r2score










