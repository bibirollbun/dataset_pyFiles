import os
test_path = ''
train_path = ''
submission_path = ''
for dirname,_,file_name in os.walk('/kaggle/input'):
    for file in file_name:
        if file.startswith('test'):
            test_path = os.path.join(dirname,file)
        elif file.startswith('train'):
            train_path = os.path.join(dirname,file)
        else:
            submission_path = os.path.join(dirname,file)
print("path created successfully!!!")
print(f"train path: {train_path}")
print(f"test path: {test_path}")
print(f"submission path: {submission_path}")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
sns.set_style('whitegrid')
import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
submission_df = pd.read_csv(submission_path)
print("All Dataframe created successfully!!!!")


train_df.head()


train_df.describe().T


train_df.isna().sum()


train_df.iloc[:,0] ## index is their 
train_df = train_df.iloc[:,1:]
train_df.head()


numerical_features = train_df.select_dtypes(include=['int64','float64']).columns
categorical_features = train_df.select_dtypes(include=['category','object']).columns
X = train_df.iloc[:,:-1]
y = train_df.iloc[:,-1]



X.head()


plt.figure(figsize=(8,6))
plt.title("KDE (annual_income)")
sns.histplot(X['annual_income'],bins=50,kde=True)
plt.xlabel("Bins")
plt.ylabel("Count")
plt.show()


plt.figure(figsize=(8,6))
plt.title(f"KDE (debt_to_income_ratio)")
sns.histplot(X['debt_to_income_ratio'],bins=50,kde=True)
plt.xlabel("Bins")
plt.ylabel("Count")
plt.show()


## instret rate per loan
irrl = X['interest_rate'] / X['loan_amount']
plt.figure(figsize=(8,6))
plt.title("KDE (interest rate per loan)")
sns.histplot(irrl,bins=50,kde=True)
plt.xlabel("Bins")
plt.ylabel("Count")
plt.show()


-3*X['loan_amount'].std()


## lets check the average loan amount
mean = X['loan_amount'].mean()
std = X['loan_amount'].std()
plt.figure(figsize=(8,6))
plt.title("KDE (loan_amount)")
sns.histplot(X['loan_amount'],bins=50,kde=True)
plt.axvline(mean,ymin=0,ymax=48959.95,linestyle = '--',linewidth=1,color = 'blue',label=f'mean = {mean:.2f}')
plt.axvline(mean+(3*std),ymin=0,ymax=48959.95/2,linestyle = '--',linewidth=1,color = 'green',label=f'+3σ = {mean + 3*std:.2f}')
plt.axvline(mean-(3*std),ymin=0,ymax=48959.95/2,linestyle = '--',linewidth=1,color = 'green',label=f'+3σ = {mean - 3*std:.2f}')
plt.xlabel("Bins")
plt.ylabel("Count")
plt.legend()
plt.show()


numerical_features


plt.figure(figsize=(8,6))
plt.title("Annual Income vs Credit Score")
sns.scatterplot(data=train_df,x='annual_income',y='credit_score',hue='loan_paid_back')
plt.xlabel("Annual Income")
plt.ylabel("Credit Score")
plt.show()


plt.figure(figsize=(8,6))
plt.title("Loan Amount vs Credit Score")
sns.scatterplot(data=train_df,x='loan_amount',y='credit_score',hue='loan_paid_back')
plt.axvline(X['loan_amount'].mean(),ymin=0,ymax=X['loan_amount'].max(),linestyle='--',linewidth=2,color='black',label=f"mean loan amount: {train_df['loan_amount'].mean():.2f}")
plt.axhline(X['credit_score'].mean(),xmin=0,xmax=X['credit_score'].max(),linestyle='--',linewidth=2,color='green',label=f"mean credit score: {train_df['credit_score'].mean():.2f}")
plt.xlabel("Loan Amount")
plt.ylabel("Credit Score")
plt.show()


corr_ = train_df[numerical_features].corr()
sns.heatmap(corr_,annot=True,fmt=".4f",linewidth=0.5)


plt.figure(figsize=(8,6))
plt.title("credit_score vs loan_paid_back")
sns.scatterplot(data=train_df,x='credit_score',y='loan_paid_back',hue='loan_paid_back')
plt.xlabel("credit_score")
plt.ylabel("loan_paid_back")
plt.show()


numerical_features


fig,ax = plt.subplots(nrows=2,ncols=3,figsize=(8,6))
numerical_feature = numerical_features[:-1]
count = 0
i,j=0,0
for i in range(2):
    for j in range(3):  
        if count >= len(numerical_feature):   # <-- Correct condition
            ax[i][j].axis('off')              # hide unused subplots
            continue
        sns.histplot(X[numerical_feature[count]],bins=50,kde=True,ax=ax[i][j])
        count+=1
        
    
plt.tight_layout()
plt.show()


X['annual_income'] = np.log1p(X['annual_income'])
X['debt_to_income_ratio'] = np.log1p(X['debt_to_income_ratio'])


X


print(X['education_level'].value_counts())
print("--"*20)
print(X['employment_status'].value_counts())


print(X['gender'].value_counts())
print("--"*20)
print(X['marital_status'].value_counts())


print(X['grade_subgrade'].value_counts())
print("--"*20)
print(X['loan_purpose'].value_counts())


### example
##X.loc[X['grade_subgrade']=='B1']
### The grade_subgrade is non impact ful feature so droping is better choice


categorical_features = categorical_features[:-1]


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
label_encoder = LabelEncoder()
one_hot = OneHotEncoder()


label_feature = ['education_level','employment_status']
one_feature = ['gender','loan_purpose','marital_status']
X_transformed = X.copy(deep=True)
X_transformed.drop(columns=['grade_subgrade'],inplace=True)
for col in label_feature:
    X_transformed[col] = label_encoder.fit_transform(X_transformed[col])

X_transformed


train_data = pd.get_dummies(X_transformed)


train_data


train_data_ = train_data.copy(deep=True)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_data_[numerical_feature] = scaler.fit_transform(train_data_[numerical_feature])


train_data_


plt.scatter(x=train_data_['credit_score'],y=train_data_['interest_rate'])







