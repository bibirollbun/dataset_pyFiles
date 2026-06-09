# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# import requrired Libraries
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
import plotly.express as px 
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder ,OneHotEncoder , OrdinalEncoder 
#from sklearn.preprocessing import LabelEncoder ,OneHotEncoder , OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection  import train_test_split
from sklearn.metrics import confusion_matrix , accuracy_score , classification_report , precision_score , recall_score, f1_score , roc_curve , roc_auc_score

import warnings
warnings.filterwarnings("ignore")


#read the dataset 
df=pd.read_csv("/kaggle/input/loan-approval-predictions/train.csv")


df.shape


df


df.head(10)


df["loan_intent"].value_counts()



df.columns


df.info()


df.describe()


df = df.drop("id" , axis=1)
df


df.isnull().sum()


#sns.pairplot( df , hue ="loan_status")
#plt.show()


for col  in df.columns :
    print (df[col].value_counts())
    print("----------------------")


num_cols = df.select_dtypes(include=["int64" ,"float64"]).columns
print(num_cols)



cat_cols= df.select_dtypes(include=["object"]).columns
print(cat_cols)


#for col in num_cols :
num = df.select_dtypes(include=("int64" ,"float64"))
num.hist(figsize=(16,20) , bins=10 ,xlabelsize=8, ylabelsize=8)
plt.title(f"Distribution of :{col}" )
plt.xlabel(col)
plt.ylabel("Frequency")
plt.show()


# Extracting bindings with loan_status
#matrix_corr=num_col.corr()
corr_matrix = df.select_dtypes(include=['float64', 'int64']).corr()
corr_with_target = corr_matrix["loan_status"].drop("loan_status")

# Taking the top 10 bindings (absolute)
top_corr = corr_with_target.sort_values(ascending=False).head(10)

# Drawing the bar graph
plt.figure(figsize=(10, 6))
sns.barplot(x=top_corr.values, y=top_corr.index, palette="coolwarm")
plt.title("Top 10 Features Correlated with is_canceled")
plt.xlabel("Absolute Correlation")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


cat= df.select_dtypes(include=["object"])
for col in cat.columns: 
    df[col].value_counts().plot.pie(autopct= '%1.1f%%', startangle=90)
    plt.title(f"distribution of : {col}")
    plt.ylabel('') 
    plt.show()


#cat= df.select_dtypes(include=["object"])
cat_cols= df.select_dtypes(include=["object"]).columns
for col in cat_cols :
    sns.countplot(data=df , x=col  ,hue="loan_status")
    plt.title(col)
    plt.xlabel(f"{col}")
    plt.ylabel("count")
    plt.xticks(rotation =45)
    plt.show()


sns.boxplot(x='loan_status', y='person_income', data=df)


sns.boxplot(x = "loan_status" , y ="person_home_ownership" , data=df)





import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x='loan_status', y='loan_amnt', data=df)
plt.title('Loan Amount vs Loan Status')
plt.xlabel('Loan Status (0 = Rejected, 1 = Approved)')
plt.ylabel('Loan Amount')
plt.show()


df.groupby('loan_status')['loan_amnt'].mean().plot(kind='bar')
plt.title('Average Loan Amount by Loan Status')
plt.xlabel('Loan Status')
plt.ylabel('Average Loan Amount')
plt.show()



df.isnull().sum()


num =df.select_dtypes(include=("int64","float64")).columns
print(num)



num =['person_age', 'person_income', 'person_emp_length', 'loan_amnt',
       'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']
for col in num :
    Q1=df[col].quantile(0.25)
    Q3=df[col].quantile(0.75)
    IQR=Q3-Q1
    lower=Q1-3*IQR
    upper=Q3+3*IQR
    df[col]=np.where(df[col]<lower ,lower ,np.where(df[col]>upper ,upper,df[col] ))


num=df.select_dtypes(include=("int64","float64"))
print(num.columns)
plt.figure(figsize=(15,30))
for i ,col in enumerate(num ,start=1):
    plt.subplot(20,2,i)
    sns.boxplot(x=df[col])
    plt.title(f"boxplot of :{col} ")
plt.tight_layout()    
plt.show()





num=df.select_dtypes(include=("int64","float64"))
print(num.columns)
plt.figure(figsize=(15,30))
for i ,col in enumerate(num ,start=1):
    plt.subplot(20,2,i)
    sns.boxplot(x=df[col])
    plt.title(f"boxplot of :{col} ")
plt.tight_layout()    
plt.show()


cat_cols= df.select_dtypes(include=["object"]).columns
print(cat_cols)


### Encoding categorical features ### 
cat_le =['loan_grade', 'cb_person_default_on_file']
LE= LabelEncoder()
for col in cat_le :
    df[col] =LE.fit_transform(df[col]) 


cat_cols= df.select_dtypes(include=["object"]).columns
print(cat_cols)


#cat_one = ["person_home_ownership", "loan_intent"]

#df = pd.get_dummies(df, columns=cat_one, drop_first=True)





cat_one=["person_home_ownership","loan_intent"]

df=pd.get_dummies(df,columns=cat_one)   



cat_cols= df.select_dtypes(include=["object"]).columns
print(cat_cols)


# Extracting bindings with loan_status 
#matrix_corr=num_col.corr()
corr_matrix = df.select_dtypes(include=['float64', 'int64']).corr()
corr_with_target = corr_matrix["loan_status"].drop("loan_status")

# Taking the top 10 bindings (absolute)
top_corr = corr_with_target.sort_values(ascending=False).head(10)

# Drawing the bar graph
plt.figure(figsize=(10, 6))
sns.barplot(x=top_corr.values, y=top_corr.index, palette="coolwarm")
plt.title("Top 10 Features Correlated with is_canceled")
plt.xlabel("Absolute Correlation")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


df


df= df.astype(int)


df


df.isnull().sum()


df.info()


num=df.select_dtypes(include=("int64","float64"))
print(num.columns)


#MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
num_scaler=['person_age', 'person_income', 'person_emp_length', 'loan_grade',
       'loan_amnt', 'loan_int_rate', 'loan_percent_income',
       'cb_person_default_on_file', 'cb_person_cred_hist_length',
        'person_home_ownership_MORTGAGE',
       'person_home_ownership_OTHER', 'person_home_ownership_OWN',
       'person_home_ownership_RENT', 'loan_intent_DEBTCONSOLIDATION',
       'loan_intent_EDUCATION', 'loan_intent_HOMEIMPROVEMENT',
       'loan_intent_MEDICAL', 'loan_intent_PERSONAL', 'loan_intent_VENTURE']
for col in num_scaler :
    df[num_scaler] =scaler.fit_transform(df[num_scaler])
df.head(30)    


for col in df.columns :
    print(df[col].value_counts())
    print("----------------------")





num=df.select_dtypes(include=("int64","float64"))
print(num.columns)
plt.figure(figsize=(15,30))
for i ,col in enumerate(num ,start=1):
    plt.subplot(20,2,i)
    sns.boxplot(x=df[col])
    plt.title(f"boxplot of :{col} ")
plt.tight_layout()    
plt.show()


df


df.duplicated().sum()



df=df.drop_duplicates()


df.duplicated().sum()


x=df.drop("loan_status" ,axis=1)# features 
y=df["loan_status"] #Target


x


y


#!pip install -U scikit-learn imbalanced-learn



#!pip install scikit-learn==1.3.2 imbalanced-learn==0.11.0



!pip install -U scikit-learn==1.2.2 imbalanced-learn==0.10.1



from imblearn.over_sampling import SMOTE
from sklearn.model_selection  import train_test_split
x_train ,x_test ,y_train , y_test =train_test_split( x, y , test_size=0.20 , random_state=42)
## SMOTE configuration 
smote =SMOTE(random_state=42)
# apply smote  of only data train
x_train_res , y_train_res =smote.fit_resample(x_train ,y_train)
print( "befor SMOTE :",y_train.value_counts())
print("after SMOTE :" ,y_train_res.value_counts())



x_train_res


y_train_res


print("x_train",x_train.shape)
print("x_test" ,x_test.shape)
print("y_train" ,y_train.shape)
print("y_train" ,y_train.shape)
print("x_train_res",x_train_res.shape)
print("y_train_res" ,y_train_res.shape)


#import kneighbors library 
from sklearn.neighbors import KNeighborsClassifier


knn =KNeighborsClassifier (n_neighbors=3 ,weights="distance" ,p=2) #difine paramters
knn.fit(x_train_res ,y_train_res)



y_pred = knn.predict(x_test)
y_pred


y_test #actual 


# confusion matrix 
from sklearn.metrics import confusion_matrix ,accuracy_score ,classification_report,precision_score , recall_score , f1_score ,roc_curve, roc_auc_score,ConfusionMatrixDisplay





knn_acc=accuracy_score (y_test ,y_pred)
knn_acc


knn_prec=precision_score(y_test ,y_pred)
knn_prec


knn_recall=recall_score(y_test ,y_pred)
knn_recall


knn_f1=f1_score(y_test ,y_pred)
knn_f1


knn_con =confusion_matrix(y_test ,y_pred)
print(f"confusion matrix :\n {knn_con}" )
disp=ConfusionMatrixDisplay(knn_con)
disp.plot()



print(f"confusion_matrix :\n",confusion_matrix(y_test ,y_pred))
print(f"accuracy :\n",accuracy_score(y_test,y_pred))
print(f"precision_score :\n",precision_score(y_test,y_pred))
print(f"recall_score :\n",recall_score(y_test ,y_pred))
print(f"f1_score :\n", f1_score(y_test,y_pred))
print(f"classification_report :\n", classification_report(y_test,y_pred))



knn1 =KNeighborsClassifier (n_neighbors=3 ,weights="distance" ,p=2) #difine paramters
knn1.fit(x_train ,y_train)






y_pred1=knn.predict(x_test)


len(x) == len(y)


print(f"confusion_matrix :\n",confusion_matrix(y_test ,y_pred))
print(f"accuracy :\n",accuracy_score(y_test,y_pred))
print(f"precision_score :\n",precision_score(y_test,y_pred))
print(f"recall_score :\n",recall_score(y_test ,y_pred))
print(f"f1_score :\n", f1_score(y_test,y_pred))
print(f"classification_report :\n", classification_report(y_test,y_pred))


from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score ,classification_report 
log_reg=LogisticRegression()
#training 
log_reg.fit(x_train ,y_train)



#prediction 
y_pred1=log_reg.predict(x_test)


print(f"confusion_matrix :\n",confusion_matrix(y_test ,y_pred1))
print(f"accuracy :\n",accuracy_score(y_test,y_pred1))
print(f"precision_score :\n",precision_score(y_test,y_pred1))
print(f"recall_score :\n",recall_score(y_test ,y_pred1))
print(f"f1_score :\n", f1_score(y_test,y_pred1))
print(f"classification_report :\n", classification_report(y_test,y_pred1))


# 
log_reg.fit(x_train_res ,y_train_res)


y_pred2=log_reg.predict(x_test)


print(f"confusion_matrix :\n",confusion_matrix(y_test ,y_pred2))
print(f"accuracy :\n",accuracy_score(y_test,y_pred2))
print(f"precision_score :\n",precision_score(y_test,y_pred2))
print(f"recall_score :\n",recall_score(y_test ,y_pred2))
print(f"f1_score :\n", f1_score(y_test,y_pred2))
print(f"classification_report :\n", classification_report(y_test,y_pred2))

