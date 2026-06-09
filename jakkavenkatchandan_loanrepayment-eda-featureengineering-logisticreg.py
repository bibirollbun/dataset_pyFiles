import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')



train.head()


sub.head(3)


train.shape


train.isnull().sum()


test.isnull().sum()


train.columns


numerical_col = ['annual_income', 'debt_to_income_ratio', 'credit_score',
       'loan_amount', 'interest_rate']
categorical_col = ['gender', 'marital_status',
       'education_level', 'employment_status', 'loan_purpose',
       'grade_subgrade']


train['loan_paid_back'].value_counts()

# A lot of 1's than 0's which can cause our dataset to be bias towards 1's



((train['loan_paid_back']==1).sum())*100 / len(train['loan_paid_back'])
# 80 % of the data is 1.


train.describe()


plt.figure(figsize = (18,len(numerical_col)+10))
for i,col in enumerate(numerical_col):
    plt.subplot(len(numerical_col)//2 +1, 3,i+1)
    plt.boxplot(col,data = train)
    plt.title(f"Distribtion of {col}")
    plt.tight_layout()


plt.figure(figsize = (18,len(numerical_col)+10))
for i,col in enumerate(numerical_col):
    plt.subplot(len(numerical_col)//2 +1, 3,i+1)
    plt.hist(col,data = train)
    plt.title(f"Distribtion of {col}")
    plt.tight_layout()


for col in categorical_col:
    print(col)
    print(train[col].value_counts())
    print("------")



import seaborn as sns
import matplotlib.pyplot as plt
LABEL = "loan_paid_back"
plt.figure(figsize = (18,len(numerical_col)+10))
for i,col in enumerate(numerical_col):
    plt.subplot(len(numerical_col)//2 +1, 3, i+1)
    sns.boxplot(x= LABEL,y = col, data = train)
    plt.title(f"{col} vs {LABEL}")
    plt.tight_layout()


import warnings
warnings.filterwarnings("ignore")
n_cols = 3
n_rows = len(numerical_col)//2 +1
plt.figure(figsize = (5*n_cols,4*n_rows))
for i,col in enumerate(numerical_col,1):
    plt.subplot(n_cols,n_rows,i)
    for label in train[LABEL].unique():
        subset = train[train[LABEL]==label]
        sns.kdeplot(subset[col],label = f"{LABEL} vs {label}",fill = True)
    plt.title(f"{col} - KDE Distribtution by {LABEL}")
    plt.xlabel(col)
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()


# Relation ship of the categorical column's to the Loan Repayment
import seaborn as sns
import matplotlib.pyplot as plt
LABEL = "loan_paid_back"
plt.figure(figsize = (18,len(categorical_col)+10))
for i,col in enumerate(categorical_col):
    plt.subplot(len(categorical_col)//2 +1, 3, i+1)
    sns.countplot(x= col, hue = LABEL, data = train)
    plt.title(f"{col} vs {LABEL}")
    plt.tight_layout()


train.columns


numerical_col_2 = ['loan_paid_back', 'annual_income', 'debt_to_income_ratio', 'credit_score',
       'loan_amount', 'interest_rate']
correlation_matrix = train[numerical_col_2].corr()
correlation_matrix


plt.figure(figsize = (8,6))
sns.heatmap(correlation_matrix,annot = True,cmap = 'coolwarm')
plt.show()



train['dataset'] = 'train'
test['dataset']  = 'test'
test['loan_paid_back'] = -1
# train = train.drop('loan_paid_back',axis = 1)
df = pd.concat([train,test],axis = 0).reset_index(drop = True)
df_with_id = df.copy()
mask_test = df_with_id['dataset'] == 'test'
df_with_id.loc[mask_test, 'id'] = test_ids.values

test_ids = test['id'].copy()
df.drop('id',axis = 1,inplace = True)
df.shape


import numpy as np
non_ensemble_df = df.copy()
# Creating 2 so as to see the power of transformation
non_ensemble_df['annual_income'] = np.log(non_ensemble_df['annual_income'])


# Before
plt.figure(figsize=(4,4))
plt.hist('annual_income', data = df)
plt.tight_layout()
#After
plt.figure(figsize=(4,4))
plt.hist('annual_income', data = non_ensemble_df)
plt.show()


plt.figure(figsize = (4,4))
plt.boxplot('annual_income',data = df)
plt.show()
plt.figure(figsize = (4,4))
plt.boxplot('annual_income',data = non_ensemble_df)
plt.show()


non_ensemble_df.shape


from scipy import stats
# Removing outliers outside of 3 standard deviations.


non_ensemble_df['zscore_dti'] = stats.zscore(non_ensemble_df['debt_to_income_ratio'])
non_ensemble_df = non_ensemble_df[(non_ensemble_df['zscore_dti'] >= -3) & (non_ensemble_df['zscore_dti'] <= 3)]
non_ensemble_df.shape


plt.figure(figsize = (4,4))
plt.boxplot('debt_to_income_ratio',data = df)
plt.show()
plt.figure(figsize = (4,4))
plt.boxplot('debt_to_income_ratio',data = non_ensemble_df)
plt.show()


# ig we need to drop the z-score cuz it doesnt really matter now
non_ensemble_df.drop(columns = 'zscore_dti',inplace = True)


df['loan_to_income_ratio'] = df['loan_amount'] / df['annual_income']
non_ensemble_df['loan_to_income_ratio'] = non_ensemble_df['loan_amount'] / non_ensemble_df['annual_income']
# Isn't debit to income the same as loan amount to annual income? - ig not


df['Loan_Burden'] = df['loan_amount'] + df['loan_amount']*(1+(df['interest_rate'])/100)
non_ensemble_df['Loan_Burden'] = non_ensemble_df['loan_amount'] + non_ensemble_df['loan_amount']*(1+(non_ensemble_df['interest_rate'])/100)



df['cs_dti_interaction'] = df['credit_score'] / (1+df['debt_to_income_ratio'])
non_ensemble_df['cs_dti_interaction'] = non_ensemble_df['credit_score'] / (1+non_ensemble_df['debt_to_income_ratio'])


df.columns


new_numerical_cols = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount',
       'interest_rate','loan_to_income_ratio', 'Loan_Burden', 'cs_dti_interaction']
newcorelation = df[new_numerical_cols].corr()
sns.heatmap(newcorelation)


encoded_map = {'A1':1,'A2':2,'A3':3,'A4':4,'A5':5,
               'B1':6,'B2':7,'B3':8,'B4':9,'B5':10,
               'C1':11,'C2':12,'C3':13,'C4':14,'C5':15,
               'D1':16,'D2':17,'D3':18,'D4':19,'D5':20,
               'E1':21,'E2':22,'E3':23,'E4':24,'E5':25,
               'F1':26,'F2':27,'F3':28,'F4':29,'F5':30
              }
df['Encoded_grade_subgrade'] = df['grade_subgrade'].map(encoded_map)
non_ensemble_df['Encoded_grade_subgrade'] = non_ensemble_df['grade_subgrade'].map(encoded_map)


# Dropping the grade_subgrade 
df = df.drop(columns = 'grade_subgrade',axis = 1)
non_ensemble_df = non_ensemble_df.drop(columns = 'grade_subgrade',axis = 1)


newcategorical_cols = ['gender', 'marital_status', 'education_level',
       'employment_status', 'loan_purpose']
df = pd.get_dummies(df,columns = newcategorical_cols,drop_first = True,dtype = float)
non_ensemble_df = pd.get_dummies(non_ensemble_df,columns = newcategorical_cols,drop_first = True,dtype = float)


# Train 2 is without transformation for ensemble methods
train2 = df[df['dataset']=='train'].drop('dataset',axis = 1)
test2 = df[df['dataset']=='test'].drop(['dataset','loan_paid_back'],axis = 1)
test2['id'] =  test_ids.values





# Train 3 is with transformation for non-ensemble methods
train3 = non_ensemble_df[non_ensemble_df['dataset']=='train'].drop('dataset',axis = 1)
test3= non_ensemble_df[non_ensemble_df['dataset']=='test'].drop(['dataset','loan_paid_back'],axis = 1)
ids_for_test3 = df_with_id.loc[test3.index, 'id']
test3['id'] = ids_for_test3.values


# 2 is for tree based algorithms.
from sklearn.model_selection import train_test_split
y2 = train2['loan_paid_back'].values
X2 = train2.drop('loan_paid_back',axis = 1).values
x2_train,x2_val,y2_train,y2_val = train_test_split(X2,y2,test_size=0.2, random_state=42, stratify=y2)



# 3 is for Distance Based ALgorithms
y3 = train3['loan_paid_back'].values
X3 = train3.drop('loan_paid_back',axis = 1).values
x3_train,x3_val,y3_train,y3_val = train_test_split(X3,y3,test_size=0.2, random_state=42,stratify=y3)



from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Scaling Important for Distance Based Algorithms
x3_train_scaled = scaler.fit_transform(x3_train)
x3_val_scaled = scaler.transform(x3_val)


class MyLogisticRegression(object):
    def __init__(self,learning_rate = 0.1,epochs = 1000):
        self.weights = None
        self.bias = None
        self.learning_rate = learning_rate
        self.epochs = epochs
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    def fit(self,x,y):
        n_samples,n_features = x.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        for i in range(self.epochs):
            linear = np.dot(x,self.weights) + self.bias
            y_pred = self.sigmoid(linear)
            dw = (1/n_samples)*np.dot(x.T,(y_pred - y))
            db = (1/n_samples)*np.sum(y_pred-y)
            self.weights = self.weights - self.learning_rate*dw
            self.bias = self.bias -self.learning_rate*db
    def predict_in_probability(self,x):
        linear = np.dot(x,self.weights)+self.bias
        return self.sigmoid(linear)
    def predict(self,x,threshold = 0.5):
        probability = self.predict_in_probability(x)
        return (probability >= threshold).astype(int)


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
my_lr = MyLogisticRegression()
my_lr.fit(x3_train_scaled,y3_train)
my_y3_pred = my_lr.predict(x3_val_scaled)


my_acc  = accuracy_score(my_y3_pred,y3_val)
my_prec = precision_score(my_y3_pred,y3_val)
my_rec  = recall_score(my_y3_pred,y3_val)
my_f1   = f1_score(my_y3_pred,y3_val)
my_auc  = roc_auc_score(my_y3_pred,y3_val)

print(f"Scratch Logistic Regression:")
print(f"  Accuracy : {my_acc:.4f}")
print(f"  Precision: {my_prec:.4f}")
print(f"  Recall   : {my_rec:.4f}")
print(f"  F1-score : {my_f1:.4f}")
print(f"  ROC-AUC  : {my_auc:.4f}")


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x3_train_scaled,y3_train)
y3_pred = my_lr.predict(x3_val_scaled)

acc  = accuracy_score(y3_pred,y3_val)
prec = precision_score(y3_pred,y3_val)
rec  = recall_score(y3_pred,y3_val)
f1   = f1_score(y3_pred,y3_val)
auc  = roc_auc_score(y3_pred,y3_val)

print(f"Logistic Regression:")
print(f"  Accuracy : {acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall   : {rec:.4f}")
print(f"  F1-score : {f1:.4f}")
print(f"  ROC-AUC  : {auc:.4f}")


y_train_pred = lr.predict(x3_train_scaled)

train_acc  = accuracy_score(y3_train, y_train_pred)
train_prec = precision_score(y3_train, y_train_pred)
train_rec  = recall_score(y3_train, y_train_pred)
train_f1   = f1_score(y3_train, y_train_pred)
train_auc  = roc_auc_score(y3_train, y_train_pred)

print("Train Metrics")
print(train_acc, train_prec, train_rec, train_f1, train_auc)


test3_pred = test3.drop('id',axis = 1)
test3_scaled = scaler.transform(test3_pred)

test_proba = lr.predict_proba(test3_scaled)[:, 1]

submission = pd.DataFrame({
    "id": test3["id"],
    "loan_paid_back": test_proba   # or 0/1 if required
})

submission.to_csv("submission.csv", index=False)




 import numpy as np
print("Unique probs (first 10):", np.unique(test_proba)[:10])
print("Min prob:", test_proba.min(), "Max prob:", test_proba.max())





