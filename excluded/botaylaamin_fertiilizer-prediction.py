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


# for remove warnings
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

# preprocessing and modelling

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection   import train_test_split

# build model
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
# evaluation 

from sklearn.metrics import classification_report


data_train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')    # upload Data 
data_test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
data_train.sample(10)               # display 10 random samples



# remove id column 
data_train.drop('id' , axis=1 ,inplace=True )
data_test.drop('id' , axis=1 ,inplace=True )


data_train.info()


data_train.duplicated().sum()    # check for duplicates


data_train.describe()  # get statistical summery for train data


numerical_cols = data_train.select_dtypes(['int','float']).columns
categorical_cols = data_train.select_dtypes('object').columns

print("Numerical Columns : ", numerical_cols)
print("Categorical Columns : ", categorical_cols)


# create Box plot for each numerical column

plt.figure()
for i,col in enumerate(numerical_cols,1):
    plt.subplot(2,3,i)        # create grid 2 x 3 for numerical features
    sns.boxplot(data_train, x= col)
    plt.title(f"box plot of {col}")
    plt.xlabel(col)

plt.tight_layout()  # prevent overlap
plt.show()    


# create histogram for each numerical column

plt.figure()
for i,col in enumerate(numerical_cols,1):
    plt.subplot(2,3,i)        # create grid 2 x 3 for numerical features
    sns.histplot(data=data_train , x =col,kde=True )
    plt.title(f"{col} distribution")
    plt.xlabel(col)

plt.tight_layout()  # prevent overlap
plt.show()    


data_train[numerical_cols].skew()               # check skewness 


# for categorical data 

plt.figure()

for i , col in enumerate(categorical_cols.drop('Fertilizer Name') , 1):
    plt.subplot(1,2,i)
    sizes = data_train[col].value_counts()
    plt.pie(sizes , autopct='%1.1f%%', labels=sizes.index)
    plt.title(f"distribution of {col}")
    


plt.tight_layout()  # prevent overlap
plt.show()      


# for target Column
counts = data_train['Fertilizer Name'].value_counts()
print(counts)


data_train['Fertilizer Name'].value_counts(True) # the precentage  of every category


counts.plot(kind='bar',color='bisque')     # plot the bar for categories
for i, v in enumerate(counts):        # display the frequency of each category
    plt.text(i, v, str(v), ha='center', va='bottom')

plt.xlabel("The Type")
plt.title("The Distribution of Target{'Fertilizer Name'}")
plt.xticks(rotation = 45)
plt.show()


corr_mat = data_train[numerical_cols].corr()
sns.heatmap(corr_mat , annot= True,cmap='coolwarm')
plt.title("Correlation Matrix ")
plt.show()


categorical_cols = categorical_cols.drop('Fertilizer Name')


def process(data ,categorical_cols, numerical_cols ,  target = None):
    # to split the data to train, test
    if target :
        y = data[target]
        data = data.drop(target, axis=1)
        
    else:
        y = None

    # to encode data 
    encoded = pd.get_dummies(data , columns=categorical_cols)

    # normalize data
    scaler = StandardScaler()
    encoded[numerical_cols] = scaler.fit_transform(encoded[numerical_cols])

    return encoded , y

# get encoded and scaled data 
data_train_processed , y_train = process(data_train ,categorical_cols, numerical_cols,'Fertilizer Name')
data_test_processed,_ = process(data_test ,categorical_cols, numerical_cols)

# encode target column in train dataset 
le = LabelEncoder()
y_train_labeld = le.fit_transform(y_train)


print("Shape of processed train data:", data_train_processed.shape)
print("Shape of processed test data:", data_test_processed.shape)
print("Unique target values:", le.classes_)


data_train_processed.shape


X_train, X_test , Y_train , Y_test = train_test_split(data_train_processed , y_train_labeld,test_size= 0.3 , random_state=41)


print("Shape of x_train " , X_train.shape)
print("Shape of y_train " , Y_train.shape)
print("Shape of x_test " , X_test.shape)
print("Shape of y_test " , Y_test.shape)


def compute_ap_at_3(preds, true_labels, n=3):
    """Compute AP@3 for a single observation."""
    ap = 0.0
    num_correct = 0
    for k in range(min(n, len(preds))):
        if preds[k] == true_labels[0]:  # Assuming one correct label
            num_correct += 1
            precision_at_k = num_correct / (k + 1)
            ap += precision_at_k
            break  # Stop after finding the correct label
    return ap

def compute_map_at_3(top_k_preds, true_labels, n=3):
    """Compute MAP@3 across all observations."""
    aps = []
    for i in range(len(true_labels)):
        ap = compute_ap_at_3(top_k_preds[i], [true_labels[i]], n)
        aps.append(ap)
    return np.mean(aps)



cls = RandomForestClassifier(n_estimators=50)
cls.fit(X_train,Y_train)
pred = cls.predict(X_test)

print(classification_report(Y_test , pred))


test_probs = cls.predict_proba(X_test)
test_labels = Y_test
top_3_preds = np.argsort(-test_probs, axis=1)[:, :3]

# Compute MAP@3
map_at_3 = compute_map_at_3(top_3_preds, test_labels, n=3)
print(f"MAP@3: {map_at_3:.4f}")


cls = LogisticRegression(class_weight='balanced')
cls.fit(X_train,Y_train)
pred = cls.predict(X_test)

print(classification_report(Y_test , pred))


test_probs = cls.predict_proba(X_test)
test_labels = Y_test
top_3_preds = np.argsort(-test_probs, axis=1)[:, :3]

# Compute MAP@3
map_at_3 = compute_map_at_3(top_3_preds, test_labels, n=3)
print(f"MAP@3: {map_at_3:.4f}")


cls = DecisionTreeClassifier()
cls.fit(X_train,Y_train)
pred = cls.predict(X_test)

print(classification_report(Y_test , pred))


test_probs = cls.predict_proba(X_test)
test_labels = Y_test
top_3_preds = np.argsort(-test_probs, axis=1)[:, :3]

# Compute MAP@3
map_at_3 = compute_map_at_3(top_3_preds, test_labels, n=3)
print(f"MAP@3: {map_at_3:.4f}")


cls = lgb.LGBMClassifier(random_state=41,is_unbalance=True, learning_rate=0.05,num_leaves=50,n_estimators=200)
cls.fit(X_train, Y_train)   


test_probs = cls.predict_proba(X_test)
test_labels = Y_test
top_3_preds = np.argsort(-test_probs, axis=1)[:, :3]
pred = cls.predict(X_test)


print("LightGBM Classification Report:")
print(classification_report(test_labels, pred))


# calc map
map_at_3 = compute_map_at_3(top_3_preds, test_labels, n=3)
print(f"LightGBM MAP@3: {map_at_3:.4f}")


lgb.plot_importance(cls, max_num_features=10)
plt.show()


test_probs = cls.predict_proba(data_test_processed)
top_3_preds_test = np.argsort(-test_probs, axis=1)[:, :3]


top_1_labels = le.inverse_transform(top_3_preds_test[:, 0])
top_2_labels = le.inverse_transform(top_3_preds_test[:, 1])
top_3_labels = le.inverse_transform(top_3_preds_test[:, 2])

top_3_preds_labels = [f"{t1}, {t2}, {t3}" for t1, t2, t3 in zip(top_1_labels, top_2_labels, top_3_labels)]

# cereate submission file
submission = pd.DataFrame({
    'id': pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')['id'],
    'Fertilizer Name': top_3_preds_labels
})

submission.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")




