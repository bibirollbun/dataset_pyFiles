pip install umap-learn


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.multioutput import MultiOutputClassifier
import umap
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#Load File
sample_submission = pd.read_excel('/kaggle/input/widsdatathon2025/SAMPLE_SUBMISSION.xlsx')
data_dict = pd.read_excel('/kaggle/input/widsdatathon2025/Data Dictionary.xlsx')
train_sol = pd.read_excel('/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAINING_SOLUTIONS.xlsx')
train_cat = pd.read_excel('/kaggle/input/widsdatathon2025/TRAIN_OLD/TRAIN_CATEGORICAL_METADATA.xlsx')
train_quant = pd.read_excel('/kaggle/input/widsdatathon2025/TRAIN_OLD/TRAIN_QUANTITATIVE_METADATA.xlsx')
funct_connect = pd.read_csv('/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv')
train_cat = pd.read_excel('/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_CATEGORICAL_METADATA_new.xlsx')
test_quant = pd.read_excel('/kaggle/input/widsdatathon2025/TEST/TEST_QUANTITATIVE_METADATA.xlsx')
test_cat = pd.read_excel('/kaggle/input/widsdatathon2025/TEST/TEST_CATEGORICAL.xlsx')
funct_test_connect = pd.read_csv('/kaggle/input/widsdatathon2025/TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv')


data_dict


funct_connect


train_sol


train_cat


train_cat.isnull().sum()


for col in train_cat:
    num_unique = train_cat[col].nunique()
    print(f"{col} has {num_unique} unique categories")
    print(f"\n Value Counts in {col}: \n{train_cat[col].value_counts()}")


train_quant


train_quant.describe()


train_quant.isnull().sum()


a = train_quant.drop('participant_id',axis = 1)
a.skew()


merge_a = train_cat.merge(train_quant,on = 'participant_id',how = 'left')
merge_b = merge_a.merge(funct_connect,on = 'participant_id',how = 'left')
train = merge_b.merge(train_sol,on = 'participant_id',how = 'left')


merge_c = test_cat.merge(test_quant,on = 'participant_id',how = 'left')
test = merge_c.merge(funct_test_connect,on = 'participant_id',how = 'left')


train.isnull().sum()


test


train.describe()


train.isnull().sum()[train.isnull().sum()>0]


missing_percentage = train.isnull().mean() * 100
missing_percentage = missing_percentage[missing_percentage > 0]
print(f"Missing Percentage:\n{missing_percentage.apply(lambda x: f'{x:.2f}%')}")



import missingno as msno

cols_with_missing = train.columns[train.isnull().sum()>0]
msno.matrix(train[cols_with_missing])


msno.heatmap(train[cols_with_missing])
plt.show()


mode_col = ["PreInt_Demos_Fam_Child_Ethnicity","PreInt_Demos_Fam_Child_Race","MRI_Track_Scan_Location"]
for col in mode_col:
    train[col].fillna(train[col].mode()[0],inplace = True)

unknown_col = na_columns = ["Barratt_Barratt_P1_Edu","Barratt_Barratt_P1_Occ","Barratt_Barratt_P2_Edu","Barratt_Barratt_P2_Occ"]
for col in unknown_col:
    train[col].fillna(1.0,inplace = True)

train["MRI_Track_Age_at_Scan"].fillna(train["MRI_Track_Age_at_Scan"].mean(), inplace=True)

train.isnull().sum()[train.isnull().sum()>0]


test.isnull().sum()[test.isnull().sum()>0]


column_test_missing = test.columns[test.isnull().sum()>0]
msno.matrix(test[column_test_missing])


# Fill na in test
mode_col = ["PreInt_Demos_Fam_Child_Ethnicity","PreInt_Demos_Fam_Child_Race",'EHQ_EHQ_Total']
for col in mode_col:
    test[col].fillna(train[col].mode()[0],inplace = True)

unknown_col = na_columns = ["Barratt_Barratt_P1_Edu","Barratt_Barratt_P1_Occ","Barratt_Barratt_P2_Edu","Barratt_Barratt_P2_Occ"]
for col in unknown_col:
    test[col].fillna(1.0,inplace = True)

test['ColorVision_CV_Score'].fillna(train['ColorVision_CV_Score'].mean(),inplace = True)

test.isnull().sum()[train.isnull().sum()>0]


test.shape


df = pd.concat([train.iloc[:,0:28],train.iloc[:,-2:]],axis = 1)
df
df.info()


b=train_cat.columns.tolist()
b.append('ADHD_Outcome')
b.append('Sex_F')
print(b)


for col in b:
   df[col]=df[col].astype(str)


df.info()


sns.countplot(x='ADHD_Outcome',hue='Sex_F',data =df)
plt.show()


df['Sex_F_ADHD'] = df['Sex_F']+ "_" +df['ADHD_Outcome']


numerical_vars = df.select_dtypes(include=['int64','float64']).columns.tolist()

for col in numerical_vars:
    plt.figure(figsize=(8,6))
    sns.boxplot(x='Sex_F_ADHD',y=col, data=df)
    plt.title(f"Distribution of {col} by Sex_F and ADHD outcome")
    plt.show()


df.info()


cat_col = df.select_dtypes(exclude=['int64','float64']).columns
cat_col = cat_col[~cat_col.isin(['Sex_F', 'ADHD_Outcome','participant_id','Sex_F_ADHD'])]

for col in cat_col:
    plt.figure(figsize=(8,6))
    sns.countplot(x=df[col],hue='Sex_F_ADHD',data=df)
    plt.title(f"Distribution of {col} by Sex_F and ADHD Outcome")
    plt.show()


X = train.drop(columns = ['ADHD_Outcome', 'Sex_F', 'participant_id'])
y = train[['ADHD_Outcome', 'Sex_F']]


X_train, X_test, y_train,y_test = train_test_split(X,y,random_state = 42, stratify = y)


xgb_model = MultiOutputClassifier(xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42))
xgb_model.fit(X_train,y_train)
y_pred = xgb_model.predict(X_test)
print('f1: ', f1_score(y_test,y_pred,average='micro'))


from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
pca_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Standardize the data
    ('pca', PCA(n_components=0.95))  # Apply PCA
])

X_train_pca = pca_pipeline.fit_transform(X_train)
X_test_pca = pca_pipeline.transform(X_test)

#train model
xgb_model_pca = MultiOutputClassifier(xgb.XGBClassifier(objective = 'binary:logistic',eval_metric='logloss',random_state = 42))
xgb_model_pca.fit(X_train_pca,y_train)

#prediction
y_pred_pca = xgb_model_pca.predict(X_test_pca)

#Evaluate F1 score
f1_pca = f1_score(y_test,y_pred_pca,average='micro')
print('F1 score with PCA',f1_pca)


#try to use UMAP 

#standardised 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#UMAP
for n in [5,10,20,30,40]:
    umap_reducer = umap.UMAP(n_components=n)
    X_train_umap = umap_reducer.fit_transform(X_train_scaled)
    X_test_umap = umap_reducer.transform(X_test_scaled)

#train model
    xgb_model_umap = MultiOutputClassifier(xgb.XGBClassifier(objective = 'binary:logistic',eval_metric='logloss',random_state = 42))
    xgb_model_umap.fit(X_train_umap,y_train)

    y_pred = xgb_model_umap.predict(X_test_umap)
    print(f'F1 score with {n} components :',f1_score(y_test,y_pred,average='micro'))




test


sample_submission


#save 'id' column for submission
test_ids = test['participant_id']

test_ids


test = test.drop('participant_id',axis =1)
y_pred_a = xgb_model.predict(test)

output = pd.DataFrame({'participant_id': test_ids,
                       'ADHD_Outcome': y_pred_a[:,0],
                      'Sex_F':y_pred_a[:,1]})

output.to_csv('submission.csv', index=False)




