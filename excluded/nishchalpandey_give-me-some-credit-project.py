# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("fivethirtyeight")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#IMPORT RELEVANT LIBRARIES

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split

# Modelling & Feature Selection
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline

# Evaluation & Metrics
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("fivethirtyeight")


train = pd.read_csv("/kaggle/input/GiveMeSomeCredit/cs-training.csv")
test = pd.read_csv("/kaggle/input/GiveMeSomeCredit/cs-test.csv")
data_excel = pd.read_excel("/kaggle/input/GiveMeSomeCredit/Data Dictionary.xls")
sample_data = pd.read_csv("/kaggle/input/GiveMeSomeCredit/sampleEntry.csv")



train_df = train.copy()
test_df = test.copy()


print("====== SUMMARY =======")
print()
print(f"Training set size: {train_df.shape}")
print(f"Testing set size:  {test_df.shape}")
print()
print(f"All features in the training dataset: {', '.join(map(str, train_df.columns))}")
print(f"All features in the testing dataset: {', '.join(map(str, test_df.columns))}")
print()
print("---Training set survivability---")
print(f"Number of Applicants: {train_df.shape[0]}")
print(f"Number of Applicants with financial distress = {len(train_df[train_df['SeriousDlqin2yrs'] == 1])}")
print(f"Percentage of Applicants with financial distress = {len(train_df[train_df['SeriousDlqin2yrs'] == 1]) / train_df.shape[0] * 100} %")
print()
print("---Missing data [%]---")
print(train_df.isnull().sum()[train_df.isnull().sum() > 0])
print()
print("---Data types---")
print(train_df.info())
print()



# rename_map = {
    #'RevolvingUtilizationOfUnsecuredLines':'Credit_Utilization','NumberOfTime30-59DaysPastDueNotWorse':'Low_Del',
    #'NumberOfTime60-89DaysPastDueNotWorse':'Mid_Del','NumberRealEstateLoansOrLines':'Mortgage_Accounts','NumberOfTimes90DaysLate':'High_Del',
    #'NumberOfDependents':'Dependents','NumberOfOpenCreditLinesAndLoans':'Active_Accounts'}
# train_df = train_df.rename(columns=rename_map)
# train_df


print("---Statistical summary---")
round(train_df.describe(),2)


plt.rc('font',size=10)
plt.rc('axes',labelsize=10,titlesize=10)
plt.rc('legend',fontsize=8)
plt.rc('xtick',labelsize=6)
plt.rc('ytick',labelsize=6)

train_df.hist(bins=50,figsize=(12,8))
plt.show()


outliers_col = ['RevolvingUtilizationOfUnsecuredLines','DebtRatio','MonthlyIncome','NumberRealEstateLoansOrLines']
plt.figure(figsize=(18, 12)) # Set overall figure size
for i, col in enumerate(train_df[outliers_col], 1):
    plt.subplot(4, 3, i) # Create subplot in 4x3 grid
    sns.boxplot(x=train_df[col])
    plt.title(f'Boxplot of {col}')
    plt.tight_layout()


dist_col = ['age','NumberOfDependents']

for col in train_df[dist_col]:
    sns.histplot(train_df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()


plt.figure(figsize=(10, 8))
sns.heatmap(train_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.show()


# Calculate correlation of each feature with SeriousDlqin2yrs
target_corr = train_df.corr()['SeriousDlqin2yrs']

# Sort correlations by absolute value (optional, for better visualization)
target_corr = target_corr.sort_values(key=abs, ascending=False).drop('SeriousDlqin2yrs')

plt.figure(figsize=(10, 6))
sns.barplot(x=target_corr.values, y=target_corr.index, palette="coolwarm")
plt.title("Correlation of Features with SeriousDlqin2yrs")
plt.xlabel("Correlation Coefficient")
plt.ylabel("Features")
plt.tight_layout()
plt.show()



def preprocess(df):

    df = df.copy()

    # Creating Age Groups and filling missing values of Monthly Income based on age group
    bins = [0,25,35,45,55,70,float('inf')]
    labels = [0,1,2,3,4,5]
    df['age_group'] = pd.cut(df['age'],bins=bins,labels=labels,include_lowest=True)

    median_income_by_age = df.groupby('age_group')['MonthlyIncome'].median()

    df['MonthlyIncome'] = df['MonthlyIncome'].replace(0, np.nan)
    df['MonthlyIncome'] = df.apply(lambda row:  median_income_by_age[row['age_group']] if pd.isnull(row['MonthlyIncome'])
                                          else row['MonthlyIncome'],axis=1)
    print("Remaining missing values in MonthlyIncome:", df['MonthlyIncome'].isnull().sum())

    # Handling outliers using IQR method 

    outliers_col = ['RevolvingUtilizationOfUnsecuredLines','DebtRatio','MonthlyIncome','NumberRealEstateLoansOrLines']

    for col in outliers_col:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    # Converting num columns into categories

    df['30-59_Due'] = df['NumberOfTime30-59DaysPastDueNotWorse'].apply(lambda x: 0 if x == 0 else 1 if x < 5 else 2)
    df['60-89_Due'] = df['NumberOfTime60-89DaysPastDueNotWorse'].apply(lambda x: 0 if x == 0 else 1 if x < 5 else 2)
    df['90+_Due'] = df['NumberOfTimes90DaysLate'].apply(lambda x: 0 if x == 0 else 1 if x < 5 else 2)
    
    df['NumberOfDependents'] = df['NumberOfDependents'].apply(lambda x: 0 if x == 0 else 1)
    df['NumberOfDependents'] = df['NumberOfDependents'].fillna(value=df['NumberOfDependents'].mode()[0])

    df['NumberOfOpenCreditLinesAndLoans'] = np.log1p(df['NumberOfOpenCreditLinesAndLoans'])
    
    df = df.drop(['NumberOfTime30-59DaysPastDueNotWorse','NumberOfTime60-89DaysPastDueNotWorse','NumberOfTimes90DaysLate','age'],axis=1)

    return df


preprocessed_train_df = preprocess(train_df)
preprocessed_test_df = preprocess(test_df)
preprocessed_train_df.info()


cat = ['NumberRealEstateLoansOrLines','NumberOfDependents','age_group','30-59_Due','60-89_Due','90+_Due']
num = ['RevolvingUtilizationOfUnsecuredLines','DebtRatio','MonthlyIncome','NumberOfOpenCreditLinesAndLoans']


y = preprocessed_train_df['SeriousDlqin2yrs']
X = preprocessed_train_df.drop(['SeriousDlqin2yrs'],axis=1)


X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2,random_state=45)


cat_encoder = OneHotEncoder(handle_unknown="ignore",sparse_output=False)
scaler = StandardScaler()
logit = LogisticRegression(random_state=45)
col_transformer = make_column_transformer((cat_encoder,cat),(scaler,num),remainder='passthrough')

pipeline_basic = Pipeline([('col',col_transformer),('clf',logit)])


model = pipeline_basic.fit(X_train,y_train)


predict_val = model.predict(X_val)


precision_logit_unbalance = precision_score(y_val,predict_val)
print(f"Precision : {precision_logit_unbalance:.04f}")

recall_logit_unbalance = recall_score(y_val,predict_val)
print(f"Recall : {recall_logit_unbalance:.04f}")

f1_logit_unbalance = f1_score(y_val,predict_val)
print(f"F1 : {f1_logit_unbalance:.4f}")


np.unique(predict_val, return_counts=True)


logit_bal = LogisticRegression(class_weight='balanced', random_state=45)
pipeline_logit = Pipeline([('col',col_transformer),('clf',logit_bal)])
model_logit_bal = pipeline_logit.fit(X_train,y_train)


pred_val_logit = model_logit_bal.predict(X_val)


precision_logit = precision_score(y_val,pred_val_logit)
print(f"Precision : {precision_logit:.4f}")

recall_logit = recall_score(y_val,pred_val_logit)
print(f"Recall : {recall_logit:.4f}")

f1_logit = f1_score(y_val,pred_val_logit)
print(f"F1 : {f1_logit:.4f}")


from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(class_weight = 'balanced',n_estimators=200,max_depth=6,min_samples_leaf=2,random_state=45)
pipeline_rf = Pipeline([('col',col_transformer),('clf',rfc)])


RFC_model = pipeline_rf.fit(X_train,y_train)


pred_val_rfc = RFC_model.predict(X_val)


precision_rfc = precision_score(y_val,pred_val_rfc)
print(f"Precision : {precision_rfc:.4f}")

recall_rfc = recall_score(y_val,pred_val_rfc)
print(f"Recall : {recall_rfc:.4f}")

f1_rfc = f1_score(y_val,pred_val_rfc)
print(f"F1 : {f1_rfc:.4f}")


y_prob = RFC_model.predict_proba(X_val)[:, 1]

# Threshold = 0.3
custom_thresh = 0.3
y_pred_thresh = (y_prob >= custom_thresh).astype(int)

P03 = precision_score(y_val, y_pred_thresh).round(4)
R03 = recall_score(y_val, y_pred_thresh).round(4)
F03 = f1_score(y_val, y_pred_thresh).round(4)

print(f"Precision: {P03}")
print(f"Recall: {R03}")
print(f"F1: {F03}")


y_prob = RFC_model.predict_proba(X_val)[:, 1]

# Threshold = 0.4
custom_thresh = 0.4
y_pred_thresh = (y_prob >= custom_thresh).astype(int)

P04 = precision_score(y_val, y_pred_thresh).round(4)
R04 = recall_score(y_val, y_pred_thresh).round(4)
F04 = f1_score(y_val, y_pred_thresh).round(4)

print(f"Precision: {P04}")
print(f"Recall: {R04}")
print(f"F1: {F04}")


y_prob = RFC_model.predict_proba(X_val)[:, 1]

# Threshold = 0.7
custom_thresh = 0.7
y_pred_thresh = (y_prob >= custom_thresh).astype(int)

P07 = precision_score(y_val, y_pred_thresh).round(4)
R07 = recall_score(y_val, y_pred_thresh).round(4)
F07 = f1_score(y_val, y_pred_thresh).round(4)

print(f"Precision: {P07}")
print(f"Recall: {R07}")
print(f"F1: {F07}")


RFC_comparision = pd.DataFrame({
    'Metric': ['Precision', 'Recall', 'F1 Score'],
    'Threshold 0.3': [
        P03,R03,F03],
    'Threshold 0.5': [
        precision_rfc,recall_rfc,f1_rfc],
    'Threshold 0.7': [
        P07,R07,F07],
    'Use Cases':[
        "For Flaging risking Profiles",
        "Balanced System",
        "Auto Rejecting High Risk Profiles"]
})

RFC_comparision = RFC_comparision.round(4)

RFC_comparision



scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    scale_pos_weight=scale_pos_weight,
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    random_state=45
)


pipeline_xgb = Pipeline([('col',col_transformer),('clf',xgb_model)])
model_xgb = pipeline_xgb.fit(X_train,y_train)


predict_val_xgb = model_xgb.predict(X_val)



precision_xgb = precision_score(y_val, predict_val_xgb).round(4)
recall_xgb = recall_score(y_val, predict_val_xgb).round(4)
f1_xgb = f1_score(y_val, predict_val_xgb).round(4)

print(f"Precision: {precision_xgb}")
print(f"Recall: {recall_xgb}")
print(f"F1: {f1_xgb}")


Model_Comparision = pd.DataFrame({
    'Metric': ['Precision', 'Recall', 'F1 Score'],
    'Logistic Regression (balanced)': [
        precision_logit,recall_logit,f1_logit],
    'Random Forest Classifier (balanced)': [
        precision_rfc,recall_rfc,f1_rfc],
    'XGBoost Classifier': [
        precision_xgb,recall_xgb,f1_xgb]
})

Model_Comparision = Model_Comparision.round(4)

Model_Comparision



test_prob = pipeline_xgb.predict_proba(preprocessed_test_df)[:, 1]

submission_df = pd.DataFrame({
    'Id': test_df['Unnamed: 0'],  
    'Probability': test_prob
})

submission_df.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")

