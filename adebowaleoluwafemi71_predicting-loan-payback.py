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


import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
import plotly.express as px 
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder ,OneHotEncoder , OrdinalEncoder 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection  import train_test_split
from sklearn.metrics import confusion_matrix , accuracy_score , classification_report , precision_score , recall_score, f1_score , roc_curve , roc_auc_score


df_Submission=pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")
df_train=pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
df_predict=pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")


df_Submission.head()


df_train.head()


df_predict.head()


df_train.info()


#Drop the ID column as it is redundant to the table
df_train = df_train.drop(['id'], axis=1)


cat_columns = df_train.select_dtypes(include="O").columns
num_columns = []
for col in df_train.columns:
    if col not in cat_columns:
        num_columns.append(col)
        
print("Numerical columns: ", num_columns)
print("Categorical columns: ", cat_columns)


df_train[cat_columns].describe()


df_train[num_columns].describe()


df_train["grade_subgrade"].unique()
df_predict["grade_subgrade"].unique()


df_train[num_columns].hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)


df_cat = df_train[cat_columns]
for col in df_cat.columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df_train, x=col, order=df_train[col].value_counts().index, alpha=0.5)
    plt.title(f'Frequency Distribution of {col}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


axis = sns.countplot(x = "gender", hue = "loan_paid_back", palette = "Set1", data = df_train, alpha=0.5 )
axis.set(title = "The loan payback rate by Sex", xlabel = "Sex",ylabel = "No.of People")
plt.show()


grid = sns.FacetGrid(df_train, col='gender',row='employment_status', hue='loan_paid_back', aspect=1.6)
grid.map(plt.hist, 'loan_paid_back', alpha=.8, bins=4)
grid.add_legend()


numerical_features = df_train[num_columns]


# Plotting scatter plots for selected numerical feature pairs
selected_pairs = [
    ('annual_income', 'loan_amount'),
    ('credit_score', 'interest_rate'),
    ('annual_income', 'debt_to_income_ratio'),
    ('loan_amount', 'interest_rate'),
    ('credit_score','loan_amount'),
    ('credit_score', 'debt_to_income_ratio')
]

for col1, col2 in selected_pairs:
    if col1 in df_train.columns and col2 in df_train.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df_train, x=col1, y=col2,hue='loan_paid_back', alpha=0.5)
        plt.title(f'Scatter Plot of {col1} vs {col2}')
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.tight_layout()
        plt.show()


all_grades = [f"{l}{n}" for l in "ABCDEF" for n in range(1, 6)]
print(all_grades)


le = LabelEncoder()
#This will encode the grade_subgrade column, then drop the original column
le.fit(all_grades)
df_train['grade_encoded'] = le.transform(df_train['grade_subgrade'])
df_train = df_train.drop(['grade_subgrade'], axis=1)
df_train.head(10)


#split the data into numerical and categorial features
numeric_features = df_train.select_dtypes(include=['int64', 'float64'])
categorical_features = df_train.select_dtypes(include=['object'])

#Encode the categorical features 
for colname in categorical_features:
    categorical_features[colname]=le.fit_transform(categorical_features[colname])

categorical_features.head(10)


#THis will create an encoded dataframe 
encoded_df = pd.concat([numeric_features, categorical_features], axis=1)
encoded_df.head(10)


#This code will check the correllation of other features to 'loan_paid_back'
corr_matrix = encoded_df.corr(numeric_only = True)
plt.figure(figsize=(10,15))
sns.heatmap(corr_matrix , cmap='inferno')

correlations = corr_matrix['loan_paid_back'].apply(abs).sort_values(ascending=False).reset_index()
print(correlations.shape)
correlations



# Extracting bindings with 'loan_paid_back' 

corr_matrix = encoded_df.select_dtypes(include=['float64', 'int64']).corr()
corr_with_target = corr_matrix["loan_paid_back"].drop("loan_paid_back")


top_corr = corr_with_target.sort_values(ascending=False).head(10)

# Drawing the bar graph
plt.figure(figsize=(10, 6))
sns.barplot(x=top_corr.values, y=top_corr.index, palette="coolwarm")
plt.title("10 Features Correlated with loan_paid_back")
plt.xlabel("Absolute Correlation")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


#This code will scale the data in the training data set
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

#Exclude the target variable from being scalled to maintain values as '1' and '0'
num = encoded_df.select_dtypes(include=("int64","float64")).drop(['loan_paid_back'], axis=1)
num_scaler= num.columns

#Scale the training data frame
for col in num_scaler :
    encoded_df[num_scaler] =scaler.fit_transform(encoded_df[num_scaler])
encoded_df.head(10)   


x=encoded_df.drop("loan_paid_back" ,axis=1)#Features 
y=encoded_df["loan_paid_back"] #Target


from sklearn.linear_model import LogisticRegression 

#split training data set for ML
x_train ,x_test ,y_train , y_test =train_test_split(x, y , test_size=0.40 , random_state=25)

#training 
log_reg=LogisticRegression()
log_reg.fit(x_train ,y_train)

#Post resul
y_pred1=log_reg.predict(x_test)
print(f"confusion_matrix :\n",confusion_matrix(y_test ,y_pred1))
print(f"accuracy :\n",accuracy_score(y_test,y_pred1))
print(f"precision_score :\n",precision_score(y_test,y_pred1))
print(f"recall_score :\n",recall_score(y_test ,y_pred1))
print(f"f1_score :\n", f1_score(y_test,y_pred1))
print(f"classification_report :\n", classification_report(y_test,y_pred1))


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
# predicted probability for class 1
y_scores = log_reg.predict_proba(x_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()


from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
ap = average_precision_score(y_test, y_scores)

plt.figure()
plt.plot(recall, precision, label='AP = %0.2f' % ap)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision–Recall Curve')
plt.legend()
plt.show()


# 't' represent the prefered threshold
t = 0.73

pos_weight = 1 / t          
neg_weight = 1 / (1 - t)    

log_reg2 = LogisticRegression(
    class_weight={0: neg_weight, 1: pos_weight},
    max_iter=1000
)

#train data with the new log_reg imput
log_reg2.fit(x_train ,y_train)
y_pred2=log_reg2.predict(x_test)

print(f"confusion_matrix :\n",confusion_matrix(y_test ,y_pred2))
print(f"accuracy :\n",accuracy_score(y_test,y_pred2))
print(f"precision_score :\n",precision_score(y_test,y_pred2))
print(f"recall_score :\n",recall_score(y_test ,y_pred2))
print(f"f1_score :\n", f1_score(y_test,y_pred2))
print(f"classification_report :\n", classification_report(y_test,y_pred2))


from xgboost import XGBClassifier

xgb = XGBClassifier(
    scale_pos_weight = (len(y_train[y_train==0]) / len(y_train[y_train==1])),
    learning_rate=0.05,
    max_depth=6,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss'
)

xgb.fit(x_train, y_train)

y_pred_xgb = xgb.predict(x_test)

print(confusion_matrix(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))


# Separating the id
ids = df_predict['id']
df_predict = df_predict.drop(['id'], axis=1)


#fit and encode the grade_subgrade
le.fit(all_grades)
df_predict['grade_encoded'] = le.transform(df_predict['grade_subgrade'])
df_predict = df_predict.drop(['grade_subgrade'], axis=1)
df_predict


#Encode the categorical data
Test_numeric_features = df_predict.select_dtypes(include=['int64', 'float64'])
Test_categorical_features = df_predict.select_dtypes(include=['object'])
for colname in Test_categorical_features:
    Test_categorical_features[colname]=le.fit_transform(Test_categorical_features[colname])

Test_categorical_features


#Merge categorical and numerical data
encoded_predictdf = pd.concat([Test_numeric_features, Test_categorical_features], axis=1)
encoded_predictdf.head()


#Scale the data
for col in num_scaler :
    encoded_predictdf[num_scaler] =scaler.fit_transform(encoded_predictdf[num_scaler])
encoded_predictdf.head(30)  


#Make predictions using the tuned Log_reg
x_compt=encoded_predictdf
predictions=log_reg2.predict(x_compt)

#predictions = np.exp(predictions)

output = pd.DataFrame({'id': ids,
                           'loan_status': predictions.squeeze()})
output.head()


# import the modules we'll need
from IPython.display import HTML
import base64

# function that takes in a dataframe and creates a text link to  
# download it (will only work for files < 2MB or so)
def create_download_link(df, title = "loan_paid_back", filename = "Submission.csv"):  
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

# create a random sample dataframe
df = pd.DataFrame({'id': ids,'loan_paid_back': predictions.squeeze()})

# create a link to download the dataframe
create_download_link(df)

# ↓ ↓ ↓  Yay, download link! ↓ ↓ ↓ 


#Make predictions using XGB
predictionsXgb = xgb.predict(x_compt)
output = pd.DataFrame({'id': ids,
                           'loan_status': predictionsXgb.squeeze()})
output.head()


# create a random sample dataframe
df2 = pd.DataFrame({'id': ids,'loan_paid_back': predictionsXgb.squeeze()})

# create a link to download the dataframe
create_download_link(df2)

# ↓ ↓ ↓  Yay, download link! ↓ ↓ ↓ 


submission = output
submission.to_csv("Titanic_submission.csv", index=False)




