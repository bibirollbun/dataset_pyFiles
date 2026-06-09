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
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler 

from sklearn.model_selection import train_test_split


train_data = pd.read_csv("/kaggle/input/playground-series-s4e10/train.csv")
train_data


test_data = pd.read_csv("/kaggle/input/playground-series-s4e10/test.csv")
test_data


train_data.info()


test_data.info()


train_data.shape


test_data.shape


train_data.columns


test_data.columns


train_data.duplicated().sum()


test_data.duplicated().sum()


train_data.isnull().sum()


test_data.isnull().sum()


train_data.sample(10)


for column in train_data.columns :
    print(train_data[column].value_counts(normalize=True))
    print("________________")


train_data["loan_status"] = train_data["loan_status"].replace({0 : "No" , 1 : "Yes"})


train_data.info()


train_data.isnull().sum()


train_data.select_dtypes(include=["object"]).columns


for column in train_data.select_dtypes(include=["object"]).columns:
    train_data[column] =  train_data[column].astype("category")

train_data.info()


# Dictionary Comprehension
test_data = test_data.astype({column : "category" for column in test_data.select_dtypes(include="object").columns })
test_data.info()


del train_data["id"]


int_flo = train_data.select_dtypes(include = ("int64" , "float64")).columns
int_flo


category = train_data.select_dtypes(include = "category").columns
category    


px.histogram(train_data , x = "person_age" , title = "Histogram for Ages distribution")


for column in int_flo :
    fig = px.histogram(train_data , x = column , color="loan_status", title = f"Histogram for {column} distribution")
    fig.show()


# supplot
plt.figure(figsize=(18,13))
for i , column in enumerate(int_flo , start=1):
    plt.subplot(4,2 , i)
    sns.boxplot(x = train_data[column])
    plt.title(f"Boxplot of {column}")
plt.tight_layout()    
plt.show()


for column in int_flo :
    print(f"Column Name : {column}")
    
    Q1 = train_data[column].quantile(0.25)
    Q3 = train_data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - (1.5 * IQR)
    upper = Q3 + (1.5 * IQR)
    
    print (f" Q1 : {Q1} , Q3 : {Q3} \nLower bound is : {lower} \nUpper Bound is : {upper}")
    outlier = train_data[(train_data[column] < lower) | (train_data[column] > upper )]
    print(f"Number of outlier : {outlier.shape[0]}")
    print("____________________")


train_data.columns


data = train_data["person_age"]


med = data.median()


data.describe()


sns.kdeplot(train_data["person_age"] , label = data , fill=True )


# median , mad , MZ 
abs_deviation = abs(data - med)
abs_deviation


mad = abs_deviation.median()
mad


modified_z_score = 0.6745 * (data - med) / mad
modified_z_score


sns.kdeplot(modified_z_score , label = data , fill=True )


modified_z_score.describe()


threshold = 3.5
outliers = train_data[(modified_z_score < -3.5) | (modified_z_score > 3.5) ]
outliers
# outlier = abs(modified_z_score > 3.5 )


def mad_outliers(column_name , threshold = 3.5): # threshold = 3.5 Defult value
    med = train_data[column_name].median()
    abs_deviation = abs (train_data[column_name] - med )
    mad = abs_deviation.median()
    modified_z_score = 0.6745 * (train_data[column_name] - med ) / mad
    outliers =  train_data[(modified_z_score < -threshold) | (modified_z_score > threshold )]
    print(f"Outlier size in {column_name} is : {outliers.shape[0]}")
    return outliers

for column in int_flo:
    mad_outliers(column , 4)


for column in int_flo:
    mad_outliers(column , 2.5)


for column in int_flo:
    mad_outliers(column)   # threshold = 3.5 Defult value


test_data


def mad_outliers(column_name , threshold = 3.5):
    med = test_data[column_name].median()
    abs_deviation = abs (test_data[column_name] - med)
    mad = abs_deviation.median()
    modified_z_score = 0.6745 * (test_data[column_name] - med) / mad
    outliers = test_data[(modified_z_score < -threshold) | (modified_z_score > threshold)]
    print(f"Outlier size in {column_name} is : {outliers.shape[0]}")

for col in test_data.select_dtypes(include=("float64" , "int64")).columns :
    mad_outliers(column , 3.5)


train_data1 = train_data.copy()
train_data1


def mad_outliers(column_name , threshold = 3.5): # threshold = 3.5 Defult value
    med = train_data1[column_name].median()
    abs_deviation = abs (train_data1[column_name] - med )
    mad = abs_deviation.median()
    modified_z_score = 0.6745 * (train_data1[column_name] - med ) / mad
    outliers =  (modified_z_score < -threshold) | (modified_z_score > threshold )
    #print(f"Outlier size in {column_name} is : {outliers.shape[0]}")
    return outliers


# deleting Outlier from column person_age 
age_outlier =  mad_outliers("person_age")
age_outlier


train_data1 = train_data1[~age_outlier]
train_data1
#dropna : drop null values 


int_flo


# supplot
# Plotting after handling outlire in age ONLY
plt.figure(figsize=(20,15))
for i , column in enumerate(int_flo , start=1):
    plt.subplot(4,2 , i)
    sns.boxplot(x = train_data[column])
    plt.title(f"Boxplot of {column}")
plt.tight_layout()    
plt.show()


px.histogram(train_data , x = "person_age", color="loan_status" , title = "Histogram for Ages distribution")


px.histogram(train_data1 , x = "person_age" ,  color="loan_status" , title = "Histogram for Ages distribution")


# person_income
income_outlier =  mad_outliers("person_income")
train_data1 = train_data1[~income_outlier]
fig1 = px.histogram(train_data , x = "person_income", color="loan_status" , title = "Histogram for Income distribution")
fig1.show()
fig2 = px.histogram(train_data1 , x = "person_income", color="loan_status" , title = "Histogram for Income distribution")
fig2.show()


train_data1


# supplot
# Plotting after handling outlire in age ONLY
plt.figure(figsize=(20,15))
for i , column in enumerate(int_flo , start=1):
    plt.subplot(4,2 , i)
    sns.boxplot(x=train_data1[column])
    plt.title(f"Boxplot of {column}")
plt.tight_layout()    
plt.show()


# cb_person_cred_hist_length
income_outlier =  mad_outliers("cb_person_cred_hist_length")
train_data1 = train_data1[~income_outlier]
fig1 = px.histogram(train_data , x = "cb_person_cred_hist_length", color="loan_status" , title = "Histogram for cp_history distribution")
fig1.show()
fig2 = px.histogram(train_data1 , x = "cb_person_cred_hist_length", color="loan_status" , title = "Histogram for cp_history distribution")
fig2.show()


train_data1


# supplot
# Plotting after handling outlire in age ONLY
plt.figure(figsize=(20,15))
for i , column in enumerate(int_flo , start=1):
    plt.subplot(4,2 , i)
    sns.boxplot(x = train_data[column])
    plt.title(f"Boxplot of {column}")
plt.tight_layout()    
plt.show()


# Define function to detect outlier 
def mad_outliers(column_name , threshold = 3.5): # threshold = 3.5 Defult value
    med = train_data1[column_name].median()
    abs_deviation = abs(train_data1[column_name] - med )
    mad = abs_deviation.median()
    modified_z_score = 0.6745 * (train_data1[column_name] - med ) / mad
    outliers = (modified_z_score < -threshold) | (modified_z_score > threshold )
    print(f"Outlier size in {column_name} is : {outliers.shape[0]}")
    return outliers

for column in int_flo: 
    income_outlier =  mad_outliers(column, 4.5)
    train_data1 = train_data1[~income_outlier]
    fig1 = px.histogram(train_data , x = column, color="loan_status" , title = f"Histogram for {column} distribution before Outlier Handling")
    fig1.show()
    fig2 = px.histogram(train_data1 , x = column, color="loan_status" , title = f"Histogram for {column} distribution After Outlier Handling")
    fig2.show()


# supplot
# Plotting after handling outlire in age ONLY
plt.figure(figsize=(20,15))
for i , column in enumerate(int_flo , start=1):
    plt.subplot(4,2 , i)
    sns.boxplot(x=train_data1[column])
    plt.title(f"Boxplot of {column}")
plt.tight_layout()    
plt.show()
train_data1


LE = LabelEncoder()
for column in category:
    train_data[column] = LE.fit_transform(train_data[column])

train_data


train_data.shape


X = train_data.drop("loan_status" , axis = 1)
y = train_data["loan_status"]


X


y


X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 , random_state = 42)


print("X_train :" , X_train.shape)
print("X_test :" , X_test.shape)
print("y_train :" , y_train.shape)
print("y_test :" , y_test.shape)


int_flo


scaler = StandardScaler()

for column in int_flo:
    train_data[[column]] = scaler.fit_transform(train_data[[column]])

train_data


int_flo = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt',
       'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']

#plt.figure(figsize=(15,20))

for column in int_flo:
    plt.figure(figsize=(5,5))
    sns.kdeplot(train_data[column] , label = column , fill=True )
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.show()


from sklearn.model_selection import train_test_split

X = train_data.drop("loan_status", axis=1)
y = train_data["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train :", X_train.shape)
print("y_train :", y_train.shape)
print("X_test :", X_test.shape)
print("y_test :", y_test.shape)


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# نموذج k-Nearest Neighbors (kNN)
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("دقة kNN:", accuracy_knn)

# نموذج Logistic Regression
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)
accuracy_log = accuracy_score(y_test, y_pred_log)
print("دقة Logistic Regression:", accuracy_log)

# نموذج Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("دقة Decision Tree:", accuracy_dt)


param_grid = {
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("أفضل معاملات:", grid_search.best_params_)
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)
print("دقة النموذج المحسن:", accuracy_best)


# استيراد المكتبات اللازمة
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# تدريب نموذج k-Nearest Neighbors (kNN)
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("kNN Accuracy:", accuracy_knn)

# تدريب نموذج Logistic Regression
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)
accuracy_log = accuracy_score(y_test, y_pred_log)
print("Logistic Regression Accuracy:", accuracy_log)

# تدريب نموذج Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("Decision Tree Accuracy:", accuracy_dt)

# تحسين المعاملات باستخدام GridSearchCV لنموذج Decision Tree
param_grid = {
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)
print("Tuned Decision Tree Accuracy:", accuracy_best)


# استيراد المكتبات اللازمة
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# دالة لتقييم النموذج
def evaluate_model(model, model_name, X_test, y_test):
    # التنبؤ
    y_pred = model.predict(X_test)
    
    # حساب المقاييس
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # حساب ROC Curve و AUC
    y_prob = model.predict_proba(X_test)[:, 1]  # احتمالية الفئة الإيجابية
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # طباعة النتائج
    print(f"\nتقييم نموذج {model_name}:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"AUC: {roc_auc:.2f}")
    print("Confusion Matrix:")
    print(cm)
    
    # رسم ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.show()

# تقييم النماذج
evaluate_model(knn, "kNN", X_test, y_test)
evaluate_model(log_reg, "Logistic Regression", X_test, y_test)
evaluate_model(dt, "Decision Tree", X_test, y_test)


# Import metrics for comparison
from sklearn.metrics import f1_score, roc_auc_score

# Get predictions from trained models
y_pred_knn = knn.predict(X_test)
y_pred_log_reg = log_reg.predict(X_test)
y_pred_dt = dt.predict(X_test)

# Calculate F1 Scores for comparison
f1_knn = f1_score(y_test, y_pred_knn)
f1_log_reg = f1_score(y_test, y_pred_log_reg)
f1_dt = f1_score(y_test, y_pred_dt)

# Calculate AUC for additional comparison
auc_knn = roc_auc_score(y_test, knn.predict_proba(X_test)[:, 1])
auc_log_reg = roc_auc_score(y_test, log_reg.predict_proba(X_test)[:, 1])
auc_dt = roc_auc_score(y_test, dt.predict_proba(X_test)[:, 1])

# Print model performances
print("Model Performances:")
print(f"kNN - F1 Score: {f1_knn:.2f}, AUC: {auc_knn:.2f}")
print(f"Logistic Regression - F1 Score: {f1_log_reg:.2f}, AUC: {auc_log_reg:.2f}")
print(f"Decision Tree - F1 Score: {f1_dt:.2f}, AUC: {auc_dt:.2f}")

# Choose the best model based on F1 Score
models_f1 = {'kNN': f1_knn, 'Logistic Regression': f1_log_reg, 'Decision Tree': f1_dt}
best_model = max(models_f1, key=models_f1.get)
print(f"\nBest Model: {best_model} with F1 Score: {models_f1[best_model]:.2f}")

# Suggestions for improvements
print("\nSuggestions for Further Improvements:")
print("- Use advanced models like Random Forest or XGBoost for better accuracy.")
print("- Increase dataset size or handle imbalanced classes with SMOTE.")
print("- Perform cross-validation for more reliable results.")
print("- Tune hyperparameters more extensively with GridSearchCV.")

