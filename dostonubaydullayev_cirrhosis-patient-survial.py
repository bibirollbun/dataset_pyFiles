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

# vizualizatsiya
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# modellar
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Pipline
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer


# data preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

# o'lchovlar
import sklearn.metrics as metrics

# modelni saqlash uchun
import joblib 


# modelni o'qish uchun dataset
df_train = pd.read_csv("/kaggle/input/multiclassificationtask/train.csv", index_col=0)  # asosiy dataframe
df_train.head()


df_train_for_duplicates = pd.read_csv("/kaggle/input/multiclassificationtask/train.csv") # duplicat qiymatlani topish uchun 'id' ustunini tekshiramiz
df_train_for_duplicates['id'].duplicated().sum()


df = df_train.copy()


df.info()


# corr_matrix = df.corrwith(df["Status"], numeric_only=True).abs().sort_values(ascending=False)
# corr_matrix


# corr_matrix = df.corr().abs()
# corr_matrix.style.background_gradient(cmap='YlOrRd')


df.describe()


df.head()


numerical_features = [
    'N_Days', 'Age', 'Bilirubin', 'Cholesterol', 'Albumin',
    'Copper', 'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets',
    'Prothrombin', 'Stage'
]

categorical_features = [
    'Drug', 'Sex', 'Ascites', 'Hepatomegaly',
    'Spiders', 'Edema'
]


for col in numerical_features:
    df[col] = df[col].fillna(df[col].mean())

for col in categorical_features:
    if df[col].mode().empty:
        df[col] = df[col].fillna("Unknown")
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

    


df.info()


df["Status"].value_counts()


status_rate = df["Status"].value_counts()/len(df)*100
status_rate


# Status ustunida Y qiymati nisbatan kamligi uchun tashlab yuboramiz
df = df[df["Status"] != "Y"] 


replaceable = {"C":0, "D":1, 'CL':2}
df["Status"] = df["Status"].map(replaceable)


status_rate = df["Status"].value_counts()/len(df)*100
status_rate


classes = df["Status"].value_counts()

fig, ax = plt.subplots(1,2, figsize=(10,6))

# 1-grafig Bar plot
ax[0].bar(classes.index, classes.values, color='skyblue')
ax[0].set_xlabel("Holat")
ax[0].set_ylabel("Soni")

# 2-grafig Pie chart
ax=ax[1].pie(classes.values, labels=["C", "D", "CL"], autopct='%1.1f%%')

plt.suptitle("Status bo'yicha taqsimot", font="Bold")
plt.show()


for col in numerical_features:
    plt.figure(figsize=(10, 6))
    # Faqat sonli qiymatlar qoldiramiz (NaN va stringlarni tashlab)
    clean_values = pd.to_numeric(df[col], errors='coerce').dropna()
    plt.hist(clean_values, bins=50)
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Soni")
    plt.grid(True)
    plt.show()




for col in categorical_features:
    if col == "Sex":
        classes = df[col].value_counts()
        
        # 1-graph Bar plot
        fig, ax = plt.subplots(1,2, figsize=(10, 6))
        ax[0].bar(classes.index, classes.values, color="skyblue")
        ax[0].set_xlabel("Jins")
        ax[0].set_ylabel('Soni')
    
        # 2-graph Pie chart
        ax[1].pie(classes.values, labels=classes.index, autopct="%1.2f%%")
    
        plt.suptitle("Insonlarni jinsi bo'yicha taqsimot")
        
    
    else: 
        classes = df[col].value_counts()
        
        # 1-graph Bar plot
        fig, ax = plt.subplots(1,2, figsize=(10,6))
        ax[0].bar(classes.index, classes.values, color="skyblue")
        ax[0].set_xlabel(col)
        ax[0].set_ylabel('Soni')
    
        # 2-graph Pie chart
        ax[1].pie(classes.values, labels=classes.index, autopct="%1.2f%%")
    
        plt.suptitle(f"{col} ustuni taqsimoti")
        plt.show()
        categorical_features = ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders',
                        'Edema', ]
  


scaler = StandardScaler()

numerical_features = [
    'N_Days', 'Age', 'Bilirubin', 'Cholesterol', 'Albumin',
    'Copper', 'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets',
    'Prothrombin', 'Stage'
]
df[numerical_features] = scaler.fit_transform(df[numerical_features])
df.head()



encoder = OrdinalEncoder()

categorical_features = [
    'Drug', 'Sex', 'Ascites', 'Hepatomegaly',
    'Spiders', 'Edema'
]
df[categorical_features] = encoder.fit_transform(df[categorical_features])
df.head()



# train / test set
X = df.drop("Status", axis=1)
y = df["Status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print(X_train.shape)
print(X_test.shape)


y_train


LR_model = LogisticRegression()
LR_model.fit(X_train, y_train)

y_pred = LR_model.predict(X_test)
y_pred_proba = LR_model.predict_proba(X_test)  # Ehtimollikni olish

print(f"Classification Report:\n\n{metrics.classification_report(y_test, y_pred)}")
print(f"Accurcy: {metrics.accuracy_score(y_test, y_pred)*100:.1f}%")
print(f"Log Loss: {metrics.log_loss(y_test, y_pred_proba, labels=LR_model.classes_)}")

# confusion metrics
metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()


tree_model = RandomForestClassifier(class_weight='balanced', random_state=42, max_depth=14)
tree_model.fit(X_train, y_train)

y_pred = tree_model.predict(X_test)
y_pred_proba = tree_model.predict_proba(X_test)  # Ehtimollikni olish

print(f"Classification Report:\n\n{metrics.classification_report(y_test, y_pred)}")
print(f"Accurcy: {metrics.accuracy_score(y_test, y_pred)*100:.1f}%")
print(f"Log Loss: {metrics.log_loss(y_test, y_pred_proba, labels=tree_model.classes_)}")

# confusion metrics
metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()


XGBoost = XGBClassifier()
XGBoost.fit(X_train, y_train)

# tekshirish
y_pred = XGBoost.predict(X_test)
y_pred_proba = XGBoost.predict_proba(X_test)  # Ehtimollikni olish
print(f"Classification Report:\n\n{metrics.classification_report(y_test, y_pred)}")
print(f"Accurcy: {metrics.accuracy_score(y_test, y_pred)*100:.1f}%")
print(f"Log Loss: {metrics.log_loss(y_test, y_pred_proba, labels=XGBoost.classes_)}")

# confusion metrics
metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()
print("\n")


# model
svm = SVC(probability=True)
svm.fit(X_train, y_train)

# tekshirish
y_pred = svm.predict(X_test)
y_pred_proba = svm.predict_proba(X_test)  # Ehtimollikni olish
print(f"Classification Report:\n\n{metrics.classification_report(y_test, y_pred)}")
print(f"Accurcy: {metrics.accuracy_score(y_test, y_pred)*100:.1f}%")
print(f"Log Loss: {metrics.log_loss(y_test, y_pred_proba, labels=svm.classes_)}")


# confusion metrics
metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()
print("\n")


# model
# barcha kategorik ustunlarni aniqlash
categorical_columns = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
model = CatBoostClassifier(
    iterations=1000,              
    learning_rate=0.05,           
    depth=6,                     
    l2_leaf_reg=3,                
    auto_class_weights='Balanced',   
    cat_features=categorical_columns,   
    eval_metric='AUC',            
    early_stopping_rounds=50,     
    task_type='CPU',              
    verbose=100,                  
    random_seed=42)   
# Modelni o‘qitish
model.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),    
    use_best_model=True         
)

# tekshirish
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)  # Ehtimollikni olish
print(f"Classification Report:\n\n{metrics.classification_report(y_test, y_pred)}")
print(f"Accurcy: {metrics.accuracy_score(y_test, y_pred)*100:.1f}%")
print(f"Log Loss: {metrics.log_loss(y_test, y_pred_proba, labels=model.classes_)}")

# confusion metrics
metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()
print("\n")



# model
D_tree = DecisionTreeClassifier()
D_tree.fit(X_train, y_train)

# tekshirish
y_pred = D_tree.predict(X_test)
y_pred_proba = D_tree.predict_proba(X_test)  # Ehtimollikni olish
print(f"Classification Report:\n\n{metrics.classification_report(y_test, y_pred)}")
print(f"Accurcy: {metrics.accuracy_score(y_test, y_pred)*100:.1f}%")
print(f"Log Loss: {metrics.log_loss(y_test, y_pred_proba, labels=D_tree.classes_)}")


# confusion metrics
metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()
print("\n")


df_for_pred = pd.read_csv("/kaggle/input/multiclassificationtask/test.csv", index_col=0)
df_for_pred.head()


df2 = df_for_pred.copy()
df2.info()


# NaN qiymatlarni to'ldirish
for col in numerical_features:
    df2[col] = df2[col].fillna(df2[col].mean())

for col in categorical_features:
    if df2[col].mode().empty:
        df2[col] = df2[col].fillna("Unknown")
    else:
        df2[col] = df2[col].fillna(df2[col].mode()[0])

    


df2.info()


scaler = StandardScaler()

numerical_features = [
    'N_Days', 'Age', 'Bilirubin', 'Cholesterol', 'Albumin',
    'Copper', 'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets',
    'Prothrombin', 'Stage'
]
df2[numerical_features] = scaler.fit_transform(df2[numerical_features])
df2.head()


encoder = OrdinalEncoder()

categorical_features = [
    'Drug', 'Sex', 'Ascites', 'Hepatomegaly',
    'Spiders', 'Edema'
]
df2[categorical_features] = encoder.fit_transform(df2[categorical_features])
df2.head()



# Qiymatlarni predict qilamiz
X = df2.copy()
proba = XGBoost.predict_proba(X)


sample_submission = pd.read_csv("/kaggle/input/multiclassificationtask/sample_submission.csv")
sample_submission


submission = pd.DataFrame({
    "id":df2.index, 
    "Status_C":proba[:, 0], 
    "Status_CL":proba[:, 1],
    "Status_D":proba[:, 2]
})
submission


submission.to_csv("submission_me.csv")

