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

# o'lchovlar
from sklearn.pipeline import Pipeline
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler

# modelni saqlash uchun
import joblib 

# warniglarni yashirish
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



# train set qilib modelni o'qitadigan dataset
df_train = pd.read_csv("/kaggle/input/binaryclassificationwithabankchurndataset/train.csv", index_col=0)
df_train.head()


# bashorat qilib beriladigan datasetimiz
df_test = pd.read_csv("/kaggle/input/binaryclassificationwithabankchurndataset/test.csv", index_col=0)
df_test.head()


df = df_train.copy()


df.head()


df.info()


df.describe()


df.Exited.value_counts()


df.Exited.value_counts()/len(df)*100



plt.figure(figsize=(4,4))
plt.pie(df.Exited.value_counts()/len(df)*100, labels=["Qolgan","Ketgan"], autopct="%1.2f%%")
plt.show()


fig, ax = plt.subplots(1,3, figsize=(15,6))

sns.histplot(data=df, x="CreditScore", hue="Exited", multiple="stack", ax=ax[0])
ax[0].set_title("Mijozning kredit qobiliyati")

sns.histplot(data=df, x="EstimatedSalary", hue="Exited", multiple="stack", ax=ax[1])
ax[1].set_title("MIjozlarning yillik daromadi")
ax[1].set_ylabel("")

sns.histplot(data=df, x="Age", hue="Exited", multiple="stack", ax=ax[2])
ax[2].set_title("Mijozlarnig yoshi")
ax[2].set_ylabel("")

plt.show()


df.corrwith(df["Exited"], numeric_only=True).abs().sort_values(ascending=False)


df = df.drop(["CustomerId", "Surname"], axis=1)
df.sample()


df["Geography"].value_counts()


replaceable = {"Male":1, "Female":0}
replaceable2 = {"France":0, "Spain":1, "Germany":2}
df["Gender"] = df["Gender"].map(replaceable)
df["Geography"] = df["Geography"].map(replaceable2)
df.head()


# Malumotlarni Scaling qilamiz
scaler = StandardScaler()

num_cols = ["CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary"]
df[num_cols] = scaler.fit_transform(df[num_cols])
df.head()


X = df.drop("Exited", axis=1)
y = df["Exited"]


# train / test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


print(X_train.shape)
print(X_test.shape)


X_train


# model
RF_model = RandomForestClassifier()
RF_model.fit(X_train, y_train)

# tekshirish
y_pred = RF_model.predict(X_test)
print(f"Classification Report:\n\n{metrics.classification_report(y_test, y_pred)}")
print(f"Accurcy: {metrics.accuracy_score(y_test, y_pred)*100:.1f}%")

# confusion metrics
metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()
print("\n")

# ROC curve
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="ROC curve")
display.plot()
plt.show()


# model
XGBoost = XGBClassifier()
XGBoost.fit(X_train, y_train)

# tekshirish
y_pred = XGBoost.predict(X_test)
print(f"Classification Report:\n\n{metrics.classification_report(y_test, y_pred)}")
print(f"Accurcy: {metrics.accuracy_score(y_test, y_pred)*100:.1f}%")

# confusion metrics
metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()
print("\n")

# ROC curve
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="ROC curve")
display.plot()
plt.show()


# model
decision_tree = DecisionTreeClassifier(max_depth=6)
decision_tree.fit(X_train, y_train)

# tekshirish
y_pred = decision_tree.predict(X_test)
print(f"Classification Report:\n\n{metrics.classification_report(y_test, y_pred)}")
print(f"Accurcy: {metrics.accuracy_score(y_test, y_pred)*100:.1f}")

# confusion metrics
metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()
print("\n")

# ROC curve
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="ROC curve")
display.plot()
plt.show()


plt.figure(figsize=(30,20))
plot_tree(decision_tree, feature_names=X.columns, filled=True)
plt.show()


# model
LR_model = LogisticRegression()
LR_model.fit(X_train, y_train)

# tekshirish
y_pred = LR_model.predict(X_test)
print(f"Classification Report:\n\n{metrics.classification_report(y_test, y_pred)}")
print(f"Accurcy: {metrics.accuracy_score(y_test, y_pred)*100:.1f}%")

# confusion metrics
metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()
print("\n")

# ROC curve
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="ROC curve")
display.plot()
plt.show()


# model
svm = SVC()
svm.fit(X_train, y_train)

# tekshirish
y_pred = svm.predict(X_test)
print(f"Classification Report:\n\n{metrics.classification_report(y_test, y_pred)}")
print(f"Accurcy: {metrics.accuracy_score(y_test, y_pred)*100:.1f}%")

# confusion metrics
metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()
print("\n")

# ROC curve
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="ROC curve")
display.plot()
plt.show()


X_train


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
    task_type='GPU',              
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
print(f"Classification Report:\n\n{metrics.classification_report(y_test, y_pred)}")
print(f"Accurcy: {metrics.accuracy_score(y_test, y_pred)*100:.1f}%")

# confusion metrics
metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()
print("\n")

# ROC curve
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="ROC curve")
display.plot()
plt.show()


df_test.head()


df_test.info()


df_test = df_test.drop(["CustomerId", "Surname"], axis=1)
df_test.sample()


replaceable = {"Male":1, "Female":0}
replaceable2 = {"France":0, "Spain":1, "Germany":2}
df_test["Gender"] = df_test["Gender"].map(replaceable)
df_test["Geography"] = df_test["Geography"].map(replaceable2)
df.head()


 # Malumotlarni Scaling qilamiz
scaler = StandardScaler()

num_cols = ["CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary"]
df_test[num_cols] = scaler.fit_transform(df_test[num_cols])
df_test.head()



df_test.shape


X = df_test.copy()


y_pred = model.predict(X)
y_pred_proba = model.predict_proba(X)[:, 1]


y_pred_proba.shape


predicted = pd.DataFrame({'id':df_test.index, 
                         "Exited":y_pred_proba})
predicted.head()


# Submission file
predicted.to_csv("Submission_me.csv")




