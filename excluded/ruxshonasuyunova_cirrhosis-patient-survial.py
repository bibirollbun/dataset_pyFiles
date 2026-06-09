# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, log_loss, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier


df = pd.read_csv('/kaggle/input/multiclassificationtask/train.csv')
df.head()


df.shape


df.info()


df.describe()


df['Age'].isnull().sum()


# Datasetni raqamli va kategorik featurlarga bo'lib olamiz
numerical_features = ['N_Days','Age', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper',
                      'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin', 'Stage']

categorical_features = ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders',
                        'Edema', ]


classes = df['Status'].value_counts()

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# 1-grafik: Bar plot
ax[0].bar(classes.index, classes.values, color='skyblue')
ax[0].set_xlabel("Holat")
ax[0].set_ylabel("Soni")

# 2-grafik: Pie chart
ax[1].pie(classes.values, labels=classes.index, autopct='%1.1f%%', colors=['lightcoral', 'lightgreen', 'lightblue'])
 

fig.suptitle("Status bo'yicha taqsimot", fontsize=15)
plt.tight_layout(pad=1)
plt.show


df = df[df['Status'] != 'Y']


for column in numerical_features:
    plt.figure(figsize=(10, 5))
    sns.histplot(data=df, x=column, bins=30, kde=True)
    plt.xlabel(column)
    plt.ylabel('Soni')
    plt.title(f' {column} taqsimoti')
    plt.tight_layout()
    plt.show()
    


for col in categorical_features:
    counts = df[col].value_counts()
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # Barchart
    ax[0].bar(counts.index, counts.values, color='skyblue')
    ax[0].set_title(f"{col} bo'yicha taqsimot")
    ax[0].set_xlabel(col)
    ax[0].set_ylabel("Soni")

    # Pie chart
    ax[1].pie(counts.values, labels=counts.index, autopct='%1.1f%%', colors=['lightcoral', 'lightgreen', 'lightblue', 'lightyellow'])
    ax[1].set_title(f"{col} bo'yicha taqsimot")

    plt.tight_layout()
    plt.show


df['Status'].value_counts()


# Status ustunini encode qilamiz
# Map status 0-D, 1-C, 2-CL
df['Status'] = df['Status'].map({'D':0, 'C':1, 'CL':2})


df['Status'].value_counts()


# sonli ustunlar uchun
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])

# kategorik ustunar uchun
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder())
])


preprocessor = ColumnTransformer([
    ('numerical', num_pipeline, numerical_features),
    ('categorical', cat_pipeline, categorical_features)
])


# Feature va target qismlarga bo'lib olamiz
X = df.drop('Status', axis=1)
y = df['Status']


X['Age'] = X['Age']//365
X


# train va test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


X_test_processed.shape


models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(class_weight='balanced'),
    'Gradient Boosting': GradientBoostingClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=1.0)
}

results = {}
# Klasslar ro'yxatini aniqlash
all_classes = np.unique(np.concatenate((y_train, y_test))) 

for name, model in models.items():
    print(f"Model: {name}")
    # Modelni o'rgatish
    model.fit(X_train_processed, y_train)
 
    #predict qilamiz
    y_pred = model.predict(X_test_processed)
    y_pred_proba = model.predict_proba(X_test_processed)
    if y_pred_proba.shape[1] == 2:
        y_pred_proba = y_pred_proba[:, 1] 
    else:
    
        pass

   # Clip qilish
    y_pred_proba_clipped = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)

    # logloss hisoblash
    logloss = log_loss(y_test, y_pred_proba_clipped, labels=all_classes)
    # accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(f"Logloss: {logloss:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    results[name] = {
        'Accuracy': acc,
        'Logloss': logloss
    }
 


y_pred_proba_clipped


# Eng yaxshi modelni log loss boyicha tanlash
best_model_name = min(results.items(), key=lambda x: x[1]['Logloss'])[0]
best_model = models[best_model_name]
print(f"\n Eng yaxshi model (LogLoss boyicha): {best_model_name} (LogLoss: {results[best_model_name]['Logloss']:.4f})")


test_df = pd.read_csv('/kaggle/input/multiclassificationtask/test.csv')
test_df.head()


test_df.shape


test_df.isnull().sum()


test_df['Age'] = test_df['Age']//365
test_df


# Yuqorida yaratib olgan pipeline orqali test setni ham transform qilib olamiz
test_processed = preprocessor.transform(test_df)


# predict qilamiz
y_test_bashorat = best_model.predict(test_processed)
y_test_bashorat_proba = best_model.predict_proba(test_processed)

# Ehtimolliklarni [1e-15, 1-1e-15] oralig‘ida chegaralash
y_test_bashorat_proba_clipp = np.clip(y_test_bashorat_proba, 1e-15, 1 - 1e-15)


y_test_bashorat_proba.shape


y_test_bashorat_proba_clipp


ids = test_df['id']
ids


model.classes_


print(len(ids))
print(y_test_bashorat_proba_clipp.shape)


submission = pd.DataFrame({
    'id':ids,
    'Status_C':y_test_bashorat_proba_clipp[:,1],
    'Status_CL':y_test_bashorat_proba_clipp[:,2],
    'Status_D':y_test_bashorat_proba_clipp[:,0]
})
submission


#'D':0, 'C':1, 'CL':2}
# CSV faylga saqlash
submission.to_csv('submission.csv', index=False)
print("Submission fayli tayyor: submission.csv")

