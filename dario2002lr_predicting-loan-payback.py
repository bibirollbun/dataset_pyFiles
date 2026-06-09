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
import matplotlib.pyplot as plt


train_data = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
train_data.head()


print("train_data : " + str(train_data.shape))
print("test_data : " + str(test_data.shape))


import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

target = train_data.loan_paid_back

counts = train_data["loan_paid_back"].value_counts().sort_index()

plt.bar(counts.index, counts.values)
plt.xticks([0, 1])
plt.xlabel("Valor")
plt.ylabel("Frecuencia")
plt.title("Distribución de loan_paid_back")
plt.show()


labels = counts.index
sizes = counts.values

colors = ['#66b3ff', '#ff9999']

# Crear gráfico de pie con 'wedgeprops' para que sea donut
plt.figure(figsize=(6,6))
plt.pie(
    sizes, 
    labels=labels, 
    autopct='%1.1f%%', 
    startangle=90,
    colors=colors,
    wedgeprops={'width':0.4, 'edgecolor':'w'})
plt.title("Distribución de loan_paid_back")
plt.show()


train_data.select_dtypes(include=["number"]).drop("loan_paid_back", axis=1).columns


train_data.select_dtypes(include=["number"]).drop("loan_paid_back", axis=1).describe().round(decimals=2)


num_attributes = train_data.select_dtypes(include=["number"]).drop("loan_paid_back", axis=1).copy()

for col in num_attributes.columns:
    plt.figure(figsize=(6,4))
    sns.histplot(num_attributes[col], kde=True, bins=30, color='skyblue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()


for col in num_attributes.columns:
    plt.figure(figsize=(6,4))
    sns.boxplot(num_attributes[col], color='skyblue')
    plt.title(f'Boxplot of {col}')
    plt.ylabel(col)
    plt.show()


for col in num_attributes.columns:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='loan_paid_back', y=col, data=train_data, palette='pastel')
    plt.title(f'Boxplot of {col} by loan_paid_back')
    plt.xlabel('loan_paid_back')
    plt.ylabel(col)
    plt.show()


num_attributes = train_data.select_dtypes(include=["number"]).copy()
corr_matrix = num_attributes.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix of Numerical Variables')
plt.show()


train_data.select_dtypes(exclude=["number"]).columns


cat_attributes = train_data.select_dtypes(exclude=["number"]).copy()

for col in cat_attributes.columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(x=col, data=train_data)
    plt.title(f'Count of categories in {col}')
    plt.xticks(rotation=45)  
    plt.show()


for col in cat_attributes.columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(x=col, hue='loan_paid_back', data=train_data, palette='pastel')
    plt.title(f'Count of categories in {col} by loan_paid_back')
    plt.xticks(rotation=45)
    plt.legend(title='loan_paid_back')
    plt.show()


for col in cat_attributes.columns:
    # Calculamos proporciones
    prop = train_data.groupby([col, 'loan_paid_back']).size().unstack(fill_value=0)
    prop = prop.div(prop.sum(axis=1), axis=0)  # normalizamos por total de la categoría
    
    prop.plot(kind='bar', stacked=True, figsize=(8,5), color=['lightcoral', 'skyblue'])
    plt.title(f'Proportion of loan_paid_back by {col}')
    plt.ylabel('Proportion')
    plt.xticks(rotation=45)
    plt.legend(title='loan_paid_back', labels=['0','1'])
    plt.show()


train_data.isna().sum().sort_values(ascending=False).head(10)


test_data.isna().sum().sort_values(ascending=False).head(10)


train_data_copy = train_data.copy()

train_data_copy = train_data_copy[
    ~((train_data_copy['credit_score'] <= 400))]


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

X = train_data_copy.drop(["id",'loan_paid_back'], axis=1)
y = train_data_copy['loan_paid_back']

X_sample, _, y_sample, _ = train_test_split(X, y, train_size=0.1, stratify=y, random_state=42)

num_features = X_sample.select_dtypes(include=['number']).columns
cat_features = X_sample.select_dtypes(exclude=['number']).columns

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_features)
])

X_train, X_val, y_train, y_val = train_test_split(
    X_sample, y_sample, test_size=0.2, stratify=y_sample, random_state=42
)

preprocessor.fit(X_train)
X_train_processed = preprocessor.transform(X_train)
X_val_processed = preprocessor.transform(X_val)

models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'RandomForest': RandomForestClassifier(n_estimators=10, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=10, use_label_encoder=False, eval_metric='logloss', random_state=42, tree_method='hist'),
    'NeuralNetwork': MLPClassifier(hidden_layer_sizes=(16,8), max_iter=50, random_state=42)
}

results = []

for name, model in models.items():
    model.fit(X_train_processed, y_train)
    y_pred = model.predict(X_val_processed)
    y_pred_prob = model.predict_proba(X_val_processed)[:,1]
    
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    roc = roc_auc_score(y_val, y_pred_prob)
    
    results.append({'Model': name, 'Accuracy': acc, 'F1': f1, 'ROC_AUC': roc})

results_df = pd.DataFrame(results).sort_values('ROC_AUC', ascending=False)
print(results_df)


X = train_data_copy.drop(["id", 'loan_paid_back', "loan_amount", "annual_income",
                         "gender", "marital_status"], axis=1)
y = train_data_copy['loan_paid_back']

X_sample, _, y_sample, _ = train_test_split(X, y, train_size=0.1, stratify=y, random_state=42)

num_features = X_sample.select_dtypes(include=['number']).columns
cat_features = X_sample.select_dtypes(exclude=['number']).columns

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_features)
])

X_train, X_val, y_train, y_val = train_test_split(
    X_sample, y_sample, test_size=0.2, stratify=y_sample, random_state=42
)

preprocessor.fit(X_train)
X_train_processed = preprocessor.transform(X_train)
X_val_processed = preprocessor.transform(X_val)

models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'RandomForest': RandomForestClassifier(n_estimators=10, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=10, use_label_encoder=False, eval_metric='logloss', random_state=42, tree_method='hist'),
    'NeuralNetwork': MLPClassifier(hidden_layer_sizes=(16,8), max_iter=50, random_state=42)
}

results = []

for name, model in models.items():
    model.fit(X_train_processed, y_train)
    y_pred = model.predict(X_val_processed)
    y_pred_prob = model.predict_proba(X_val_processed)[:,1]
    
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    roc = roc_auc_score(y_val, y_pred_prob)
    
    results.append({'Model': name, 'Accuracy': acc, 'F1': f1, 'ROC_AUC': roc})

results_df = pd.DataFrame(results).sort_values('ROC_AUC', ascending=False)
print(results_df)


from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [3, 5, 7],
    'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'classifier__subsample': [0.6, 0.8, 1.0],
    'classifier__colsample_bytree': [0.6, 0.8, 1.0],
    'classifier__gamma': [0, 0.1, 0.2],
    'classifier__min_child_weight': [1, 3, 5]
}

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, tree_method='hist'))
])

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=20,           
    scoring='roc_auc',   
    cv=3,                
    verbose=1,
    n_jobs=-1,
    random_state=42
)

search.fit(X_train, y_train)

print("Best ROC_AUC:", search.best_score_)
print("Best parameters:", search.best_params_)

best_model = search.best_estimator_
y_val_pred_prob = best_model.predict_proba(X_val)[:,1]

from sklearn.metrics import roc_auc_score
print("Validation ROC_AUC:", roc_auc_score(y_val, y_val_pred_prob))


X_train_sample, _, y_train_sample, _ = train_test_split(
    train_data_copy.drop(["id", 'loan_paid_back', "loan_amount", "annual_income",
                         "gender", "marital_status"], axis=1),
    train_data_copy['loan_paid_back'],
    train_size=0.5,
    stratify=train_data_copy['loan_paid_back'],
    random_state=42
)

final_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        tree_method='hist',
        n_estimators=200,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.6,
        gamma=0.1,
        min_child_weight=1
    ))
])

final_pipeline.fit(X_train_sample, y_train_sample)


X = test_data.drop(["id", "loan_amount", "annual_income",
                         "gender", "marital_status"], axis=1)

y_pred_prob = final_pipeline.predict_proba(X)[:,1]

results_df = test_data[["id"]].copy()
results_df["loan_paid_back"] = y_pred_prob

results_df.to_csv("loan_predictions.csv", index=False)

print(results_df.head())


from lightgbm import LGBMClassifier

X_train_full = train_data_copy.drop(["id", 'loan_paid_back', "loan_amount", "annual_income",
                                     "gender", "marital_status"], axis=1)
y_train_full = train_data_copy['loan_paid_back']

final_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LGBMClassifier(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.6,
        random_state=42
    ))
])

final_pipeline.fit(X_train_full, y_train_full)


X = test_data.drop(["id", "loan_amount", "annual_income",
                         "gender", "marital_status"], axis=1)

y_pred_prob = final_pipeline.predict_proba(X)[:,1]

results_df = test_data[["id"]].copy()
results_df["loan_paid_back"] = y_pred_prob

results_df.to_csv("loan_predictions_lgbm.csv", index=False)

print(results_df.head())

