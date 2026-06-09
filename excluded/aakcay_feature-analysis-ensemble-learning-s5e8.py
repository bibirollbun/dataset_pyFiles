import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings("ignore")



train_data = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


# rename y column
train_data.rename(columns={'y': 'target'}, inplace=True)



train_data.info()


train_data.head()


for i in train_data.columns:
    print(f"{i}: \n{train_data[i].unique()} \n {'*'*50}")


train_data.isnull().sum()


test_data.isnull().sum()


print(sum(train_data['education']=='unknown'))
print(sum(train_data['job']=='unknown'))
print(sum(train_data['contact']=='unknown'))
print(sum(train_data['poutcome']=='unknown'))


print(sum(test_data['education']=='unknown'))
print(sum(test_data['job']=='unknown'))
print(sum(test_data['contact']=='unknown'))
print(sum(test_data['poutcome']=='unknown'))


print(train_data['education'].mode()) 
print(train_data['job'].mode())
print(train_data['contact'].mode()) 
print(train_data['poutcome'].mode())


#train_data['education'].replace('unknown', train_data['education'].mode()[0], inplace=True)
#test_data['education'].replace('unknown', test_data['education'].mode()[0], inplace=True)


f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(train_data.select_dtypes(include=int).corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)
plt.show()


import seaborn as sns

plt.figure(figsize=(15,15))
col=train_data.select_dtypes(include=int).columns     
for i in range(len(col)):
        
    plt.subplot(3,3,i+1)
    
    sns.boxplot(x=train_data[col[i]])
       
    plt.xlabel(col[i].replace('_'," "))
    plt.ylabel(' ')


cat_cols = train_data.select_dtypes(include='object').columns

plt.figure(figsize=(18, 20))
for i, col in enumerate(cat_cols):
    plt.subplot(3, 3, i + 1)
    
    sns.countplot(data=train_data, x=col, order=train_data[col].value_counts().index, palette="viridis")
    
    plt.xlabel(col.replace('_', ' '))
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.title(f"{col} Distribution")

plt.tight_layout()
plt.show() 


cat_cols = train_data.select_dtypes(include='object').columns
cat_cols = [col for col in cat_cols if col != 'marital']  

plt.figure(figsize=(18, 25))
for i, col in enumerate(cat_cols):
    plt.subplot(4, 2, i + 1)

    sns.countplot(data=train_data, x=col, hue='marital', order=train_data[col].value_counts().index, palette="viridis")
    
    plt.xlabel(col.replace('_', ' '))
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.title(f"{col} by Marital Status")

plt.tight_layout()
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

categorical_cols = train_data.select_dtypes(include='object').columns
numerical_cols = train_data.select_dtypes(include='int').drop('target', axis=1).columns

X = train_data.drop('target', axis=1)
y = train_data['target']

all_data = pd.concat([X, test_data], axis=0)

all_data_encoded = pd.get_dummies(all_data, columns=categorical_cols, drop_first=True)

X_train_full = all_data_encoded.iloc[:len(train_data), :]
X_test_final = all_data_encoded.iloc[len(train_data):, :]



X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y, test_size=0.2, random_state=42)


from xgboost import XGBClassifier
import numpy as np
import matplotlib.pyplot as plt

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)

importances = xgb_model.feature_importances_
features = X_train.columns
indices = np.argsort(importances)[::-1]

top_n = 30  

plt.figure(figsize=(12, 8))
plt.title("Top Feature Importances (XGBoost)")
plt.barh(range(top_n), importances[indices[:top_n]][::-1])
plt.yticks(range(top_n), [features[i] for i in indices[:top_n]][::-1])
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

selected_features = [features[i] for i in indices[:top_n]]
X_train_selected = X_train[selected_features]
X_valid_selected = X_valid[selected_features]
X_test_selected = X_test_final[selected_features]



from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

models = {
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "LightGBM": LGBMClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "CatBoostClassifier":CatBoostClassifier(verbose=0, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_valid_selected)
    y_proba = model.predict_proba(X_valid_selected)[:, 1]
    
    results[name] = {
        'Accuracy': accuracy_score(y_valid, y_pred),
        'F1 Score': f1_score(y_valid, y_pred),
        'ROC AUC': roc_auc_score(y_valid, y_proba)
    }

import pandas as pd
results_df = pd.DataFrame(results).T
print(results_df.sort_values(by='ROC AUC', ascending=False))



best_model = CatBoostClassifier(verbose=0, random_state=42)
best_model.fit(X_train_full[selected_features], y)

test_preds = best_model.predict_proba(X_test_selected)[:, 1]

submission = pd.DataFrame({
    'id': test_data['id'],
    'y': test_preds
})
submission.to_csv('submission.csv', index=False)


