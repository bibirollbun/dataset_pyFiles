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
import seaborn as sns
from pandas.api.types import CategoricalDtype
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

train_set = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
train_set.head(10)
train_set.info()


#check NA
train_set.isna().sum()
#transfer id, day to object
train_set['id'] = train_set['id'].astype('object')
train_set['day'] = train_set['day'].astype('object')
#target variable inspection
y_counts = train_set['y'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(y_counts, labels=y_counts.index, autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#ff9999'])
plt.title("Distribution of 'y' variable")
plt.show()


numcols = train_set.select_dtypes(include=['int64','float64']).columns.tolist()
train_set[numcols].head()
train_set[numcols].describe()


#correlation plot for numerical variables
corr = train_set[numcols].corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            cmap=sns.diverging_palette(220, 10, as_cmap=True),
            annot=True)
plt.show()


#chi squared test for categorical variables
catcols = train_set.select_dtypes(include=['object','category']).columns.tolist()
catcols = [c for c in catcols if c != "id"]
train_set[catcols].head()

for i in range(len(catcols)):
    for j in range(i + 1, len(catcols)):
        contingency_table = pd.crosstab(train_set[catcols[i]], train_set[catcols[j]])
        chi2, p, _, _ = chi2_contingency(contingency_table)
        print(f"Chi-squared test between {catcols[i]} and {catcols[j]}: chi2 = {chi2}, p-value = {p}")



#chi squared test for target variable
for cat in catcols:
    contingency_table = pd.crosstab(train_set[cat], train_set['y'])
    chi2, p, _, _ = chi2_contingency(contingency_table)
    print(f"Chi-squared test between y and {cat}: chi2 = {chi2:.2f}, p-value = {p:.4f}")


#plots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for idx, cat in enumerate(catcols[:6]):
    ax = axes[idx // 3, idx % 3]
    sns.countplot(data=train_set, x=cat, hue='y', ax=ax)
    ax.set_title(f'Subscription (y) by {cat}')
    ax.set_xlabel(cat)
    ax.set_ylabel('Count')
    ax.legend(title='y', loc='upper right')
    ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()


#logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
numcols = ['age','balance','campaign','pdays','previous','duration']          
catcols = ['job','marital','education','default','housing','loan','contact','day','month','poutcome']


X = train_set[numcols + catcols]
y = train_set['y']
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numcols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), catcols),
], sparse_threshold=0.1)

logit_pipe = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(
        solver="saga", penalty="l2", C=1.0, max_iter=200, n_jobs=-1, random_state=42, class_weight="balanced"
    ))
])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)
logit_pipe.fit(X_train, y_train)
y_proba = logit_pipe.predict_proba(X_test)[:, 1]
y_pred  = (y_proba >= 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC :", roc_auc_score(y_test, y_proba))
print("Classification report:\n", classification_report(y_test, y_pred))


# ROC AUC curve plot
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc_score(y_test, y_proba):.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUC ROC Curve')
plt.legend(loc='lower right')
plt.show()


#predict test set
test_set = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
test_set['id'] = test_set['id'].astype('object')
test_set['day'] = test_set['day'].astype('object')
X_test_final = test_set[numcols + catcols]
y_test_proba = logit_pipe.predict_proba(X_test_final)[:, 1]
y_test_pred  = (y_test_proba >= 0.5).astype(int)   


#extract submission.csv
submission = pd.DataFrame({
    "id": test_set["id"],
    "y": y_test_proba
})
submission.to_csv("submission.csv", index=False)

