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
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder



train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)


train.head()


test.head()


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
sns.countplot(data=train, x="loan_paid_back", palette="viridis")
plt.title("Loan Payback Distribution")
plt.show()



train.hist(figsize=(15, 12), bins=30)
plt.suptitle("Numerical Feature Distributions", y=1.02)
plt.show()



plt.figure(figsize=(6,4))
sns.boxplot(data=train, x="loan_paid_back", y="loan_amount")
plt.title("Feature vs Loan Payback")
plt.show()



# Target
target = "loan_paid_back"

# Separate X and y
X = train.drop(columns=[target])
y = train[target]


# Identify numeric and categorical columns
num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = X.select_dtypes(include=['object']).columns.tolist()

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


# Numeric pipeline
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])


# Categorical pipeline
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])


# Combined preprocessor (THIS IS WHAT WAS MISSING)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ]
)


# Train-validation split
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "LightGBM": LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        random_state=42
    ),
}



results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    
    preds = pipe.predict_proba(X_val)[:,1]
    auc = roc_auc_score(y_val, preds)
    ll = log_loss(y_val, preds)
    
    results[name] = (auc, ll)
    print(f"{name} -> AUC: {auc:.4f} | LogLoss: {ll:.4f}")



best_model_name = max(results, key=lambda x: results[x][0])
best_model = models[best_model_name]

print("\nBest model based on AUC:", best_model_name)

final_pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", best_model)])
final_pipe.fit(X, y)


# Make predictions on test data (probability of class = 1)
test_pred = final_pipe.predict_proba(test)[:, 1]

# Create submission file
submission = pd.DataFrame({
    "id": test["id"],
    "loan_paid_back": test_pred
})

# Save to CSV
submission.to_csv("output.csv", index=False)

print("\nSubmission file saved as output.csv")

# Show first few rows
submission.head()



from sklearn.metrics import accuracy_score

val_pred_labels = final_pipe.predict(X_val)
accuracy = accuracy_score(y_val, val_pred_labels)
print("Accuracy:", accuracy)



from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_val, val_pred_labels)
print(cm)



import matplotlib.pyplot as plt

plt.hist(val_pred_prob, bins=50)
plt.title("Prediction Probability Distribution")
plt.xlabel("Probability")
plt.ylabel("Count")
plt.show()


