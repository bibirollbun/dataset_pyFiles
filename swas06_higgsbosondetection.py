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


df_train = pd.read_csv("/kaggle/input/higgs-boson-detection-2025/train.csv")
df_test = pd.read_csv("/kaggle/input/higgs-boson-detection-2025/test.csv")


df_train.head(3)


df_train.info()


df_train.shape,df_test.shape


df_train.isnull().sum()


X=df_train.drop("label",axis=1)
y=df_train['label']



from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()

# Fit and transform the data (scale the features)
X_scaled = scaler.fit_transform(X)

# If you want to check the scaled data:
print(X_scaled)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)


import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score,recall_score,roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
from matplotlib import pyplot as plt
import seaborn as sns



classifiers = {
    "Logistic Regression": LogisticRegression(class_weight='balanced'),
    "Random Forest": RandomForestClassifier(class_weight='balanced'),
    "XGBoost": xgb.XGBClassifier(scale_pos_weight=5)
}

# Iterate through classifiers and apply recall threshold adjustment
for name, model in classifiers.items():
    # Train the model
    model.fit(X_train, y_train)

    # Get predicted probabilities for the positive class (class 1)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Set a new threshold to increase recall (lower than 0.5)
    threshold = 0.3  # Adjust this threshold value to see how recall changes
    y_pred_new = (y_prob >= threshold).astype(int)

    # Evaluate recall with the new threshold
    recall = recall_score(y_test, y_pred_new)
    print(f"{name} - Recall at threshold {threshold}: {recall}")

    # Optionally, plot the ROC curve to visualize the trade-off between recall and precision
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{name} ROC Curve')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title(f'ROC Curve - {name}')
    plt.legend()
    plt.show()


from sklearn.metrics import accuracy_score
model =  xgb.XGBClassifier(scale_pos_weight=5)

model.fit(X_train, y_train)
# Test on the test set
best_preds = model.predict(X_test)
final_accuracy = recall_score(y_test, best_preds)

print("Final Test Accuracy:", final_accuracy)


df_test_scaled = scaler.fit_transform(df_test)


y_test_pred = model.predict_proba(df_test_scaled)[:, 1]


submission_df = pd.read_csv('/kaggle/input/higgs-boson-detection-2025/sample_submission.csv')


submission_df["Predicted"] = y_test_pred
submission_df['Id'] = submission_df['Id'].astype(np.int64)
submission_df['Id'] = submission_df['Id'].apply(lambda x: f"{float(x):.18e}")
submission_df.to_csv("submission.csv", index=False)
submission_df.head()

