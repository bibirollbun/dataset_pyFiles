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


import warnings
warnings.filterwarnings("ignore")
df_train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
df_test =pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


df_train['temp_range'] = df_train['maxtemp'] - df_train['mintemp']
df_test['temp_range'] = df_test['maxtemp'] - df_test['mintemp']


 df_train['temp_from_dewpoint'] = df_train['temparature'] - df_train['dewpoint']
 df_test['temp_from_dewpoint'] = df_test['temparature'] - df_test['dewpoint']


df_train.head(3)


df_test.head(3)


df_train = df_train.drop(['mintemp','maxtemp','dewpoint'],axis="columns")
df_test = df_test.drop(['mintemp','maxtemp','dewpoint'],axis="columns")


df_train.info(),df_test.info()


df_train.isnull().sum(),df_test.isnull().sum()


df_test['winddirection'] = df_test['winddirection'].fillna(df_test['winddirection'].mean())


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


X=df_train.drop('rainfall',axis=1)
y=df_train['rainfall']




# Initialize StandardScaler
scaler = StandardScaler()

# Fit and transform the data (scale the features)
X_scaled = scaler.fit_transform(X)

# If you want to check the scaled data:
print(X_scaled)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# Define the classifiers with parameters
classifiers = {
    "Logistic Regression": LogisticRegression(class_weight='balanced', C=1.0, penalty='l2', solver='liblinear', max_iter=100),
    "Random Forest": RandomForestClassifier(class_weight='balanced', n_estimators=100, max_depth=10, min_samples_split=2),
    "Support Vector Machine": SVC(class_weight='balanced', C=1.0, kernel='rbf', gamma='scale', probability=True),
    "Decision Tree": DecisionTreeClassifier(class_weight='balanced', criterion='gini', max_depth=5, min_samples_split=2),
    "XGBoost": XGBClassifier(class_weight='balanced', learning_rate=0.1, n_estimators=100, max_depth=5)
}




kfold = StratifiedKFold(10, shuffle=True, random_state=0)

# Initialize lists to store results
auc_scores = {model_name: [] for model_name in classifiers.keys()}
fpr_dict = {model_name: [] for model_name in classifiers.keys()}
tpr_dict = {model_name: [] for model_name in classifiers.keys()}

# Perform cross-validation
for fold, (train_idx, val_idx) in enumerate(kfold.split(X_scaled, y)):
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    print(f"\nFold {fold+1}:")
    
    # Train models
    for model_name, model in classifiers.items():
        model.fit(X_train, y_train)

        # Predict probabilities
        y_pred_proba = model.predict_proba(X_val)[:, 1]

        # Calculate AUC
        auc = roc_auc_score(y_val, y_pred_proba)
        auc_scores[model_name].append(auc)
        
        print(f"Model: {model_name}, AUC: {auc}")

        # Calculate fpr and tpr for plotting
        fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
        fpr_dict[model_name].append(fpr)
        tpr_dict[model_name].append(tpr)

# Calculate average AUC for each model
for model_name, scores in auc_scores.items():
    avg_auc = np.mean(scores)
    print(f"\nModel: {model_name}, Average AUC: {avg_auc}")


cmap = plt.get_cmap('BrBG')
colors = [cmap(0.1), cmap(0.3), cmap(0.7), cmap(0.9), cmap(0.6)]

plt.figure(figsize=(10, 8))
for i, model_name in enumerate(classifiers.keys()):
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.zeros_like(mean_fpr)
   
    for fpr, tpr in zip(fpr_dict[model_name], tpr_dict[model_name]):
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        mean_tpr += interp_tpr
    mean_tpr /= len(fpr_dict[model_name])
  
    plt.plot(mean_fpr, mean_tpr, label=f"{model_name} (AUC = {np.mean(auc_scores[model_name]):.2f})", color=colors[i])

plt.plot([0, 1], [0, 1], linestyle="--", color="r", label="Chance")
plt.title('ROC Curves for K-Fold Cross Validation')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(color='gray', linestyle='--', linewidth=0.7)
plt.show()


best_model_name = max(auc_scores, key=lambda x: np.mean(auc_scores[x]))
print(f"Best Model: {best_model_name}")
best_model = classifiers[best_model_name]


best_model.fit(X_scaled, y)


df_test.head(3)


df_test_scaled = scaler.fit_transform(df_test)


df_test_scaled


test_pred_proba = best_model.predict_proba(df_test_scaled)[:, 1]


submission_df = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
submission_df = pd.DataFrame({'id': df_test.id, 'rainfall': test_pred_proba})

submission_df.to_csv('submission.csv', index=False)
submission_df.head(10)



import seaborn as sns
plt.figure(figsize=(12, 4))
sns.histplot(test_pred_proba, kde=True, bins=20, color=colors[3])
plt.title('Distribution of Predicted Probabilities on Test Set')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.grid(color='gray', linestyle='--', linewidth=0.7)
plt.show()

