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


dfo=pd.read_csv("/kaggle/input/playground-series-s3e10/train.csv")


from sklearn.model_selection import train_test_split

df, df_test= train_test_split(
    dfo, 
    test_size=0.2,  
    random_state=42 
)


df.size


df.shape


df.head()


df.info()


df.isnull().sum()


df.describe().T


df.drop('id',axis=1,inplace=True)


for i in df.columns:
    print(i,'---',df[i].skew())


import seaborn as sns
import matplotlib.pyplot as plt


for col in df.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col], kde=True, bins=30, color="skyblue")
        plt.title(f'Histogram of {col}', fontsize=12)
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()





for col in df.columns:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df[col], color="lightgreen")
        plt.title(f'Boxplot of {col}', fontsize=12)
        plt.xlabel(col)
        plt.tight_layout()
        plt.show()
    




import itertools


num_cols = df.select_dtypes(include=['float64']).columns


for col1, col2 in itertools.combinations(num_cols, 2):
    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=df, x=col1, y=col2, alpha=0.6)
    plt.title(f'Scatter Plot: {col1} vs {col2}', fontsize=12)
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.tight_layout()
    plt.show()



df_iqr=df.copy()



df_iqr.info()


import numpy as np

def iqr_trim_and_count(df, col, factor=1.5):
    """
    Removes rows where column values are outside the IQR bounds.
    """
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lb = Q1 - factor * IQR
    ub = Q3 + factor * IQR
    
    # Count outliers
    below_count = (df[col] < lb).sum()
    above_count = (df[col] > ub).sum()
    
    print(f"Column: {col}")
    print(f"  Q1 = {Q1:.2f}, Q3 = {Q3:.2f}, IQR = {IQR:.2f}")
    print(f"  Lower Bound = {lb:.2f}, Upper Bound = {ub:.2f}")
    print(f"  Values below LB: {below_count}, Values above UB: {above_count}")
    
    # Trim the outliers
    df_trimmed = df[(df[col] >= lb) & (df[col] <= ub)].copy()
    
    return df_trimmed, below_count, above_count

# Apply to all numeric columns
numeric_cols = df_iqr.select_dtypes(include=['float64']).columns
df_trimmed = df_iqr.copy()

for col in numeric_cols:
    df_trimmed, below, above = iqr_trim_and_count(df_trimmed, col)



import numpy as np

def iqr_cap_and_count(df, col, factor=1.5):

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lb = Q1 - factor * IQR
    ub = Q3 + factor * IQR
    
    # Count outliers
    below_count = (df[col] < lb).sum()
    above_count = (df[col] > ub).sum()
    
    print(f"Column: {col}")
    print(f"  Q1 = {Q1:.2f}, Q3 = {Q3:.2f}, IQR = {IQR:.2f}")
    print(f"  Lower Bound = {lb:.2f}, Upper Bound = {ub:.2f}")
    print(f"  Values below LB: {below_count}, Values above UB: {above_count}")
    
    # Cap the outliers
    df[col] = np.where(df[col] < lb, lb, df[col])
    df[col] = np.where(df[col] > ub, ub, df[col])
    
    return df[col], below_count, above_count



numeric_cols = df_iqr.select_dtypes(include=['float64']).columns
for col in numeric_cols:
    df_iqr[col], below, above = iqr_cap_and_count(df_iqr, col)


for col in df_iqr.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(df_iqr[col], kde=True, bins=30, color="skyblue")
        plt.title(f'Histogram of {col}', fontsize=12)
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()


for col in df_iqr.columns:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df_iqr[col], color="lightgreen")
        plt.title(f'Boxplot of {col}', fontsize=12)
        plt.xlabel(col)
        plt.tight_layout()
        plt.show()
    




import itertools


num_cols = df_iqr.select_dtypes(include=['float64']).columns


for col1, col2 in itertools.combinations(num_cols, 2):
    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=df_iqr, x=col1, y=col2, alpha=0.6)
    plt.title(f'Scatter Plot: {col1} vs {col2}', fontsize=12)
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.tight_layout()
    plt.show()



df['Class'].value_counts()


df_iqr['Class'].value_counts()


df_iqr.shape


pip install imbalanced-learn xgboost



from imblearn.over_sampling import SMOTE

X = df_iqr.drop(columns=['Class'])  
y = df_iqr['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)




smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)



X_train_smote, y_train_smote, X_test, y_test


from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix



rf = RandomForestClassifier(n_estimators=10, random_state=42)
rf.fit(X_train_smote, y_train_smote)




y_pred_rf = rf.predict(X_test)



print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))



cm = confusion_matrix(y_test, y_pred_rf)
cm



xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb.fit(X_train_smote, y_train_smote)



y_pred_xgb = xgb.predict(X_test)



print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))



cm = confusion_matrix(y_test, y_pred_xgb)
cm



scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)




logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train_scaled, y_train_smote)




y_pred_log = logreg.predict(X_test_scaled)




print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))



cm = confusion_matrix(y_test, y_pred_log)
cm


df_test


X_df = df_test.drop(columns=['Class','id'])
y_df = df_test['Class']


X_df_scaled = scaler.transform(X_df)


y_pred_rf = rf.predict(X_df)
y_pred_xgb = xgb.predict(X_df)
y_pred_log = logreg.predict(X_df_scaled)



print("Random Forest Accuracy:", accuracy_score(y_df, y_pred_rf))
print(classification_report(y_df, y_pred_rf))



cm = confusion_matrix(y_df, y_pred_rf)
cm



print("XGBoost Accuracy:", accuracy_score(y_df, y_pred_xgb))
print(classification_report(y_df, y_pred_xgb))



cm = confusion_matrix(y_df, y_pred_xgb)
cm



print("Logistic Regression Accuracy:", accuracy_score(y_df, y_pred_log))
print(classification_report(y_df, y_pred_log))



cm = confusion_matrix(y_df, y_pred_log)
cm


from sklearn.model_selection import GridSearchCV


rf = RandomForestClassifier(random_state=42)
param_grid_rf = {
    'n_estimators': [15, 20, 25],
    'max_depth': [None,6,7],
    'min_samples_split': [2,3],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['log2', 'sqrt']
}

grid_rf = GridSearchCV(
    estimator=rf,
    param_grid=param_grid_rf,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)





grid_rf.fit(X_train_smote, y_train_smote)

print("Best RF Parameters:", grid_rf.best_params_)




y_pred_rf = grid_rf.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))



y_pred_rf = grid_rf.predict(X_df)

print("Random Forest Accuracy:", accuracy_score(y_df, y_pred_rf))
print(classification_report(y_df, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_df, y_pred_rf))


best_params = grid_rf.best_params_



rf_new = RandomForestClassifier(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    max_features=best_params['max_features'],
    random_state=42
)

rf_new.fit(X_train_smote, y_train_smote)




y_pred_rf_new = rf_new.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_rf_new))
print(classification_report(y_test, y_pred_rf_new))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf_new))



y_pred_rf = rf_new.predict(X_df)

print("Random Forest Accuracy:", accuracy_score(y_df, y_pred_rf))
print(classification_report(y_df, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_df, y_pred_rf))


X_train_smote.columns


import joblib
model_dict = {
    'model': rf_new ,
    'columns': X.columns.tolist()
}

joblib.dump(model_dict, 'model.pkl')




