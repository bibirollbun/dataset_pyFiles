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


#Read data file
train =pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test =pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
sample =pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')

#show missing in test & train data
print(f'missing value in train data: {train.isnull().sum()}')
print(f'missing value in test data: {test.isnull().sum()}')

#check about duplicated data
print(f'duplicated data in training data = {train.duplicated().sum()}')
print(f'duplicated data in tset data = {test.duplicated().sum()}')

#shape about train and test data 
print(f'tain shape = {train.shape}')
print(f'test shape = {test.shape}')


#show NULL value in data
data_null = round(train.isna().sum() / train.shape[0] * 100, 2)
data_null.to_frame(name = 'percentÂ NULLÂ train dataÂ (%)')


train.head()


train.describe()


train.info()


# Importing important libraries for analysis and EDA
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Set the aesthetic style of the plots
sns.set(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (10, 6)


# Display the first few rows of the dataset
print(train.head())

# Check for missing values
missing_values = train.isnull().sum()
print("Missing Values:\n", missing_values[missing_values > 0])

# Visualize the distribution of the target variable
plt.figure(figsize=(6, 4))
sns.countplot(x="y", data=train, palette=["#3498db", "#e74c3c"])
plt.title("Target Variable Distribution (y)", fontsize=14)
plt.xlabel("y (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

# Show Outliers using Boxplots
num_cols = train.select_dtypes(include=['int64', 'float64']).columns

for col in num_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=train[col], color="#9b59b6")
    plt.title(f"Outlier Detection - {col}", fontsize=14)
    plt.xlabel(col)
    plt.show()

# Visualize numerical data distributions
for col in num_cols:
    if col != "y":
        plt.figure(figsize=(6, 4))
        sns.histplot(data=train, x=col, hue="y", kde=True, stat="density", common_norm=False, palette=["#3498db", "#e74c3c"])
        plt.title(f"Distribution of {col} by Target Variable (y)", fontsize=14)
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.legend(title='y', labels=['No', 'Yes'])
        plt.show()

# Visualize categorical data distributions
cat_cols = train.select_dtypes(include=['object']).columns

for col in cat_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=col, hue="y", data=train, palette=["#3498db", "#e74c3c"])
    plt.title(f"Count of {col} by Target Variable (y)", fontsize=14)
    plt.xticks(rotation=45)
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.legend(title='y', labels=['No', 'Yes'])
    plt.show()

# Correlation Heatmap
corr = train[num_cols].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap", fontsize=14)
plt.show()





# Identify categorical and numerical features
#num_cols = list(train.select_dtypes(exclude=['object']).columns.difference(['y']))
#cat_cols = list(train.select_dtypes(include=['object']).columns)


#Encoding the category data
#from sklearn.preprocessing import LabelEncoder
#le = LabelEncoder()
#for feature in cat_cols:
 #   train[feature] = le.fit_transform(train[feature])
 #  test[feature] = le.transform(test[feature])


#normalization outliers 
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#train[num_cols] = scaler.fit_transform(train[num_cols])
#est[num_cols] = scaler.transform(test[num_cols])


#X = train.drop('y',axis=1)
#y = train['y']


# i will use xgboost & lightgbm
#from lightgbm import LGBMClassifier 
#from xgboost import XGBClassifier
# importing accuraty measures 
#from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,f1_score,recall_score,roc_auc_score
#from sklearn.model_selection import cross_val_score,KFold,StratifiedKFold
#from sklearn.model_selection import train_test_split



#X.shape


#y.shape


#create x ,y (train , test)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#train LGB Model
#lgb_model = LGBMClassifier(random_state=42)
#lgb_model.fit(X_train, y_train)

#y_pred_lgb = lgb_model.predict(X_test)
#lgb_acc = accuracy_score(y_test, y_pred_lgb)



# train XGBoost model 
#xgb_model = XGBClassifier(eval_metric="logloss", random_state=42, use_label_encoder=False)
#xgb_model.fit(X_train, y_train)
#y_pred_xgb = xgb_model.predict(X_test)
#xgb_acc = accuracy_score(y_test, y_pred_xgb)


#from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#print("ğŸ“Š LightGBM Accuracy:", lgb_acc)
#print("ğŸ“Š XGBoost Accuracy:", xgb_acc)

#print("\nLightGBM Classification Report:\n", classification_report(y_test, y_pred_lgb))
#print("\nXGBoost Classification Report:\n", classification_report(y_test, y_pred_xgb))


#lgb_importance = pd.DataFrame({
#    'feature': X.columns,
#    'importance': lgb_model.feature_importances_
#}).sort_values(by="importance", ascending=False)

#plt.figure(figsize=(8,6))
#sns.barplot(data=lgb_importance.head(10), x="importance", y="feature", palette="viridis")
#plt.title("Top 10 Important Features - LightGBM")
#plt.show()



#xgb_importance = pd.DataFrame({
#    'feature': X.columns,
#    'importance': xgb_model.feature_importances_
#}).sort_values(by="importance", ascending=False)

#plt.figure(figsize=(8,6))
#sns.barplot(data=xgb_importance.head(10), x="importance", y="feature", palette="magma")
#plt.title("Top 10 Important Features - XGBoost")
#plt.show()


#best_model = "LightGBM" if lgb_acc > xgb_acc else "XGBoost"
#best_pred = y_pred_lgb if best_model == "LightGBM" else y_pred_xgb

#cm = confusion_matrix(y_test, best_pred)
#sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#plt.title(f"Confusion Matrix - {best_model}")
#plt.xlabel("Predicted")
#plt.ylabel("Actual")
#plt.show()


#sample.head()


#from sklearn.model_selection import StratifiedKFold

#cat_cols = train.select_dtypes('object').columns.to_list()
#for col in cat_cols:
#    le = LabelEncoder()
#    le.fit(pd.concat([train[col], test[col]], axis=0).astype(str))
#    train[col] = le.transform(train[col].astype(str))
#    test[col] = le.transform(test[col].astype(str))

# Features / target
#X = train.drop('y', axis=1)   
#X_test = test.copy()        

# Stratified KFold
#kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#oof_preds = np.zeros(len(X))
#test_preds = np.zeros(len(X_test))

#for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
 #   print(f"== Fold {fold+1} ==")
 #   X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
 #   y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
 #   model = XGBClassifier(
 #       n_estimators=1000,
 #       learning_rate=0.05,
 #       max_depth=6,
 #       colsample_bytree=0.8,
 #       subsample=0.8,
 #       random_state=42,
 #       use_label_encoder=False,   
 #       eval_metric='auc'
 #   )
    
 #   model.fit(
 #       X_train, y_train,
 #       eval_set=[(X_val, y_val)],
 #      early_stopping_rounds=10,   
 #       verbose=50                  #
 #   )
    
 #   oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
 #   test_preds += model.predict_proba(X_test)[:, 1] / kf.n_splits

# Local validation AUC
#auc_score = roc_auc_score(y, oof_preds)
#print(f"CV ROC AUC: {auc_score:.4f}")
#submission
#submission = pd.DataFrame({
#    "id": test["id"],        
#    "y": test_preds          
#})

#submission.to_csv("submission_xgb.csv", index=False)
#print("âœ… Saved submission_xgb.csv with", len(submission), "rows")


#submission = pd.DataFrame({
 #   "id": range(750000, 750000 + len(test_preds)),
  #  "y": test_preds
#})


#submission.to_csv("submission_3.csv", index=False)
#print("âœ… Saved submission.csv")


#print(submission.head())


from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import pandas as pd
import numpy as np

#encoding for object & text data
cat_cols = train.select_dtypes('object').columns.to_list()
for col in cat_cols:
    le = LabelEncoder()
    le.fit(pd.concat([train[col], test[col]], axis=0).astype(str))
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# Features / Target
X = train.drop('y', axis=1)
y = train['y']
X_test = test.copy()

# Stratified KFold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"== Fold {fold+1} ==")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = XGBClassifier(
     n_estimators=2000,
     learning_rate=0.02,          #Less than before so we can benefit from the 2000 trees
     max_depth=8,                # A little deeper with grow_policy
     min_child_weight=2,
     colsample_bytree=0.9,
     subsample=0.9,
     gamma=1.5,
     reg_alpha=0.2,
     reg_lambda=2,
     scale_pos_weight=1,          #If there is an imbalance, change it = (negative number / positive number)
     tree_method='hist',          # Faster and better in handling large data
     max_bin=256,                 #More bins = higher resolution
     grow_policy='lossguide',     # Focus on important divisions
     random_state=42,
     use_label_encoder=False,
     eval_metric='auc'
     )

    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=100
    )
    
    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds += model.predict_proba(X_test)[:, 1] / kf.n_splits

# AUC on Cross Validation
auc_score = roc_auc_score(y, oof_preds)
print(f"CV ROC AUC: {auc_score:.4f}")



submission = pd.DataFrame({
     "id": range(750000, 750000 + len(test_preds)),
     "y": test_preds
})


submission.to_csv("submission_4.csv", index=False)
print("âœ… Saved submission.csv")


print(submission.head())




