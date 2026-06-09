import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, warnings
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score
import lightgbm as lgb
import catboost as cb


train=pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test=pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
submission=pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")


train.head()


train.isnull().sum()


train.describe()


train.drop(['id'],inplace=True,axis=1)



# Set up subplots for histograms
fig, axes = plt.subplots(nrows=len(train.columns)//3 + 1, ncols=3, figsize=(15, len(train.columns)*1.5))
axes = axes.flatten()

for idx, feature in enumerate(train.columns):
    sns.histplot(train[feature], bins=30, kde=True, ax=axes[idx])
    axes[idx].set_title(f"Distribution of {feature}")
# Hide any unused subplots
for idx in range(len(train.columns) - 1, len(axes)):  # -1 because we excluded 'dataset'
    fig.delaxes(axes[idx])
plt.tight_layout()
plt.show()


train['dataset'] = 'Train'
test['dataset'] = 'Test'

# Concatenating train and test sets for comparison
combined_df = pd.concat([train, test])

# Determine number of rows for subplots
rows = int(np.ceil((len(train.columns) - 1) / 3))  # -1 to exclude 'dataset' column
cols = 3

# Creating figure and setting size
fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5))
axes = axes.flatten()

# Loop through features and create subplots
for idx, feature in enumerate(train.columns.drop('dataset','rainfall')): 
    sns.boxplot(x=feature, y='dataset', data=combined_df, ax=axes[idx])
    axes[idx].set_title(feature)
    axes[idx].set_xlabel(feature)  
    axes[idx].set_ylabel('dataset')

# Hide any unused subplots
for idx in range(len(train.columns) - 1, len(axes)):  # -1 because we excluded 'dataset'
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.show()
train.drop(['dataset'],inplace=True,axis=1)
test.drop(['dataset'],inplace=True,axis=1)


train['rainfall'].value_counts()


sns.countplot(x='rainfall', data=train)
plt.xlabel("Rainfall")
plt.title("Count of Rainfall")



# Calculate the correlation matrix
corr_matrix = train.corr()

# Visualize the correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


def new_features(df):
    df['Temp_Range'] = df['maxtemp'] - df['mintemp']  # Temperature range
    df = df.drop(columns=['maxtemp', 'mintemp'],inplace=True,axis=1)  



new_features(train)
train.head()


new_features(test)


plt.figure(figsize=(10, 6))
sns.heatmap(train.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap After Feature Engineering")
plt.show()


test.isnull().sum()


test['winddirection']=test['winddirection'].fillna(test['winddirection'].median())
test.drop(['id'],inplace=True,axis=1)


X=train.drop(['rainfall'],axis=1)
Y=train['rainfall']


from imblearn.over_sampling import SMOTE
from collections import Counter


counter=Counter(Y)
print('Before oversampling y {}'.format(counter))
smt=SMOTE()
X_resampled,Y_resampled=smt.fit_resample(X,Y)


counter2=Counter(Y_resampled)
print('After oversampling y {}'.format(counter2))


X_resampled.shape


X_train,X_test,y_train,y_test=train_test_split(X_resampled,Y_resampled,test_size=0.2,random_state=42)


def model_trying(model):
    model.fit(X_train,y_train)
    pred=model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred)
    report = classification_report(y_test, pred)
    print("Accuracy :",acc)
    print("Confusion matrix:",cm)
    print("Classification report",report)
    y_pred_proba = model.predict_proba(X_test)  # Returns probabilities for both classes

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])  # Use probabilities for the positive class (rainy day)

    # Compute AUC score
    auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
    print(f"AUC-ROC Score: {auc_score:.4f}")

    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'AUC = {auc_score:.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonal line (random model)
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()



model_trying(LogisticRegression())


model_trying(RandomForestClassifier(n_estimators=100, random_state=42))


model_trying(XGBClassifier())


model_trying(cb.CatBoostClassifier(iterations=500, learning_rate=0.05, depth=7, random_seed=42, verbose=100))


model=cb.CatBoostClassifier(iterations=500, learning_rate=0.05, depth=7, random_seed=42, verbose=100)
model.fit(X_train,y_train)
pred=model.predict_proba(test)


submission['rainfall']=pred[:,1]


submission.to_csv("submission.csv",index=False)
print("Submission file is ready ")

