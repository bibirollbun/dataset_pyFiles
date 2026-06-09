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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix,classification_report
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


samplesubmision=pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
train=pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')


samplesubmision.head()


train.head()


train.info()


test.head()


test.nunique()


test.info()


train.drop_duplicates(inplace=True)


train.isnull().sum()


train.describe().T


train['age'].max()


train['age'].min()


train['job'].value_counts()


sns.countplot(data=train, x='y', hue='job')


train['marital'].value_counts()


train['education'].value_counts()


train.info()


# distribution of y variable
train.y.value_counts()*100/len(train)


#Visualitazion histogram
fig,ax = plt.subplots(4,2, figsize=(20,20))
sns.histplot(train.age,bins=20 ,ax=ax[0,0],color='red',kde=True)
sns.histplot(train.balance,bins=20,ax=ax[0,1],color='red',kde=True)
sns.histplot(train.day,bins=20 ,ax=ax[1,0],color='red',kde=True)
sns.histplot(train.duration,bins=20,ax=ax[1,1],color='red',kde=True)
sns.histplot(train.campaign,bins=20 ,ax=ax[2,0],color='red',kde=True)
sns.histplot(train.pdays,bins=20,ax=ax[2,1],color='red',kde=True)
sns.histplot(train.previous,bins=20 ,ax=ax[3,0],color='red',kde=True)
sns.histplot(train.y,bins=20,ax=ax[3,1],color='red',kde=True)

plt.tight_layout()
plt.show()


train.groupby("y").agg({'balance':'mean'})


train.groupby('y').agg({'balance':'max'})


train.groupby('y').agg({'campaign':'mean'})


train.groupby('y').agg({'campaign':'max'})


train.head()


train.groupby('y')['housing'].value_counts()


train.groupby('y')['loan'].value_counts()


train.groupby('y')['poutcome'].value_counts()


sns.countplot(data=train, x='y', hue='poutcome')


sns.histplot(train.pdays,color='red')


sns.scatterplot(x='age', y='balance', data=train)
plt.title("Korelation age vs balance")
plt.show()


corr = train[['age','balance','day','duration','campaign','pdays','previous','y']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


# Data Preprocessing
train_ids = train['id'].copy()
test_ids = test['id'].copy()


# Remove ID columns
train_clean = train.drop('id', axis=1)
test_clean = test.drop('id', axis=1)


train_clean.head()


# Prepare features and target
X = train_clean.drop('y', axis=1)
y = train_clean['y']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def feature_engineering(df, fit_params=None):
    df_processed = df.copy()
    
    # Winsorizing outliers
    if fit_params is None:
        # Fit on training data
        balance_low = df_processed['balance'].quantile(0.01)
        balance_high = df_processed['balance'].quantile(0.99)
        campaign_low = df_processed['campaign'].quantile(0.01)
        campaign_high = df_processed['campaign'].quantile(0.99)
        fit_params = {
            'balance_low': balance_low, 'balance_high': balance_high,
            'campaign_low': campaign_low, 'campaign_high': campaign_high
        }
    
    # Apply winsorizing
    df_processed['balance'] = np.clip(df_processed['balance'], 
                                    fit_params['balance_low'], fit_params['balance_high'])
    df_processed['campaign'] = np.clip(df_processed['campaign'], 
                                     fit_params['campaign_low'], fit_params['campaign_high'])
    
    # Duration scaling
    scaler = StandardScaler()
    if 'duration_scaler' not in fit_params:
        df_processed['duration_scaled'] = scaler.fit_transform(df_processed[['duration']])
        fit_params['duration_scaler'] = scaler
    else:
        df_processed['duration_scaled'] = fit_params['duration_scaler'].transform(df_processed[['duration']])
    
    # One-hot encoding
    categorical_cols = df_processed.select_dtypes(include='object').columns
    df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
    
    return df_processed, fit_params


# Apply feature engineering
print("Applying feature engineering...")
X_train_processed, fit_params = feature_engineering(X_train)
X_test_processed, _ = feature_engineering(X_test, fit_params)


train.info()


X_test_processed.info()


X_train_processed.info()


# Choose classification models
lr=LogisticRegression()
rf=RandomForestClassifier()
gb=GradientBoostingClassifier()
dt = DecisionTreeClassifier()
xgb_model = xgb.XGBClassifier()
lgb_model = lgb.LGBMClassifier()
cat_model = CatBoostClassifier()


lr.fit(X_train_processed, y_train)


rf.fit(X_train_processed,y_train)


gb.fit(X_train_processed, y_train)


dt.fit(X_train_processed,y_train)


xgb_model.fit(X_train_processed,y_train)


lgb_model.fit(X_train_processed,y_train)


cat_model.fit(X_train_processed,y_train)


print('Accuracy Data Train',accuracy_score(y_train, lr.predict(X_train_processed)))
print('Accuracy Data Test',accuracy_score(y_test, lr.predict(X_test_processed)))


print('Accuracy Data Train',accuracy_score(y_train, rf.predict(X_train_processed)))
print('Accuracy Data Test',accuracy_score(y_test, rf.predict(X_test_processed)))


print('Accuracy Data Train',accuracy_score(y_train, gb.predict(X_train_processed)))
print('Accuracy Data Test',accuracy_score(y_test, gb.predict(X_test_processed)))


print('Accuracy Data Train',accuracy_score(y_train, dt.predict(X_train_processed)))
print('Accuracy Data Test',accuracy_score(y_test, dt.predict(X_test_processed)))


print('Accuracy Data Train',accuracy_score(y_train, xgb_model.predict(X_train_processed)))
print('Accuracy Data Test',accuracy_score(y_test, xgb_model.predict(X_test_processed)))


print('Accuracy Data Train',accuracy_score(y_train, lgb_model.predict(X_train_processed)))
print('Accuracy Data Test',accuracy_score(y_test, lgb_model.predict(X_test_processed)))


print('Accuracy Data Train',accuracy_score(y_train, cat_model.predict(X_train_processed)))
print('Accuracy Data Test',accuracy_score(y_test, cat_model.predict(X_test_processed)))


# linear regresion
y_pred_prob_lr = lr.predict_proba(X_test_processed)[:, 1]
roc_auc_lr = roc_auc_score(y_test, y_pred_prob_lr)
print("ROC AUC Linear Regresion:", roc_auc_lr)

# Random Forest
y_pred_prob_rf = rf.predict_proba(X_test_processed)[:, 1]
roc_auc_rf = roc_auc_score(y_test, y_pred_prob_rf)
print("ROC AUC Random Forest:", roc_auc_rf)

# Gradient Boosting
y_pred_prob_gb = gb.predict_proba(X_test_processed)[:, 1]
roc_auc_gb = roc_auc_score(y_test, y_pred_prob_gb)
print("ROC AUC Gradient Boosting:", roc_auc_gb)

# Decision Tree
y_pred_prob_dt = dt.predict_proba(X_test_processed)[:, 1]
roc_auc_dt = roc_auc_score(y_test, y_pred_prob_dt)
print("ROC AUC Decision Tree:", roc_auc_dt)

# Xgb Model
y_pred_prob_xgb = xgb_model.predict_proba(X_test_processed)[:, 1]
roc_auc_xgb = roc_auc_score(y_test, y_pred_prob_xgb)
print("ROC AUC Xgb Model:", roc_auc_xgb)

# Lgb Model
y_pred_prob_lgb = lgb_model.predict_proba(X_test_processed)[:, 1]
roc_auc_lgb = roc_auc_score(y_test, y_pred_prob_lgb)
print("ROC AUC Lgb Model:", roc_auc_lgb)

# Cat Model
y_pred_prob_cat = cat_model.predict_proba(X_test_processed)[:, 1]
roc_auc_cat = roc_auc_score(y_test, y_pred_prob_cat)
print("ROC AUC Cat Model:", roc_auc_cat)


# Prepare Test Data and Make Predictions
print("\nProcessing test data and making predictions...")
test_processed, _ = feature_engineering(test_clean, fit_params)


# Align test features with training features
missing_cols = set(X_train_processed.columns) - set(test_processed.columns)
for col in missing_cols:
    test_processed[col] = 0


# Reorder columns to match training data
test_processed = test_processed[X_train_processed.columns]

# Make predictions
test_predictions = cat_model.predict_proba(test_processed)[:, 1]

# Create Submission File
submission = pd.DataFrame({
    'id': test_ids,
    'y': test_predictions
})


# Display prediction statistics
print(f"\nPrediction Statistics:")
print(f"Min probability: {test_predictions.min():.4f}")
print(f"Max probability: {test_predictions.max():.4f}")
print(f"Mean probability: {test_predictions.mean():.4f}")
print(f"Predictions > 0.5: {(test_predictions > 0.5).sum()}")


sns.set(style='whitegrid')

plt.figure(figsize=(8, 5))
sns.histplot(test_predictions, bins=20, kde=True, color='skyblue')
plt.title('Distribusi Probabilitas Kelas Positif (1)')
plt.xlabel('Probabilitas')
plt.ylabel('Frekuensi')
plt.show()


# Save submission
submission.to_csv('submission.csv', index=False)
print(f"\nSubmission file created!")
print(f"Shape: {submission.shape}")
print(f"Sample predictions:")
print(submission.head())

