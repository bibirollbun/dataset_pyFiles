#importing important libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler

import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
df.head()


df_test =pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
df_test.head()


df.shape


df.info()


df_test.info()


df.describe().T


df.isnull().sum()


df_test.isnull().sum()


df.fillna(df.median(), inplace=True)
df_test.fillna(df_test.median(), inplace=True)


df.columns


df.rename(str.strip,
          axis='columns', 
          inplace=True)

df.columns


for col in df.columns:
  
  # Checking if the column contains
  # any null values
  if df[col].isnull().sum() > 0:
    val = df[col].mean()
    df[col] = df[col].fillna(val)
    
df.isnull().sum().sum()


plt.pie(df['rainfall'].value_counts().values,
        labels = df['rainfall'].value_counts().index,
        autopct='%1.1f%%')
plt.show()


df.groupby('rainfall').mean()


features = list(df.select_dtypes(include = np.number).columns)
features.remove('day')
print(features)


plt.subplots(figsize=(15,8))

for i, col in enumerate(features):
  plt.subplot(3,4, i + 1)
  sb.distplot(df[col])
plt.tight_layout()
plt.show()


plt.subplots(figsize=(15,8))

for i, col in enumerate(features):
  plt.subplot(3,4, i + 1)
  sb.boxplot(df[col])
plt.tight_layout()
plt.show()


df.replace({'yes':1, 'no':0}, inplace=True)


plt.figure(figsize=(10,10))
sb.heatmap(df.corr() > 0.8,
           annot=True,
           cbar=False)
plt.show()


df.drop(['maxtemp', 'mintemp'], axis=1, inplace=True)


features = df.drop(['id','day', 'rainfall'], axis=1)
target = df.rainfall


X_train, X_val, \
    Y_train, Y_val = train_test_split(features,
                                      target,
                                      test_size=0.2,
                                      stratify=target,
                                      random_state=2)

# As the data was highly imbalanced we will
# balance it by adding repetitive rows of minority class.
ros = RandomOverSampler(sampling_strategy='minority',
                        random_state=22)
X, Y = ros.fit_resample(X_train, Y_train)


df_test1=df_test.drop(['id','day'], axis=1)


df_test1


df_test1=df_test1.drop(['maxtemp','mintemp'], axis=1)


# Normalizing the features for stable and fast training.
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_val = scaler.transform(X_val)
X_test = scaler.transform(df_test1)


models = [LogisticRegression(), XGBClassifier(), SVC(kernel='rbf', probability=True)]

for i in range(3):
  models[i].fit(X, Y)

  print(f'{models[i]} : ')

  train_preds = models[i].predict_proba(X) 
  print('Training Accuracy : ', metrics.roc_auc_score(Y, train_preds[:,1]))

  val_preds = models[i].predict_proba(X_val) 
  print('Validation Accuracy : ', metrics.roc_auc_score(Y_val, val_preds[:,1]))
  print()


models = [LogisticRegression(), XGBClassifier(), SVC(kernel='rbf', probability=True)]
auc_scores = {}

for model in models:
    model.fit(X, Y)
    train_preds = model.predict_proba(X)[:, 1]
    val_preds = model.predict_proba(X_val)[:, 1]
    
    train_auc = metrics.roc_auc_score(Y, train_preds)
    val_auc = metrics.roc_auc_score(Y_val, val_preds)
    
    print(f'{model} : ')
    print(f'Training AUC: {train_auc}')
    print(f'Validation AUC: {val_auc}\n')
    
    auc_scores[model] = val_auc

# Find the best model
best_model = max(auc_scores, key=auc_scores.get)
print(f'Best Model: {best_model} with AUC: {auc_scores[best_model]}')



import matplotlib.pyplot as plt 
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import metrics

ConfusionMatrixDisplay.from_estimator(models[2], X_val, Y_val)
plt.show()


# Make final predictions on test set
test_predictions = best_model.predict_proba(df_test1)[:, 1]

# Save submission file
submission = pd.DataFrame({"id": df_test["id"], "rainfall": test_predictions})
submission.to_csv("submission123.csv", index=False)
print("Submission file saved!")




