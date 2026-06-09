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


df=pd.read_csv('/kaggle/input/anomaly-detection/train.csv')


df.info()


df.head(2)


df['is_anomaly_encoded']=df['is_anomaly'].astype(int)


df.head()


df.is_anomaly_encoded.value_counts()


from sklearn.preprocessing import MinMaxScaler


# Normalising Features
scaler=MinMaxScaler()
scaled_feature=scaler.fit_transform(df[['value','predicted']])


# Preparing for K-means
df_scaled=pd.DataFrame(scaled_feature, columns=['value_scaled','predicted_scaled'])


from sklearn.cluster import KMeans


# Number of cluster is 2 (True / False)
kmeans=KMeans(n_clusters=2,n_init=10,random_state=42)


df_scaled['cluster']=kmeans.fit_predict(df_scaled)


from sklearn.metrics import f1_score


f1=f1_score(df['is_anomaly_encoded'],df_scaled['cluster'])
print(f"F1 Score: {f1}")


df['is_anomaly_encoded'].value_counts()


# List of imports
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


# Standard Scaling
scaler=StandardScaler()
X=scaler.fit_transform(df[['value','predicted']])


# Target variable
y=df['is_anomaly_encoded']


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Initialize the Random Forest Classifier
rf = RandomForestClassifier(class_weight='balanced', random_state=42)



# Train the model
rf.fit(X_train, y_train)


# Predictions
y_pred = rf.predict(X_test)


# Probabilities (optional for threshold-based decisions)
y_prob = rf.predict_proba(X_test)[:, 1]


# Metrics
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"F1 Score: {f1}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Confusion Matrix:\n{conf_matrix}")



# Test data loading
df_test=pd.read_csv('/kaggle/input/anomaly-detection/test.csv')
df_submission=pd.read_csv('/kaggle/input/anomaly-detection/Submission.csv')


df_test


df_test[['scaled_value', 'scaled_predicted']] = scaler.transform(df_test[['value', 'predicted']])





df_test['is_anomaly']=rf.predict(df_test[['scaled_value','scaled_predicted']].values)


df_test['is_anomaly']=df_test['is_anomaly'].astype(bool)


df_test=df_test[['timestamp','is_anomaly']]


df_test


df_submission


df_test


df_test.to_csv('submission.csv', index=False)




