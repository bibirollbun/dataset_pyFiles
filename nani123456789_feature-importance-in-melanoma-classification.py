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
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


data = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv') 


data.info()


print("Missing values:\n", data.isnull().sum())


# Encode categorical variables
le = LabelEncoder()
categorical_columns = ['sex', 'anatom_site_general_challenge', 'diagnosis', 'benign_malignant']
for col in categorical_columns:
    # Fill NaNs with a placeholder before encoding
    data[col] = data[col].fillna('unknown')
    data[col] = le.fit_transform(data[col])


X = data.drop(columns=['image_name', 'patient_id', 'target', 'benign_malignant'])
y = data['target']


print(X.shape)
print(y.shape)


pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  
    ('scaler', StandardScaler()),  
    ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))
])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



pipeline.fit(X_train,y_train)


y_pred = pipeline.predict(X_test)
y_pred


print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print(f"\nAccuracy Score: {accuracy_score(y_test, y_pred):.2f}")





clf = pipeline.named_steps['classifier']
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance in Melanoma Classification')
plt.tight_layout()
plt.show()




