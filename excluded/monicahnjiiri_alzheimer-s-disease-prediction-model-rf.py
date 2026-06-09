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


train = pd.read_csv('/kaggle/input/alzheimers-disease-risk-prediction-eu-business/train.csv')


train.head()


train.info()


train.describe()


train.columns


X = train[['Age', 'Gender', 'Ethnicity', 'EducationLevel', 'BMI',
        'Smoking', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality',
        'SleepQuality', 'FamilyHistoryAlzheimers', 'CardiovascularDisease',
        'Diabetes', 'Depression', 'HeadInjury', 'Hypertension', 'SystolicBP',
        'DiastolicBP', 'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL',
        'CholesterolTriglycerides', 'MMSE', 'FunctionalAssessment',
        'MemoryComplaints', 'BehavioralProblems', 'ADL', 'Confusion',
        'Disorientation', 'PersonalityChanges', 'DifficultyCompletingTasks',
        'Forgetfulness']]
y  = train['Diagnosis']




from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


from sklearn.preprocessing import LabelEncoder


categorical_cols = ['Gender', 'Ethnicity', 'EducationLevel']


encoder_dict = {}
for col in categorical_cols:
    encoder = LabelEncoder()
    X_train[col] = encoder.fit_transform(X_train[col])
    X_test[col] = encoder.transform(X_test[col])  
    encoder_dict[col] = encoder


from sklearn.ensemble import RandomForestClassifier


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)


rf_model.fit(X_train, y_train)


from sklearn.metrics import f1_score, classification_report


y_pred = rf_model.predict(X_test)


y_test = y_test.reset_index(drop=True)


results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})


print(results.head(10))


f1 = f1_score(y_test, y_pred, average='weighted')  
print(f"F1 Score: {f1:.4f}")


print(classification_report(y_test, y_pred))


from sklearn.metrics import confusion_matrix


cm = confusion_matrix(y_test, y_pred)


plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No Alzheimer', 'Alzheimer'], yticklabels=['No Alzheimer', 'Alzheimer'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


test = pd.read_csv("/kaggle/input/alzheimers-disease-risk-prediction-eu-business/test.csv")


test.head()


test.info()


print(test.columns)


from sklearn.preprocessing import LabelEncoder


categorical_cols = ['Gender', 'Ethnicity', 'EducationLevel']


for col in categorical_cols:
    test[col] = encoder_dict[col].transform(test[col])


test = test[X_train.columns]


predictions = rf_model.predict(test)


test.loc[:, 'Predicted_Diagnosis'] = predictions


test.to_csv("predicted_results.csv", index=False)


print(test.head(10))



print("Predictions saved as 'predicted_results.csv'")


test['PatientID'] = train['PatientID'] 


test.head()

