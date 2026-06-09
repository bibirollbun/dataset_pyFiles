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


train=pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')


test=pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')


train.head()


train.info()


train['Fertilizer Name'].value_counts()



train['Soil Type'].value_counts()



train['Crop Type'].value_counts()



from sklearn.preprocessing import LabelEncoder


le=LabelEncoder()


columns_to_encode = ['Soil Type','Crop Type','Fertilizer Name']


for column in columns_to_encode:
    train[column] = le.fit_transform(train[column])


columns_to_encodes = ['Soil Type','Crop Type']


for column in columns_to_encodes:
    test[column] = le.fit_transform(test[column])


X=train.drop(labels=['id','Fertilizer Name'],axis=1)


Y=train['Fertilizer Name']


X.head()


Y.head()


print(X.shape)
print(Y.shape)


from imblearn.under_sampling import RandomUnderSampler


rus = RandomUnderSampler(random_state=42)


X_resampled, y_resampled = rus.fit_resample(X, Y)



from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)


import xgboost as xgb


dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)


num_classes = len(set(Y))
params = {
    'objective': 'multi:softmax',  # or 'multi:softprob' for probabilities
    'num_class': num_classes,
    'eval_metric': 'mlogloss',
    'max_depth': 3,
    'eta': 0.1,
    'seed': 42
}


bst = xgb.train(params, dtrain, num_boost_round=100)


preds = bst.predict(dtest)



from sklearn.metrics import classification_report, accuracy_score


print("Accuracy:", accuracy_score(y_test, preds))
print("\nClassification Report:\n", classification_report(y_test, preds))


print(X_test.shape[0])   # Should be 250000
print(preds.shape[0])    # Should also be 250000



ids = test["id"]

# Drop 'id' to get feature matrix
X_test = test.drop("id", axis=1)

# Convert to DMatrix
dtest = xgb.DMatrix(X_test)

# Predict using the trained model
preds = bst.predict(dtest)  # Use .argmax(axis=1) if using softprob

preds = preds.astype(int)

class_names = ['14-35-14', '10-26-26', '17-17-17','28-28','20-20','DAP','Urea']  # Replace with your actual class names
le = LabelEncoder()
le.fit(class_names)
pred_labels = le.inverse_transform(preds)

# Create submission DataFrame
submission = pd.DataFrame({
    "id": ids,
    "Fertilizer Name": pred_labels  # Ensure it's integer labels
})

# Save to CSV
submission.to_csv("submissiof.csv", index=False)

print("Submission file 'submission.csv' created successfully!")




