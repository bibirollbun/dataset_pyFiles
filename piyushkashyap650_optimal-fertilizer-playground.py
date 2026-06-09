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


train=pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv',index_col='id')
test=pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
sample_submission=pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')
train.head() #to have a look at our data


train.isnull().sum()


train.duplicated().sum()


cat_feature= ['Soil Type', 'Crop Type']
feature = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous','Soil Type', 'Crop Type']


X= train[['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous','Soil Type', 'Crop Type']]


X


y=train['Fertilizer Name']


from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val= train_test_split(X,y,test_size= 0.2,random_state= 1)


from sklearn.preprocessing import LabelEncoder

#applying it on cat_features
le= LabelEncoder()
y_train=le.fit_transform(y_train)
y_val= le.fit_transform(y_val)


y_train


soil_le= LabelEncoder()
X_train['Soil Type']= soil_le.fit_transform(X_train['Soil Type'])
X_val['Soil Type']= soil_le.fit_transform(X_val['Soil Type'])


crop_le=LabelEncoder()
X_train['Crop Type']=crop_le.fit_transform(X_train['Crop Type'])
X_val['Crop Type']=crop_le.fit_transform(X_val['Crop Type'])


X_val


test.drop('id',axis=1,inplace=True)
test['Soil Type'] = soil_le.transform(test['Soil Type'])
test['Crop Type'] = crop_le.transform(test['Crop Type'])


test


test = test[X_train.columns]
test


from xgboost import XGBClassifier


model = XGBClassifier(
    objective='multi:softprob',
    num_class=len(np.unique(y_train)),
    n_estimators=4000,
    learning_rate=0.04,         
    max_depth=7,                
    colsample_bytree=0.6,       
    colsample_bylevel=0.8,      
    subsample=0.8,
    tree_method='hist',
)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=30,
    verbose=False
)
best_iter = model.best_iteration
print("Best iteration:", best_iter)


y_test_pred_probs = model.predict_proba(X_val)
y_test_pred = model.predict(X_val)


# MAP@3 function
def mapk(true_labels, predicted_labels, k=3):
    map_total = 0.0
    for true, preds in zip(true_labels, predicted_labels):
        score = 0.0
        for i, pred in enumerate(preds[:k]):
            if pred == true:
                score = 1.0 / (i + 1)
                break
        map_total += score
    map_score = map_total / len(true_labels)
    print(f"MAP@{k} Score: {map_score:.4f}")
    return map_score


# Get top 3 predictions
top_3_indices = np.argsort(y_test_pred_probs, axis=1)[:, -3:][:, ::-1]

# Flatten -> inverse transform -> reshape
flat_indices = top_3_indices.flatten()
flat_labels = le.inverse_transform(flat_indices)
top_3_labels = flat_labels.reshape(top_3_indices.shape)

# Prepare predictions
predicted_labels = [list(row) for row in top_3_labels]

# Get true labels
true_labels = le.inverse_transform(y_val)

# Evaluating the prediction for X_test
mapk(true_labels, predicted_labels, k=3)


sample_submission


## prediction for test data
test_pred_probs = model.predict_proba(test)
test_pred = model.predict(test)

# Predict class probabilities for test set
test_pred_probs = model.predict_proba(test)

# Get top 3 predicted indices
top_3_test_indices = np.argsort(test_pred_probs, axis=1)[:, -3:][:, ::-1]

# Flatten → inverse_transform → reshape to get original fertilizer names
flat_test_indices = top_3_test_indices.flatten()
flat_test_labels = le.inverse_transform(flat_test_indices)
top_3_test_labels = flat_test_labels.reshape(top_3_test_indices.shape)

# Join top 3 fertilizer names with space for each test sample
top_3_test_joined = [' '.join(row) for row in top_3_test_labels]

# Assign predictions
sample_submission['Fertilizer Name'] = top_3_test_joined

# Save to CSV
sample_submission.to_csv('xgbclassifierprediction.csv', index=False)

print("✅ Submission file saved as XGBClassifier_prediction.csv")




