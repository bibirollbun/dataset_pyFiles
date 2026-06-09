import warnings
warnings.simplefilter('ignore')
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score
from lightgbm import LGBMClassifier, early_stopping
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


import pandas as pd

data1=pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
data2=pd.read_csv("/kaggle/input/extrovert-vs-introvert-behavior-data/personality_datasert.csv")
data3=pd.read_csv("/kaggle/input/extrovert-vs-introvert-behavior-data/personality_dataset.csv")
data1=data1.iloc[:,1:]

train = pd.concat([data1, data2,data3], axis=0, ignore_index=True)
train = train.drop_duplicates().reset_index(drop=True)
test =pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

Stage_fear=LabelEncoder()
Drained_after_socializing=LabelEncoder()
Personality=LabelEncoder()

train['Stage_fear']=Stage_fear.fit_transform(train['Stage_fear'].astype(str))
train['Drained_after_socializing']=Drained_after_socializing.fit_transform(train['Drained_after_socializing'].astype(str))
train['Personality']=Personality.fit_transform(train['Personality'].astype(str))
test['Stage_fear']=Stage_fear.fit_transform(test['Stage_fear'].astype(str))
test['Drained_after_socializing']=Drained_after_socializing.fit_transform(test['Drained_after_socializing'].astype(str))

x=train.iloc[:,:-1]
y=train['Personality']
x_train,x_temp,y_train,y_temp=train_test_split(x,y,test_size=0.3,random_state=42)
x_val,x_test,y_val,y_test=train_test_split(x_temp,y_temp,test_size=0.33,random_state=42)


params = {
    'objective': 'binary:logistic', 
    'max_depth': 7,
    'learning_rate': 0.03,
    'subsample': 0.8,
    'max_bin': 128,
    'colsample_bytree': 0.3, 
    'colsample_bylevel': 1,  
    'colsample_bynode': 1,  
    'tree_method': 'hist',  
    'random_state': 42,
    'eval_metric': 'logloss',  
    'device': "cuda",
    'enable_categorical': True,
    'n_estimators': 10000,
    'early_stopping_rounds': 50,
}


# Check validation score
model = XGBClassifier(**params)
model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)


val_predictions = model.predict(x_val)
val_accuracy = accuracy_score(y_val, val_predictions)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Get predicted probabilities for log loss
val_proba = model.predict_proba(x_val)
val_logloss = log_loss(y_val, val_proba)
print(f"Validation Log Loss: {val_logloss:.4f}")

# Create submission file
# First, prepare test data (remove 'id' column)
test_features = test.iloc[:, 1:]  # Remove the first column (id)

# Make predictions on test set
test_predictions = model.predict(test_features)

# Convert predictions back to original labels
test_predictions_labels = Personality.inverse_transform(test_predictions)

# Create submission dataframe
submission = pd.DataFrame({
    'id': test['id'],  # Use the id column from test data
    'Personality': test_predictions_labels
})

# Save submission file
submission.to_csv('submission.csv', index=False)
print("Submission file created successfully!")
print(f"Submission shape: {submission.shape}")
print("\nFirst few rows of submission:")
print(submission.head())

