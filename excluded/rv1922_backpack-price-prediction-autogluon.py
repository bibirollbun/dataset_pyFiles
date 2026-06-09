!pip install autogluon
!pip install -U ipywidgets


import pandas as pd
import numpy as np
import optuna
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
from autogluon.tabular import TabularDataset, TabularPredictor
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')


train.head()


cat_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']


for col in cat_cols:
    train[col] = train[col].fillna(train[col].mode()[0])
    test[col] = test[col].fillna(test[col].mode()[0])


train['Compartments'] = train['Compartments'].astype(int)
test['Compartments'] = test['Compartments'].astype(int)


train['Waterproof'] = train['Waterproof'].map({'Yes': 1, 'No': 0})
train['Laptop Compartment'] = train['Laptop Compartment'].map({'Yes': 1, 'No': 0})

size_mapping = {'Small': 1, 'Medium': 2, 'Large': 3}
train['Size'] = train['Size'].map(size_mapping)


test['Waterproof'] = test['Waterproof'].map({'Yes': 1, 'No': 0})
test['Laptop Compartment'] = test['Laptop Compartment'].map({'Yes': 1, 'No': 0})

size_mapping = {'Small': 1, 'Medium': 2, 'Large': 3}
test['Size'] = test['Size'].map(size_mapping)


cat_col = ['Brand', 'Material','Style', 'Color']


le = LabelEncoder()

for col in cat_col:
    train[col] = le.fit_transform(train[col])
    test[col] = le.fit_transform(test[col])


train.head()


X = train.drop(columns=['Price'])
y = train['Price']


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)


label = "Price"


predictor = TabularPredictor(label=label,eval_metric ='rmse',
                            problem_type="regression").fit( train,
                                                            time_limit=1500,verbosity=3,
                                                            presets='best_quality',  
                                                            ag_args_fit={'num_gpus': 1})  
results = predictor.fit_summary()


predictor.leaderboard()


test.head()


df = predictor.predict(test).to_frame(name=label)
df.head()


submission['Price']=df[label]
submission.head()


submission.to_csv('submission.csv',index=False)
print("File Saved!!")

