# Importing Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error


# This is not related to the project. It is to filter future warnings by seaborn

import warnings
warnings.filterwarnings('ignore')


# Importing traning data

train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')

train.head()


# Importing testing data

test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')

test.head()


# Importing Sample Submission data

sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')

sample_submission.head()


# Information
print("Information : \n")
print(train.info())

# Missing Values
print("\nMissing Values : \n")
print(train.isnull().sum())

# Duplicate Entries
print("\nDuplicate Entries : \n")
print(train.duplicated().sum())


# Correlations between features

plt.figure(figsize=(12,7))
corr = train.corr(numeric_only = True)
sns.heatmap(corr, annot = True, linewidth = 0.2)
plt.show()



# Distributions of features

plt.figure(figsize = (20,20))

features = ['RhythmScore','AudioLoudness','VocalContent','AcousticQuality','InstrumentalScore','LivePerformanceLikelihood','MoodScore','TrackDurationMs','Energy']
n = 0
for col in features:
    n = n+1
    plt.subplot(3,3,n)
    sns.histplot(data = train, x = col, kde = True)
    plt.title(f"'{col}' Distribution")
plt.tight_layout()
plt.show()



train = train.drop('id', axis = 1)

test_ids = test['id']
test = test.drop('id', axis = 1)


train['has_vocal'] = (train['VocalContent']>0).astype('int')
test['has_vocal'] = (test['VocalContent']>0).astype('int')

train['is_acoustic'] = (train['AcousticQuality']>0).astype('int')
test['is_acoustic'] = (test['AcousticQuality']>0).astype('int')

train['is_instrumental'] = (train['InstrumentalScore']>0).astype('int')
test['is_instrumental'] = (test['InstrumentalScore']>0).astype('int')

train['is_live'] = (train['LivePerformanceLikelihood']).astype('int')
test['is_live'] = (test['LivePerformanceLikelihood']).astype('int')

train.head()


# splitting 'train' dataset to training and testing
x = train.drop('BeatsPerMinute', axis = 1)
y = train['BeatsPerMinute']

x_train,x_test,y_train,y_test = train_test_split(
    x,y,
    test_size = 0.2,
    random_state = 30
)



minmax = MinMaxScaler()

features_to_scale = ['RhythmScore','AudioLoudness','VocalContent','AcousticQuality','InstrumentalScore','LivePerformanceLikelihood','MoodScore','TrackDurationMs','Energy']

x_train_scaled = minmax.fit_transform(x_train[features_to_scale])
x_test_scaled = minmax.transform(x_test[features_to_scale])

test_scaled = minmax.transform(test[features_to_scale])

x_train[features_to_scale] = x_train_scaled
x_test[features_to_scale] = x_test_scaled

test[features_to_scale] = test_scaled



# Importing Model

from lightgbm import LGBMRegressor


from catboost import CatBoostRegressor

cat = CatBoostRegressor(
    iterations=5000,
    learning_rate=0.02,
    depth=8,
    l2_leaf_reg=3,
    random_seed=42,
    eval_metric='MAE',
    verbose=100 
)

cat.fit(
    x_train, y_train,
    eval_set=(x_test, y_test),
    early_stopping_rounds=200
)

print("Best iteration:", cat.get_best_iteration())



# submitting Predictions

cat_predictions = cat.predict(test)

submission_data = pd.DataFrame({
    "id" : test_ids,
    "BeatsPerMinute" : cat_predictions
})


submission_data.to_csv('submission.csv',header = True,index = False)



