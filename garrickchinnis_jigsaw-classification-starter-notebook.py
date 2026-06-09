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


#XGBoost 
from xgboost import XGBRegressor, XGBClassifier, plot_importance, plot_tree
#Importing other useful packages
import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
import statistics
#Sklearn/RandomForest
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense



# Load training data
train = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
train.info()


train.head()


train['subreddit_num'] = pd.factorize(train['subreddit'])[0]


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# Tokenizing
tokenizer = Tokenizer(num_words=5272,oov_token="<unk>")
tokenizer.fit_on_texts(train['body'].values)
sequences = tokenizer.texts_to_sequences(train['body'])
vocab_size = len(tokenizer.word_index) + 1
encoded_docs = tokenizer.texts_to_sequences(train['body'])

# Pad sequences
max_len = train['body'].str.len().mean().astype('int64')

X = pad_sequences(sequences, maxlen=max_len, padding='post')
y = train['rule_violation'].values


#Split training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


# Define base model
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
xgb = XGBClassifier(objective='binary:logistic', random_state=42)

'''''# Define hyperparameter grid
param_grid = {
    'n_estimators': [50, 80, 100, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7, 10],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Define scorer (lower MAE is better, so set greater_is_better=False)
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

# Use 5-fold cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Run GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring=mae_scorer,
    cv=cv,
    n_jobs=-1,
    verbose=2
)

grid_search.fit(train_X, train_y)

# Best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best MAE (CV):", -grid_search.best_score_)'''''


#Add Parameters
model = XGBClassifier(colsample_bytree = 0.8, learning_rate = 0.01, max_depth = 10, n_estimators = 100, subsample = 0.8)
#model2 = RandomForestRegressor()

#Fitting data to model
model.fit(train_X, train_y)

#Making predictions on data
val_predictions = model.predict(val_X)

#Getting MAE and Accuracy scores
val_mae = mean_absolute_error(val_predictions, val_y)

#Printing results
print("Validation MAE for the model: {:,.0f}".format(val_mae))
print('The accuracy of the model is: ', model.score(val_X, val_y)) 
print('The accuracy of the training model is: ', model.score(train_X, train_y))


#Create finalized model determined hyperparameters from above cell
final_model = XGBClassifier(n_estimators = 120, learning_rate = 0.1, max_depth = 4,eval_metric = 'mae')
#Fit model to full training data
final_model.fit(X,y)


#load test data
test = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')

tokenizer = Tokenizer(num_words=5272,oov_token="<unk>")
tokenizer.fit_on_texts(test['body'].values)
sequences = tokenizer.texts_to_sequences(test['body'])
vocab_size = len(tokenizer.word_index) + 1
encoded_docs = tokenizer.texts_to_sequences(test['body'])

# Pad sequences
max_len = train['body'].str.len().mean().astype('int64')
X_test = pad_sequences(sequences, maxlen=max_len, padding='post')

#Predicting test data's missing y target value
y_pred = final_model.predict(X_test)


#Generic submission formatting 

output = pd.DataFrame({'row_id': test['row_id'],
                       'rule_violation': y_pred})
output.to_csv('submission.csv', index=False)

