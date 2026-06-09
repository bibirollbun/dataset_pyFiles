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


# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.impute import KNNImputer


train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


# Info of training data
print(train.info())
train.sample(5)


# describe the training data
print(test.info())
test.sample(5)


# Describe the data

# trainset
print("TrainSet:\n",train.describe(include='all'), end = "\n\n")

# trainset
print("TestSet:\n",test.describe(include='all'))


# Lets first check for missing values

#trainset
print("TrainSet: \n\n",train.isnull().sum(), end = "\n\n")

#testset
print("TestSet: \n\n",test.isnull().sum())


# encode the object data
train['Stage_fear'] = train['Stage_fear'].map({'No':0 , 'Yes':1})
train['Drained_after_socializing'] = train['Drained_after_socializing'].map({'No':0 , 'Yes':1})
train['Personality'] = train['Personality'].map({'Extrovert':0 , 'Introvert':1})

test['Stage_fear'] = test['Stage_fear'].map({'No':0 , 'Yes':1})
test['Drained_after_socializing'] = test['Drained_after_socializing'].map({'No':0 , 'Yes':1})


# initialize the imputer
imputer = KNNImputer(n_neighbors = 7)

# train the imputer model
imputed = imputer.fit_transform(train)

train = pd.DataFrame(imputed, columns = train.columns)


# Same for test data
imputed = imputer.fit_transform(test)
test = pd.DataFrame(imputed, columns = test.columns)


# imports
from xgboost import XGBClassifier 
import optuna
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split


#Seperate the data]
X = train.drop(['Personality'], axis = 1)
Y = train['Personality']


X_train,X_test, Y_train,Y_test = train_test_split(X,
                                                 Y,
                                                 test_size = 0.2,
                                                 random_state = 42)


#optuna hobjective function
def objective(trial):
    param = {
    'booster':trial.suggest_categorical('booster',['gbtree','dart','gblinear']),
    'objective':'binary:logistic',
    'eval_metric':'auc',
    'tree_method':trial.suggest_categorical('tree_method',['hist','exact','approx']),
    'scale_pos_weight':trial.suggest_float('scale_pos_weight',1,100),
    'max_depth':trial.suggest_int('max_depth',3,12),
    'learning_rate':trial.suggest_float('learning_rate',0.001, 0.3),
    'subsample':trial.suggest_float('subsample',0.5,1.0),
    'colsample_bytree':trial.suggest_float('colsample_bytree', 0.5, 1.0),
    'lambda':trial.suggest_float('lambda',1e-3, 10.0),
    'alpha':trial.suggest_float('alpha',1e-3,10.0)
    }

    model = XGBClassifier(**param, use_label_encoder = False)
    model.fit(X_train, Y_train)
    preds = model.predict(X_test)
    return precision_score(Y_test, preds)


#create study object
study = optuna.create_study(direction='maximize')
optuna.logging.set_verbosity(optuna.logging.INFO)
study.optimize(objective, n_trials = 100)


# Now we use the best model available from the study object
best_params = study.best_params
model = XGBClassifier(**best_params)
model.fit(X_train,Y_train)
preds = model.predict(X_test)
preds_proba = model.predict_proba(X_test)[:,1]


print(f"Model: XGBClassifier with hyperparams tuned with Optuna")
print(f"Acuracy Score : {accuracy_score(Y_test, preds)}")
print(f"ROC_AUC Score : {roc_auc_score(Y_test, preds_proba)}")
print(f"Precision Score : {precision_score(Y_test, preds)}")
print(f"Recall Score : {recall_score(Y_test, preds)}")
print(f"Classification Report : \n{classification_report(Y_test, preds)}")
print(f"Confusion Matrix : \n")
cm = confusion_matrix(Y_test, preds)
cm_disp = ConfusionMatrixDisplay(confusion_matrix = cm)
cm_disp.plot()
plt.show()


# full train
model.fit(X,Y)
preds = model.predict(test)


output = pd.DataFrame({'id': test.id.astype(int),
                      'Personality': preds})

output['Personality'] = output['Personality'].map({0:'Extrovert', 1:'Introvert'})

output.to_csv('submission.csv', index = False)


output




