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


# Import necessary packages and ignore warnings

import pandas as pd
import numpy as np
import seaborn as sns
import optuna
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
    
warnings.filterwarnings('ignore')


# Read datasets and drop unecessary columns
train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')

# ID is required for submission purpose
ids = test['id']
train.drop(columns='id',inplace=True,axis=1)
test.drop(columns='id',inplace=True,axis=1)


# Understanding the dataset

print("Dataset shape: ",train.shape)
print("\nDaraset columns: ",train.columns)
print("\n",train.info())
print("\n",train.describe())


# Seperating numerical and categorical data

numerics = train.select_dtypes(include='number').columns.tolist()
categorical = train.select_dtypes(exclude='number').columns.tolist()
print("Numerical columns: ",numerics)
print("Categorical columns: ",categorical)


# Visualizing the correlation among numerical features of the dataset

sns.heatmap(train[numerics].corr())
sns.pairplot(train[numerics])


# Label Encoding the categorical features

def preprocessing(data):
    soil_encoder = LabelEncoder()
    crop_encoder = LabelEncoder()
    data['Soil Type'] = soil_encoder.fit_transform(data['Soil Type'])
    data['Crop Type'] = crop_encoder.fit_transform(data['Crop Type'])
    return data

# Label Encoding target variable
fertilizer_encoder = LabelEncoder()
train['Fertilizer Name'] = fertilizer_encoder.fit_transform(train['Fertilizer Name'])        
train = preprocessing(train)
test = preprocessing(test)


# Divide the data into features and targets

X = train.drop(columns = 'Fertilizer Name')
y = train['Fertilizer Name']

X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8)

print(X_train.shape, X_test.shape)


"""

# Define the number of classes
n_classes = 7

# Objective function for Optuna
def objective(trial):
    params = {
        "objective": "multi:softmax",  # For multi-class classification
        "num_class": n_classes,
        "eval"
        "tree_method": "gpu_hist",  # GPU acceleration
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "n_estimators": trial.suggest_int("n_estimators", 50, 300)
    }

    model = xgb.XGBClassifier(**params, use_label_encoder=False, verbosity=0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy  # maximize accuracy

# Run optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# Print best result
print("✅ Best Hyperparameters:")
print(study.best_params)

"""


params = {'max_depth': 9, 'learning_rate': 0.09612941234145238, 'subsample': 0.8189360849126304, 'colsample_bytree': 0.6816763279208065, 'gamma': 0.6736633274662291, 'reg_alpha': 1.540306330437053, 'reg_lambda': 1.693551657182308, 'min_child_weight': 5, 'n_estimators': 228}


# Train final model with best params

model = xgb.XGBClassifier(*params, eval_metric='mlogloss', use_label_encoder=False, verbosity=1)
model.fit(X_train, y_train)


# 1. Get predicted probabilities for each class
y_val_xgb_proba = model.predict_proba(X_test)  # (n_samples, n_classes)

# 2. Extract the indices of the top 3 classes with the highest probabilities
y_val_xgb_pred_top3_idx = np.argsort(y_val_xgb_proba, axis=1)[:, -3:][:, ::-1]

# 3. Convert indices to fertilizer names (strings)
y_val_xgb_pred_top3_label = fertilizer_encoder.inverse_transform(y_val_xgb_pred_top3_idx.ravel()).reshape(y_val_xgb_pred_top3_idx.shape)
y_val_label = fertilizer_encoder.inverse_transform(y_test)

# 4. Compare predictions with actual values
xgb_pred_vs_real = np.column_stack((y_val_xgb_pred_top3_label, y_val_label))


# prediction accuracy
xgb_correct = [row[-1] in row[:3] for row in xgb_pred_vs_real]
xgb_accuracy = sum(xgb_correct) / len(xgb_correct)

print("Accuracy:", xgb_accuracy)


# Predict the probabilities for the test dataset

test_proba = model.predict_proba(test)

test_pred_top3_idx = np.argsort(test_proba, axis=1)[:, -3:][:, ::-1]
test_pred_top3_label = fertilizer_encoder.inverse_transform(test_pred_top3_idx.ravel()).reshape(test_pred_top3_idx.shape)


# Prepare submission
submission = pd.DataFrame({
    'id': ids, 
    'Fertilizer Name': [' '.join(row) for row in test_pred_top3_label]
})

submission.to_csv('submission.csv', index=False)

