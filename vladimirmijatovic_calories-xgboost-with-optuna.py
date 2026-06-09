import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler # for scaling
import math
import optuna

from sklearn.model_selection import train_test_split, RandomizedSearchCV,cross_val_score,StratifiedKFold
from sklearn.impute import SimpleImputer

from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_log_error


# plotting
import matplotlib.pyplot as plt
import seaborn as sns


#Ignore warnings
import warnings
warnings.filterwarnings('ignore')






train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')


# create already now an empty submission file 
submission = pd.DataFrame()

# dd id column to submission file
submission['id'] = test['id']


train.head()


train.info()


print("Shape of train: ", train.shape)

print("Shape of test: ", test.shape)


train.describe()


# Body Mass Index
train['BMI'] = train['Weight'] / ((train['Height'] / 100) ** 2)
test['BMI'] = test['Weight'] / ((test['Height'] / 100) ** 2)

# add BMI categories
train['BMI_Category'] = pd.cut(train['BMI'], 
                               bins=[0, 18.5, 25, 30, 100],
                               labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
test['BMI_Category'] = pd.cut(test['BMI'], 
                              bins=[0, 18.5, 25, 30, 100],
                              labels=['Underweight', 'Normal', 'Overweight', 'Obese'])




# heart rate and body temperature

train['Heart_x_Body_temp'] = train['Heart_Rate'] * train['Body_Temp']
test['Heart_x_Body_temp'] = test['Heart_Rate'] * test['Body_Temp']


# encode Sex with dummy variable
train['Sex'] = train['Sex'].map({'male': 1, 'female': 0})
test['Sex'] = test['Sex'].map({'male': 1, 'female': 0})



# add one-hot-encoding categorical for BMI

train = pd.get_dummies(train, columns = ['BMI_Category'], drop_first = True)
test = pd.get_dummies(test, columns = ['BMI_Category'], drop_first = True)




train.head()


X_train = train.copy()
X_test = test.copy()

# remove ID column from the dataset

X_train = X_train.drop("id", axis = 1)
X_test = X_test.drop('id', axis = 1)





# create X_train and Y_train

Y_train = X_train['Calories']
X_train = X_train.drop('Calories', axis = 1)



X_train_initial, X_val_final, Y_train_initial, y_val_final = train_test_split(
    X_train, 
    Y_train, 
    test_size=0.2, 
    random_state=42, 
    stratify = Y_train
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# define XGBoost Optuna 


def rmsle(y_true, y_pred):
    """
    Compute Root Mean Squared Logarithmic Error.
    """
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

def objective_xgb_regressor(trial):
        # Define hyperparameters to tune

    params = {
    # hyperparameters for XGBClassifier
    #'n_estimators' : trial.suggest_int("n_estimators", 500, 1000),
    'n_estimators' : 500, 
    #'learning_rate' : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
    'learning_rate': 0.1221598364599878,
    'max_depth' : trial.suggest_int("max_depth", 3, 10),
    'subsample' : trial.suggest_float("subsample", 0.5, 1.0),
    'colsample_bytree' : trial.suggest_float("colsample_bytree", 0.5, 1.0),
    'random_state' : 20
    }

    
    model = XGBRegressor(**params)
    
    # We need to make predictions to calculate RMSLE
    scores = []

    for train_index, val_index in cv.split(X_train_initial, Y_train_initial):
        X_train_fold, X_val_fold = X_train_initial.iloc[train_index], X_train_initial.iloc[val_index]
        Y_train_fold, Y_val_fold = Y_train_initial.iloc[train_index], Y_train_initial.iloc[val_index]

        model.fit(X_train_fold, Y_train_fold)
        preds = model.predict(X_val_fold)
        preds[preds < 0] = 0
        score = rmsle(Y_val_fold, preds)
        scores.append(score)

    return np.mean(scores)


study_xgb = optuna.create_study(direction="minimize")


# optimize with 200 trials 

study_xgb.optimize(objective_xgb_regressor, n_trials = 20)


print("Number of finished trials: {}".format(len(study_xgb.trials)))
print("Best trial:")
trial = study_xgb.best_trial
print("  Value: {}".format(trial.value))
print("  Params: {}".format(trial.params))


# Train the final XGBoost model with the best hyperparameters on the full training set

best_params = trial.params

best_model = XGBRegressor(
    **best_params, 
    random_state=42, 
    n_jobs=-1
)  


best_model.fit(
    X_train, 
    Y_train
)





# Evaluate the final model on the test set

Y_pred_test = best_model.predict(X_val_final)

Y_pred_test_clipped = np.maximum(Y_pred_test, 0) # Ensure non-negative






test_rmsle = rmsle(y_val_final, Y_pred_test_clipped)

print(f"Test RMSLE: {test_rmsle}")


pred = best_model.predict(X_test)


# add prediction column to submission file 

submission["Calories"] = pred


# write to csv
submission.to_csv(
    "submission.csv", 
    index = False
)  # remove index, otherwise submission will fail

