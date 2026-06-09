import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import *
from sklearn.model_selection import (train_test_split, GridSearchCV, KFold, 
RepeatedKFold,RandomizedSearchCV, cross_val_score)
from sklearn.metrics import mean_squared_error, mean_squared_log_error
import optuna


import sklearn
sklearn.metrics??


target = 'Calories'


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv', index_col='id')
train.head()


test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv', index_col='id')
test.head()


orig = pd.read_csv('/kaggle/input/calories-burnt-prediction/calories.csv')
orig = orig.drop(columns =['User_ID'])
orig.columns = train.columns
orig.head()


train_comb = pd.concat([train, orig])
train_comb.tail()


sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')


for df in [train, test, orig, train_comb]:
    df['heart_rate_x_duration'] = df['Heart_Rate']*df['Duration']
    df['bmi'] = df['Weight']/(0.01*df['Height'])**2
    df['Weight_group'] = pd.cut(df['Weight'], bins=4, labels=range(1, 5)).astype('int')
    df['Height_group'] = pd.cut(df['Height'], bins=4, labels=range(1, 5)).astype('int')
    df['body_temp_x_duration'] = df['Body_Temp']*df['Duration']
    df['body_temp_/_duration'] = df['Body_Temp']/df['Duration']
    df['log_weight_x_height'] = np.log(df['Weight']*df['Height'])
    df['weight_x_height'] = df['Weight']*df['Height']/100
    # df['age_group'] = pd.cut(df['Age'], bins=10, labels=range(1, 11)).astype('int')
    df['age_group'] = pd.cut(df['Age'], bins=[0, 25, 35, 50, 100], labels=range(1, 5)).astype('int')
    df['body_temp_x_weight'] = df['Body_Temp']*df['Weight']
    df['Sex'] = (df['Sex'] == 'female')*1


test.head()


# test.var()


# test.skew()


X = train.copy()
X_or = orig.copy()
X_cb = train_comb.copy()

y = X.pop(target)
y_or = X_or.pop(target)
y_cb = X_cb.pop(target)

feat_of_interest = [
                    'Sex', 
                    'Age',
                    'Weight',
                    'Height_group',
                    'Body_Temp', 
                    'body_temp_x_duration',
                    'body_temp_/_duration',
                    'bmi',
                    'log_weight_x_height',
                    'Heart_Rate', 
                    'heart_rate_x_duration',
                    'body_temp_x_weight',
                   ]
X = X[feat_of_interest]
X_or = X_or[feat_of_interest]
X_test = test[feat_of_interest]
X_cb = X_cb[feat_of_interest]
X.head()


plt.figure(figsize=(9, 8))
x_corr = X.corr(numeric_only=True).abs()
sns.heatmap(x_corr, annot=True, fmt='.2f', cbar=False, cmap='RdYlGn_r', square=True, vmin=0.6)
plt.show()


from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=4)


# scaler = PowerTransformer()
scaler = MinMaxScaler()


# Define the objective function
def objective_xgb(trial):
    xgb_param_grid = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1),
        "max_depth": trial.suggest_int("max_depth", 1, 15),
        "subsample": trial.suggest_float("subsample", 0.5, 1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),  # L1 regularization
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),  # L2 regularization
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }

    # Data preprocessing and model initialization
    model = make_pipeline(scaler,XGBRegressor(**xgb_param_grid))

    # fit the model
    model.fit(X_tr, y_tr)
    
    # Evaluate the model using RMSE
    try:
        preds = np.clip(model.predict(X_va), 
                        orig[target].min(), 
                        orig[target].max())
        score = np.sqrt(mean_squared_log_error(y_va, preds))
        return score
    except:
        return 100

# Define the function to run the study
def Run_Pass_xgb_study(n_trials=1):
    if n_trials > 1:
        # Create and run the study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective_xgb, n_trials=n_trials, 
                       show_progress_bar=True)
        best_study_params = study.best_params

        # Print results
        print(f"Number of finished trials: {len(study.trials)}")
        trial = study.best_trial
        print(f"Best trial RMSE score: {trial.value:.6f}")
    else:
        print("No need to run Optuna, we will use the parameters obtained earlier.")       
        best_study_params = {'n_estimators': 386, 
                             'learning_rate': 0.043880833926294216, 
                             'max_depth': 14, 
                             'subsample': 0.9252956001901687, 
                             'colsample_bytree': 0.7643282360969083, 
                             'reg_alpha': 1.876119289691811, 
                             'reg_lambda': 9.784188433027413, 
                             'min_child_weight': 1}

    print(f"Best parameters: {best_study_params}")
    return best_study_params

xgb_best_params = Run_Pass_xgb_study(50)


estimator = XGBRegressor(**xgb_best_params, enable_categorical = True)
# estimator = LogisticRegression()


model = make_pipeline(scaler, estimator)
model.fit(X_tr, y_tr)


y_va_hat = model.predict(X_va)
va_score = np.sqrt(mean_squared_log_error(y_va, y_va_hat))
# va_score = model.score(X_va, y_va)
y_or_hat = model.predict(X_or)
or_score = np.sqrt(mean_squared_log_error(y_or, y_or_hat))
# or_score = model.score(X_or, y_or)

plt.figure(figsize=(9,4))
plt.subplot(121)
sns.scatterplot(x=y_va, y=y_va_hat, palette='Set2', hue=X_va['Sex'])
plt.xlabel('True')
plt.ylabel('preds')
plt.title('rmsle on validation data: {:.6}'.format(va_score))
plt.subplot(122)
sns.scatterplot(x=y_or, y=y_or_hat, palette='Set2', hue=X_or['Sex'])
plt.xlabel('True')
plt.title('rmsl on original data: {:.6}'.format(or_score))
plt.tight_layout()


def rmsle_scorer(y_true, y_pred):
    # Ensure the inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate the logarithm of the true and predicted values
    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred)
    
    # Calculate the squared differences
    squared_diff = np.square(log_true - log_pred)
    
    # Calculate the mean of the squared differences
    mean_squared_diff = np.mean(squared_diff)
    
    # Calculate the root of the mean squared differences
    rmsle_value = np.sqrt(mean_squared_diff)
    
    return rmsle_value


# Define RMSLE metric
def rmsle_metric(y_true, y_pred):
    y_true = np.maximum(y_true, 1e-6)  # Avoid log(0) issues
    y_pred = np.maximum(y_pred, 1e-6)

    log_true = np.log(y_true)
    log_pred = np.log(y_pred)

    rmsle = np.sqrt(np.mean((log_true - log_pred) ** 2))
    return rmsle


final_model = make_pipeline(scaler, estimator).fit(X_cb, y_cb)


preds = np.clip(final_model.predict(X_test),
                orig[target].min(),
                orig[target].max()
               )

sample_submission[target] = preds

display(sample_submission.head(10))

sample_submission.to_csv('submission.csv', index=False)
print('Your predictions are ready for submission!')

