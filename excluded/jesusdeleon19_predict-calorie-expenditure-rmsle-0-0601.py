import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import lightgbm as lgb
import optuna
from sklearn.metrics import mean_squared_log_error
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
df.head()


df_test.head()


df.isna().sum()


df_test.isna().sum()


df.info()


df_test.info()


oe = OrdinalEncoder()
df["Sex"] = oe.fit_transform(df[["Sex"]])
df_test["Sex"] = oe.fit_transform(df_test[["Sex"]])
df.info()


fig, axs = plt.subplots(2, 4, figsize=(17, 10))

sex = df['Sex'].value_counts()
axs[0,0].pie(sex, labels=sex.index, autopct='%1.1f%%')
axs[0,0].set_title('Pie Plot of Sex')

sns.histplot(df['Age'], ax=axs[0,1], kde=True)
axs[0,1].set_title('Distribution of Age')

sns.histplot(df['Heart_Rate'], ax=axs[0,2], kde=True)
axs[0,2].set_title('Distribution of Heart_Rate')

sns.histplot(df['Weight'], ax=axs[1,0], kde=True)
axs[1,0].set_title('Distribution of Weight')

sns.histplot(df['Duration'], ax=axs[1,1], kde=True)
axs[1,1].set_title('Distribution of Duration')

sns.histplot(df['Height'], ax=axs[1,2], kde=True)
axs[1,2].set_title('Distribution of Height')

sns.histplot(df['Body_Temp'], ax=axs[1,3], kde=True)
axs[1,3].set_title('Distribution of Body_Temp')

sns.histplot(df['Calories'], ax=axs[0,3], kde=True)
axs[0,3].set_title('Distribution of Calories (Target)')

plt.tight_layout()
plt.show()


fig, axs = plt.subplots(2, 4, figsize=(17, 10))

axs[0,0].boxplot(df['Age'])
axs[0,0].set_title('Box Plot of Age')

axs[0,1].boxplot(df['Heart_Rate'])
axs[0,1].set_title('Box Plot of Heart_Rate')

axs[0,2].boxplot(df['Weight'])
axs[0,2].set_title('Box Plot of Weight')

axs[0,3].boxplot(df['Duration'])
axs[0,3].set_title('Box Plot of Duration')

axs[1,0].boxplot(df['Calories'])
axs[1,0].set_title('Box Plot of Calories (Target)')

axs[1,1].boxplot(df['Height'])
axs[1,1].set_title('Box Plot of Height')

axs[1,2].boxplot(df['Body_Temp'])
axs[1,2].set_title('Box Plot of Body_Temp')

axs[1,3].boxplot(df['Sex'])
axs[1,3].set_title('Box Plot of Sex')

plt.tight_layout()
plt.show()


df['Height_m'] = df['Height'] / 100
df['BMI'] = df['Weight'] / (df['Height_m']**2)
df['Max_Est_Heart_Rate'] = 220 - df['Age']
df['HR_to_MaxHR_Ratio'] = df['Heart_Rate'] / df['Max_Est_Heart_Rate']
df['Cardiac_Work_Proxy'] = df['Duration'] * df['Heart_Rate']
df['BMI_x_Age'] = df['BMI'] * df['Age']
df['BMI_x_Temp'] = df['BMI'] * df['Body_Temp']
df['BMI_x_Duration'] = df['BMI'] * df['Duration']
df['BMI_x_HeartRate'] = df['Age'] * df['Heart_Rate']
df['Age_x_HeartRate'] = df['Age'] * df['Heart_Rate']
df['Age_x_Duration'] = df['Age'] * df['Duration']
df['Age_x_Tempe'] = df['Age'] * df['Body_Temp']
df['BodyTemp_x_Duration'] = df['Body_Temp'] * df['Duration']
df['BodyTemp_x_HeartRate'] = df['Body_Temp'] * df['Heart_Rate']
df['BodyTemp_x_Age'] = df['Body_Temp'] * df['Age']
# log of most positive related features 
df['duration_log'] = np.log1p(df['Duration'])
df['BodyTemp_log'] = np.log1p(df['Body_Temp'])
df['Age_log'] = np.log1p(df['Age'])
df['Heart_Rate_log'] = np.log1p(df['Heart_Rate'])

df['Age_x_Duration_log'] = np.log1p(df['Age_x_Duration'])
df['Age_x_HeartRate_log'] = np.log1p(df['Age_x_HeartRate'])
df['Age_x_BodyTemp_log'] = np.log1p(df['Age_x_Tempe'])
df['BodyTemp_x_Duration_log'] = np.log1p(df['BodyTemp_x_Duration'])
df['BodyTemp_x_HeartRate_log'] = np.log1p(df['BodyTemp_x_HeartRate'])
df['BodyTemp_x_Age_log'] = np.log1p(df['BodyTemp_x_Age'])
df['BMI_x_Age_log'] = np.log1p(df['BMI_x_Age'])
df['BMI_x_Duration_log'] = np.log1p(df['BMI_x_Duration'])
df['BMI_x_HeartRate_log'] = np.log1p(df['BMI_x_HeartRate'])
df['BMI_x_BodyTemp_log'] = np.log1p(df['BMI_x_Temp'])
# Sqr of most related features
df['BMI_sq'] = df['BMI']**2
df['Age_sq'] = df['Age']**2
df['Height_sq'] = df['Height']**2
df['Weight_sq'] = df['Weight']**2
df['Duration_sq'] = df['Duration']**2
df['Heart_Rate_sq'] = df['Heart_Rate']**2
df['Body_Temp_sq'] = df['Body_Temp']**2
# Ratio of features
df['Max_Est_HR'] = 220 - df['Age']
df['HR_as_Perc_Max_Est'] = df['Heart_Rate'] / df['Max_Est_HR']
# Handle potential division by zero or negative Max_Est_HR if age is very high/problematic
df['HR_as_Perc_Max_Est'].replace([np.inf, -np.inf], np.nan, inplace=True)

# Test data, feature extraction
df_test['Height_m'] = df_test['Height'] / 100
df_test['BMI'] = df_test['Weight'] / (df_test['Height_m']**2)
df_test['Max_Est_Heart_Rate'] = 220 - df_test['Age']
df_test['HR_to_MaxHR_Ratio'] = df_test['Heart_Rate'] / df_test['Max_Est_Heart_Rate']
df_test['Cardiac_Work_Proxy'] = df_test['Duration'] * df_test['Heart_Rate']
df_test['BMI_x_Age'] = df_test['BMI'] * df_test['Age']
df_test['BMI_x_Temp'] = df_test['BMI'] * df_test['Body_Temp']
df_test['BMI_x_Duration'] = df_test['BMI'] * df_test['Duration']
df_test['BMI_x_HeartRate'] = df_test['Age'] * df_test['Heart_Rate']
df_test['Age_x_HeartRate'] = df_test['Age'] * df_test['Heart_Rate']
df_test['Age_x_Duration'] = df_test['Age'] * df_test['Duration']
df_test['Age_x_Tempe'] = df_test['Age'] * df_test['Body_Temp']
df_test['BodyTemp_x_Duration'] = df_test['Body_Temp'] * df_test['Duration']
df_test['BodyTemp_x_HeartRate'] = df_test['Body_Temp'] * df_test['Heart_Rate']
df_test['BodyTemp_x_Age'] = df_test['Body_Temp'] * df_test['Age']
# Log of test data
df_test['duration_log'] = np.log1p(df_test['Duration'])
df_test['BodyTemp_log'] = np.log1p(df_test['Body_Temp'])
df_test['Age_log'] = np.log1p(df_test['Age'])
df_test['Heart_Rate_log'] = np.log1p(df_test['Heart_Rate'])

df_test['Age_x_Duration_log'] = np.log1p(df_test['Age_x_Duration'])
df_test['Age_x_HeartRate_log'] = np.log1p(df_test['Age_x_HeartRate'])
df_test['Age_x_BodyTemp_log'] = np.log1p(df_test['Age_x_Tempe'])
df_test['BodyTemp_x_Duration_log'] = np.log1p(df_test['BodyTemp_x_Duration'])
df_test['BodyTemp_x_HeartRate_log'] = np.log1p(df_test['BodyTemp_x_HeartRate'])
df_test['BodyTemp_x_Age_log'] = np.log1p(df_test['BodyTemp_x_Age'])
df_test['BMI_x_Age_log'] = np.log1p(df_test['BMI_x_Age'])
df_test['BMI_x_Duration_log'] = np.log1p(df_test['BMI_x_Duration'])
df_test['BMI_x_HeartRate_log'] = np.log1p(df_test['BMI_x_HeartRate'])
df_test['BMI_x_BodyTemp_log'] = np.log1p(df_test['BMI_x_Temp'])
# Srt of test data
df_test['BMI_sq'] = df_test['BMI']**2
df_test['Age_sq'] = df_test['Age']**2
df_test['Height_sq'] = df_test['Height']**2
df_test['Weight_sq'] = df_test['Weight']**2
df_test['Duration_sq'] = df_test['Duration']**2
df_test['Heart_Rate_sq'] = df_test['Heart_Rate']**2
df_test['Body_Temp_sq'] = df_test['Body_Temp']**2
# Ratio of test data
df_test['Max_Est_HR'] = 220 - df_test['Age']
df_test['HR_as_Perc_Max_Est'] = df_test['Heart_Rate'] / df_test['Max_Est_HR']
# Handle potential division by zero or negative Max_Est_HR if age is very high/problematic
df_test['HR_as_Perc_Max_Est'].replace([np.inf, -np.inf], np.nan, inplace=True)


plt.figure(figsize=(32, 22))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.1%')
plt.title('Correlation Heatmap')
plt.show()


# Divide train data into train and test datasets for training
X = df.drop(['Calories','id'], axis=1)
y = df['Calories']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

len(X_train), len(y_train)


# if using sklearn version 1.4 and up can simply import 
# from sklearn.metrics import root_mean_squared_log_error

def rmsle(y_true, y_pred):
    """
    Calculates the Root Mean Squared Log Error (RMSLE).

    Args:
        y_true (np.ndarray): Array of true values.
        y_pred (np.ndarray): Array of predicted values.

    Returns:
        float: RMSLE value.
    """
    # Ensure no negative values and add 1 to avoid log(0) errors
    y_pred_adjusted = np.clip(y_pred, 0, None) + 1
    y_true_adjusted = np.clip(y_true, 0, None) + 1

    # Calculate the squared logarithmic errors
    squared_log_errors = (np.log(y_pred_adjusted) - np.log(y_true_adjusted)) ** 2

    # Calculate the mean of the squared logarithmic errors
    mean_squared_log_error = np.mean(squared_log_errors)

    # Calculate the square root of the mean squared log error
    rmsle_value = np.sqrt(mean_squared_log_error)
    
    return rmsle_value


model = LinearRegression()
model.fit(X_train, y_train)
rmsle_model = rmsle(y_test, model.predict(X_test))
print("Root Mean Squared Log Error (RMSLE):", rmsle_model)


model_2 = lgb.LGBMRegressor()
model_2.fit(X_train, y_train)
rmsle_model_2 = rmsle(y_test, model_2.predict(X_test))
print("Root Mean Squared Log Error (RMSLE):", rmsle_model_2)


# Use of GPU is recommended as training can be time consuming 
import optuna

def objective(trial):
    """
    Optuna objective function to tune XGBoost Regressor hyperparameters.
    """
    params = {
        'objective': 'reg:squarederror',  # Regression task
        'eval_metric': 'rmse',            # Evaluation metric
        'verbosity': 0,                   # Suppress XGBoost messages during tuning

        # Parameters for GPU usage
        'tree_method': 'hist',
        # Conditionally set device based on availability
        'device': 'cuda',

        # Hyperparameters to be tuned by Optuna
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.5, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0), # Fraction of samples used for training each tree
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0), # Fraction of features used for training each tree
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'gamma': trial.suggest_float('gamma', 0, 1.0), # Minimum loss reduction required to make a further partition
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True), # L2 regularization term
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),   # L1 regularization term
        'seed': 42 # For reproducibility
    }

    optuna_model = xgb.XGBRegressor(**params)

    optuna_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    rmsle_op = rmsle(y_test, optuna_model.predict(X_test))

    return rmsle_op


# optuna hyperparameters using xgboost 

#study_name = 'xgboost-regression-tuning'

#study = optuna.create_study(direction='minimize', study_name=study_name)

#print(f"\nStarting Optuna study: {study_name}")

#try:
    #study.optimize(objective, n_trials=200) 
#except Exception as e:
    #print(f"An error occurred during Optuna optimization: {e}")

#print("\nOptimization Finished!")
#print("Number of finished trials: ", len(study.trials))

#best_trial = study.best_trial
#print("Best trial:")
#print(f"  Value (RMSE): {best_trial.value}")
#print("  Params: ")
#for key, value in best_trial.params.items():
#    print(f"    {key}: {value}")


# evaluate with best params

#model_3 = xgb.XGBRegressor(**best_trial.params)
#model_3.fit(X_train, y_train)
#rmsle = rmsle(y_test, model_3.predict(X_test))
#print("Root Mean Squared Log Error (RMSLE):", rmsle)


sub = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
sub.head()


# Submission dataframe

#sub['Calories'] = np.clip(model_3.predict(df_test.drop('id',axis=1)), a_min=0.1, a_max=None)
#sub.head()


# submission csv

#sub.to_csv('predict_calories_xgb_op_fe_2.csv', index=False)




