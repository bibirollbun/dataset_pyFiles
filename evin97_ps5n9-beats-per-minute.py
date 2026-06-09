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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv',index_col =0)
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv',index_col =0)
sub = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')


def eda(df):
  print('head of dataset\n')
  print(df.head(10))
  print('\n description about dataset\n')
  print(df.describe())
  print('\n info about dataset\n')
  print(df.info())
  print('\n shape of dataset\n')
  print(df.shape)
  print('\n columns of dataset\n')
  print(df.columns)
  print('\n null values \n')
  print(df.isnull().sum())
  print('')

eda(train)
print("---------------------------------------------------------\n about test dataset")
eda(test)


num_features = train.select_dtypes(include=[np.number]).columns.to_list()

for feature in num_features:
    plt.figure(figsize=(4,3))
    plt.hist(train[feature], bins=30, edgecolor='black', color='skyblue')
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.show()


import seaborn as sns

plt.figure(figsize =(10,6))
sns.heatmap(train[num_features].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

print("\n--- Correlation with Target ---")
print(train.corr()["BeatsPerMinute"].sort_values(ascending=False))


for col in num_features:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=train[col])
    plt.title(f"Outliers in {col}")
    plt.show()


def remove_outlier(df,col,iqr=1.5):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3-Q1

    lower_bound = Q1 -(iqr*IQR)
    upper_bound = Q3 +(iqr*IQR)
    #capping or winsorization
    df.loc[df[col]<lower_bound,col]=lower_bound
    df.loc[df[col]>upper_bound,col] = upper_bound
    return df


# outlier removal
'''
BeatsPerMinute               1.000000
MoodScore                    0.007059
TrackDurationMs              0.006637
RhythmScore                  0.005440
VocalContent                 0.004876
LivePerformanceLikelihood    0.003471
InstrumentalScore            0.001900
AcousticQuality             -0.000820
AudioLoudness               -0.003327
Energy                      -0.004375
'''

train =remove_outlier(train,'RhythmScore')
train =remove_outlier(train,'MoodScore')
train =remove_outlier(train,'TrackDurationMs')
train =remove_outlier(train,'VocalContent')
train =remove_outlier(train,'LivePerformanceLikelihood')

test=remove_outlier(test,'RhythmScore')
test=remove_outlier(test,'MoodScore')
test=remove_outlier(test,'TrackDurationMs')
test=remove_outlier(test,'VocalContent')
test=remove_outlier(test,'LivePerformanceLikelihood')


# lgbm baseline model :-> version 1   26.38
X = train.drop('BeatsPerMinute',axis=1)
y = train['BeatsPerMinute']

kf = KFold(n_splits= 5,shuffle = True ,random_state=42)
y_test_prediction =np.zeros(len(test)) #stores test predictions


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold # Add KFold for clarity if needed

# Assume 'train' and 'test' DataFrames are loaded and have had outlier handling applied
# using the 'replace_outliers' function (winsorization).

# Step 1: Define features (X) and target (y)
# Ensure these features are consistent throughout your process.
features = ['RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
            'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
            'TrackDurationMs', 'Energy']

X_train = train[features]
y_train = train['BeatsPerMinute']

# Prepare X_test for final prediction (using the preprocessed 'test' DataFrame)
X_test_final = test[features] # Assuming 'test' is your preprocessed test DataFrame


# Step 2: Perform Cross-Validation to estimate performance (RMSE)
lin_reg_cv = LinearRegression() # Create a new instance for CV to avoid confusion if lin_reg was trained elsewhere
mse_scores = -cross_val_score(lin_reg_cv, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
rmse_scores = np.sqrt(mse_scores)

average_rmse = np.mean(rmse_scores)
std_dev_rmse = np.std(rmse_scores)

print(f"Estimated average RMSE from cross-validation: {average_rmse:.2f}")
print(f"Standard deviation of RMSE: {std_dev_rmse:.2f}")

# Step 3: Train the final model on the entire training data
# Use a fresh model instance or the one from CV if you haven't done anything else with it.
final_lin_reg_model = LinearRegression() 
final_lin_reg_model.fit(X_train, y_train)

# Step 4: Make predictions on the preprocessed test set
y_test_prediction = final_lin_reg_model.predict(X_test_final)

# Step 5: Create the submission file
# 'sub' should be a DataFrame with the correct index for submission.
# If 'sub' is just the original 'test' DataFrame structure, ensure its index matches 'test_final'.
sub = pd.DataFrame({'BeatsPerMinute': y_test_prediction}, index=X_test_final.index) # Use X_test_final's index
sub.to_csv('linear_regression_submission.csv', index=False)

# The KFold setup (kf = KFold(...)) is not directly used for a single Linear Regression model's prediction/submission,
# but would be used if you were implementing a more complex cross-validation strategy like k-fold stacking.
# If that's your intention, you would integrate it within a loop to train multiple models and average their predictions.



#lists for error of each fold
'''fold_rmses =[]
fold_maes = []
fold_r2s =[]

for folds, (train_idx, val_idx) in enumerate(kf.split(X,y)):
    print(f"Training fold {folds+1}/5>>>")
    X_train,y_train = X.iloc[train_idx],y.iloc[train_idx]
    X_val,y_val = X.iloc[val_idx],y.iloc[val_idx]

    model = lgb.LGBMRegressor(
        objective='regression',
        n_estimators = 10000,
        learning_rate = 0.005,
        num_leaves = 100,
        max_depth =7,
        min_child_samples=9,
        subsample=0.8,
        colsample_bytree=0.5,
        reg_alpha=0.79,
        reg_lambda=3.0,
        max_bin=4523,
        random_state=42,
        verbosity=-1
    )

    model.fit(
         X_train,y_train,
         eval_set =[(X_val,y_val)],
         eval_metric = 'rmse',
         callbacks=[
            lgb.early_stopping(500),
            lgb.log_evaluation(period=500)
        ])
    y_val_predict = model.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val,y_val_predict))
    mae = mean_absolute_error(y_val,y_val_predict)
    r2 = r2_score(y_val,y_val_predict)
    
    print(f"Fold {folds+1} Validation RMSE: {rmse:.4f}")
    print(f"Fold {folds+1} Validation MAE: {mae:.4f}")
    print(f"Fold {folds+1} Validation R-squared: {r2:.4f}")

    fold_rmses.append(rmse)
    fold_maes.append(mae)
    fold_r2s.append(r2)
    
    y_test_prediction += model.predict(test)/5 

print(f"\nAverage RMSE across folds: {np.mean(fold_rmses):.4f}")
print(f"Average MAE across folds: {np.mean(fold_maes):.4f}")
print(f"Average R-squared across folds: {np.mean(fold_r2s):.4f}")
'''


import matplotlib.pyplot as plt
lgb.plot_importance(model,max_num_features=20)
plt.show()


sub['BeatsPerMinute'] = y_test_prediction
sub.to_csv('lgbm_submission.csv',index=False)
print(sub)


''' version 1 with optimisation                 accuracy : 26.46
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import optuna

# Assuming 'train' and 'test' DataFrames are already defined
X = train.drop('BeatsPerMinute', axis=1)
y = train['BeatsPerMinute']

# Function to be optimized by Optuna
def objective(trial, X, y):
    # Define the hyperparameter search space using the trial object
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'n_estimators': trial.suggest_int('n_estimators', 2000, 20000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.005, 0.05),
        'num_leaves': trial.suggest_int('num_leaves', 32, 256),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 20),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 0.9),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 10.0),
        'max_bin': 4523,  # You can keep this constant if it works well
        'random_state': 42,
        'verbosity': -1,
        'n_jobs': -1
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []
    
    for train_idx, val_idx in kf.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        model = lgb.LGBMRegressor(**params)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(500, verbose=False)]
        )
        
        y_val_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        rmse_scores.append(rmse)
        
    return np.mean(rmse_scores)


# Run the Optuna optimization
print("Starting Optuna hyperparameter optimization...")
study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: objective(trial, X, y), n_trials=50) # Use more trials for better results

print("\nOptimization finished.")
print("Best hyperparameters found: ", study.best_params)
print("Best average RMSE: ", study.best_value)


# --- Train the final model with the best parameters ---
print("\nTraining final model with best hyperparameters...")
best_params = study.best_params
best_params['objective'] = 'regression'
best_params['metric'] = 'rmse'
best_params['random_state'] = 42
best_params['verbosity'] = -1
best_params['n_jobs'] = -1
best_params['max_bin'] = 4523

final_model = lgb.LGBMRegressor(**best_params)
final_model.fit(X, y)

# --- Make predictions on the test set ---
print("Making predictions on the test set...")
y_test_prediction = final_model.predict(test)
print("Predictions complete.")

'''




