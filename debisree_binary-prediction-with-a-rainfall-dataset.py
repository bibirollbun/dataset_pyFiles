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


#import libraries:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import shap


from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelBinarizer
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.pipeline import Pipeline


from sklearn.model_selection import RandomizedSearchCV

import optuna

from imblearn.over_sampling import SMOTE

from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,  roc_auc_score




import warnings
warnings.simplefilter("ignore")
pd.options.mode.chained_assignment = None  

pd.set_option('display.max_columns', None)


train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
print("Train shape", train.shape )


train.head()


train.dtypes


test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
print("Test shape:", test.shape )
test.head()


train.isnull().sum()


test.isnull().sum()


test['winddirection'].fillna(test['winddirection'].median(), inplace=True)


test.isnull().sum()


test.columns


train.columns


# Make 'id' index:

train.set_index('id', inplace= True)
test.set_index('id', inplace= True)


sns.histplot(data= train, x= 'day', hue = 'rainfall', bins =50, kde = True)
plt.show()


train['day'].describe()


#Target

rain = train['rainfall'].value_counts(normalize=True)

sns.barplot(x=rain.index, y=rain.values*100)
plt.xlabel('rainfall')
plt.ylabel('proportion(%)')
plt.title('Class imbalance')
plt.show()


feat_col = ['pressure', 'maxtemp', 'temparature', 'mintemp',
       'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection',
       'windspeed']


def plot_histograms_with_kde(df):
    
    target_col = 'rainfall'
    for col in feat_col:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[df[target_col] == 0][col], kde=True, color='blue', label='Target = 0', bins=30, alpha = 0.7, stat='density')
        sns.histplot(df[df[target_col] == 1][col], kde=True, color='red', label='Target = 1', bins=30, alpha = 0.3,  stat='density')
        
        plt.title(f'Distribution of {col} by {target_col}')
        plt.xlabel(col)
        plt.ylabel('Density')
        plt.legend()
        plt.show()
        plt.close()


plot_histograms_with_kde(train)


train['cloud'].describe()


train['humidity'].describe()



# Convert degrees to radians
train['winddirection'] = np.deg2rad(train['winddirection'])
test['winddirection'] = np.deg2rad(test['winddirection'])

# train['sin_wind'] = np.sin(train['winddirection'])
# train['cos_wind'] = np.cos(train['winddirection'])

# test['sin_wind'] = np.sin(test['winddirection'])
# test['cos_wind'] = np.cos(test['winddirection'])



# train['sin_day'] = np.sin(2 * np.pi * train['day'] / 365)
# train['cos_day'] = np.cos(2 * np.pi * train['day'] / 365)


# test['sin_day'] = np.sin(2 * np.pi * test['day'] / 365)
# test['cos_day'] = np.cos(2 * np.pi * test['day'] / 365)


train["cloud_humidity_product"] = train["cloud"] * train["humidity"]

test["cloud_humidity_product"] = test["cloud"] * test["humidity"]



train['effective_sunshine'] = train['sunshine'] * (1 - train['cloud']/100)
test['effective_sunshine'] = test['sunshine'] * (1 - test['cloud']/100)


def calculate_relative_humidity(temp, dewpoint):
    # Constants for the Magnus formula (assuming temperatures in Celsius)
    A = 17.625
    B = 243.04
    # Calculate the exponent parts
    numerator = np.exp((A * dewpoint) / (B + dewpoint))
    denominator = np.exp((A * temp) / (B + temp))
    RH = 100 * (numerator / denominator)
    return RH

# Apply the function to our DataFrame
train['relative_humidity'] = calculate_relative_humidity(train['temparature'], train['dewpoint'])
test['relative_humidity'] = calculate_relative_humidity(test['temparature'], test['dewpoint'])


train['dewpoint_depression'] = train['temparature'] - train['dewpoint']
test['dewpoint_depression'] = test['temparature'] - test['dewpoint']


# # Calculate the diurnal temperature range
# train['temp_range'] = train['maxtemp'] - train['mintemp']

# # Compute the average temperature from min and max (if needed)
# train['computed_mean_temp'] = (train['maxtemp'] + train['mintemp']) / 2

# # Calculate the deviation of the provided temperature from the computed mean temperature
# train['temp_deviation'] = train['temparature'] - train['computed_mean_temp']


# # Calculate the diurnal temperature range
# test['temp_range'] = test['maxtemp'] - test['mintemp']

# # Compute the average temperature from min and max (if needed)
# test['computed_mean_temp'] = (test['maxtemp'] + test['mintemp']) / 2

# # Calculate the deviation of the provided temperature from the computed mean temperature
# test['temp_deviation'] = test['temparature'] - test['computed_mean_temp']


# train['u_wind'] = train['windspeed'] * np.cos(train['winddirection'])
# train['v_wind'] = train['windspeed'] * np.sin(train['winddirection'])

# test['u_wind'] = test['windspeed'] * np.cos(test['winddirection'])
# test['v_wind'] = test['windspeed'] * np.sin(test['winddirection'])


# train['pressure_anomaly'] = train['pressure'] - train['pressure'].mean()
# train['pressure_humidity_interaction'] = train['pressure_anomaly'] * train['humidity']


# test['pressure_anomaly'] = test['pressure'] - test['pressure'].mean()
# test['pressure_humidity_interaction'] = test['pressure_anomaly'] * test['humidity']


# # # Compute correlation matrix
corr_matrix = train.corr()

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix - Features with the target")
plt.show()


test.isnull().sum()


train.columns


#Dropping:

# train.drop(['day', 'maxtemp', 'temparature', 'mintemp'], axis =1, inplace= True)
# test.drop(['day', 'maxtemp', 'temparature', 'mintemp'], axis =1, inplace= True)
# train.head()


train.columns


# train.drop(['maxtemp', 'temparature', 'windspeed'], axis=1, inplace=True)
# test.drop(['maxtemp', 'temparature', 'windspeed'], axis=1, inplace=True)
# # train.columns


# Split data (if not already done)
y = train['rainfall']
X = train.drop('rainfall', axis=1)


# # Feature Scaling (optional but can help)
# scaler = StandardScaler()
# X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# smote = SMOTE(random_state=42)
# X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


# Define cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()


# Create the model with class balancing
model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=pos_weight
)

# Define StratifiedKFold to maintain class distribution in each fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

roc_auc_list = []

# Manual cross-validation loop
for train_index, test_index in skf.split(X, y):
    X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
    
    # Fit the model on the training fold
    model.fit(X_train_fold, y_train_fold)
    
    # Make predictions on the validation fold
    y_pred = model.predict(X_test_fold)
    roc_auc = roc_auc_score(y_test_fold, y_pred)
    print("XGBoost ROC AUC:", roc_auc)
    roc_auc_list.append(roc_auc)


# Compute class imbalance ratio
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# Parameter grid for tuning
param_dist = {
    "max_depth": [3, 4, 5, 6, 7],
    "min_child_weight": [1, 3, 5],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_alpha": [0, 0.1, 1, 10],     # L1 regularization
    "reg_lambda": [1, 1.5, 2, 3],     # L2 regularization
    "learning_rate": [0.01, 0.05, 0.1, 0.3]
}

# Initialize the XGBoost classifier with class balancing
model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=pos_weight,
    random_state=42
)

# Set up RandomizedSearchCV with 5-fold CV
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=50,
    scoring="roc_auc",
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# Fit the search to training data
random_search.fit(X_train, y_train)

# Output best parameters and score
print("Best Hyperparameters:", random_search.best_params_)
print("Best ROC AUC Score:", random_search.best_score_)


# Retrieve the best hyperparameters from RandomizedSearchCV
best_params = random_search.best_params_

# Recalculate class imbalance for full train set
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# Initialize a new XGBoost model with best parameters and class balancing
best_model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    scale_pos_weight=pos_weight,
    **best_params
)

# Train the model
best_model.fit(X_train, y_train)

# Predict probabilities for ROC AUC
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Calculate ROC AUC score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print("XGBoost ROC AUC:", roc_auc)



# def objective(trial):
#     param = {
#         "max_depth": trial.suggest_int("max_depth", 3, 10),
#         "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
#         "n_estimators": trial.suggest_int("n_estimators", 50, 300),
#         "subsample": trial.suggest_float("subsample", 0.5, 1.0),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
#         "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
#         # Optionally include scale_pos_weight if needed:
#         "scale_pos_weight": np.sum(y_train_resampled == 0) / np.sum(y_train_resampled == 1)
#     }
    
#     model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, **param)
#     auc_scores = cross_val_score(model, X_train_resampled, y_train_resampled, scoring="roc_auc", cv=cv)
#     return np.mean(auc_scores)

# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=50)

# print("Best Hyperparameters:", study.best_params)





# Train final model with best parameters
# final_model = xgb.XGBClassifier(**study.best_params, use_label_encoder=False, eval_metric='logloss', random_state=42)
# final_model.fit(X_train_resampled, y_train_resampled)

# # Evaluate ROC AUC on test set
# y_pred_proba = final_model.predict_proba(X_test)[:, 1]
# roc_auc = roc_auc_score(y_test, y_pred_proba)
# print("XgBoost ROC AUC:", roc_auc)


# Initialize the SHAP explainer

explainer = shap.TreeExplainer(best_model)
shap_values = explainer(X_test) 

#  Visualize the SHAP summary plot
shap.summary_plot(shap_values, X_test, plot_type="bar")


# # Define cross-validation strategy
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# # Set up Sequential Forward Selection to select the best subset of features
# # Adjust 'n_features_to_select' as needed or set it to 'auto' if available
# sfs = SequentialFeatureSelector(final_model,
#                                 n_features_to_select='auto',
#                                 direction='forward',
#                                 scoring='roc_auc',
#                                 cv=cv,
#                                 n_jobs=-1)

# # Fit the feature selector on the resampled training data
# sfs.fit(X_train_resampled, y_train_resampled)

# # Get the selected features
# selected_features = X_train_resampled.columns[sfs.get_support()].tolist()
# print("Selected features:", selected_features)

# # Evaluate model performance using only the selected features on the resampled data
# X_selected = X_train_resampled[selected_features]
# cv_scores = cross_val_score(final_model, X_selected, y_train_resampled, scoring='roc_auc', cv=cv)
# print("Mean ROC AUC with selected features:", np.mean(cv_scores))



# # Instantiate individual models

# model_xgb = final_model
# model_rf = RandomForestClassifier(n_estimators=200, random_state=42)
# model_lr = LogisticRegression(max_iter=1000, random_state=42)
# model_lgbm = LGBMClassifier(random_state=42)



# estimators = [
#     ('xgb', final_model),        # Pre-tuned XGBoost model
#     ('rf', model_rf),          # RandomForest (can be base or tuned)
#     ('lr', model_lr),          # Logistic Regression
#     ('lgbm', model_lgbm)       # LightGBM (optional)
# ]



# stack_model = StackingClassifier(
#     estimators=estimators,
#     final_estimator=LogisticRegression(max_iter=2000, solver='liblinear', random_state=42),
#     passthrough=True,  # feeds original features to meta-model (can improve performance)
#     cv=5,
#     n_jobs=-1
# )


# param_grid = {
#     # Random Forest parameters
#     'rf__n_estimators': [100, 200, 300],
#     'rf__max_depth': [None, 10, 20],

#     # LightGBM parameters (if not yet tuned)
#     'lgbm__learning_rate': [0.01, 0.1],
#     'lgbm__n_estimators': [100, 200],

#     # Meta-model (Logistic Regression) parameters
#     'final_estimator__C': [0.01, 0.1, 1, 10],
#     'final_estimator__penalty': ['l1', 'l2']
# }



# # Define the cross-validation strategy for the RandomizedSearchCV
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# # Set up the RandomizedSearchCV
# random_search = RandomizedSearchCV(
#     estimator=stack_model,
#     param_distributions=param_grid,
#     n_iter=20,  # number of parameter settings sampled
#     scoring='roc_auc',
#     cv=cv,
#     random_state=42,
#     n_jobs=-1,
#     verbose=1
# )

# # Fit the RandomizedSearchCV on your resampled training data
# random_search.fit(X_train_resampled, y_train_resampled)

# # Output the best hyperparameters and best ROC AUC score from tuning
# print("Best parameters:", random_search.best_params_)
# print("Best ROC AUC (cv):", random_search.best_score_)



#best_model = random_search.best_estimator_


#subset the test/inference data:


# test_selected = test[selected_features]
# test_selected.shape


#Prediction
# test_scaled = pd.DataFrame(scaler.transform(test_selected), columns=test_selected.columns)
# test_pred = best_model.predict_proba(test_scaled)[:, 1]

test_pred_proba = best_model.predict_proba(test)[:, 1]



submission = pd.DataFrame({'id': test.index, 'rainfall': test_pred_proba})
submission.to_csv('/kaggle/working/submission.csv', index=False)
display(submission)




