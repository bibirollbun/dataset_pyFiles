# Importing and loading necessary libraries and packages
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style = 'white', palette = 'Set2')
pal = sns.color_palette('Set2')

import catboost
from catboost import Pool, CatBoostClassifier
from catboost.utils import eval_metric

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score, classification_report
from sklearn.preprocessing import MinMaxScaler

import xgboost as xgb
xgb.set_config(verbosity=0)

import optuna
from optuna.samplers import TPESampler

import shap
shap.initjs()

import lightgbm
from lightgbm import plot_importance

import warnings
warnings.filterwarnings('ignore')


# Loading in the datasets
df_train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
df_original = pd.read_csv("/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv")


# Viewing first 5 entries of 'df_train'
df_train.head()


# Looking at the info of 'df_train'
df_train.info()


# Creating a function to show a summary of a given dataset
def show_sum(df):
    sum_df = pd.DataFrame(index = list(df))
    sum_df['Dtype'] = df.dtypes
    sum_df['Count'] = df.count()
    sum_df['#Unique'] = df.nunique()
    sum_df['%Unique'] = sum_df['#Unique'] / len(df) * 100
    sum_df['#Null'] = df.isnull().sum()
    sum_df['%Null'] = sum_df['#Null'] / len(df) * 100
    print(sum_df)


# Examining summary of 'df_train'
show_sum(df_train)


# Examining summary statistics of each column in 'df_train'
df_train.describe()


# Viewing first 5 entries of 'df_test'
df_test.head()


# Looking at the info of 'df_test'
df_test.info()


# Examining summary of 'df_test'
show_sum(df_test)


# Examining summary statistics of each column in 'df_test'
df_test.describe()


# Creating countplot for target variable 'rainfall'
ax = sns.countplot(x='rainfall', data=df_train, palette='Set2')
for label in ax.containers:
  ax.bar_label(label)
ax.set_ylabel('Count')
ax.set_xlabel('Rainfall')
ax.set_title('Rainfall Outcome Distribution')
ax.set_ylim(0, 1800)
plt.show()


# Creating donut chart for target variable 'rainfall'
plt.plot(1, 2, 1)
df_train['rainfall'].value_counts().plot.pie(
        autopct='%1.1f%%', colors=['lightpink', 'lightblue'], wedgeprops=dict(width=0.3), startangle=180)
plt.ylabel(None)
plt.title('Rainfall Outcome Percentage')
plt.show()


# Visualizing distributions of numerical features (histograms with KDE line)
fig, ax = plt.subplots(6, 2, figsize = (15, 25), dpi = 300)
ax = ax.flatten()
data_numerical = df_train.drop(['rainfall'], axis=1)
features = data_numerical.columns

for i, column in enumerate(features):

    sns.histplot(df_train[column], ax=ax[i], color=pal[0], fill=True, kde=True, bins=30)
    sns.histplot(df_test[column], ax=ax[i], color=pal[2], fill=True, kde=True, bins=30)
    ax[i].set_title(f'{column}', size = 14)
    ax[i].set_xlabel(None)

fig.suptitle('Distributions of Numerical Features', fontsize = 24, fontweight = 'bold')
plt.tight_layout()


# Visualizing correlation heatmap 
plt.figure(figsize=(14,10))
corr=df_train.corr()
sns.heatmap(corr,annot=True,cmap='coolwarm',mask=np.triu(corr), linewidths=0.5, fmt=',.2f')
plt.suptitle('Correlation Heatmap', fontsize=16, fontweight='bold')
plt.show()


# Visualizing pairplot of 'df_train'
sns.pairplot(df_train.drop('id', axis=1), hue='rainfall', corner=True)
plt.suptitle('Pairplot of Training Data')
plt.show()


# Visualizing violin plot of 'rainfall' & 'humidity'
plt.figure(figsize=(10, 6))
sns.violinplot(data=df_train, x='rainfall', y='humidity')
plt.title('Rainfall & Humidity')
plt.xlabel('Rainfall (No: 0, Yes: 1)')
plt.ylabel('% Humidity')
plt.show()


# Visualizing violin plot of 'rainfall' & 'cloud'
plt.figure(figsize=(10, 6))
sns.violinplot(data=df_train, x='rainfall', y='cloud')
plt.title('Rainfall & Cloud Coverage')
plt.xlabel('Rainfall (No: 0, Yes: 1)')
plt.ylabel('% Cloud Coverage')
plt.show()


# Visualizing violin plot of 'rainfall' & 'sunshine'
plt.figure(figsize=(10, 6))
sns.violinplot(data=df_train, x='rainfall', y='sunshine')
plt.title('Rainfall & Hours of Sunshine')
plt.xlabel('Rainfall (No: 0, Yes: 1)')
plt.ylabel('Hours of Sunshine')
plt.show()


# Viewing the first 5 entries of 'df_original'
df_original.head()


# Looking at the info of 'df_original'
df_original.info()


# Viewing column names of 'df_original'
df_original.columns


# Removing spaces in column names
df_original.columns = df_original.columns.str.strip()


# Converting 'rainfall' column in 'df_original' to int (yes=1, no=0)
df_original['rainfall'] = df_original['rainfall'].map({'yes': 1, 'no': 0})


# Verifying changes make to 'rainfall' column
df_original['rainfall']


# Declaring feature columns
features = (['day', 'pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint',
            'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed'])


# Creating new target column 'label', where all train set samples are labeled with 0, and all orignal set samples with 1
df_train['label'] = 0
df_original['label'] = 1
target = 'label'

# Combinging features and target label into 'all_cols'
all_cols = features + [target]


# Checking out the shape of 'all_cols' in train and original datasets
df_train[all_cols].shape, df_original[all_cols].shape


# Defining a function to create adversarial data: combines, shuffles, and re-splits the two datasets
# The resulting datasets include a mixture of the train and orignal data
def create_adversarial_data(df_train, df_original, cols, N_val=2000):
    df_master = pd.concat([df_train[cols], df_original[cols]], axis=0)
    adversarial_test = df_master.sample(N_val, replace=False)
    adversarial_train = df_master[~df_master.index.isin(adversarial_test.index)]
    return adversarial_train, adversarial_test

# Applying function to train and orignal data, checking out the resulting shapes
adversarial_train, adversarial_test = create_adversarial_data(df_train, df_original, all_cols)
adversarial_train.shape, adversarial_test.shape


# Setting up the Catboost model for adversarial validation
train_data = Pool(
    data=adversarial_train[features],
    label=adversarial_train[target]
)
holdout_data = Pool(
    data=adversarial_test[features],
    label=adversarial_test[target]
)

# Establishing parameters for the Catboost classifier
params = {
    'iterations': 100,
    'eval_metric': 'AUC',
    'od_type': 'Iter',
    'od_wait': 50,
    'random_seed': 42,
    'verbose': 0
}

# Fitting the model to the data
model = CatBoostClassifier(**params)
_ = model.fit(train_data, eval_set=holdout_data)


# Setting up ROC Curve plot
def plot_roc(y_trues, y_preds, labels, x_max=1.0):
    fig, ax = plt.subplots()
    for i, y_pred in enumerate(y_preds):
        y_true = y_trues[i]
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        ax.plot(fpr, tpr, label='%s; AUC=%.3f' % (labels[i], auc), marker='o', markersize=1)

    ax.legend()
    ax.grid()
    ax.plot(np.linspace(0, 1, 20), np.linspace(0, 1, 20), linestyle='--')
    ax.set_title('ROC curve')
    ax.set_xlabel('False Positive Rate')
    ax.set_xlim([-0.01, x_max])
    _ = ax.set_ylabel('True Positive Rate')
    
# Plotting
plot_roc(
    [holdout_data.get_label()],
    [model.predict_proba(holdout_data)[:,1]],
    ['Baseline']
)


# Defining function to plot feature importance
def plot_importances(model, holdout_data, features):
    shap_values = model.get_feature_importance(holdout_data, type='ShapValues')
    expected_value = shap_values[0,-1]
    shap_values = shap_values[:,:-1]
    shap.summary_plot(shap_values, holdout_data, feature_names=features, plot_type='bar')
    
# Plotting feature importance
plot_importances(model, holdout_data, features)


# Removing 'day' and retraining the model
params2 = dict(params)
params2.update({"ignored_features": ['day']})
model2 = CatBoostClassifier(**params2)
_ = model2.fit(train_data, eval_set=holdout_data, plot=False, verbose=False)


# Plotting updated ROC Curve plot
plot_roc(
    [holdout_data.get_label()]*2,
    [model.predict_proba(holdout_data)[:,1], model2.predict_proba(holdout_data)[:,1]],
    ['Baseline', "Removing 'day'"]
)


# Re-loading in training and testing sets
df_train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv", index_col=0)
df_test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
# Re-loading in the original data
df_original = pd.read_csv("/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv")


# Extracting ids from 'df_test' for later submission
ids = df_test['id']


# Dropping 'id' column from 'df_test'
df_test.drop(['id'], axis=1, inplace=True)


# Reapplying cleaning steps to 'df_original'
df_original.columns = df_original.columns.str.strip()
df_original['rainfall'] = df_original['rainfall'].map({'yes': 1, 'no': 0})


# Combining the 'df_train' and 'df_original' datasets
df_combined = pd.concat([df_train, df_original], axis=0, ignore_index=True)


# Checking for nulls in 'df_combined'
df_combined.info()


# Filling in null values with the column mean for 'winddirection' and 'windspeed'
df_combined['winddirection'].fillna(df_combined['winddirection'].mean(), inplace=True)
df_combined['windspeed'].fillna(df_combined['windspeed'].mean(), inplace=True)


# Checking for nulls in 'df_test'
df_test.info()


# Filling in null values with the column mean for 'winddirection'
df_test['winddirection'].fillna(df_test['winddirection'].mean(), inplace=True)


# Declaring which columns should be checked for outliers
cols = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint',
       'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']

# Creating function to remove outliers by applying the Inter Quartile method
def remove_outliers(data, column):
    # Calculating the Inter Quartile Range (IQR) 
    Q1 = data[column].quantile(0.15)
    Q3 = data[column].quantile(0.85)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filtering the data, including all data between the lower and upper bounds
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    # Calculating the number of rows deleted
    rows_deleted = len(data) - len(filtered_data)
    return filtered_data, rows_deleted

# Applying the 'remove_outliers' function to each column in 'train' dataset
rows_deleted_total = 0

for column in cols:
    df_combined, rows_deleted = remove_outliers(df_combined, column)
    rows_deleted_total += rows_deleted
    print(f"{column}: {rows_deleted}")

print(f"Total rows deleted: {rows_deleted_total}")


# This code chunk has been adapted from a great work by Sanad Alali (Kudos!)

# Creating a function to create new features 
def create_features(df):

    # Creating a copy of the original dataframe
    df_new = df.copy()
    
    # Creating temperature-based features
    df_new['temp_range'] = df_new['maxtemp'] - df_new['mintemp']
    df_new['temp_ratio'] = df_new['temparature'] / df_new['maxtemp']
    df_new['temp_from_dewpoint'] = df_new['temparature'] - df_new['dewpoint']
    df_new['max_min_temp_ratio'] = df_new['maxtemp'] / df_new['mintemp']
    
    # Creating humidity-based features
    df_new['humid_temp_interaction'] = df_new['humidity'] * df_new['temparature']
    df_new['humid_pressure_interaction'] = df_new['humidity'] * df_new['pressure']
    df_new['humid_dewpoint_interaction'] = df_new['humidity'] * df_new['dewpoint']
    
    # Creating cloud-based features
    df_new['cloud_sunshine_ratio'] = df_new['cloud'] / (df_new['sunshine'] + 1)  # Adding 1 to avoid division by zero
    df_new['cloud_coverage_rate'] = df_new['cloud'] / 100  # Normalize to 0-1 range
    
    # Creating pressure-based features
    df_new['pressure_temp_ratio'] = df_new['pressure'] / df_new['temparature']
    df_new['pressure_humidity_ratio'] = df_new['pressure'] / df_new['humidity']
    
    # Creating combined weather features
    df_new['weather_severity'] = (df_new['cloud'] * df_new['humidity']) / (df_new['pressure'] * (df_new['sunshine'] + 1))
    df_new['temp_humidity_index'] = (df_new['temparature'] * df_new['humidity']) / 100
    df_new['pressure_temp_humidity'] = (df_new['pressure'] * df_new['temparature']) / df_new['humidity']
    
    return df_new

# Applying 'create_features' function to train and test datasets
df_combined = create_features(df_combined)
df_test = create_features(df_test)


# Initializing the MinMaxScaler
scaler = MinMaxScaler()

# Fitting the 'df_combined' data to the scaler
train_scaled = scaler.fit_transform(df_combined)
# Scaling the data
train = pd.DataFrame(train_scaled, columns=df_combined.columns)


# Viewing scaling changes to 'train'
train.head()


# Applying scaling steps to 'df_test'
test_scaled = scaler.fit_transform(df_test)
test = pd.DataFrame(test_scaled, columns=df_test.columns)


# Viewing scaling changes to 'test'
test.head()


# Splitting 'train' dataset into features (X) and target (y)
X = train.drop(['rainfall'], axis=1)
y = train['rainfall']
# Splitting data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Fitting the baseline model to the training data
baseline_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc', random_state=42)
baseline_model.fit(X_train, y_train)


# Evaluating the baseline model's performance (auc score) on the validation data
y_val_pred_proba = baseline_model.predict_proba(X_valid)[:, 1]
auc_score = roc_auc_score(y_valid, y_val_pred_proba)
print(f'Baseline Model ROC-AUC Score: {auc_score:.4f}')


# Plotting the AUC-ROC curve of the baseline model
fpr, tpr, thresholds = roc_curve(y_valid, y_val_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("AUC-ROC: Baseline Model")
plt.legend()
plt.show()


# Plotting feature importance of the baseline model
xgb.plot_importance(baseline_model, max_num_features=12)


# Creating 'objective' function which will trial different parameter values and combinations
def objective(trial):
    params = {
        'objective': 'binary:logistic',
        'device': 'cpu',
        'eval_metric': 'auc',
        'random_state': 42,
        'verbosity': 1,
        'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'max_depth': trial.suggest_int('max_depth', 1, 15),
        'subsample': trial.suggest_float('subsample', 0.25, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),  
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True) 
    }

    # Fitting XGBoost model with parameters from the trials
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)

    # Making predictions on the validation set
    y_pred_proba = model.predict_proba(X_valid)[:, 1]
    score = roc_auc_score(y_valid, y_pred_proba)
    print('ROC-AUC:', score)
    return score

# When set to 1, optuna will create a study to find the optimal parameters for the model
run=0

if run==1:

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    print('Best trial:')
    trial = study.best_trial

    print('Value: {}'.format(trial.value))
    print('Params: ')
    for key, value in trial.params.items():
        print(' {}: {}'.format(key, value))


# Recording best parameters from trial #1
best_params = {
    'objective': 'binary:logistic',
    'device': 'cpu',
    'eval_metric': 'auc',
    'random_state': 42,
    'n_estimators': 727,
    'learning_rate': 0.02065809145104903,
    'colsample_bytree': 0.6844525799009806,
    'max_depth': 1,
    'subsample': 0.519006802168094,
    'min_child_weight': 5,
    'reg_alpha': 0.00015480424761672249,
    'reg_lambda': 0.2256543295563229
    }


# Recording best parameters from trial #2
best_params_2 = {
    'objective': 'binary:logistic',
    'device': 'cpu',
    'eval_metric': 'auc',
    'random_state': 42,
    'n_estimators': 528,
    'learning_rate': 0.010070446663886614,
    'colsample_bytree': 0.3755118358739974,
    'max_depth': 9,
    'subsample': 0.43796953742273803,
    'min_child_weight': 6,
    'reg_alpha': 2.01368738947277e-08,
    'reg_lambda': 9.989919055463442e-08
    }


# Fitting the model with the best parameters!
final_model = xgb.XGBClassifier(**best_params)


# Training the model
final_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=40, verbose=False)


# Making predictions on the validation set
y_val_pred_proba = final_model.predict_proba(X_valid)[:, 1]  
y_val_pred = (y_val_pred_proba >= 0.5).astype(int)  


# Evaluating the performance of the final model
auc_score = roc_auc_score(y_valid, y_val_pred_proba)
print(f'Validation ROC-AUC Score: {auc_score:.4f}')


# Plotting the AUC-ROC curve of the final model
fpr, tpr, thresholds = roc_curve(y_valid, y_val_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("AUC-ROC: Final Model")
plt.legend()
plt.show()


# Plotting feature importance of the final model
xgb.plot_importance(final_model, max_num_features=12)


# Making predictions on the test data
preds = final_model.predict_proba(test)[:,1]


# Creating 'submission' dataframe to store predictions with ids
submission = pd.DataFrame({'id': ids, 'rainfall': preds})
submission


# Creating .csv file for submissions and scoring
run = 1

if run == 1:
    submission.to_csv('submission.csv', index=False)

