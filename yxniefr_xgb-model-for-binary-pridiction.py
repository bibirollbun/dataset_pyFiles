import warnings
warnings.filterwarnings("ignore")

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna 
from sklearn.model_selection import train_test_split,GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.feature_selection import mutual_info_classif

print("Libraries imported successfully!")


# loading the training data
train_path = '/kaggle/input/playground-series-s5e3/train.csv'
train_df = pd.read_csv(train_path)

# loading the testing data
test_path = '/kaggle/input/playground-series-s5e3/test.csv'
test_df = pd.read_csv(test_path)


train_df.head(5)


train_df.tail(5)


# displaying the cols of train data,  shape 
train_df.columns, train_df.shape


# Displaying the info and description of the train data
print('Train data information:- \n')
train_df.info()

print('\nTrain data description:- ')
train_df.describe().T


# checking for missing values in the train data
train_df.isnull().sum()

# checking for zeros in the train data
train_df.isin([0]).sum()



# checiking for duplicates in the train data
train_df.duplicated().sum()


print('test data head:- \n')
test_df.head(5)


print('test data tail:- \n')
test_df.tail(5)


# checikng the cols of test data, shape
test_df.columns, test_df.shape


# checking the info and description of the test data
print('Test data information:- \n')
test_df.info()

print('\nTest data description:- ')
test_df.describe().T


# checking for missing values in the test data
test_df.isnull().sum()



# checking for zeros in the test data
test_df.isin([0]).sum()


# filling the missing values in the test data with the mean
test_df['winddirection'].fillna(test_df['winddirection'].mean(), inplace=True)


# checing for duplicates in the test data
test_df.duplicated().sum()


# creating a new temp dataframe for the train data
train_temp = train_df.copy()
test_temp = test_df.copy()
train_temp.columns


# checking if the data is imbalance or not 
plt.Figure(figsize=(8, 5))
plt.title('Count of target variable rainfall')
plt.xlabel('Rainfall')
plt.ylabel('Count')
sns.countplot(x='rainfall', data=train_temp)


# rainfall counts
rainfall_count = train_temp['rainfall'].value_counts()

# collecting negative and positive classes
minor_class = rainfall_count[0]
major_class = rainfall_count[1]

# printing values
print(f"Minor Class (No Rain): {minor_class}")
print(f"Major Class (Rain): {major_class}")


# Pie chart data
labels = ['Minority Class', 'Majority Class']
sizes = [minor_class, major_class]
colors = ['lightblue', 'orange']

# Plotting the pie chart
plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
plt.title('Distribution of Target Variable (Rainfall)')


# SETTING THE SCALE_POS_WEIGHT

SCALE_POS_WEIGHT = minor_class / major_class

(SCALE_POS_WEIGHT)


train_temp['rainfall'].value_counts()


# numerical columns
numerical_col = ['day', 'pressure', 'maxtemp', 'temparature', 'mintemp','dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection','windspeed']

COLS = 3
ROWS = math.ceil(len(numerical_col)/COLS)

hist_param = {
    'kde': True,
    'bins': 'auto',
    'stat': 'percent'
}

# plotting the numerical columns
fig, ax = plt.subplots(ROWS, COLS, figsize=(30, 20))
ax = ax.ravel()

for i, col in enumerate(numerical_col):
    sns.histplot(x = col, ax=ax[i], **hist_param, hue='rainfall', data=train_temp)
    ax[i].set_title(f'{col} Distribution', fontsize=15)
    ax[i].set_xlabel(None, fontsize=16)  
    ax[i].set_ylabel(None, fontsize=16)

fig.suptitle(f'Numerical Features Distributions\n\n\n', ha='center', fontweight='bold', fontsize=25, y=0.93)


# correlation matrix
corr = train_temp.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(corr, annot=True, cmap='Purples', fmt='.2f', vmin=-1, vmax=1, square=True,)
plt.title('Correlation Matrix', fontsize=20)


def detect_outliers_iqr(df, columns):
    outlier_summary = []

    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = len(outliers)

        outlier_summary.append({
            'Column': col,
            'Lower Bound': lower_bound,
            'Upper Bound': upper_bound,
            'Outlier Count': outlier_count
        })

    outlier_df = pd.DataFrame(outlier_summary)
    return outlier_df


outlier_summary = detect_outliers_iqr(train_temp, numerical_col)

print('Outlier Summary:- \n')
outlier_summary


train_df.columns


# def get_season(month):
#     if month in [12, 1, 2]:
#         return 0 
#     elif month in [3, 4, 5]:
#         return 1 
#     elif month in [6, 7, 8]:
#         return 2 
#     else:
#         return 3 

def new_features(data):

    # data['day'] = pd.to_datetime(data['day'])
    # data['month'] = data['day'].dt.month
    # data['season'] = data['month'].apply(get_season)
    # data['day_of_week']=data['day'].dt.weekday
    # data['is_weekend'] =data['day_of_week'].isin([5,6]).astype(int)

    # data['season'] = data['day'].apply(get_season)
    # data['day_of_year'] = data['day'].dt.dayofyear
    # data['sin_day'] = np.sin(2 * np.pi * data['day_of_year'] / 365)
    # data['cos_day'] = np.cos(2 * np.pi * data['day_of_year'] / 365)

    data['temp_range'] = data['maxtemp'] - data['mintemp']
    data['temp_variablity'] = data[['maxtemp', 'mintemp']].std(axis=1)
    data['temp_dev_from_min'] = data['temparature'] - data['mintemp']
    data['temp_dev_from_max'] = data['maxtemp'] - data['temparature']
    data['temp_total_avg'] = data[['maxtemp', 'mintemp', 'temparature']].mean(axis=1)
    data['temp_max_min_ratio'] = data['maxtemp'] / data['mintemp']

    # data['wind_effect'] = data['windspeed'] * data['winddirection'] 
    
    data['cloud_per_sunshine'] = data['cloud'] / (data['sunshine'] + 1)

    data['humidity_temp_interaction'] = data['humidity']*data['temparature']
    data['humidity_cloud_interaction'] = data['humidity'] * data['cloud']
    data['humidity_dewpoint_interaction'] = data['humidity'] * data['dewpoint']
    
    # wind direction in radians 
    # data['winddirection_rad'] = np.radians(data['winddirection'])
    # data['wind_dir_sin'] = np.sin(data['winddirection_rad'])
    # # data['wind_dir_cos'] = np.cos(data['winddirection_rad'])
    # data.drop('winddirection_rad', axis=1, inplace=True)

    # moving average 
    data['MA_temp'] = data['temp_total_avg'].rolling(window=8).mean()
    # data['MA_windspeed'] = data['windspeed'].rolling(window=8).mean()
    data['MA_humidity'] = data['humidity'].rolling(window=8).mean()

    # lag features
    data['lag_temp'] = data['temp_total_avg'].shift(1)
    # data['lag_humidity'] = data['humidity'].shift(1)
    # data['lag_windspeed'] = data['windspeed'].shift(1)
    
    return data



train_df = new_features(train_df)
test_df = new_features(test_df)


# I'm adding new features to train_temp to check mutual information score
train_temp = new_features(train_temp)

# handling missing values
train_temp = train_temp.apply(lambda x: x.fillna(x.mean()), axis=0)

# 'train_temp' is your dataframe and 'rainfall' is your target column
train_temp_features = train_temp.drop(columns=['rainfall', 'id'], axis=1)  # Features
train_temp_target = train_temp['rainfall']  # Target variable


# # checikng the mutual information scores
mi_scores = mutual_info_classif(train_temp_features, train_temp_target)

# # Create a DataFrame for better visualization
mutual_info_df = pd.DataFrame(mi_scores, index=train_temp_features.columns, columns=['Mutual Information'])
mutual_info_df = mutual_info_df.sort_values(by='Mutual Information', ascending=False)

# # print the mi_scores
print(mutual_info_df)


train_df.columns


new_temp_col = [col for col in train_df.columns if col.startswith('temp_')]

fig, ax = plt.subplots(3,2, figsize=(20, 15))
ax = ax.ravel()


for i, col in enumerate(new_temp_col):
    sns.histplot(x = col, ax=ax[i], data=train_df, kde=True)
    ax[i].set_title(f'{col} Distribution', fontsize=15)
    ax[i].set_xlabel(None, fontsize=16)
    ax[i].set_ylabel(None, fontsize=16)


# dropping some cols
drop_col = ['id']
train_df.drop(drop_col, axis=1, inplace=True)
test_df.drop(drop_col, axis=1, inplace=True)


train_df.fillna(train_df.mean(), inplace=True)
test_df.fillna(test_df.mean(), inplace=True)


# selecting the features and target variable
X = train_df.drop('rainfall', axis=1)
y = train_df['rainfall']


X.shape, y.shape


# splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# XGBoost Model
xgb_model = XGBClassifier(scale_pos_weight=SCALE_POS_WEIGHT)



# defining stratified k-folds for cv
skf = StratifiedKFold(n_splits = 5, random_state=10, shuffle=True)

# checking cv_scores
cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=skf, scoring='roc_auc')

# printing scores
print(f"CV AUC Scores: {cv_scores}")
print(f"Mean AUC: {np.mean(cv_scores)}")


xgb_model.fit(X_train,y_train)



xgb_pred = xgb_model.predict(X_test)

# checking the roc_auc score on test data
score = roc_auc_score(y_test, xgb_pred)

print(f"ROC-AUC score on test data: {score}")


# optimized hyperparameter tunning with optuna

def objective(trial):
    # params for tuning
    params ={
        'n_estimators': trial.suggest_int('n_estimators', 100,600),
        'eta': trial.suggest_float('eta', 0.01,0.2),
        'gamma': trial.suggest_float('gamma', 0,2),
        'lambda': trial.suggest_float('lambda', 0.2,0.6),
        'alpha': trial.suggest_float('alpha', 0.5,1),
        'eval_metric': 'auc'
    }

    # model
    model = XGBClassifier(**params, scale_pos_weight=SCALE_POS_WEIGHT)

    # CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=skf, scoring='roc_auc').mean()

    return score



study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials = 100)


best_params = study.best_params
best_params


xgb_final = XGBClassifier(scale_pos_weight=SCALE_POS_WEIGHT, random_state=19, **best_params)

xgb_final.fit(X_train, y_train)

y_pred_proba = xgb_final.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f'ROC_AUC score is {roc_auc}')


xgb_final.fit(X, y)


# storing ID from the test data copy
test_id = test_temp['id']
test_id


# making predictions
test_predictions = xgb_final.predict_proba(test_df)[:, 1]

# creating a dataframe for the submission
submission = pd.DataFrame({'id': test_id, 'rainfall': test_predictions})

# Save to CSV
submission.to_csv('submission.csv', index=False)

