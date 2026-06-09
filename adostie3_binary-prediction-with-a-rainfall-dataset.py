# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedShuffleSplit


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
directory = "/kaggle/input/playground-series-s5e3"
train = pd.read_csv(directory+"/train.csv")
test = pd.read_csv(directory+"/test.csv")
for x in (train, test): print(x.head())




test.columns = test.columns.str.strip()
train.drop_duplicates(inplace=True)
test['winddirection'].fillna(test['winddirection'].median(), inplace=True)
test.isna().any()


train.info()
plt.hist(train['rainfall'])
plt.title("correlations")
plt.show()
#histogram of classes
sns.heatmap(train.corr())
#heatmap of correlations
for col in list(test.columns)[2:]:
    plt.scatter(data=train, x="day", y=col, c='rainfall')
    plt.title(f"{col} over time")
    plt.show()



import numpy as np
import pandas as pd
import scipy.stats as stats  # Importing for Box-Cox and Yeo-Johnson transformations

# Define function to categorize wind direction into sectors 
def wind_sector(direction):
    if pd.isna(direction):
        return np.nan  # Preserve missing values for later handling
    direction = float(direction)
    if direction >= 315 or direction < 45:
        return 'North'
    elif direction >= 45 and direction < 135:
        return 'East'
    elif direction >= 135 and direction < 225:
        return 'South'
    else:
        return 'West'

def perform_feature_engineering(df):
    """
    Applies feature engineering to the dataframe, creating new features for weather prediction.
    """
    
    # 1. Seasonal Features using 'day' (cyclical representation of the year)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 365)

    # 2. Lagged Features (previous day's values for key predictors)
    #    Shift by 1, then fill any remaining NaNs with 0 (or a median if desired)
    df['cloud_lag1'] = df['cloud'].shift(1).fillna(0)
    df['sunshine_lag1'] = df['sunshine'].shift(1).fillna(0)
    df['humidity_lag1'] = df['humidity'].shift(1).fillna(0)

    # 3. Rolling Statistics (3-day trends for key predictors)
    #    Use rolling(window=3, min_periods=1) so the first 1-2 rows won't be NaN. Backfill if needed.
    df['cloud_roll3_mean'] = df['cloud'].rolling(window=3, min_periods=1).mean().fillna(method='bfill')
    df['sunshine_roll3_mean'] = df['sunshine'].rolling(window=3, min_periods=1).mean().fillna(method='bfill')
    df['humidity_roll3_mean'] = df['humidity'].rolling(window=3, min_periods=1).mean().fillna(method='bfill')

    # 4. Interaction Features (combinations of highly correlated features)
    df['cloud_humidity'] = (df['cloud'] * df['humidity']).fillna(0)  # Replace missing with 0
    df['sunshine_cloud_ratio'] = (df['sunshine'] / (df['cloud'] + 1e-5)).fillna(0)

    # 5. Meteorological Features
    #    Compute temperature range and pressure difference
    df['temp_range'] = (df['maxtemp'] - df['mintemp']).fillna(df['maxtemp'].median())
    df['pressure_diff'] = df['pressure'].diff().fillna(0)

    # 6. Additional Time-Based Interactions with 'day'
    df['cloud_day_sin'] = (df['cloud'] * df['day_sin']).fillna(0)
    df['sunshine_day_cos'] = (df['sunshine'] * df['day_cos']).fillna(0)
    df['humidity_roll3_day_sin'] = (df['humidity_roll3_mean'] * df['day_sin']).fillna(0)

    # 7. Categorical Feature: Wind Direction
    #    Map wind direction to bins and replace missing with 'Unknown'
    df['wind_sector'] = df['winddirection'].apply(wind_sector).fillna('Unknown')
    
    # 7.1. Wind and Cloud Interaction Features (NEW)
    #    Captures how changes in wind and cloud metrics interact.
    df['change_in_direction'] = abs(df['winddirection'] - df['winddirection'].shift(1)).fillna(0)
    df['cloud_wind_interaction'] = df['cloud'] * np.log1p(df['windspeed'])
    df['wind_cloud_interaction'] = np.log1p(df['cloud']) * df['windspeed']

    # 8. Logarithmic and Transform Features for 'cloud' variable (NEW)
    df['cloud_log'] = np.log1p(df['cloud'])  # Log transformation to handle skewness
    df['cloud_sqrt'] = np.sqrt(df['cloud'])    # Square root transformation
    # Box-Cox transformation (requires strictly positive values; add 1 to avoid zero)
    df['cloud_boxcox'], lambda_bc = stats.boxcox(df['cloud'] + 1)
    # Yeo-Johnson transformation (handles negative values as well)
    df['cloud_yeojohnson'], lambda_yj = stats.yeojohnson(df['cloud'])

    # 9. Additional Meteorological Features (NEW)
    #    Combining logarithmic transformations for pressure and dewpoint, and cloud & sunshine
    df['log_pressure_dewpoint'] = np.log1p(df['pressure']) + np.log1p(df['dewpoint'])
    df['log_cloud_sunshine'] = np.log1p(df['cloud']) + np.log1p(df['sunshine'])
    df['cloudtest'] = (df['cloud'] == 88).astype(int)  # Binary flag if cloud equals 88
    df['sin_day2'] = np.sin(2 * np.pi * df['day'] / (365 * 2))  # Alternative cyclical feature (half frequency)
    df['cos_day2'] = np.cos(2 * np.pi * df['day'] / (365 * 2))
    df['wet_bulb'] = (2/3 * df['temparature'] + 1/3 * df['dewpoint'])  # Weighted average for wet bulb temperature
    
    return df

# ----------------------
# Apply Feature Engineering to Combined Train & Test Data
# ----------------------
id_test = test['id']

# Concatenate train & test, apply transformations, then split back
full_data = pd.concat([train, test], axis=0).sort_values('id')
full_data = perform_feature_engineering(full_data)

# Split back into train & test
train = full_data[full_data['rainfall'].notna()]
test = full_data[full_data['rainfall'].isna()]
selected_features = ['windspeed','cloud', 'sunshine', 'cloud_humidity', 'sunshine_day_cos', 'log_pressure_dewpoint', 'humidity_roll3_mean', 'temp_range']

train = train[selected_features + ['rainfall']]
test = test[ selected_features ]


sns.heatmap(train.corr())
#heatmap of correlation


def remove_outliers_iqr_with_plot(data, column):
    Q1 = data[column].quantile(0.10)
    Q3 = data[column].quantile(0.90)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter the data
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
random_state = 222
for col in train.columns:
    train = remove_outliers_iqr_with_plot(train, col)
#Use minmax scaler for normalization
scaler = MinMaxScaler().set_output(transform='pandas')
scaling_train = train.drop('rainfall', axis=1)
y = train['rainfall']
print(scaler.fit(scaling_train))
scaling_train = scaler.transform(scaling_train)
test = scaler.transform(test)
#Oversampling with SMOTE
sm = SMOTE(random_state = 222)
X, y = sm.fit_resample(scaling_train, y)




import itertools

def Get_Scatters(feature_1, feature_2, axs):
    '''Plot classifier type for each feature pair to determine separability'''
    axs.scatter(x=X[feature_1], y=X[feature_2], c=y)
    axs.set_xlabel(feature_1)
    axs.set_ylabel(feature_2)
    axs.set_title(np.corrcoef(X[feature_1], X[feature_2])[1][0])
f, axs = plt.subplots(7,4, figsize = (30,10))
f.tight_layout(pad=5.0)
plt.rcParams.update({'axes.titlesize': 'large', 'axes.labelsize': 'large'})
i = 0
for subset in itertools.combinations(X.columns, 2): #plot correlation coefficients of feature pairs
    x = i % 4
    z = i % 7
    Get_Scatters(subset[0], subset[1], axs[z, x])
    i += 1



from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
def SSSCV(X, y, n_splits=5):
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=222)
    for train_idx, val_idx in sss.split(X, y):
        print(train_idx, val_idx)
        yield X.iloc[train_idx], X.iloc[val_idx], y.iloc[train_idx], y.iloc[val_idx]
best_score = -np.inf

#loop through all single features and find best score
for feature in X.columns:
    avg_scores = []
    std_scores = []
    plt.figure()
    Scores = LogisticRegressionCV(cv=5, max_iter=10000, scoring='roc_auc_ovr', random_state=random_state, solver='liblinear', Cs=10).fit(X[[feature]], y)
    print(Scores.scores_)
    averages = np.mean(Scores.scores_[1], axis=0)
    std_dev = np.std(Scores.scores_[1], axis=0)
    plt.errorbar(Scores.Cs_, averages, yerr=std_dev, fmt='-o')
    plt.xscale('log')
    plt.xlabel(feature)
    plt.ylabel('AUC')
    feature_max = np.max(averages)
    plt.title(feature_max)
    if feature_max > best_score:
        best_score = feature_max
        best_feature = feature

print(f"{best_feature} had a best score of {best_score}")




def plotLogreg2feat(X, featname_1, featname_2, model):
    '''
    Inputs:
      X - Input DataFrame (assumes Nx2 for N data points and 2 features)
      featname_1, featname_2 - String containing feature names
      model - Fitted LogisticRegressionCV model

    Outputs:
      ax - Returns figure axis object
    '''

    # make grid
    x_min, x_max = X[featname_1].min() - 0.5*X[featname_1].std(), X[featname_1].max() + 0.5*X[featname_1].std()
    y_min, y_max = X[featname_2].min() - 0.5*X[featname_2].std(), X[featname_2].max() + 0.5*X[featname_2].std()
    h = 0.02  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    #plt.figure(1, figsize=(4, 3))
    fig, ax = plt.subplots()
    ax.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    # Plot also the training points
    #plt.scatter(X_train[features[i]][ Y_train == 0 ], X_train[features[j]][ Y_train == 0 ], c='r')
    #plt.scatter(X_train[features[i]][ Y_train == 1 ], X_train[features[j]][ Y_train == 1 ], c='g')
    #plt.scatter(X_train[features[i]][ Y_train == 2 ], X_train[features[j]][ Y_train == 2 ], c='b')

    ax.scatter(X[featname_1][ y == 0 ], X[featname_2][ y == 0 ], c='r')
    ax.scatter(X[featname_1][ y == 1 ], X[featname_2][ y == 1 ], c='g')
    ax.scatter(X[featname_1][ y == 2 ], X[featname_2][ y == 2 ], c='b')

    ax.set_xlabel(featname_1)
    ax.set_ylabel(featname_2)


    #plt.xlim(xx.min(), xx.max())
    #plt.ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    return ax


def get_subsets():
  '''Train and plot classifier results for each feature pairing'''
  for subset in itertools.combinations(X.columns, 2):
    Scores = LogisticRegressionCV(cv=5, max_iter=10000, penalty = 'l2', scoring='roc_auc_ovr', random_state=random_state, solver='liblinear', Cs=10).fit(X[[subset[0], subset[1]]], y)
    averages = np.mean(Scores.scores_[1.0], axis=0)
    if np.max(averages)>best_score:
        plotLogreg2feat(X, subset[0], subset[1], Scores)
        plt.title(np.max(averages))

get_subsets()
auc_scores = []
for i, (X_train, X_val, y_train, y_val) in enumerate(SSSCV(X, y)):
    LR = LogisticRegression(penalty='l2', solver='liblinear', C=10, class_weight=None).fit(X_train, y_train)
    y_val_proba = LR.predict_proba(X_val)[:, 1]
    print(y_val_proba)
    y_val_pred = (y_val_proba >= 0.5).astype(int)
    
    # Compute ROC-AUC
    fold_auc = roc_auc_score(y_val, y_val_proba)
    
    # Print metrics
    print(f"Fold {i} - ROC-AUC: {fold_auc:.4f}")
    print(classification_report(y_val, y_val_pred))
    print("-" * 40)
    
    auc_scores.append(fold_auc)
avg_auc = np.mean(auc_scores)
print(f"Average ROC-AUC for Logistic Regression: {avg_auc:.4f}\n")
LR.fit(X, y)
test_prob = LR.predict_proba(test)[:, 1]
print(test_prob)
output = pd.DataFrame({'id': id_test, 'rainfall': test_prob})
print(output.head())
print(output.value_counts())
output.to_csv('/kaggle/working/submission.csv', index=False)


