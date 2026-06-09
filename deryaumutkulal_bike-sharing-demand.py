# Install the AutoGluon library, which is used for automatic machine learning (AutoML).
!pip install autogluon

# If needed, install AutoGluon with a specific version of scikit-learn. Uncomment if you face version issues.
# !pip install autogluon scikit-learn==1.5.2

# Install Packages and Libraries

# Statistical functions
from scipy import stats

# Visualize missing values in the dataset
import missingno as msno

# For probability plots
import pylab

# For working with dates (like getting day names)
import calendar

# Numerical computing
import numpy as np

# Data manipulation and analysis
import pandas as pd 

# AutoML library for training and evaluating models automatically
from autogluon.tabular import TabularPredictor

# Data visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Ignore warnings to keep the output clean
import warnings
warnings.filterwarnings("ignore")

# For working with datetime objects
from datetime import datetime

# Avoids warnings related to chained assignments in pandas
pd.options.mode.chained_assignment = None

# Enables inline plotting in Jupyter notebooks
%matplotlib inline

# Lists all the files in the '/kaggle/input' directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Load the training dataset into a pandas DataFrame
train = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv")

# Load the test dataset into a pandas DataFrame
test = pd.read_csv("/kaggle/input/bike-sharing-demand/test.csv")

# Load the sample submission file (used to format our predictions correctly)
submission = pd.read_csv("/kaggle/input/bike-sharing-demand/sampleSubmission.csv")


# Display the first 5 rows of the training dataset to get a quick look at the data
train.head()


# Show the data types of each column
train.dtypes


# Display basic statistical details like mean and standard deviation values for each numerical column
train.describe()


# Create histograms for all numerical features to visualize their distributions
train.hist();


# Show how many unique values exist in each column
train.apply(lambda x: len(x.unique()))


# Drop the 'casual' and 'registered' columns from the training set
# Since the test dataset does not have these columns and we want to take a quick look, we try to remove these columns before doing EDA.
train_for_autogluon = train.drop(["casual", "registered"], axis=1)

# Create and train an AutoGluon TabularPredictor model
predictor = TabularPredictor(label="count", eval_metric="rmse").fit(
    train_for_autogluon, 
    time_limit=600,
    presets="best_quality"
)


# Display a summary of the training process
predictor.fit_summary()


# Plot a bar chart of the validation scores (e.g., RMSE) for each model trained by AutoGluon
predictor.leaderboard(silent=True).plot(kind="bar", x="model", y="score_val");


# Display the name of the best-performing model chosen by AutoGluon
predictor.model_best


# Use the trained AutoGluon model to predict bike rental counts on the test dataset
predictions = predictor.predict(test)

# Show the first few predictions
predictions.head()


# Check if there are any negative predictions
print(predictions[predictions < 0])

# Print the total of all negative predictions
print(predictions[predictions < 0].sum())


# Create a submission DataFrame by copying the sample submission format
first_submission = submission

# Replace the placeholder 'count' column with our actual predictions
first_submission["count"] = predictions

# Display the first few rows of the submission file
first_submission.head()


# Save the submission DataFrame as a CSV file
first_submission.to_csv("first_submission.csv", index=False)


# Convert 'datetime' column to actual datetime format
train_for_autogluon['datetime'] = pd.to_datetime(train_for_autogluon['datetime'])
test['datetime'] = pd.to_datetime(test['datetime'])

# Extract day, month, and hour from the datetime â€” useful time-based features
train_for_autogluon["day"] = train_for_autogluon["datetime"].dt.day
train_for_autogluon["month"] = train_for_autogluon["datetime"].dt.month
train_for_autogluon["hour"] = train_for_autogluon["datetime"].dt.hour
test["day"] = test["datetime"].dt.day
test["month"] = test["datetime"].dt.month
test["hour"] = test["datetime"].dt.hour

# Convert 'season' and 'weather' columns to categorical data types
# This helps AutoGluon understand that these are not continuous numeric values but categories
train_for_autogluon["season"] = train_for_autogluon["season"].astype("category")
train_for_autogluon["weather"] = train_for_autogluon["weather"].astype("category")
test["season"] = test["season"].astype("category")
test["weather"] = test["weather"].astype("category")


# Display the first 2 rows of the train_for_autogluon file
train_for_autogluon.head(2)


# Create and train an AutoGluon TabularPredictor model
predictor_new_features = TabularPredictor(label="count", eval_metric="root_mean_squared_error").fit(
    train_for_autogluon, 
    time_limit=600,
    presets="best_quality"
)

# Display a summary of the training process
predictor_new_features.fit_summary()


# Plot a bar chart of the validation scores (e.g., RMSE) for each model trained by AutoGluon
predictor.leaderboard(silent=True).plot(kind="bar", x="model", y="score_val");


# Use the trained AutoGluon model to predict bike rental counts on the test dataset
predictions_new_features = predictor_new_features.predict(test)

# Check if there are any negative predictions
print(predictions_new_features[predictions_new_features < 0])

# Create a submission DataFrame by copying the sample submission format
submission_new_features = submission

# Replace the placeholder 'count' column with our actual predictions
submission_new_features["count"] = predictions_new_features

# Display the first few rows of the submission file
submission_new_features.head()


# Save the submission DataFrame as a CSV file
submission_new_features.to_csv("submission_new_features.csv", index=False)


from autogluon.common import space

nn_options = {  # specifies non-default hyperparameter values for neural network models
    'num_epochs': 30,  # number of training epochs (controls training time of NN models)
    'learning_rate': space.Real(1e-4, 1e-2, default=5e-4, log=True),  # learning rate used in training (real-valued hyperparameter searched on log-scale)
    'activation': space.Categorical('relu', 'softrelu', 'tanh'),  # activation function used in NN (categorical hyperparameter, default = first entry)
    'dropout_prob': space.Real(0.0, 0.5, default=0.1),  # dropout probability (real-valued hyperparameter)
}

gbm_options = {  # specifies non-default hyperparameter values for lightGBM gradient boosted trees
    'num_boost_round': 100,  # number of boosting rounds (controls training time of GBM models)
    'num_leaves': space.Int(lower=26, upper=66, default=36),  # number of leaves in trees (integer hyperparameter)
}

hyperparameters = {  # hyperparameters of each model type
                   'GBM': gbm_options,
                   'NN_TORCH': nn_options,
                  }  # When these keys are missing from hyperparameters dict, no models of that type are trained

time_limit = 600  # train various models for ~10 min
num_trials = 5  # try at most 5 different hyperparameter configurations for each type of model
search_strategy = 'auto'  # to tune hyperparameters using random search routine with a local scheduler

hyperparameter_tune_kwargs = {  # HPO is not performed unless hyperparameter_tune_kwargs is specified
    'num_trials': num_trials,
    'scheduler' : 'local',
    'searcher': search_strategy,
}  # Refer to TabularPredictor.fit docstring for all valid values

# Train a new AutoGluon model with hyperparameter tuning
# We are using the same training data as before, but this time tuning the model's internal parameters
predictor_new_hpo = TabularPredictor(label="count", eval_metric="root_mean_squared_error").fit(
    train_for_autogluon,
    time_limit=time_limit,
    hyperparameters=hyperparameters, # Model types and their hyperparameters to tune
    hyperparameter_tune_kwargs=hyperparameter_tune_kwargs, # Tuning strategy and settings
)


# Display a summary of the training process
predictor_new_hpo.fit_summary()


# Plot a bar chart of the validation scores (e.g., RMSE) for each model trained by AutoGluon
predictor.leaderboard(silent=True).plot(kind="bar", x="model", y="score_val");


# Use the trained AutoGluon model to predict bike rental counts on the test dataset
predictions_new_hpo = predictor_new_hpo.predict(test)

# If there is a count value less than 0, we set it equal to 0.
predictions_new_hpo[predictions_new_hpo < 0] = 0

# Create a submission DataFrame by copying the sample submission format
submission_new_hpo = submission

# Replace the placeholder 'count' column with our actual predictions
submission_new_hpo["count"] = predictions_new_hpo

# Display the first few rows of the submission file
submission_new_hpo.head()


# Save the submission DataFrame as a CSV file
submission_new_hpo.to_csv("submission_new_hpo.csv", index=False)


# Create a DataFrame summarizing RMSE scores from different modeling stages
# and plot them to visually compare performance
fig = pd.DataFrame(
    {
        "model": ["initial", "add_features", "hpo"],
        "score_val(RMSE)": [-53.029175, -30.196934, -36.574802]
    }
).plot(x="model", y="score_val(RMSE)", figsize=(8, 6)).get_figure()

# Save the plot as an image file for reference or inclusion in a report
fig.savefig('model_train_score.png')


# Create a DataFrame summarizing final test scores from each model version
# and plot them for a clear visual comparison
fig = pd.DataFrame(
    {
        "test_eval": ["initial", "add_features", "hpo"],
        "score": [1.79214, 0.64351, 0.51168]
    }
).plot(x="test_eval", y="score", figsize=(8, 6)).get_figure()

# Save the test score comparison plot
fig.savefig('model_test_score.png')


# Load the training dataset into a pandas DataFrame
train = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv")

# Load the test dataset into a pandas DataFrame
test = pd.read_csv("/kaggle/input/bike-sharing-demand/test.csv")

# Load the sample submission file (used to format our predictions correctly)
submission = pd.read_csv("/kaggle/input/bike-sharing-demand/sampleSubmission.csv")


# Check for missing values in each column of the training set
train.isna().sum()


# Visualize missing data using a matrix plot
# Each line represents a row in the dataset, and missing values are shown as gaps
msno.matrix(train, figsize=(12, 5));


from datetime import datetime
import calendar

# Function to extract time-based features from the 'datetime' column
def time_process(df):
    df['datetime'] = pd.to_datetime(df['datetime'])
    # Year, month, day, hour feature extraction
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour
    # Label the number of weeks of the date to explore the characteristics of weekdays
    df['week'] = df['datetime'].dt.isocalendar().week
    df['weekday'] = df['datetime'].dt.dayofweek
    return df

# Apply time feature extraction to both train and test sets
train = time_process(train)
test = time_process(test)


# month_map = {
#     'January': 1, 'February': 2, 'March': 3, 'April': 4,
#     'May': 5, 'June': 6, 'July': 7, 'August': 8,
#     'September': 9, 'October': 10, 'November': 11, 'December': 12
# }

# train['month'] = train['month'].map(month_map).astype(int)

# You don't need to use these, but you can use if you want it.


# Display the first 5 rows of the training dataset to get a quick look at the data
train.head()


# Display summary information about the training dataset
train.info()


# Use Random Forest Classifier to fill windspeed values that are zero (likely missing or incorrect)
from sklearn.ensemble import RandomForestClassifier
pd.options.mode.chained_assignment = None

def wind_0_fill(df):
    wind_0 = df[df['windspeed'] == 0]
    wind_not0 = df[df['windspeed'] != 0]
    y_label = wind_not0['windspeed']

    # We assume windspeed depends on these weather/time-related features
    windcolumns = ['season', 'weather', 'temp', 'atemp', 'humidity', 'hour', 'month', "year"]

    # Train a Random Forest Classifier (not Regressor) to predict discrete windspeed values
    clf = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0)
    clf.fit(wind_not0[windcolumns], y_label.astype('int')) # Cast target to int

    # Predict windspeed for rows where it was originally 0
    pred_y = clf.predict(wind_0[windcolumns])
    wind_0['windspeed'] = pred_y

    # Merge the corrected rows back with the rest of the dataset
    df_rfw = pd.concat([wind_0, wind_not0])
    df_rfw.reset_index(inplace=True)
    # df_rfw.drop(columns=['index'], inplace=True, axis=1)
    return df_rfw

# Apply windspeed imputation to train and test sets
train = wind_0_fill(train)
test = wind_0_fill(test)


# Display the first 5 rows of the training dataset to get a quick look at the data
train.head()


# Set datetime as index for both train and test datasets
train.set_index(pd.to_datetime(train["datetime"]), inplace=True)
test.set_index(pd.to_datetime(test["datetime"]), inplace=True)

# Returns a range of all 24 hours in a given day
def get_day(day_start):
    day_end = day_start + pd.offsets.DateOffset(hours=23)
    return pd.date_range(day_start, day_end, freq="H")

# Tax Day, Still Need to Work
train.loc[get_day(datetime(2011, 4, 15)), "workingday"] = 1
train.loc[get_day(datetime(2012, 4, 16)), "workingday"] = 1
# Thanksgiving without work
test.loc[get_day(datetime(2011, 11, 25)), "workingday"] = 0
test.loc[get_day(datetime(2012, 11, 23)), "workingday"] = 0
#Christmas, no work
test.loc[get_day(datetime(2011, 12, 24)), "workingday"] = 0
test.loc[get_day(datetime(2011, 12, 31)), "workingday"] = 0
test.loc[get_day(datetime(2012, 12, 26)), "workingday"] = 0
test.loc[get_day(datetime(2012, 12, 31)), "workingday"] = 0
# Tax Day, No Holiday
train.loc[get_day(datetime(2011, 4, 15)), "holiday"] = 0
train.loc[get_day(datetime(2012, 4, 16)), "holiday"] = 0
# Thanksgiving. Vacation.
test.loc[get_day(datetime(2011, 12, 24)), "holiday"] = 1
test.loc[get_day(datetime(2011, 12, 31)), "holiday"] = 1
test.loc[get_day(datetime(2012, 12, 31)), "holiday"] = 1
# Heavy rain
test.loc[get_day(datetime(2012, 5, 21)), "holiday"] = 1
# Tsunami
train.loc[get_day(datetime(2012, 6, 1)), "holiday"] = 1


# Create readable season and weather labels in new columns to improve clarity during data exploration and visualization
def name_process(df):
    df['season2'] = df['season']
    df['weather2'] = df['weather']
    df['season2'] = df['season2'].map({1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'})
    df['weather2'] = df['weather2'].map({1: 'Clear', 2: 'Mist', 3: 'Light_Snow', 4: 'Heavy_Rain'})
    # df['month'] = df['month'].map({1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'})   
    return df

train = name_process(train)
test = name_process(test)


# Display the first 5 rows of the training dataset to get a quick look at the data
train.head()


# Visualizing the distribution of 'count' across multiple categorical features using boxplots
fig, axes = plt.subplots(nrows=3, ncols=2)
fig.set_size_inches(12, 15)
sns.boxplot(data=train, y="count", orient="v", ax=axes[0][0])
sns.boxplot(data=train, y="count", x="season", orient="v", ax=axes[0][1])
sns.boxplot(data=train, y="count", x="hour", orient="v", ax=axes[1][0])
sns.boxplot(data=train, y="count", x="workingday", orient="v", ax=axes[1][1])
sns.boxplot(data=train, y='count', x='month', orient="v", ax=axes[2][0])
sns.boxplot(data=train, y='count', x='weekday', orient="v", ax=axes[2][1])

axes[0][0].set(ylabel="Count", title="Box Plot on Count")
axes[0][1].set(xlabel="Season", ylabel="Count", title="Box Plot on Count Across Season")
axes[1][0].set(xlabel="Hour of the Day", ylabel="Count", title="Box Plot on Count Across Hour of the Day")
axes[1][1].set(xlabel="Working Day", ylabel="Count", title="Box Plot on Count Across Working Day")
axes[2][0].set(xlabel="Month", ylabel="Count", title="Box Plot on Count Across Month")
axes[2][1].set(xlabel="Weekday", ylabel="Count", title="Box Plot on Count Across Weekday")
plt.show()


# 147 outliers removed using the 3-sigma rule
outliers = np.abs(train['count'] - train['count'].mean()) > (3 * train['count'].std())
outliers_num = len(train[outliers])
train.drop(index=train[outliers].index)
print("Deleted",outliers_num,"Outliers")


# Interpret peak usage times based on data
train['peak'] = train[['hour', 'workingday']].apply(lambda x: (0, 1)[(x['workingday'] == 1 and  ( (x['hour'] == 8) or (17 <= x['hour'] <= 18) or (12 <= x['hour'] <= 12))) or ((x['workingday'] == 0) and  (10 <= x['hour'] <= 19))], axis = 1)
test['peak'] = test[['hour', 'workingday']].apply(lambda x: (0, 1)[(x['workingday'] == 1 and  ( (x['hour'] == 8) or (17 <= x['hour'] <= 18) or (12 <= x['hour'] <= 12))) or ((x['workingday'] == 0) and  (10 <= x['hour'] <= 19))], axis = 1)


# Display the first 5 rows of the training dataset to get a quick look at the data
train.head()


# Shows how hourly bike rental counts vary across different weather conditions
fig, ax = plt.subplots(figsize=(15, 6))
sns.pointplot(data=train, x="hour", y="count", hue="weather2", ax=ax)
ax.set(title="Count of Bikes During Different Weathers");


# Create a categorical plot showing how bike counts vary by weather
sns.catplot(data=train, x='weather', y='count', ax=ax);


# Gives insight into how bike usage varies through the week
fig, ax = plt.subplots(figsize=(20, 6))
sns.barplot(data=train, x="weekday", y="count", ax=ax)

ax.set(title="Count of Bikes During Different Weekdays");


# A heatmap to visualize the correlation between numerical variables
corrMat = train[["temp", "atemp", "casual", "registered", "humidity", "windspeed", "count"]].corr()
mask = np.array(corrMat)
mask[np.tril_indices_from(mask)] = False # Hide the lower triangle
fig, ax = plt.subplots()
fig.set_size_inches(20, 10)
sns.heatmap(corrMat, mask=mask, vmax=8, square=True, annot=True);


# Regression plots showing how bike rentals correlate with weather-related features.
fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4)
fig.set_size_inches(20, 5)

sns.regplot(x="temp", y="count", data=train, ax=ax1)
sns.regplot(x="windspeed", y="count", data=train, ax=ax2)
sns.regplot(x="humidity", y="count", data=train, ax=ax3)
sns.regplot(x="atemp", y="count", data=train, ax=ax4)
plt.show()


# Check normality of the target variable and assess the effect of log transformation
# Raw distribution of 'count' is usually skewed; log-transform helps normalize it
fig, axes = plt.subplots(ncols=2, nrows=2)
fig.set_size_inches(12, 10)

# Raw count distribution and Q-Q plot
sns.distplot(train["count"], ax=axes[0][0])
stats.probplot(train["count"], dist="norm", fit=True, plot=axes[0][1])

# Log-transformed distributions and Q-Q plot
sns.distplot(np.log(train["count"]), ax=axes[1][0])
stats.probplot(np.log1p(train["count"]), dist="norm", fit=True, plot=axes[1][1])
plt.show()

# Sample normal distribution in general
# train['count'].plot(kind='kde');
# Perform log1p transformations
# import math
# train['count_log'] = train['count'].apply(lambda x: math.log(x+1))
# train['count_log'].plot(kind='kde');


fig,(ax1, ax2, ax3, ax4) = plt.subplots(nrows=4)
fig.set_size_inches(12, 20)
# sortOrder = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
# hueOrder = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

# Average Count by Month
monthAggregated = pd.DataFrame(train.groupby("month")["count"].mean()).reset_index()
monthSorted = monthAggregated.sort_values(by="count", ascending=False)
sns.barplot(data=monthSorted, x="month", y="count", ax=ax1,) # order=sortOrder
ax1.set(xlabel='Month', ylabel='Average Count', title="Average Count By Month")

# Average Count by Hour & Season
hourAggregated = pd.DataFrame(train.groupby(["hour", "season"], sort=True)["count"].mean()).reset_index()
sns.pointplot(x=hourAggregated["hour"], y=hourAggregated["count"], hue=hourAggregated["season"], data=hourAggregated, join=True, ax=ax2)
ax2.set(xlabel='Hour Of The Day', ylabel='Users Count', title="Average Users Count By Hour Of The Day Across Season", label='big')

# Average Count by Hour & Weekday
hourAggregated = pd.DataFrame(train.groupby(["hour", "weekday"], sort=True)["count"].mean()).reset_index()
sns.pointplot(x=hourAggregated["hour"], y=hourAggregated["count"], hue=hourAggregated["weekday"], data=hourAggregated, join=True, ax=ax3) # hue_order=hueOrder
ax3.set(xlabel='Hour Of The Day', ylabel='Users Count',title="Average Users Count By Hour Of The Day Across Weekdays",label='big')

# Average Count by Hour & User Type
hourTransformed = pd.melt(train[["hour", "casual", "registered"]], id_vars=['hour'], value_vars=['casual', 'registered'])
hourAggregated = pd.DataFrame(hourTransformed.groupby(["hour", "variable"],sort=True)["value"].mean()).reset_index()
sns.pointplot(x=hourAggregated["hour"], y=hourAggregated["value"], hue=hourAggregated["variable"], hue_order=["casual", "registered"], data=hourAggregated, join=True, ax=ax4)
ax4.set(xlabel='Hour Of The Day', ylabel='Users Count', title="Average Users Count By Hour Of The Day Across User Type", label='big')
plt.show()


# Convert categorical features into binary (one-hot) encoded format
# while retaining original columns for later analysis
train = pd.get_dummies(train, columns=['season2'])
train = pd.get_dummies(train, columns=['weather2'])

test = pd.get_dummies(test, columns=['season2'])
test = pd.get_dummies(test, columns=['weather2'])


# All possible features, useful for global exploration or baseline models
All_feature_columns = ['season','weather','temp','atemp','humidity','windspeed',
                        'year','holiday','workingday','month','day','hour','week','weekday','peak',
                       'season2_Fall','season2_Spring','season2_Summer','season2_Winter',
                       'weather2_Clear','weather2_Heavy_Rain','weather2_Light_Snow','weather2_Mist']

# Random Forest Regressor â€” insensitive to multicollinearity, but benefits from categorical clarity
RFR_feature_columns = ['weather','temp','atemp','windspeed',
                       'workingday','season','holiday',
                       'hour','weekday','week','peak',
                       'season2_Fall','season2_Spring','season2_Summer','season2_Winter',
                      'weather2_Clear','weather2_Heavy_Rain','weather2_Light_Snow','weather2_Mist']

# Gradient Boosting Regressor â€” more sensitive to noise, so humidity included and fewer redundant fields
GBR_feature_columns =['weather','temp','atemp','humidity','windspeed',
                       'holiday','workingday','season',
                       'hour','weekday','year',
                      'season2_Fall','season2_Spring','season2_Summer','season2_Winter',
                       'weather2_Clear','weather2_Heavy_Rain','weather2_Light_Snow','weather2_Mist']


# Splitting the dataset into feature matrices for RFR and GBR based on selected columns
RFR_X_train = train[RFR_feature_columns].values
RFR_X_test = test[RFR_feature_columns].values

GBR_X_train = train[GBR_feature_columns].values
GBR_X_test = test[GBR_feature_columns].values

# Applying log1p transformation to target variables to reduce skewness
y_casual = train['casual'].apply(lambda x: np.log1p(x)).values
y_registered = train['registered'].apply(lambda x: np.log1p(x)).values
y_count = train['count'].apply(lambda x: np.log1p(x)).values

# Keeping datetime values separately for final prediction output
X_date = test['datetime'].values


# Custom function to calculate Root Mean Squared Logarithmic Error (RMSLE)
def rmsle(y_real, y_pre):    
    log1 = np.log(y_real + 1)
    log2 = np.log(y_pre + 1)    
    calc = (log1 - log2) ** 2
    return np.sqrt(np.mean(calc))


# Split model training set
from sklearn.model_selection import train_test_split
X_train = train[All_feature_columns].values
xd_train, xd_test, yd_train, yd_test = train_test_split(X_train, y_count, random_state=0)
# Train, tune, and test various regression models
# LGBM
from lightgbm import LGBMRegressor
def LGBM_model():
    LGBM = LGBMRegressor(boosting_type='gbdt', objective='regression', num_leaves=1200,
                                learning_rate=0.17, n_estimators=1000, max_depth=10,
                                metric='rmse', bagging_fraction=0.8, feature_fraction=0.8, reg_lambda=0.9)
    LGBM.fit(xd_train, yd_train)
    pre_test = LGBM.predict(xd_test)
    score = rmsle(yd_test, pre_test)
    return score
# Random Forest
from sklearn.ensemble import RandomForestRegressor
def RandomForest_model():
    RFR = RandomForestRegressor(n_estimators=1000, max_depth=15, random_state=0, n_jobs=-1)
    RFR.fit(xd_train, yd_train)
    pre_test = RFR.predict(xd_test)
    score = rmsle(yd_test, pre_test)
    return score
# Decision Tree
from sklearn.tree import DecisionTreeRegressor
def DecisionTree_model():
    DTR = DecisionTreeRegressor(max_features='sqrt', splitter='random', min_samples_split=4, max_depth=10)
    DTR.fit(xd_train, yd_train)
    pre_test = DTR.predict(xd_test)
    score = rmsle(yd_test, pre_test)
    return score
#Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor
def GradientBoosting_model():
    GBR = GradientBoostingRegressor(n_estimators=1000, max_depth=5, random_state=0)
    GBR.fit(xd_train, yd_train)
    pre_test = GBR.predict(xd_test)
    score = rmsle(yd_test, pre_test)
    return score
# Logistic Regression --> Not ideal for regression (you might consider removing it here).
# from sklearn.linear_model import LogisticRegression
# def Logistic_model():
#     LG = LogisticRegression(penalty="l2", tol=0.0001, C=1.0, solver="lbfgs", max_iter=3000, multi_class='ovr', verbose=0)
#     LG.fit(xd_train, yd_train)
#     pre_test = LG.predict(xd_test)
#     score = rmsle(yd_test, pre_test)
#     return score
# AdaBoost
from sklearn.ensemble import AdaBoostRegressor
def AdaBoost_model():
    ABR = AdaBoostRegressor(learning_rate=0.1, loss='square', n_estimators=1000)
    ABR.fit(xd_train, yd_train)
    pre_test = ABR.predict(xd_test)
    score = rmsle(yd_test, pre_test)
    return score
# Linear Regression
from sklearn.linear_model import LinearRegression
def LinearRegression_model():
    LR = LinearRegression(n_jobs=-1)
    LR.fit(xd_train, yd_train)
    pre_test = LR.predict(xd_test)
    score = rmsle(yd_test, pre_test)
    return score
# Ridge
from sklearn.linear_model import Ridge
# from sklearn.model_selection import GridSearchCV
# from sklearn.linear_model import ElasticNetCV
def Ridge_model():
    RM = Ridge(max_iter=3000, alpha=0.1)
    # ridge_params_ = {'max_iter':[3000], 'alpha':[0.1, 1, 2, 3, 4, 10, 30,100,200,300,400,800,900,1000]}
    # rmsle_scorer = metrics.make_scorer(rmsle, greater_is_better=False)
    # grid_ridge_m = GridSearchCV(ridge_m_,
    #                       ridge_params_,
    #                       scoring=rmsle_scorer,
    #                       cv=5)
    RM.fit(xd_train, yd_train)
    pre_test = RM.predict(xd_test)
    score = rmsle(yd_test, pre_test)
    return score
# Lasso
from sklearn.linear_model import Lasso
def Lasso_model():
    LM = Lasso(max_iter=3000, alpha=0.1)
    # alpha  = 1/np.array([0.1, 1, 2, 3, 4, 10, 30, 100, 200, 300, 400, 800, 900, 1000])
    # lasso_params_ = { 'max_iter': [3000],'alpha': alpha}
    # grid_lasso_m = GridSearchCV(lasso_m_,lasso_params_,scoring=rmsle_scorer, cv=5)
    LM.fit(xd_train, yd_train)
    pre_test = LM.predict(xd_test)
    score = rmsle(yd_test, pre_test)
    return score
# Huber Regressor
from sklearn.linear_model import HuberRegressor
def HuberRegressor_model():
    HR = HuberRegressor(max_iter=3000, alpha=0.01)
    HR.fit(xd_train, yd_train)
    pre_test = HR.predict(xd_test)
    score = rmsle(yd_test, pre_test)
    return score
# ExtraTreeRegressor 
from sklearn.ensemble import ExtraTreesRegressor
def ExtraTreeRegressor_model():
    ETR = ExtraTreesRegressor(max_depth=10,
                              n_estimators=50,
                              n_jobs=-1)
    ETR.fit(xd_train, yd_train)
    pre_test = ETR.predict(xd_test)
    score = rmsle(yd_test, pre_test)
    return score


# model_functions = {
#     'LGBM': LGBM_model,
#     'Random Forest': RandomForest_model,
#     'Decision Tree': DecisionTree_model,
#     'Gradient Boosting': GradientBoosting_model,
#     'AdaBoost': AdaBoost_model,
#     'Linear Regression': LinearRegression_model,
#     'Ridge': Ridge_model,
#     'Lasso': Lasso_model,
#     'Huber': HuberRegressor_model,
#     'Extra Trees': ExtraTreeRegressor_model
# }

# for name, func in model_functions.items():
#     try:
#         print(f"{name}: RMSLE = {func():.4f}")
#     except Exception as e:
#         print(f"{name} failed: {e}")


# RMSLE evaluation of various models
print("LGBM_model:             ", LGBM_model())
print("RandomForest_model:     ", RandomForest_model())
print("DecisionTree_model:     ", DecisionTree_model())
print("GradientBoosting_model: ", GradientBoosting_model())
# print("Logistic_Regression_model:  ", Logistic_model())
print("AdaBoost_model:         ", AdaBoost_model())
print("Linear_Regression_model:         ", LinearRegression_model())
print("Ridge_model:         ", Ridge_model())
print("Lasso_model:         ", Lasso_model())
print("HuberRegressor_model:         ", HuberRegressor_model())
print("ExtraTreeRegressor_model:         ", ExtraTreeRegressor_model())


# Random Forest
from sklearn.ensemble import RandomForestRegressor
params = {'n_estimators': 1000, #5000
          'max_depth': 15, # 20
          'random_state': 0, 
          'min_samples_split': 5, # 8
          'n_jobs': -1}

# Train Random Forest model on casual users
RFR1 = RandomForestRegressor(**params)
RFR1.fit(RFR_X_train, y_casual)
print("Degree of the model fit:",RFR1.score(RFR_X_train, y_casual))

# Train Random Forest model on registered users
RFR2 = RandomForestRegressor(**params)
RFR2.fit(RFR_X_train, y_registered)
print("Degree of the model fit:",RFR2.score(RFR_X_train, y_registered))

# Train Random Forest model on total count
RFR3 = RandomForestRegressor(**params)
RFR3.fit(RFR_X_train, y_count)
print("Degree of the model fit:",RFR3.score(RFR_X_train, y_count))


from sklearn.model_selection import RandomizedSearchCV

# Define a grid of hyperparameter values to sample from during randomized search
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

# Store the hyperparameter ranges in a dictionary
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

print(random_grid)


# Train a Random Forest model on the casual component with random hyperparameter combinations
RFR_Random_1 = RandomForestRegressor()
RFR_Random_CV_1 = RandomizedSearchCV(estimator=RFR_Random_1, param_distributions=random_grid, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
# Fit the random search model
RFR_Random_CV_1.fit(RFR_X_train, y_casual)
print("Degree of the model fit:",RFR_Random_CV_1.score(RFR_X_train, y_casual))


# Train on registered component
RFR_Random_2 = RandomForestRegressor()
RFR_Random_CV_2 = RandomizedSearchCV(estimator=RFR_Random_2, param_distributions=random_grid, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
# Fit the random search model
RFR_Random_CV_2.fit(RFR_X_train, y_registered)
print("Degree of the model fit:",RFR_Random_CV_2.score(RFR_X_train, y_registered))


# Train on total count
RFR_Random_3 = RandomForestRegressor()
RFR_Random_CV_3 = RandomizedSearchCV(estimator=RFR_Random_3, param_distributions=random_grid, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
# Fit the random search model
RFR_Random_CV_3.fit(RFR_X_train, y_count)
print("Degree of the model fit:",RFR_Random_CV_3.score(RFR_X_train, y_count))


# Gradient Boost
from sklearn.ensemble import GradientBoostingRegressor

params2 = {'n_estimators': 150, # 300
           'max_depth': 5, # 15
           'random_state': 0, 
           'min_samples_leaf' : 10, 
           'learning_rate': 0.1, # 0.01
           'subsample': 0.7, 
           'loss': 'squared_error'}

# Train Gradient Boosting model on casual users
GBR1 = GradientBoostingRegressor(**params2)
GBR1.fit(GBR_X_train, y_casual)
print("Degree of the model fit:",GBR1.score(GBR_X_train, y_casual))


# Train Gradient Boosting model on registered users
GBR2 = GradientBoostingRegressor(**params2)
GBR2.fit(GBR_X_train, y_registered)
print("Degree of the model fit:",GBR2.score(GBR_X_train, y_registered))


# Train Gradient Boosting model on total count
GBR3 = GradientBoostingRegressor(**params2)
GBR3.fit(GBR_X_train, y_count)
print("Degree of the model fit:",GBR3.score(GBR_X_train, y_count))


# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.model_selection import GridSearchCV

# param_grid = {
#     'n_estimators': [50, 100, 200, 300, 500],
#     'learning_rate': [0.01, 0.1, 0.2, 0.001],
#     'max_depth': [3, 5, 7, 10, 15],
#     'min_samples_leaf' : [3, 5, 10, 15]
# }

# GBRCV_1 = GradientBoostingRegressor()
# GBR_GridSearch_1 = GridSearchCV(estimator=GBRCV_1, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
# GBR_GridSearch_1.fit(GBR_X_train, y_casual)
# print("Degree of the model fit:", GBR_GridSearch_1.score(GBR_X_train, y_casual))


# GBR2 = GradientBoostingRegressor(**params2)
# GBR2.fit(GBR_X_train, y_registered)
# print("Degree of the model fit:",GBR2.score(GBR_X_train, y_registered))

# GBR3 = GradientBoostingRegressor(**params2)
# GBR3.fit(GBR_X_train, y_count)
# print("Degree of the model fit:",GBR3.score(GBR_X_train, y_count))


# Make predictions with the trained Random Forest models and reverse log1p transformation
RFR_pre_casual = RFR1.predict(RFR_X_test)
RFR_pre_casual = np.exp(RFR_pre_casual) - 1

RFR_pre_registered = RFR2.predict(RFR_X_test)
RFR_pre_registered = np.exp(RFR_pre_registered) - 1

# Combine casual and registered user predictions
RFR_pre = RFR_pre_casual + RFR_pre_registered

# Make predictions with the trained Gradient Boosting models and reverse log1p transformation
GBR_pre_casual = GBR1.predict(GBR_X_test)
GBR_pre_casual = np.exp(GBR_pre_casual) - 1

GBR_pre_registered = GBR2.predict(GBR_X_test)
GBR_pre_registered = np.exp(GBR_pre_registered) - 1

# Combine casual and registered user predictions
GBR_pre = GBR_pre_casual + GBR_pre_registered

# Weighted blend of Random Forest and Gradient Boosting predictions
submit1 = pd.DataFrame({'datetime': X_date, 'count': 0.2 * RFR_pre + 0.8 * GBR_pre})

# Export the submission file
submit1.to_csv('/kaggle/working/submisssion_1.csv', index=False)


# Predict total count directly using Random Forest and reverse log1p transformation
RFR_pre_count = RFR3.predict(RFR_X_test)
RFR_pre_count = np.exp(RFR_pre_count) - 1

# Predict total count directly using Gradient Boosting and reverse log1p transformation
GBR_pre_count = GBR3.predict(GBR_X_test)
GBR_pre_count = np.exp(GBR_pre_count) - 1


# Weighted average of the two model predictions
pre_count= 0.2 * RFR_pre_count + 0.8 * GBR_pre_count

# ðŸ’¾ Save as a second submission file
submit2 = pd.DataFrame({'datetime': X_date, 'count': pre_count})
submit2.to_csv('/kaggle/working/submisssion_2.csv', index=False)


# Predict the casual users using the best Random Forest model from RandomizedSearchCV
RFRCV_pre_casual = RFR_Random_CV_1.predict(RFR_X_test)

# The target variable was log-transformed, so we reverse the transformation using exp() and subtract 1
RFRCV_pre_casual = np.exp(RFRCV_pre_casual) - 1

# Predict the registered users using the best Random Forest model from RandomizedSearchCV
RFRCV_pre_registered = RFR_Random_CV_2.predict(RFR_X_test)
RFRCV_pre_registered = np.exp(RFRCV_pre_registered) - 1

# Total count prediction from Random Forest = casual + registered
RFRCV_pre = RFRCV_pre_casual + RFRCV_pre_registered

# Predict casual users using Gradient Boosting model
GBR_pre_casual = GBR1.predict(GBR_X_test)
GBR_pre_casual = np.exp(GBR_pre_casual) - 1

# Predict registered users using Gradient Boosting model
GBR_pre_registered = GBR2.predict(GBR_X_test)
GBR_pre_registered = np.exp(GBR_pre_registered) - 1

# Total count prediction from Gradient Boosting = casual + registered
GBR_pre = GBR_pre_casual + GBR_pre_registered

# Combine the predictions from Random Forest and Gradient Boosting models
# Weighted averaging: 20% weight to Random Forest and 80% weight to Gradient Boosting
# This helps leverage the strengths of both models and often improves final prediction accuracy
submit1 = pd.DataFrame({'datetime': X_date, 'count': 0.2 * RFRCV_pre + 0.8 * GBR_pre})
submit1.to_csv('/kaggle/working/submisssion_1.csv', index=False)


# Predict the total count directly using the third Random Forest model (tuned with RandomizedSearchCV)
RFRCV_pre_count = RFR_Random_CV_3.predict(RFR_X_test)
RFRCV_pre_count = np.exp(RFRCV_pre_count) - 1

# Predict the total count directly using the Gradient Boosting model
GBR_pre_count = GBR3.predict(GBR_X_test)
GBR_pre_count = np.exp(GBR_pre_count) - 1

# Again, combine both predictions using weighted averaging
# We give more trust to Gradient Boosting (80%) and less to Random Forest (20%)
pre_count = 0.2 * RFRCV_pre_count + 0.8 * GBR_pre_count

submit2 = pd.DataFrame({'datetime': X_date, 'count': pre_count})
submit2.to_csv('/kaggle/working/submisssion_2.csv', index=False)

