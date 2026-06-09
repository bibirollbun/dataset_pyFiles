# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# Data manipulation and analysis
import pandas as pd  # Equivalent to tidyverse (dplyr, tidyr)
import numpy as np   # For numerical operations

import pyarrow.parquet as pq

# Data visualization
import matplotlib.pyplot as plt  # Equivalent to ggplot2
import seaborn as sns            # For advanced visualizations

# Machine learning and modeling
from sklearn.model_selection import train_test_split  # For data splitting
from sklearn.preprocessing import StandardScaler      # For preprocessing
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor  # Equivalent to xgboost
from sklearn.metrics import accuracy_score, mean_squared_error  # For model evaluation

# SHAP for model interpretation
import shap  # Equivalent to shapviz

# Date and time manipulation
from datetime import datetime  # Equivalent to lubridate
from datetime import timedelta 

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Define the path to the directory containing the files
path = "/kaggle/input/neo-bank-non-sub-churn-prediction/"


# List all training files matching the pattern "train_*"
training_files = [os.path.join(path, f) for f in os.listdir(path) if f.startswith("train_")]


# Read all training files into a list of DataFrames
df_list = [pd.read_parquet(f) for f in training_files]



# Combine all training DataFrames into one
data_train = pd.concat(df_list, ignore_index=True)


# Read the test file
data_test = pd.read_parquet(os.path.join(path, "test.parquet"))


# Display the first 5 rows and structure of the training data
print("Training Data:")
print(data_train.head(5))  # Display the first 5 rows
print(data_train.info())   # Display the structure (column names, data types, etc.)


# Display the first 5 rows and structure of the test data
print("Test Data:")
print(data_test.head(5))   # Display the first 5 rows
print(data_test.info())    # Display the structure (column names, data types, etc.)


# Calculate mean of numeric columns grouped by date
data_train_mean = data_train.groupby('date').mean(numeric_only=True).reset_index()


# Filter data for special customers
special_customer_data = data_train[data_train['customer_id'] == 3367].select_dtypes(include='number').drop(columns=['Id', 'customer_id', 'interest_rate'])
special_customer_data['date'] = data_train[data_train['customer_id'] == 3367]['date']

special_customer_data2 = data_train[data_train['customer_id'] == 41422].select_dtypes(include='number').drop(columns=['Id', 'customer_id', 'interest_rate'])
special_customer_data2['date'] = data_train[data_train['customer_id'] == 41422]['date']


# Melt data for plotting
data_train_mean_melted = data_train_mean.melt(id_vars=['date'], var_name='key', value_name='value')
special_customer_data_melted = special_customer_data.melt(id_vars=['date'], var_name='key', value_name='value')
special_customer_data2_melted = special_customer_data2.melt(id_vars=['date'], var_name='key', value_name='value')


print(data_train_mean_melted.isin([np.inf, -np.inf]).sum())
print(special_customer_data_melted.isin([np.inf, -np.inf]).sum())
print(special_customer_data2_melted.isin([np.inf, -np.inf]).sum())


import warnings 
# Suppress the specific warning
warnings.filterwarnings("ignore", category=FutureWarning, message="use_inf_as_na option is deprecated")

# Plot
plt.figure(figsize=(15, 10))
sns.set_theme(style="whitegrid")

# Plot mean values
sns.lineplot(data=data_train_mean_melted, x='date', y='value', color='grey', label='Mean')

# Plot special customer 1
sns.scatterplot(data=special_customer_data_melted, x='date', y='value', color='red', label='Customer 3367')

# Plot special customer 2
sns.scatterplot(data=special_customer_data2_melted, x='date', y='value', color='black', label='Customer 41422')

# Add facets
g = sns.FacetGrid(data_train_mean_melted, col='key', col_wrap=3, sharey=False, height=4, aspect=1.5)
g.map(sns.lineplot, 'date', 'value', color='grey')
g.map_dataframe(sns.scatterplot, x='date', y='value', color='red', data=special_customer_data_melted)
g.map_dataframe(sns.scatterplot, x='date', y='value', color='black', data=special_customer_data2_melted)

# Add titles and labels
g.set_titles("{col_name}")
g.set_axis_labels("", "")
plt.suptitle("Averaged Customer Behaviour", y=1.02)
plt.tight_layout()
plt.show()



# Filter the data for customer_id == 3367
filtered_data = data_train[data_train['customer_id'] == 3367]

# Select specific columns
selected_columns = filtered_data[['atm_transfer_in', 'atm_transfer_out', 
                                  'bank_transfer_in_volume', 'bank_transfer_out_volume',
                                  'crypto_in_volume', 'crypto_out_volume']]

# Summarize the data by calculating the sum of each column
summary = selected_columns.sum(axis=0, numeric_only=True, skipna=True).to_frame().T

# Display the result
print(summary)


# For data_train
data_train = data_train.groupby('customer_id').apply(
    lambda x: x.assign(activity_gap=x['tenure'] - x['tenure'].shift())
).reset_index(drop=True)

# Replace NA values in activity_gap with 0
data_train['activity_gap'] = data_train['activity_gap'].fillna(0)

# For data_test
data_test = data_test.groupby('customer_id').apply(
    lambda x: x.assign(activity_gap=x['tenure'] - x['tenure'].shift())
).reset_index(drop=True)

# Replace NA values in activity_gap with 0
data_test['activity_gap'] = data_test['activity_gap'].fillna(0)


print(data_train.head(5))


print(data_test.head(5))


print(data_train['activity_gap'].isna().sum())


print(data_train[['customer_id', 'tenure', 'activity_gap']].head())
print(data_train[['customer_id', 'tenure', 'activity_gap']].head())



print(data_train_mean.head(10))   # Display the first 5 rows


# List of loyal customer IDs
loyal_customer_ids = [74330, 41422, 98852, 41620, 84714, 84703, 84692, 41857]

# Filter the dataframe to include only rows where customer_id is in loyal_customer_ids
filtered_data = data_train[data_train['customer_id'].isin(loyal_customer_ids)]

# Get the maximum value in the 'activity_gap' column of the filtered dataframe
max_activity_gap = filtered_data['activity_gap'].max()

print(max_activity_gap)


#create a time series plot using matplotlib and seaborn 
# Filter the dataframe for customer_id == 1
customer_1_data = data_train[data_train['customer_id'] == 1]

# Create the plot
plt.figure(figsize=(10, 6))  # Set the figure size
sns.scatterplot(x='date', y='bank_transfer_in', data=customer_1_data, alpha=0.3, s=100, color='black')  # Scatter plot
sns.lineplot(x='date', y='bank_transfer_in', data=customer_1_data, alpha=0.3, color='black')  # Line plot

# Add title and labels
plt.title("Time Series of Customer 1")
plt.xlabel("Date")
plt.ylabel("Bank Transfer In")

# Apply a black-and-white theme
sns.set_style("whitegrid")

# Show the plot
plt.show()


# Find the last date in the dataset
last_date = data_train['date'].max()


print(last_date)


# Group by customer_id and calculate max_date and inactivity
data_train = data_train.groupby('customer_id').apply(
    lambda group: group.assign(
        max_date=group['date'].max(),
        inactivity=(last_date - group['date'].max()).days
    )
).reset_index(drop=True)


print(data_test)


print(data_train[['date', 'tenure', 'activity_gap']].head())


# Create the churned column
data_train['churned'] = ((data_train['inactivity'] >= 500) & 
                         (data_train['date'] == data_train['max_date'])).astype(int)



print(data_train[['date', 'tenure', 'activity_gap', 'churned']].head())


# Calculate the cutoff date (500 days before last_date)
last_train_date = last_date - timedelta(days=500)


print(last_train_date)


# Filter the dataframe to include only rows before the cutoff date
data_train_reduced = data_train[data_train['date'] < last_train_date]


# Count the number of churned customers in the filtered dataframe
num_churners = data_train_reduced[data_train_reduced['churned'] == 1].shape[0]


# Print the result
print(f"We have {num_churners} churners in the training data.")


# Calculate mean transactions per customer in the training data
mean_transactions_train = len(data_train_reduced) / data_train_reduced['customer_id'].nunique()
print(f"Mean transactions per customer in training data: {mean_transactions_train}")


# Calculate mean transactions per customer in the testing data
mean_transactions_test = len(data_test) / data_test['customer_id'].nunique()
print(f"Mean transactions per customer in testing data: {mean_transactions_test}")


# Calculate the transaction factor
transactions_factor = mean_transactions_train / mean_transactions_test
print(f"Transaction Factor: {transactions_factor}")


#Create time series plot showing number of customers over time 
#Sort the dataframe by date 
data_train_reduced = data_train_reduced.sort_values(by='date')
#Group by customer ID and mark new customers 
data_train_reduced['new_customer'] = data_train_reduced.groupby('customer_id')['date'].transform(
    lambda x: (x == x.min()).astype(int)
)






# Calculate cumulative sums
data_train_reduced['all_customers'] = data_train_reduced['new_customer'].cumsum()
data_train_reduced['churners'] = data_train_reduced['churned'].cumsum()
data_train_reduced['number_of_loyal_customers'] = (
    data_train_reduced['all_customers'] - data_train_reduced['churners']
)


# Convert date to datetime
data_train_reduced['date'] = pd.to_datetime(data_train_reduced['date'])


# Melt the dataframe for plotting
plot_data = data_train_reduced.melt(
    id_vars=['date'], 
    value_vars=['all_customers', 'churners', 'number_of_loyal_customers'],
    var_name='key', 
    value_name='value'
)


# Create the plot
plt.figure(figsize=(15, 5))
sns.lineplot(data=plot_data, x='date', y='value', hue='key', palette={
    'churners': 'red',
    'all_customers': 'grey',
    'number_of_loyal_customers': 'black'
})

# Add vertical lines
plt.axvline(pd.to_datetime('2019-01-10'), linestyle='--', color='blue')
plt.axvline(pd.to_datetime('2020-12-30'), linestyle='--', color='blue')

# Add title and labels
plt.title("Number of customers over time")
plt.xlabel("Date")
plt.ylabel("Number of customers")
plt.legend(title="Customers", loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=3)

# Apply theme
sns.set_style("whitegrid")

# Show the plot
plt.show()


#Calculate touchpoints 
#Calculate the number of touchpoints for each row in the training data to understand how many times contact was made 
data_train_reduced['contact_count'] = data_train_reduced['touchpoints'].apply(len)

#same for testing data
data_test['contact_count'] = data_test['touchpoints'].apply(len)


# Ensure csat_scores contains only dictionaries
data_train_reduced['csat_scores'] = data_train_reduced['csat_scores'].apply(
    lambda x: x if isinstance(x, dict) else {}
)

# Calculate the composite CSAT score
def calculate_csat(x):
    if not isinstance(x, dict):
        return np.nan  # Return NaN if x is not a dictionary
    values = [
        x.get('appointment', np.nan),
        x.get('email', np.nan),
        x.get('phone', np.nan),
        x.get('whatsapp', np.nan)
    ]
    # Convert non-numeric values to np.nan
    values = [v if isinstance(v, (int, float)) else np.nan for v in values]
    return np.nansum(values)

data_train_reduced['csat'] = data_train_reduced['csat_scores'].apply(calculate_csat)


print(data_train_reduced['csat_scores'].apply(type).value_counts())
print(data_train_reduced['csat'].isna().sum())


print(data_train_reduced['csat_scores'].head())
print(data_train_reduced['csat_scores'].apply(type).unique())


data_test['csat_scores'] = data_test['csat_scores'].apply(
    lambda x: x if isinstance(x, dict) else {}
)

def calculate_csat(x): 
    if not isinstance(x, dict):
        return np.nan # return Nan if x is not a dictionary 
    values = [
        x.get('appointment', np.nan), 
        x.get('email', np.nan), 
        x.get('phone', np.nan), 
        x.get('whatsapp', np.nan)
    ]

    #convert non numeric values to np-nan 
    values = [v if isinstance(v, (int, float)) else np.nan for v in values]
    return np.nansum(values)

data_test['csat'] = data_test['csat_scores'].apply(calculate_csat)


#function pre-process data - data preprocessing steps on the data set The function creates new features, groups data by customer_id and activity_cluster, and calculates cumulative sums and means for various columns. 
def preprocess_data(data):
    new_data = data.copy()
    
    # Create activity_cluster column
    new_data['activity_cluster'] = (new_data['activity_gap'] > 250).astype(int)
    
    # Calculate age
    new_data['age'] = ((datetime.now() - pd.to_datetime(new_data['date_of_birth'])) / pd.Timedelta(days=365)).astype(int)
    
    # Create discrete_date column
    new_data['discrete_date'] = pd.cut(
        new_data['date'],
        bins=[pd.Timestamp.min, pd.Timestamp('2019-01-10'), pd.Timestamp('2020-12-30'), pd.Timestamp.max],
        labels=[1, 2, 3]
    )
    
    # Group by customer_id and calculate cumulative activity_cluster
    new_data['activity_cluster'] = new_data.groupby('customer_id')['activity_cluster'].cumsum()
    
    # Group by customer_id and activity_cluster, then calculate cumulative sums and means
    new_data = new_data.groupby(['customer_id', 'activity_cluster']).apply(
        lambda group: group.assign(
            sum_bank_transfer_out=group['bank_transfer_out'].cumsum(),
            sum_bank_transfer_in=group['bank_transfer_in'].cumsum(),
            sum_crypto_in=group['crypto_in'].cumsum(),
            sum_crypto_out=group['crypto_out'].cumsum(),
            sum_atm_in=group['atm_transfer_in'].cumsum(),
            sum_atm_out=group['atm_transfer_out'].cumsum(),
            sum_crypto_in_volume=group['crypto_in_volume'].cumsum(),
            sum_crypto_out_volume=group['crypto_out_volume'].cumsum(),
            sum_bank_transfer_out_volume=group['bank_transfer_out_volume'].cumsum(),
            sum_bank_transfer_in_volume=group['bank_transfer_in_volume'].cumsum(),
            sum_complaints=group['complaints'].cumsum(),
            sum_csat=group['csat'].cumsum(),
            sum_contacts=group['contact_count'].cumsum(),
            mean_age=group['age'].expanding().mean(),
            mean_date=group['discrete_date'].expanding().mean(),
            mean_tenure=group['tenure'].expanding().mean(),
            num_activities_per_cluster=group.groupby(['customer_id', 'activity_cluster']).cumcount() + 1
        )
    ).reset_index(drop=True)
    
    return new_data


# Apply the function to the training and testing datasets
data_train_reduced = preprocess_data(data_train_reduced)



data_test_processed = preprocess_data(data_test)


# Save your processed datasets
#data_train_reduced.to_csv('/kaggle/working/data_train_reduced.csv', index=False)
#data_test_processed.to_csv('/kaggle/working/data_test_processed.csv', index=False)

# Save your model
#import joblib
#joblib.dump(xgb_model, '/kaggle/working/xgb_model.pkl')


# Reload saved data and model
#data_train_reduced = pd.read_csv('/kaggle/input/your-dataset-name/data_train_reduced.csv')
#data_test_processed = pd.read_csv('/kaggle/input/your-dataset-name/data_test_processed.csv')


# Select columns from data_train_reduced and stores them in new df called activity_cludtered_data 
#The selected columns are related to customer activity clusters, churn status, and various cumulative transaction metrics.

# Selection of the relevant columns 
activity_clustered_data = data_train_reduced[[
    'activity_cluster', 'churned', 'sum_bank_transfer_out',
    'sum_bank_transfer_in', 'sum_crypto_in', 'sum_crypto_out',
    'sum_atm_in', 'sum_atm_out', 'sum_crypto_in_volume',
    'sum_crypto_out_volume', 'sum_bank_transfer_out_volume',
    'sum_bank_transfer_in_volume', 'sum_complaints', 'sum_contacts',
    'sum_csat', 'mean_tenure', 'num_activities_per_cluster'
]].copy()



# Create a grouped bar plot to visualize the mean values of various features grouped by activity_cluster and churned status. 

# Set the plot size
plt.figure(figsize=(15, 15))

# Group by activity_cluster and churned, then calculate the mean of all columns
grouped_data = activity_clustered_data.groupby(
    ['activity_cluster', 'churned'], as_index=False
).mean()

# Melt the dataframe for plotting
melted_data = pd.melt(
    grouped_data,
    id_vars=['activity_cluster', 'churned'],
    var_name='key',
    value_name='value'
)

# Create the plot
g = sns.FacetGrid(
    melted_data, 
    col='key', 
    col_wrap=3, 
    sharey=False, 
    height=4, 
    aspect=1.5
)
g.map_dataframe(
    sns.barplot, 
    x='activity_cluster', 
    y='value', 
    hue='churned', 
    palette={0: 'grey', 1: 'black'},
    dodge=True
)
g.set_axis_labels("Activity Cluster", "Mean Value")
g.add_legend(title="Churner")
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle("Mean Values by Activity Cluster and Churn Status")

# Apply theme
sns.set_style("whitegrid")

# Show the plot
plt.show()


# Prepares the training data for modeling by selecting specific columns and converting the churned column to a factor 
#Selection of the relevant columns 
train_data_model = data_train_reduced[[
    'activity_cluster', 'churned', 'sum_bank_transfer_out',
    'sum_bank_transfer_in', 'sum_crypto_in', 'sum_crypto_out',
    'sum_atm_in', 'sum_atm_out', 'sum_crypto_in_volume',
    'sum_crypto_out_volume', 'sum_bank_transfer_out_volume',
    'sum_bank_transfer_in_volume', 'sum_complaints', 'sum_contacts',
    'sum_csat', 'num_activities_per_cluster', 'mean_age', 'mean_tenure',
    'bank_transfer_in', 'bank_transfer_out', 'crypto_in', 'crypto_out',
    'bank_transfer_out_volume', 'bank_transfer_in_volume',
    'crypto_in_volume', 'crypto_out_volume', 'from_competitor',
    'contact_count', 'csat', 'tenure', 'mean_date'
]].copy()


# Convert churned to a categorical variable
train_data_model['churned'] = train_data_model['churned'].astype('category')


# Prepares the testing data for modeling by selecting specific columns from the data_test_processed dataframe. 
#Similar to the previous step for the training data but excludes the churned column (since this is the target variable and may not be available in the testing data). 
# Selection of the relevant columns 
test_data_model = data_test_processed[[
    'activity_cluster', 'sum_bank_transfer_out',
    'sum_bank_transfer_in', 'sum_crypto_in', 'sum_crypto_out',
    'sum_atm_in', 'sum_atm_out', 'sum_crypto_in_volume',
    'sum_crypto_out_volume', 'sum_bank_transfer_out_volume',
    'sum_bank_transfer_in_volume', 'sum_complaints', 'sum_contacts',
    'sum_csat', 'num_activities_per_cluster', 'mean_age', 'mean_tenure',
    'bank_transfer_in', 'bank_transfer_out', 'crypto_in', 'crypto_out',
    'bank_transfer_out_volume', 'bank_transfer_in_volume',
    'crypto_in_volume', 'crypto_out_volume', 'from_competitor',
    'contact_count', 'csat', 'tenure', 'mean_date'
]].copy()



#Modelling
#cross validation 
#Stratified K Fold 

from sklearn.model_selection import StratifiedKFold

#set the random seed for reproduceability 
random_seed = 42 

#Split the data into training and validation sets (85% training and 15% validation )
train_data, val_data = train_test_split(
    train_data_model, 
    test_size=0.15,
    random_state=random_seed, 
    stratify=train_data_model['churned']
)

#create 5 stratified cross-validation folds 
sk = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed) 
folds = sk.split(train_data, train_data['churned'])


#parameters 
#def for hyperparameters for an XGBoost model
from xgboost import XGBClassifier

# Define the XGBoost model with the same hyperparameters
xgb_model = XGBClassifier(
    n_estimators=1000,  # Number of trees (boosting rounds)
    max_depth=6,        # Maximum depth of each tree
    min_child_weight=5, # Minimum sum of instance weights (similar to min_n)
    gamma=1e-6,         # Minimum loss reduction required to make a split
    subsample=0.8,      # Proportion of rows to sample for each tree
    colsample_bytree=6 / train_data.shape[1],  # Proportion of features to sample for each tree
    learning_rate=0.05, # Learning rate (shrinkage)
    objective='binary:logistic',  # Objective function for binary classification
    random_state=42     # Random seed for reproducibility
) 


#Preprocessing of data before fitting a model. Specify transformations and feature engineering steps. 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

# Define a function to convert 'from_competitor' to numeric
def convert_from_competitor(df):
    df['from_competitor'] = pd.to_numeric(df['from_competitor'])
    return df

# Create a ColumnTransformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('convert_from_competitor', FunctionTransformer(convert_from_competitor), ['from_competitor'])
    ],
    remainder='passthrough'  # Keep all other columns unchanged
)

# Apply the preprocessing to the training data
train_data_preprocessed = preprocessor.fit_transform(train_data_model)



#fit 
#Defines workflows A workflow combines a model (xgb_params) with a recipe (xgb_rec0 or xgb_rec) to create a reusable pipeline for training and prediction. 
from sklearn.pipeline import Pipeline 
#define the first pipeline (xgb_wf0)
xgb_pipeline0 = Pipeline([
    ('preprocessor', preprocessor), #POreprocessing steps 
    ('model', xgb_model)            #XGBoost model
])

#define the second pipeline (xgb_wf)
xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', xgb_model)
])


#Perform a cross-validated fitting of the XGBoost workflow (xgb_wf0) using the predefined folds (folds). It evaluates the model using the log loss metric and collects the evaluation metrics. 
from sklearn.model_selection import cross_val_score 
from sklearn.metrics import make_scorer, log_loss 
import numpy as np 

#define the log loss scorer
log_loss_scorer = make_scorer(log_loss, greater_is_better=False, needs_proba=True)

#perform cross-validated fitting 
log_loss_scores = cross_val_score(
    xgb_pipeline0,   #the pipeline equivalent to xgb_wf0
    train_data.drop(columns=['churned']),
    train_data['churned'],  #target variable 
    cv=folds,   #Cross validation folds 
    scoring=log_loss_scorer   #log loss metric 
)



#print the log loss for each score 
print("Log loss scores for each fold:", log_loss_scores)
print("Mean log loss:", np.mean(log_loss_scores))


#Fits the XGBoost workflow (xgb_wf) to the entire training dataset (train_data_model). It trains the model on all available data (without cross-validation) for final model training or deployment. 
# Fit the pipeline to the entire training dataset
xgb_fit_total = xgb_pipeline.fit(
    train_data_model.drop(columns=['churned']),  # Features
    train_data_model['churned']  # Target variable
)


#explainability 
#Fits the XGBoost workflow (xgb_wf0) to the training portion of the split dataset (training(data_split)). It is typically used for model explanation or analysis on a subset of the data. 
# Fit the pipeline to the training portion of the split dataset
xgb_fit_explain = xgb_pipeline0.fit(
    train_data.drop(columns=['churned']),  # Features
    train_data['churned']  # Target variable
)


"""Makes predictions using the fitted XGBoostmodel (xgb_fit_explain) on the testing portion of the split dataset (testing(data_split)). 
It then calculates and prints:
The maximum predicted probability of churn.
The mean predicted probability of churn per sample.
The actual churning frequency in the testing data.
"""

# Make predictions on the testing data
predictions = xgb_fit_explain.predict_proba(val_data.drop(columns=['churned']))[:, 1]

# Calculate and print the maximum predicted probability
max_predicted_prob = np.max(predictions)
print(f"Maximal predicted probability: {max_predicted_prob}")

# Calculate and print the mean predicted probability per sample
mean_predicted_prob = np.mean(predictions)
print(f"Mean predicted probability per sample: {mean_predicted_prob}")

# Calculate and print the actual churning frequency in the testing data
actual_churn_freq = np.mean(pd.to_numeric(val_data['churned']))
print(f"Actual churning frequency per sample: {actual_churn_freq}")


#Calculate SHAP values on the testing split 
"""
calculates SHAP (SHapley Additive exPlanations) values for the XGBoost model using the shap package. 
SHAP values help explain the contribution of each feature to the model's predictions. 
"""
import shap

# Extract the XGBoost model from the pipeline
xgb_model = xgb_fit_explain.named_steps['model']

# Prepare the testing data (exclude the target column)
new_data = val_data.drop(columns=['churned'])

# Convert the testing data to a matrix format
new_data_matrix = new_data.values

# Initialize the SHAP explainer
explainer = shap.TreeExplainer(xgb_model)

# Calculate SHAP values
shap_values = explainer.shap_values(new_data_matrix)

# Visualize SHAP values for a single observation (e.g., the first observation)
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0, :], new_data.iloc[0, :])


#waterflow nplots for the 5 loyal customers 

"""
A SHAP waterfall plot is a powerful tool for visualizing how individual features contribute to a single prediction. 
The base value E[f(x)] is the average model output over the training dataset, however not given in probabilities, but in log-odds. 
The base value represents the starting point for the prediction. Each bar in the plot represents the contribution of a feature to the prediction. 
Positive contributions (yellow bars going to the right) increase the prediction towards not churning, 
while negative contributions (red bars going to the left) decrease the prediction towards churning. The values of the features are displayed alongside their contributions.
"""


import shap
import matplotlib.pyplot as plt

# Set the plot size
plt.figure(figsize=(15, 5))

# Plot SHAP values for the first row
shap.force_plot(explainer.expected_value, shap_values[0, :], new_data.iloc[0, :], matplotlib=True)
plt.title(f"Estimated churning probability: {round(predictions[0], 2)}")
plt.show()
# This sample corresponds to the first customer who is in activity cluster 0 and in this cluster has a mean tenure of 0.5 days. 
# These features pointing to a new customer, together with a large number of daily bank transfers push the churning estimate towards non churning.

# Plot SHAP values for the 1001st row
shap.force_plot(explainer.expected_value, shap_values[1000, :], new_data.iloc[1000, :], matplotlib=True)
plt.title(f"Estimated churning probability: {round(predictions[1000], 2)}")
plt.show()
# This sample also corresponds to a new customer. We can see that besides the new customer indicators, 
# ordering only a small number of bank_transfer_out contributes towards non churning.

# Plot SHAP values for the 13277th row
shap.force_plot(explainer.expected_value, shap_values[13276, :], new_data.iloc[13276, :], matplotlib=True)
plt.title(f"Estimated churning probability: {round(predictions[13276], 2)}")
plt.show()
# A small `tenure` still contributes towards not churning, but less than a very small `tenure`.

# Plot SHAP values for the 30000th row
shap.force_plot(explainer.expected_value, shap_values[29999, :], new_data.iloc[29999, :], matplotlib=True)
plt.title(f"Estimated churning probability: {round(predictions[29999], 2)}")
plt.show()
# A medium bank_transfer_in_volume contributes towards non churning, whereas a relatively large crypto_out_volume 
# and small bank_transfer_out_volume pushes the churning probability towards churning.

# Plot SHAP values for the 106298th row
shap.force_plot(explainer.expected_value, shap_values[106297, :], new_data.iloc[106297, :], matplotlib=True)
plt.title(f"Estimated churning probability: {round(predictions[106297], 2)}")
plt.show()
# A tenure of around one year still contributes towards non churning. A small bank_transfer_in_volume 
# and a small sum_bank_transfer_out_volume pushes the probability towards churning.


#Waterfall plot for 5 churners 
# Plot SHAP values for the 63584th row
shap.force_plot(explainer.expected_value, shap_values[63583, :], new_data.iloc[63583, :], matplotlib=True)
plt.title(f"Estimated churning probability: {round(predictions[63583], 2)}")
plt.show()
# Features, which contribute towards churning are a median `tenure` and median `activity_cluster`.

# Plot SHAP values for the 149008th row
shap.force_plot(explainer.expected_value, shap_values[149007, :], new_data.iloc[149007, :], matplotlib=True)
plt.title(f"Estimated churning probability: {round(predictions[149007], 2)}")
plt.show()
# Interestingly, a very high `bank_transfer_in_volume` also contributes towards churning.

# Plot SHAP values for the 388914th row
shap.force_plot(explainer.expected_value, shap_values[388913, :], new_data.iloc[388913, :], matplotlib=True)
plt.title(f"Estimated churning probability: {round(predictions[388913], 2)}")
plt.show()
# Several contacts to the bank contribute towards churning.

# Plot SHAP values for the 429022nd row
shap.force_plot(explainer.expected_value, shap_values[429021, :], new_data.iloc[429021, :], matplotlib=True)
plt.title(f"Estimated churning probability: {round(predictions[429021], 2)}")
plt.show()
# Being in the third date segment (after the banking strategy changed the second time) also contributes towards churning.

# Plot SHAP values for the 472185th row
shap.force_plot(explainer.expected_value, shap_values[472184, :], new_data.iloc[472184, :], matplotlib=True)
plt.title(f"Estimated churning probability: {round(predictions[472184], 2)}")
plt.show()
# A large number of bank transfers (`bank_transfer_out`) contribute towards churning.



#Beeswarm over all validation samples using SHAP and Matplotlib 
# Adjust plot size
plt.figure(figsize=(15, 10))

# Create the beeswarm plot
shap.summary_plot(shap_values, plot_type="bar", max_display=15)


#prediction 
#We now predict the test data with the model trained on all training data. 
#predict_proba method from XGBoost's Scikit-learn API.

# Make predictions and get probabilities
prediction_test = xgb_fit_total.predict_proba(test_data_model)



#Churning rules 
#Rule 1: Constant Probability of Churning 
#Calculate Churn Probablility by activity_gap 


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Group by `activity_gap` and calculate the frequency of churned customers creating a new dataframe churning_probability 
churning_probability = (
    data_train_reduced
    .groupby('activity_gap')
    .apply(lambda x: x['churned'].sum() / len(x))
    .reset_index(name='frequency_churned')
)

# Calculate Churn Frequency Relative to Total Churned Customers:
total_churned = len(data_train_reduced[data_train_reduced['churned'] == 1])

# Group by `activity_gap` and calculate churn frequency relative to total churned
churning_probability2 = (
    data_train_reduced
    .groupby('activity_gap')
    .apply(lambda x: x['churned'].sum() / total_churned)
    .reset_index(name='frequency_churned')
)

# Set plot dimensions
plt.figure(figsize=(15, 5))

# Plot cumulative density for churners
plt.plot(
    churning_probability['activity_gap'],
    np.cumsum(churning_probability['frequency_churned']) / np.sum(churning_probability['frequency_churned']),
    label="All Customers",
    color="black"
)

# Plot cumulative density for total churners
plt.plot(
    churning_probability2['activity_gap'],
    np.cumsum(churning_probability2['frequency_churned']),
    label="Churners",
    color="darkgrey"
)

# Add annotations
plt.text(100, 0.9, "Churners", color="darkgrey")
plt.text(100, 0.1, "All Customers", color="black")

# Labels and theme
plt.xlabel("Days since last activity")
plt.ylabel("Churning density")
plt.title("Churning Probability Density")
plt.grid(visible=True, linestyle='--', linewidth=0.5)
plt.legend()
plt.show()



#Most churners seem to decide to churn after a few days of activity.
#Calculates the average churn frequency over 11 intervals, each spanning a large date range

# Initialize variables
next_date = data_train_reduced['date'].min()
churner_freq_ave = 0

# Loop through 11 intervals
for i in range(11):
    start = next_date
    next_date = start + pd.Timedelta(days=500)

    # Define churned and total sets based on the date range
    churned_set = data_train_reduced[
        (data_train_reduced['churned'] == 1) &
        (data_train_reduced['date'] >= start) &
        (data_train_reduced['date'] < next_date)
    ]

    total_set = data_train_reduced[
        (data_train_reduced['date'] >= start) &
        (data_train_reduced['date'] < next_date)
    ]

    # Calculate unique customer ratios and accumulate
    churned_ratio = churned_set['customer_id'].nunique() / total_set['customer_id'].nunique()
    churner_freq_ave += churned_ratio / 10

# Output result
print(churner_freq_ave)


# Calculate the mean of predicted probabilities for the target class (e.g., class 1)
test_mean = np.mean(prediction_test[:, 1])  # Assuming prediction_test[:, 1] contains probabilities for class 1

# Calculate the final result
result = len(data_test['customer_id']) * test_mean / data_test['customer_id'].nunique()

# Output the result
print(result)


import pandas as pd

# Convert prediction_test to a DataFrame
prediction_test = pd.DataFrame(prediction_test, columns=['.pred', '.pred_1'])

# Adjust the predicted probabilities with the transactions factor
prediction_test['.pred_1'] *= transactions_factor

# Clamp the values to be between 0 and 1
prediction_test['.pred_1'] = prediction_test['.pred_1'].clip(0, 1)


#Rule 2: Churned due to fraud 
# Get the unique customer IDs where churn_due_to_fraud is True
churned_during_test = data_test[data_test['churn_due_to_fraud'] == True]['customer_id'].unique()

# Print the number of customers and the message
print(f"{len(churned_during_test)} customers churned due to fraud.")



#SUBMISSION 
submit = pd.read_csv(f"{path}/sample_submission.csv")
#apply 2 rules churned due to churning in the train data and the modeled churning probability.
tested_churning_rate = 0.05153
score = (tested_churning_rate -1) * np.log(1 - prediction_test['.pred'].mean()) - tested_churning_rate * np.log(prediction_test['.pred_1'].mean())
print(f"Score: {score}")

# When using the modeled churner frequency computed above.
data_test['churn'] = np.where(data_test['churn_due_to_fraud'] == True, 1, prediction_test['.pred_1'])

# Number of churning entries
print(f"Churning entries: {len(data_test[data_test['churn'] == 1])}")

# Churning entries in the public leaderboard
print(f"Churning entries in public: {len(data_test[(data_test['churn'] == 1) & (data_test['Usage'] == 'Public')])}")

# 67 entries were marked as churned due to fraud in the test data, 4 of them are evaluated in the public leaderboard.
# Update the submission file with churn predictions
submit['churn'] = data_test['churn']

# Save the submission file
submit.to_csv("submission.csv", index=False, quoting=2)  # quoting=2 corresponds to quote = FALSE in R

# Preview the first few rows of the submission file
print(submit.head())






