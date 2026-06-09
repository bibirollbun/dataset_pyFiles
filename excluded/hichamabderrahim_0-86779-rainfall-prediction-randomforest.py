import seaborn as sns
import pandas as pd
import warnings
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.dummy import DummyClassifier
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap


# Seaborn theme
sns.set_theme(style="whitegrid")

# Useful line of code to set the display option so we could see all the columns in pd dataframe
pd.set_option('display.max_columns', None)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Disable all Optuna logs except warnings and errors
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Global random seed
global_seed = 42


# Detect outliers in specified columns of a dataset using the Interquartile Range (IQR) method
def detect_outliers_iqr(dataset, columns):
    bounds = pd.DataFrame(index=['lower_bound', 'upper_bound'])
    outliers = {}
    for column in columns:
        # Calculate Q1, Q3 and IQR
        Q1 = dataset[column].quantile(0.25)
        Q3 = dataset[column].quantile(0.75)
        IQR = Q3 - Q1
        
        # Calculate the lower and upper bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Identify lower bounds and upper bounds
        bounds[column] = [lower_bound, upper_bound]
        
        # Identify outliers
        outliers[column] = dataset[(dataset[column] < lower_bound) | (dataset[column] > upper_bound)]
    
    return outliers, bounds


# Prints a separator line
def print_sl():
    print("=" * 50)


# Splits the DataFrame into features (X_train) and target (y_train)
def features_target_split(train_df):
    X_train = np.array(train_df.iloc[:,:-1])
    y_train = np.array(train_df.iloc[:,[-1]])
    return X_train, y_train


# Define file paths for the training and testing datasets
TRAIN_PATH = '/kaggle/input/playground-series-s5e3/train.csv'
TEST_PATH = '/kaggle/input/playground-series-s5e3/test.csv'


# Load the training and testing datasets into DataFrames
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)


train_df.set_index('id',inplace=True)
test_df.set_index('id',inplace=True)


target = 'rainfall'


train_df.head()


test_df.head()


print(f'Are there any null values in train ? : {train_df.isnull().any().any()}\n')
print(f'Are there any null values in test ? : {test_df.isnull().any().any()}\n')


# Count of Null Values in Test Data
test_df.isnull().sum()


# Extract non-null wind direction data
wind_direction_data = test_df['winddirection'].dropna()

# Convert the wind direction data to radians
wind_direction_data_rad = np.radians(wind_direction_data)

# Compute the circular mean
circular_mean = np.arctan2(np.sin(wind_direction_data_rad).mean(), np.cos(wind_direction_data_rad).mean())

# Convert the circular mean back to degrees
circular_mean_deg = np.degrees(circular_mean)

# Fill NaN in winddirection with the circular mean
test_df.fillna({'winddirection':circular_mean_deg},inplace=True)


# Preview of Data Types of Train Dataset Columns
train_df.dtypes


# Preview of Data Types of Test Dataset Columns
test_df.dtypes


# Transforming Rainfall Column to Boolean Format
train_df['rainfall'] = train_df['rainfall'].map({1: True, 0: False})


# Check Skewness of Train Dataset Columns
train_df.iloc[:,1:-1].skew()


# Check Skewness of Test Dataset Columns
test_df.iloc[:,1:].skew()


# Transform skewed columns

train_df['dewpoint'] = train_df['dewpoint'] ** 2 # Square Transformation
train_df['cloud'] = train_df['cloud'] ** 2 # Square Transformation
train_df['windspeed'] = np.log1p(train_df['windspeed']) # Log transformation

test_df['dewpoint'] = test_df['dewpoint'] ** 2 # Square Transformation
test_df['cloud'] = test_df['cloud'] ** 2 # Square Transformation
test_df['humidity'] = test_df['humidity'] ** 2 # Square Transformation


# Check Skewness Again of Train Dataset Columns
train_df.iloc[:,1:-1].skew()


# Check Skewness Again of Test Dataset Columns
test_df.iloc[:,1:].skew()


potential_outlier_columns = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']


# Detect outliers of train dataset
train_outliers, train_bounds = detect_outliers_iqr(train_df, potential_outlier_columns)

# Display the outliers
print("Train dataset outliers detected:")
for column, outlier_data in train_outliers.items():
    if not outlier_data.empty:
        print(outlier_data.loc[outlier_data.index,column])
        print_sl()
        


# Detect outliers of test dataset
test_outliers, test_bounds = detect_outliers_iqr(test_df, potential_outlier_columns)

# Display the outliers
print("Test dataset outliers detected:")
for column, outlier_data in test_outliers.items():
    if not outlier_data.empty:
        print(outlier_data.loc[outlier_data.index,column])
        print_sl()


# Impute Outliers with Median
for column in potential_outlier_columns:
    train_df[column] = train_df[column].apply(lambda x: train_df[column].median() if x < train_bounds.loc['lower_bound',column] or x > train_bounds.loc['upper_bound',column] else x)
    test_df[column] = test_df[column].apply(lambda x: test_df[column].median() if x < test_bounds.loc['lower_bound',column] or x > test_bounds.loc['upper_bound',column] else x)


# Train Dataset
train_df.describe()


# Test Dataset
test_df.describe()


# Pearson Correlation Matrix
train_correlation = train_df.corr()
test_correlation = test_df.corr()

# Create subplots (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# Heatmap for the train dataset
sns.heatmap(train_correlation, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[0])
axes[0].set_title("Train Dataset Correlation")

# Heatmap for the test dataset
sns.heatmap(test_correlation, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[1])
axes[1].set_title("Test Dataset Correlation")

# Adjust layout
plt.tight_layout()
plt.show()


# Create subplots for the train dataset
num_columns = len(train_df.columns)
fig, axes = plt.subplots(num_columns, 2, figsize=(12, num_columns * 5))

# Loop through all columns to generate histograms and boxplots
for i, column in enumerate(train_df.columns):
    # Check if the column is continuous or discrete
    if train_df[column].dtype in ['float64', 'int64']:  # Continuous columns
        # Histogram with KDE for continuous data
        sns.histplot(train_df[column], kde=True, bins=20, ax=axes[i, 0], color='blue')
        axes[i, 0].set_title(f'{column} Histogram')
        
        # Boxplot for continuous data
        sns.boxplot(x=train_df[column], ax=axes[i, 1], color='lightgreen')
        axes[i, 1].set_title(f'{column} Boxplot')

    else:  # Discrete columns (Categorical or other types)
        
        # Strip Plot for discrete data
        sns.stripplot(x=train_df[column], ax=axes[i, 0], color='lightblue', jitter=True)
        axes[i, 0].set_title(f'{column} Strip Plot')

        # Count plot for discrete data
        sns.countplot(x=train_df[column], ax=axes[i, 1], color='lightgreen')
        axes[i, 1].set_title(f'{column} Countplot')

# Adjust layout
plt.tight_layout()
plt.show()


# Create subplots for the test dataset
num_columns = len(test_df.columns)
fig, axes = plt.subplots(num_columns, 2, figsize=(12, num_columns * 5))

# Loop through all columns to generate histograms and boxplots
for i, column in enumerate(test_df.columns):
    # Check if the column is continuous or discrete
    if test_df[column].dtype in ['float64', 'int64']:  # Continuous columns
        # Histogram with KDE for continuous data
        sns.histplot(test_df[column], kde=True, bins=20, ax=axes[i, 0], color='blue')
        axes[i, 0].set_title(f'{column} Histogram')
        
        # Boxplot for continuous data
        sns.boxplot(x=test_df[column], ax=axes[i, 1], color='lightgreen')
        axes[i, 1].set_title(f'{column} Boxplot')

    else:  # Discrete columns (Categorical or other types)
        
        # Count plot for discrete data
        sns.countplot(x=test_df[column], ax=axes[i, 0], color='blue')
        axes[i, 0].set_title(f'{column} Countplot')

# Adjust layout
plt.tight_layout()
plt.show()


# Create pairplot for the train dataset
sns.pairplot(train_df,hue=target,corner=True)
plt.show()


# Plot temperature trends over days of the year
plt.figure(figsize=(10, 6))
plt.plot(train_df['day'], train_df['temparature'], label='Temperature')
plt.plot(train_df['day'], train_df['maxtemp'], label='Max Temp')
plt.plot(train_df['day'], train_df['mintemp'], label='Min Temp')
plt.title('Temperature Trend Over Days of the Year')
plt.xlabel('Day of Year')
plt.ylabel('Temperature')
plt.legend()
plt.show()


# Get the percentage distribution of each class in the target column
class_counts = train_df[target].value_counts(normalize=True) * 100
class_counts


# Visualize class percentages
class_counts.plot(kind='bar', color='skyblue', figsize=(8, 5))
plt.title("Class Distribution", fontsize=16)
plt.xlabel("Class", fontsize=14)
plt.ylabel("Percentage (%)", fontsize=14)
plt.show()


# Compute entropy of the target distribution
class_counts = train_df[target].value_counts(normalize=True)
entropy = -np.sum(class_counts * np.log2(class_counts))
is_imbalanced = entropy < 0.5  # A threshold of 0.5 can indicate imbalance
print(f'Is the train dataset imbalanced ? : {is_imbalanced}\n')


train_df['day_sin'] = np.sin(2 * np.pi * train_df['day'] / 365)
train_df['day_cos'] = np.cos(2 * np.pi * train_df['day'] / 365)
train_df['temp_dew_diff'] = train_df['temparature'] - train_df['dewpoint']
train_df['temp_diff'] = train_df['maxtemp'] - train_df['mintemp']
train_df['high_temp'] = train_df['temparature'].apply(lambda x: True if x > 35 else False)
train_df['low_temp'] = train_df['temparature'].apply(lambda x: True if x < -10 else False)
train_df['wind_speed_dir'] = train_df['windspeed'] * train_df['winddirection']
train_df['rolling_temp_mean'] = train_df['temparature'].rolling(window=7).mean() # 7-day rolling mean
train_df['rolling_temp_mean'].fillna(0,inplace=True)
train_df['rolling_wind_speed_std'] = train_df['windspeed'].rolling(window=7).std() # 7-day rolling mean
train_df['rolling_wind_speed_std'].fillna(0,inplace=True)
train_df['temp_humidity_interaction'] = train_df['temparature'] * train_df['humidity']

test_df['day_sin'] = np.sin(2 * np.pi * test_df['day'] / 365)
test_df['day_cos'] = np.cos(2 * np.pi * test_df['day'] / 365)
test_df['temp_dew_diff'] = test_df['temparature'] - test_df['dewpoint']
test_df['temp_diff'] = test_df['maxtemp'] - test_df['mintemp']
test_df['high_temp'] = test_df['temparature'].apply(lambda x: 1 if x > 35 else 0)
test_df['low_temp'] = test_df['temparature'].apply(lambda x: 1 if x < -10 else 0)
test_df['wind_speed_dir'] = test_df['windspeed'] * test_df['winddirection']
test_df['rolling_temp_mean'] = test_df['temparature'].rolling(window=7).mean() # 7-day rolling mean
test_df['rolling_temp_mean'].fillna(0,inplace=True)
test_df['rolling_wind_speed_std'] = test_df['windspeed'].rolling(window=7).std() # 7-day rolling mean
test_df['rolling_wind_speed_std'].fillna(0,inplace=True)
test_df['temp_humidity_interaction'] = test_df['temparature'] * test_df['humidity']

train_df[target] = train_df.pop(target)


scaler = MinMaxScaler()

# List of columns to scale
columns_to_scale = ['day','pressure','maxtemp','temparature',
                    'mintemp','dewpoint','humidity','cloud',
                    'sunshine','winddirection','windspeed',
                    'day_sin','day_cos','temp_dew_diff',
                    'temp_diff','wind_speed_dir','rolling_temp_mean',
                    'rolling_wind_speed_std','temp_humidity_interaction']

# Apply scaling to both train and test sets
train_df[columns_to_scale] = scaler.fit_transform(train_df[columns_to_scale])
test_df[columns_to_scale] = scaler.transform(test_df[columns_to_scale])


# Divides training dataset into features and target variables
X, y = features_target_split(train_df)

# Calculate Mutual Information
mi_scores = mutual_info_classif(X, y, random_state=global_seed)

# Convert scores to a DataFrame for easier interpretation
mi_df = pd.DataFrame({'MI Score': mi_scores}, index=train_df.columns[:-1],).sort_values(by='MI Score',ascending=False)

# Set a threshold
threshold = mi_df['MI Score'].mean()

# Filter features with MI scores
selected_features = mi_df[mi_df['MI Score'] > threshold].index.tolist()

# Divides training dataset into selected features and target variables
X, y = features_target_split(train_df[selected_features + [target]])


# TimeSeriesSplit with 5 splits
tscv = TimeSeriesSplit(n_splits=5)


# Initialize logistic regression model
model = LogisticRegression(random_state=global_seed)

auc_scores = []
# Perform time-series cross-validation
for train_index, test_index in tscv.split(X):
    # Split data into train and test sets
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict_proba(X_test)[:,1] > 0.5

    # Calculate AUC score
    auc = roc_auc_score(y_test, y_pred)  # Compare with binary target (y > 5)
    auc_scores.append(auc)

# Print the Mean AUC Score
print(f"Mean AUC: {np.mean(auc_scores)}")

# Plot the results
plt.plot(range(1, len(auc_scores) + 1), auc_scores, marker='o')
plt.title("AUC Scores")
plt.xlabel("Fold")
plt.ylabel("AUC")
plt.show()


# Initialize Dummy Classifier (random strategy)
random_model = DummyClassifier(strategy="uniform", random_state=global_seed)

auc_scores = []

# Perform time-series cross-validation
for train_index, test_index in tscv.split(X):
    # Split data into train and test sets
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Train the random model
    random_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_random = random_model.predict(X_test)
    
    # Calculate AUC score
    auc_random = roc_auc_score(y_test, y_pred_random)
    auc_scores.append(auc_random)

# Print the Mean AUC Score
print(f"Mean AUC: {np.mean(auc_scores)}")

# Plot the results
plt.plot(range(1, len(auc_scores) + 1), auc_scores, marker='o')
plt.title("AUC Scores")
plt.xlabel("Fold")
plt.ylabel("AUC")
plt.show()


# # Define the objective function for Optuna
# def objective(trial):
#     # Suggest hyperparameters for the Random Forest model
#     n_estimators = trial.suggest_int("n_estimators", 500, 1000)
#     max_depth = trial.suggest_int("max_depth", 2, 10)
#     min_samples_split = trial.suggest_int("min_samples_split", 15, 100)

#     # Define the Random Forest model
#     model = RandomForestClassifier(
#         n_estimators=n_estimators,
#         max_depth=max_depth,
#         min_samples_split=min_samples_split,
#         min_samples_leaf=2,
#         max_features='log2',
#         criterion='entropy',
#         n_jobs=1,  # Utilize one CPU core to avoids variability caused by parallelization
#         random_state=global_seed
#     )

#     # Calculate auc scores
#     auc_scores = []
#     # Time-series cross-validation
#     for train_index, test_index in tscv.split(X):
#         X_train, X_test = X[train_index], X[test_index]
#         y_train, y_test = y[train_index], y[test_index]

#         # Fit the Random Forest model
#         model.fit(X_train, y_train)

#         # Make predictions on the test set
#         y_pred = model.predict_proba(X_test)[:, 1]

#         # Calculate AUC score
#         auc = roc_auc_score(y_test, y_pred)
#         auc_scores.append(auc)

#     # Return the mean AUC score
#     return np.mean(auc_scores)

# # Create a study for hyperparameter optimization
# tpe_sampler = TPESampler(seed=global_seed) # Define a seed for Optuna's sampler to ensure reproducibility
# study = optuna.create_study(direction="maximize",sampler=tpe_sampler)
# study.optimize(objective, n_trials=1000, n_jobs=1, show_progress_bar=True)

# # Print the best AUC score
# print(f"Best AUC: {study.best_value}")

# # Print the best hyperparameters
# print(f"Best Hyperparameters: {study.best_params}")


# Define the Random Forest model
model = RandomForestClassifier(
    n_estimators=891,  # Number of trees
    max_depth=6,  # Maximum depth of trees
    min_samples_split=15,  # Minimum samples required to split a node
    min_samples_leaf=2,  # Minimum samples required at a leaf node
    max_features='log2',  # Number of features considered for splitting
    criterion='entropy',
    n_jobs=1,
    random_state=global_seed  # Ensure reproducibility
)

# Calculate residuals
residuals = []
# Identify misclassified samples
misclassified_samples = []
# Calculate auc scores
auc_scores = []
# Time-series cross-validation
for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Fit the Random Forest model
    model.fit(X_train, y_train)

    # Make predictions on the test set (probabilities for the positive class)
    y_pred = model.predict_proba(X_test)[:, 1]
    
    # Calculate residuals
    residuals.extend(y_test - y_pred)
    
    # Identify misclassified samples
    misclassified = np.where(y_test.flatten() != (y_pred >= 0.5))[0]
    misclassified_samples.extend(test_index[misclassified])
    
    # Calculate the AUC score
    auc = roc_auc_score(y_test, y_pred)
    auc_scores.append(auc)

# Print the mean AUC score
print(f"Mean AUC: {np.mean(auc_scores)}")


# Plot the results
plt.plot(range(1, len(auc_scores) + 1), auc_scores, marker='o')
plt.title("AUC Scores")
plt.xlabel("Fold")
plt.ylabel("AUC")
plt.show()


# Plot residuals
plt.hist(np.ravel(residuals), bins=50, alpha=0.75, color='blue')
plt.title('Residual Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# Display misclassified samples
print(f"Number of Misclassified samples: {len(misclassified_samples)}")
print(f"Misclassified samples indices: {misclassified_samples}")


# Get feature importance
importances = model.feature_importances_
feature_importance = sorted(zip(importances, selected_features), reverse=True)

print("Features contributed most to the misclassified samples:")
for importance, name in feature_importance[:3]:  # Top 3 features
    print(f"{name}: {importance}")


# Initialize SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Summary plot
shap.summary_plot(shap_values[1], X, feature_names=selected_features)


# Make predictions with the model on the test set
y_test_pred = model.predict_proba(test_df[selected_features].values)[:, 1]
y_test_pred = np.array(y_test_pred).flatten()

# Create the submission DataFrame
submission = pd.DataFrame({'id': test_df.index, 'rainfall': y_test_pred})

# Save the submission file
submission.to_csv('submission.csv', index=False)

print("Submission file created successfully.")


submission.head()

