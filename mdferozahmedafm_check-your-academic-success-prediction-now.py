import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



#Install the ucimlrepo package
!pip install ucimlrepo


# !pip install catboost
# !pip install kmodes
# !pip install optuna
!pip install kmodes



# Import Libraries

import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import math
from scipy.stats import skew
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, PolynomialFeatures,StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import  KFold, StratifiedKFold, cross_val_score, cross_validate, GridSearchCV
from sklearn import metrics
from ucimlrepo import fetch_ucirepo
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, make_scorer, f1_score
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.utils.class_weight import compute_class_weight
from kmodes.kmodes import KModes
import optuna
from optuna.samplers import TPESampler


import warnings
warnings.filterwarnings("ignore")


# fetch original dataset
predict_students_dropout_and_academic_success = fetch_ucirepo(id=697)

# data (as pandas dataframes)
X_original = predict_students_dropout_and_academic_success.data.features
y_original = predict_students_dropout_and_academic_success.data.targets

# metadata
print(predict_students_dropout_and_academic_success.metadata)

# variable information
print(predict_students_dropout_and_academic_success.variables)


print('Shape of train data is : ' , X_original.shape)
X_original.head(5)


column_mapping = {'Marital Status': 'Marital status'}
X_original.rename(columns=column_mapping, inplace=True)


#Loading the Dataset
train_data = pd.read_csv('/kaggle/input/playground-series-s4e6/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s4e6/test.csv')

#check the head of dataset
train_data.head(10)


train_data = train_data.drop(['id'] , axis=1)

print('Shape of train data is : ' , train_data.shape)
print('train data columns: ' , train_data.columns)


original_data = X_original.copy()
original_data['Target'] = y_original

train_data = pd.concat([train_data, original_data], ignore_index=True)
print('Shape of X is : ' , train_data.shape)


duplicated_rows = train_data.duplicated()
sum(duplicated_rows)


#some information about the attributes(datatypes & null values)
train_data.info()


#Check statistical information of numerical values

numerical_features = train_data.select_dtypes(include=[np.number])
train_data.describe(include=[np.number]).transpose()


#Check statistical information of categorical values

categorial_features = train_data.select_dtypes(include=object)
train_data.describe(include=object)


train_data['Target'] = train_data['Target'].replace({'Graduate': 0, 'Dropout':1, 'Enrolled': 2})

# Graduate    38491
# Dropout     26717
# Enrolled    15734


# Get the number of unique values for each column
unique_counts = train_data.nunique()
print(unique_counts)


from IPython.display import display, HTML

html_output = "<details><summary><strong> Click to View Value Frequencies Output</strong></summary><pre>"

for col in train_data.columns:
    html_output += f"Column '{col}':\n"
    html_output += f"Number of unique values: {train_data[col].nunique()}\n"
    html_output += f"Value frequencies:\n{train_data[col].value_counts().to_string()}\n\n"

html_output += "</pre></details>"

display(HTML(html_output))



categorical_features = [
    'Marital status', 'Application mode', 'Course', 'Daytime/evening attendance',
    'Previous qualification', 'Nacionality', 'Mother\'s qualification', 'Father\'s qualification',
    'Mother\'s occupation', 'Father\'s occupation', 'Displaced', 'Educational special needs',
    'Debtor', 'Tuition fees up to date', 'Gender', 'Scholarship holder', 'International'
]


train_data2 = train_data.drop(['Target'] , axis=1)
# Initialize dictionaries to store value frequencies, differences, and indices
value_frequencies_train = {}
value_frequencies_test = {}
unique_in_train_not_in_test = {}
indices_unique_in_train_not_in_test = {}

# Initialize a set to collect all unique indices
all_unique_indices = set()

# Loop through columns to get value frequencies and unique values
for col in train_data2.columns:
    value_frequencies_train[col] = train_data2[col].value_counts()
    if col in test_data.columns:
        value_frequencies_test[col] = test_data[col].value_counts()
    else:
        value_frequencies_test[col] = pd.Series()

    if col in categorical_features:
        unique_values = set(train_data2[col].unique()) - set(test_data[col].unique())
        unique_in_train_not_in_test[col] = unique_values

        # Find indices of samples with these unique values in train set
        if unique_values:
            indices = train_data2[train_data2[col].isin(unique_values)].index.tolist()
            indices_unique_in_train_not_in_test[col] = indices
            all_unique_indices.update(indices)  # Add these indices to the set

# Convert set to list to get all unique indices
all_unique_indices_list = list(all_unique_indices)

# Print results
for col in train_data2.columns:

    if col in categorical_features:
      if unique_in_train_not_in_test[col]:
        print(f"Column '{col}':")
        print(f"Values in train but not in test: {unique_in_train_not_in_test[col]}")
        print(f"Indices of samples with these unique values in train:")
        print(indices_unique_in_train_not_in_test[col])
        print()


# Print all unique indices across all categorical features
print("\n All indices of samples with unique categorical values in train but not in test:")
print(all_unique_indices_list)


target_values = train_data.loc[all_unique_indices_list, 'Target']
target_values.value_counts()


# Initialize dictionaries to store unique values and indices
unique_in_train_not_in_test = {}
indices_unique_in_train_not_in_test = {}

# Initialize a set to collect all unique indices
all_unique_indices = set()

# Loop through categorical columns to get unique values and indices
for col in categorical_features:
    unique_values = set(train_data[col].unique()) - set(test_data[col].unique())
    unique_in_train_not_in_test[col] = unique_values

    if unique_values:
        indices = train_data[train_data[col].isin(unique_values)].index.tolist()
        indices_unique_in_train_not_in_test[col] = indices
        all_unique_indices.update(indices)  # Add these indices to the set

# Convert set to list to get all unique indices
all_unique_indices_list = list(all_unique_indices)

# Function to get the third most frequent value
def get_third_most_frequent(series):
    counts = series.value_counts()
    if len(counts) >= 3:
        return counts.index[2]  # Third most frequent
    elif len(counts) > 0:
        return counts.index[-1]  # Use the least frequent if fewer than 3
    else:
        return None  # In case there are no valid values

# Impute unique values based on the third most frequent value within the same target group
for col in categorical_features:
    if col in unique_in_train_not_in_test and unique_in_train_not_in_test[col]:
        for index in indices_unique_in_train_not_in_test[col]:
            # Get the target value for the current index
            target_value = train_data.loc[index, 'Target']
            # Compute third most frequent value of the column for the specific target group
            third_most_value = get_third_most_frequent(train_data[train_data['Target'] == target_value][col])
            # Impute the unique value with the third most frequent value
            train_data.at[index, col] = third_most_value

# Example for checking the imputed values with target labels:
print("Imputed values for columns with unique values:")

for col in categorical_features:
    if col in unique_in_train_not_in_test and unique_in_train_not_in_test[col]:
        print(f"Column '{col}': Imputed with third most frequent value for corresponding target value")
        imputed_values = train_data.loc[indices_unique_in_train_not_in_test[col], [col, 'Target']]
        # Print each imputed value with its corresponding target label
        for idx, row in imputed_values.iterrows():
            print(f"{idx}: {row[col]} (Target: {row['Target']})")


test_ids = test_data['id']
test_data = test_data.drop('id', axis=1)

# Combine train and test for consistent label encoding (excluding target column in train)
combined = pd.concat([train_data.drop(columns=['Target']), test_data], ignore_index=True)
y_train = train_data['Target']

# Label encode categorical features
for col in categorical_features:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col].astype(str))

# Separate back into train and test
train_data = combined.iloc[:len(train_data)]
test_data = combined.iloc[len(train_data):]

# Add back the target column to train dataset only
train_data['Target'] = y_train


train_data = train_data[~((train_data['Target'] == 0) & (train_data['Curricular units 1st sem (grade)'] == 0) & (train_data['Curricular units 2nd sem (grade)'] == 0))]
train_data = train_data[~((train_data['Target'] == 0) & (train_data['Curricular units 1st sem (evaluations)'] == 0)  & (train_data['Curricular units 1st sem (approved)'] == 0) & (train_data['Curricular units 1st sem (without evaluations)'] == 0) & (train_data['Curricular units 1st sem (grade)'] != 0) & (train_data['Curricular units 2nd sem (grade)'] == 0))]


def Find_inconsistencies(df):

  for index,row in df.iterrows():
    #semester1 :
    #A:
    if (row['Curricular units 1st sem (approved)'] > row['Curricular units 1st sem (enrolled)']) & (row['Curricular units 1st sem (approved)'] > row['Curricular units 1st sem (evaluations)']):
      if (row['Curricular units 1st sem (enrolled)'] == 0 ) & (row['Curricular units 1st sem (evaluations)'] == 0 ) & \
         (row['Curricular units 1st sem (grade)'] == 0):
         row['Curricular units 1st sem (approved)'] = 0
      elif  (row['Curricular units 1st sem (enrolled)'] == 0 ) & (row['Curricular units 1st sem (evaluations)'] == 0 ) & \
         (row['Curricular units 1st sem (grade)'] != 0):
         row['Curricular units 1st sem (enrolled)'] = row['Curricular units 1st sem (approved)']
         row['Curricular units 1st sem (evaluations)'] = row['Curricular units 1st sem (approved)']
      else:
        row['Curricular units 1st sem (approved)'] = row['Curricular units 1st sem (enrolled)']
    #B:
    if (row['Curricular units 1st sem (approved)'] > row['Curricular units 1st sem (enrolled)']):
      if (row['Curricular units 1st sem (enrolled)'] == 0 ) & (row['Curricular units 1st sem (grade)'] != 0):
        row['Curricular units 1st sem (enrolled)'] = row['Curricular units 1st sem (approved)']
      else:
        row['Curricular units 1st sem (approved)'] = row['Curricular units 1st sem (enrolled)']
    #C:
    if (row['Curricular units 1st sem (approved)'] > row['Curricular units 1st sem (evaluations)']):
      if (row['Curricular units 1st sem (evaluations)'] == 0 ) & (row['Curricular units 1st sem (grade)'] == 0):
        row['Curricular units 1st sem (approved)'] = 0
      elif (row['Curricular units 1st sem (grade)'] != 0):
        row['Curricular units 1st sem (evaluations)'] = row['Curricular units 1st sem (approved)']

    #semester21 :
    #A:
    if (row['Curricular units 2nd sem (approved)'] > row['Curricular units 2nd sem (enrolled)']) & (row['Curricular units 2nd sem (approved)'] > row['Curricular units 2nd sem (evaluations)']):
      if (row['Curricular units 2nd sem (enrolled)'] == 0 ) & (row['Curricular units 2nd sem (evaluations)'] == 0 ) & \
         (row['Curricular units 2nd sem (grade)'] == 0):
         row['Curricular units 2nd sem (approved)'] = 0
      elif  (row['Curricular units 2nd sem (enrolled)'] == 0 ) & (row['Curricular units 2nd sem (evaluations)'] == 0 ) & \
         (row['Curricular units 2nd sem (grade)'] != 0):
         row['Curricular units 2nd sem (enrolled)'] = row['Curricular units 2nd sem (approved)']
         row['Curricular units 2nd sem (evaluations)'] = row['Curricular units 2nd sem (approved)']
      else:
        row['Curricular units 2nd sem (approved)'] = row['Curricular units 2nd sem (enrolled)']
    #B:
    if (row['Curricular units 2nd sem (approved)'] > row['Curricular units 2nd sem (enrolled)']):
      if (row['Curricular units 2nd sem (enrolled)'] == 0 ) & (row['Curricular units 2nd sem (grade)'] != 0):
        row['Curricular units 2nd sem (enrolled)'] = row['Curricular units 2nd sem (approved)']
      else:
        row['Curricular units 2nd sem (approved)'] = row['Curricular units 2nd sem (enrolled)']
    #C:
    if (row['Curricular units 2nd sem (approved)'] > row['Curricular units 2nd sem (evaluations)']):
      if (row['Curricular units 2nd sem (evaluations)'] == 0 ) & (row['Curricular units 2nd sem (grade)'] == 0):
        row['Curricular units 2nd sem (approved)'] = 0
      elif (row['Curricular units 2nd sem (grade)'] != 0):
        row['Curricular units 2nd sem (evaluations)'] = row['Curricular units 2nd sem (approved)']

  return df

train_data = Find_inconsistencies(train_data)
test_data = Find_inconsistencies(test_data)


train_data.shape


categorical_features = [
    'Marital status', 'Application mode', 'Course', 'Daytime/evening attendance',
    'Previous qualification', 'Nacionality', 'Mother\'s qualification', 'Father\'s qualification',
    'Mother\'s occupation', 'Father\'s occupation', 'Displaced', 'Educational special needs',
    'Debtor', 'Tuition fees up to date', 'Gender', 'Scholarship holder', 'International'
]
print('Count of categorical features is: ', len(categorical_features))

numerical_features = train_data.drop(columns=categorical_features, axis=1)
numerical_features = numerical_features.drop(columns=['Target'], axis=1)
num_features = numerical_features.columns.tolist()
print('Count of numerical features is: ', numerical_features.shape[1])


def plot_feature_distributions(data, target):
    cat_cols = [col for col in data.columns if data[col].dtype == 'O' or data[col].nunique() < 100 and col != target]
    num_cols = [col for col in data.columns if col not in cat_cols and col != target]

    # Number of subplots including the pie chart for the target column
    total_plots = len(cat_cols) + len(num_cols) + 1  # +1 for the pie chart
    n_cols = 2
    n_rows = int(np.ceil(total_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 5))
    axes = axes.flatten()

    # Plot pie chart for the target variable
    target_counts = data[target].value_counts(normalize=True)
    colors = ['#66c2a5', '#fc8d62', '#8da0cb']  # Add more colors for additional classes
    axes[0].pie(target_counts, labels=target_counts.index, autopct='%1.1f%%', colors=colors[:len(target_counts)], startangle=90)
    axes[0].set_title(f"Distribution of {target}")
    
    # Loop through categorical columns
    for idx, col in enumerate(cat_cols):
        contingency_table = pd.crosstab(data[col], data[target], normalize='index')
        contingency_table.plot(kind="bar", stacked=True, color=colors[:len(target_counts)], ax=axes[idx+1])  # +1 to account for the pie chart
        axes[idx+1].set_title(f"Percentage Distribution of {target} across {col}")
        axes[idx+1].set_xlabel(col)
        axes[idx+1].set_ylabel("Percentage")
        axes[idx+1].legend(title=target, loc='upper right')

    # Loop through numerical columns
    for idx, col in enumerate(num_cols, start=len(cat_cols) + 1):  # +1 to account for the pie chart
        # Check skewness and compressed to axis condition
        if data[col].dtype != 'O' and skew(data[col]) > 0.75:
            sns.histplot(data=data, x=col, hue=target, kde=True, ax=axes[idx], palette=colors[:len(target_counts)], bins=50, kde_kws={'bw_adjust': 0.5})
        else:
            sns.histplot(data=data, x=col, hue=target, kde=True, ax=axes[idx], palette=colors[:len(target_counts)], bins='auto', kde_kws={'bw_adjust': 0.5})

        axes[idx].set_title(f"Distribution of {col} colored by {target}")
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel("Density")

    # Remove any extra empty plots if the number of subplots is odd
    for i in range(total_plots, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

plot_feature_distributions(train_data, 'Target')


skew_features = numerical_features.skew().sort_values(ascending=False)
skew_features


skew_newfeatures = numerical_features.skew().sort_values(ascending=False)

skew_limit = 2
# Showing the skewed columns
skew_cols = (skew_newfeatures
             .sort_values(ascending=False)
             .to_frame()
             .rename(columns={0:'Skew'})
             .query('abs(Skew) > {}'.format(skew_limit)))
skew_cols


# #perrform the skew transformation:
# for col in skew_cols.index.values:
#     train_data[col] = train_data[col].apply(np.log1p


def plot_boxplots(data, n_cols=3):

    # Filter columns with more than two unique values
    filtered_cols = [col for col in data.columns if data[col].nunique() > 2]

    # Calculate the number of rows needed
    n_rows = (len(filtered_cols) + n_cols - 1) // n_cols

    # Create a figure and axes with the calculated number of subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 5))
    axes = axes.flatten()

    # Loop through each filtered column and create a boxplot
    for i, col in enumerate(filtered_cols):
        sns.boxplot(y=data[col], ax=axes[i], palette=['#66c2a5'])
        axes[i].set_title(f'Boxplot of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Value')

    # Remove any extra empty subplots if the number of columns doesn't divide evenly
    for j in range(len(filtered_cols), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


plot_boxplots(train_data)


# Define a function to find outliers based on IQR
def find_outliers(df):
    outliers = {}
    imputed_df = df.copy()
    for col in df.columns:
        v = df[col]
        q1 = v.quantile(0.25)
        q3 = v.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers_count = ((v < lower_bound) | (v > upper_bound)).sum()
        perc = outliers_count * 100.0 / len(df)
        outliers[col] = (perc, outliers_count)
        print(f"Column {col} outliers = {perc:.2f}% ({outliers_count} out of {len(df)})")

    return outliers

# Find outliers in the DataFrame
outliers = find_outliers(numerical_features)


# # Compute the correlation matrix
correlation_matrix = train_data.corr()
target_correlation = correlation_matrix['Target'].sort_values(ascending=False)
print(target_correlation)


# Plot the heatmap
plt.figure(figsize=(12, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".3f", annot_kws={"size": 6})
plt.title('Correlation Matrix')
plt.show()


X_train = train_data.drop('Target', axis=1)
y_train = train_data['Target']


# Feature Engineering

def Feature_Engineering(data,clean_data):

    epsilon = 1e-9

    # Total Number of Evaluations in 1st and 2nd Semesters
    clean_data['Total_Evaluations_1st_Semester'] = clean_data['Curricular units 1st sem (evaluations)'] + clean_data['Curricular units 1st sem (without evaluations)']
    clean_data['Total_Evaluations_2nd_Semester'] = clean_data['Curricular units 2nd sem (evaluations)'] + clean_data['Curricular units 2nd sem (without evaluations)']
    # Total Number of Curricular Units in 1st and 2nd Semesters
    clean_data['Total_Curricular_Units_1st_Semester'] = clean_data['Curricular units 1st sem (credited)'] + clean_data['Curricular units 1st sem (enrolled)']
    clean_data['Total_Curricular_Units_2nd_Semester'] = clean_data['Curricular units 2nd sem (credited)'] + clean_data['Curricular units 2nd sem (enrolled)']
    # Total Number of Curricular Units Approved in Both Semesters
    clean_data['Total_approved_Curricular_Units'] = clean_data['Curricular units 1st sem (approved)'] + clean_data['Curricular units 2nd sem (approved)']
    # Total Number of Curricular Units Enrolled in Both Semesters
    clean_data['Total_enrolled_Curricular_Units'] = clean_data['Curricular units 1st sem (enrolled)'] + clean_data['Curricular units 2nd sem (enrolled)']

    data['Total_approved_Curricular_Units'] = clean_data['Curricular units 1st sem (approved)'] + clean_data['Curricular units 2nd sem (approved)']
    data['Average_Grade'] = (clean_data['Curricular units 1st sem (grade)'] + clean_data['Curricular units 2nd sem (grade)']) / 2
    data['Approval_Ratio'] = clean_data['Total_approved_Curricular_Units'] / (clean_data['Total_Curricular_Units_1st_Semester'] + clean_data['Total_Curricular_Units_2nd_Semester'] + 1)
    data['Total_Academic_Load'] = clean_data['Total_Curricular_Units_1st_Semester'] + clean_data['Total_Curricular_Units_2nd_Semester'] + clean_data['Total_Evaluations_1st_Semester'] + clean_data['Total_Evaluations_2nd_Semester']
    # Performance_Ratio: Reflects the efficiency of students in passing the curricular units relative to their academic load.
    data['Performance_Ratio'] = clean_data['Total_approved_Curricular_Units'] / (data['Total_Academic_Load'] + 1)
    # Total Enrolled Units by Age Group
    data['Total_Enrolled_Units_by_Age'] = clean_data.groupby('Age at enrollment')['Total_enrolled_Curricular_Units'].transform('sum')


    return data

cleaned_data_df = X_train.copy()
cleaned_test_df = test_data.copy()


X_train = Feature_Engineering(X_train, cleaned_data_df)
test_data = Feature_Engineering(test_data, cleaned_test_df)


# Install kmodes (only needed once)
!pip install kmodes

# Then import
from kmodes.kmodes import KModes



import kmodes
print(kmodes.__version__)



from kmodes.kmodes import KModes

categoricalKmodes_columns = ['Mother\'s qualification', 'Father\'s qualification', 'Mother\'s occupation', 'Father\'s occupation']

# Extract the categorical features
categorical_Kmodesdata = X_train[categoricalKmodes_columns].copy()
categorical_Kmodesdata_test = test_data[categoricalKmodes_columns].copy()

# Determine number of clusters (k)
k = 3  # Initial value based on number of classes
k_range = range(2, 6)  # Range to test if desired

# Dictionary to store inertia for different k values
inertia_dict = {}

# Perform K-Modes clustering for different values of k
for k_val in k_range:
    km = KModes(n_clusters=k_val, init='Cao', n_init=5, verbose=1, random_state=42)
    clusters = km.fit_predict(categorical_Kmodesdata)
    inertia_dict[k_val] = km.cost_
    print(f'K: {k_val}, Inertia: {km.cost_}')

# Select best k based on inertia (elbow method can be applied here)
best_k = min(inertia_dict, key=inertia_dict.get)
print(f'Selected K: {best_k}')

# Fit the final model with selected k
km_final = KModes(n_clusters=best_k, init='Cao', n_init=5, verbose=1, random_state=42)
clusters_final = km_final.fit_predict(categorical_Kmodesdata)
clusters_final_test = km_final.predict(categorical_Kmodesdata_test)

# Add the clusters as a new feature in X_train
X_train['kmodes_cluster'] = clusters_final
test_data['kmodes_cluster'] = clusters_final_test


new_features = ['Total_approved_Curricular_Units', 'Average_Grade', 'Approval_Ratio', 'Total_Academic_Load', 'Performance_Ratio','Total_Enrolled_Units_by_Age', 'kmodes_cluster']
X_new = X_train[new_features].copy()
X_new['Target'] = y_train


plot_feature_distributions(X_new, 'Target')


def evaluation(model_name, pipeline_model, X, y):

  scoring = {
        'accuracy': 'accuracy',
        'f1_score_macro': make_scorer(f1_score, average='macro')
    }

  skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)   
  scores = cross_validate(pipeline_model, X, y, cv=skf, scoring=scoring , n_jobs=-1 , return_train_score=True, return_estimator=True )


  # Print evaluation results
  print(f"\n{model_name} Cross-Validation Scores:")
  print(f"Train Accuracy: {scores['train_accuracy'].mean():.6f} Â± {scores['train_accuracy'].std():.6f}")
  print(f"Train F1 Score : {scores['train_f1_score_macro'].mean():.6f} Â± {scores['train_f1_score_macro'].std():.6f}")
  print(f"Test Accuracy: {scores['test_accuracy'].mean():.6f} Â± {scores['test_accuracy'].std():.6f}")
  print(f"Test F1 Score : {scores['test_f1_score_macro'].mean():.6f} Â± {scores['test_f1_score_macro'].std():.6f}")


classes = np.unique(y_train)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weights_dict = {i: class_weights[i] for i in range(len(classes))}


# def objective(trial, model_class, X_train, y_train, n_splits=5, random_state=42):
def objective(trial, model_class, X_train, y_train, class_weights, n_splits=5, random_state=42):
    """
    Objective function for Optuna hyperparameter tuning.

    Parameters:
    - trial: Optuna trial object
    - model_class: The classifier class (e.g., XGBClassifier, LGBMClassifier, CatBoostClassifier)
    - X_train: Training features
    - y_train: Training labels
    - class_weights: Dictionary of class weights
    - n_splits: Number of splits for Stratified K-Folds cross-validation
    - random_state: Random seed

    Returns:
    - Mean accuracy score across cross-validation folds
    """
    if model_class == XGBClassifier:
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000), 
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),   
            'max_depth': trial.suggest_int('max_depth', 3, 10), 
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 0.01, 10.0), 
            'reg_lambda': trial.suggest_loguniform('reg_lambda',0.01, 10.0), 
            'scale_pos_weight': trial.suggest_categorical('scale_pos_weight', [class_weights[1] / class_weights[0]])
        }
        model = model_class(**params, objective='multi:softprob', num_class=len(np.unique(y_train)),
                            random_state=random_state)

    elif model_class == LGBMClassifier:
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100,2000),  
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01,0.3),  
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 0.01, 10.0), 
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 0.01, 10.0),
            'class_weight': class_weights_dict
        }
        model = model_class(**params, objective='multiclass', random_state=random_state)

    elif model_class == CatBoostClassifier:
        params = {
            'iterations': trial.suggest_int('iterations', 1000, 2000),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.1, 0.15),
            'depth': trial.suggest_int('depth', 3, 7),
            'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 2, 5),
            'class_weights': class_weights
        }
        model = model_class(**params, loss_function='MultiClass', random_state=random_state, verbose=0)

    # Create pipeline with StandardScaler and model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scores = []
    for train_index, test_index in skf.split(X_train, y_train):
        X_tr, X_te = X_train.iloc[train_index].copy(), X_train.iloc[test_index].copy()  # Use iloc for integer indexing
        y_tr, y_te = y_train.iloc[train_index], y_train.iloc[test_index]
        pipeline.fit(X_tr, y_tr)
        y_pred = pipeline.predict(X_te)
        scores.append(accuracy_score(y_te, y_pred))

    return np.mean(scores)


def tune_hyperparameters(X_train, y_train, model_class, class_weights, n_trials=20):
    """
    Tune hyperparameters using Optuna for a given model class.

    Parameters:
    - X_train: Training features
    - y_train: Training labels
    - model_class: The classifier class (e.g., XGBClassifier, LGBMClassifier, CatBoostClassifier)
    - class_weights: Dictionary of class weights
    - n_trials: Number of optimization trials

    Returns:
    - Dictionary of best hyperparameters
    """
    study = optuna.create_study(direction='maximize', sampler=TPESampler())
    study.optimize(lambda trial: objective(trial, model_class, X_train, y_train, class_weights), n_trials=n_trials)
    return study.best_params


# Create scaler
scaler = StandardScaler()

# Tune hyperparameters
xgb_best_params = tune_hyperparameters(X_train, y_train, XGBClassifier, class_weights_dict)

# Create pipeline
pipeline_model_xgb = Pipeline([
    ('scaler', scaler),
    ('model', XGBClassifier(**xgb_best_params,
                            objective='multi:softprob',
                            num_class=len(np.unique(y_train)),
                            random_state=42))
])

print("Best parameters for XGBoost:", xgb_best_params)

# Evaluate
evaluation('XGBoost_clf', pipeline_model_xgb, X_train, y_train)


