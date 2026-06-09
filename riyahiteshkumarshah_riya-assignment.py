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


 # use pandas to load your data to dataframe objects
import pandas as pd

# load the training, testing, and sample submission data
training_data = pd.read_csv(r'/kaggle/input/playground-series-s4e6/train.csv')
testing_data = pd.read_csv(r'/kaggle/input/playground-series-s4e6/test.csv')
sample_submission = pd.read_csv(r'/kaggle/input/playground-series-s4e6/sample_submission.csv')


# verify the data was loaded
for dataset in [training_data, testing_data, sample_submission]:
    print(f"dataset shape: {dataset.shape}")


# verify the data was loaded
for dataset in [training_data, testing_data, sample_submission]:
    print(dataset.info())


# verify the data was loaded
print("Training_data:")
print(training_data.head(5))
print("\nTesting_data:")
print(testing_data.head(5))
print("\nsample_submission:")
print(sample_submission.head(5))


# verify the data for missing values
for dataset_name, dataset in zip(["Training Data", "Testing Data", "Sample Submission"], 
                                 [training_data, testing_data, sample_submission]):
    
    print(f"\nðŸ”¹ {dataset_name} - Missing Values:")
    print(dataset.isnull().sum()) 


# Data Profiling Function
def create_data_profiling_df(data: pd.DataFrame) -> pd.DataFrame:
    data_profiling_df = pd.DataFrame(columns=[
        "column_name", "data_type", "values", "null_values", "percent_null", "unique_values",
        "duplicate_values", "min", "max", "mean", "median", "stdev", "IQR", "skewness", "most_common_value", "outliers"
    ])
    for column in data.columns:
        column_dict = {}
        try:
            column_dict["column_name"] = column
            column_dict["data_type"] = data[column].dtypes
            column_dict["values"] = data[column].notnull().sum()
            column_dict["null_values"] = data[column].isna().sum()
            column_dict["percent_null"] = round(data[column].isna().sum() / len(data[column]), 2)
            column_dict["unique_values"] = len(data[column].unique())
            column_dict["duplicate_values"] = data[column].duplicated().sum()
            column_dict["min"] = data[column].min() if np.issubdtype(data[column].dtype, np.number) else "NA"
            column_dict["max"] = round(data[column].max(), 1) if np.issubdtype(data[column].dtype, np.number) else "NA"
            column_dict["mean"] = round(data[column].mean(), 1) if np.issubdtype(data[column].dtype, np.number) else "NA"
            column_dict["median"] = round(data[column].median(), 1) if np.issubdtype(data[column].dtype, np.number) else "NA"
            column_dict["stdev"] = round(data[column].std(), 1) if np.issubdtype(data[column].dtype, np.number) else "NA"
            column_dict["IQR"] = round(data[column].quantile(.75) - data[column].quantile(.25), 1) if np.issubdtype(data[column].dtype, np.number) else "NA"
            column_dict["most_common_value"] = data[column].mode().iloc[0] if not data[column].mode().empty else "NA"
            column_dict["skewness"] = data[column].skew(skipna=True) if np.issubdtype(data[column].dtype, np.number) else "NA"
            if np.issubdtype(data[column].dtype, np.number):
                Q1 = data[column].quantile(0.25)
                Q3 = data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)][column]
                column_dict["outliers"] = len(outliers)
            else:
                column_dict["outliers"] = "NA"
        except Exception as e:
            print(f"Unable to read column: {column}, error: {e}")
        data_profiling_df = pd.concat([data_profiling_df, pd.DataFrame([column_dict])], ignore_index=True)
    print(f"Data profiling complete, dataframe contains {len(data_profiling_df)} columns")
    return data_profiling_df


# run the data profiling function
data_profiling_df = create_data_profiling_df(data = training_data)

# print the dataframe
data_profiling_df


# import needed libraries
import matplotlib.pyplot as plt
import seaborn as sns

# define function to plot histogram and identify outliers
def plot_histogram(df: pd.DataFrame,
                   variable: str,
                   bins=10,
                   color='grey',
                   edgecolor='black',
                   figsize=(7, 2),
                   iqr_on = False):

    # set the figure size
    plt.figure(figsize=figsize)

    # plot the histogram
    plt.hist(df[variable],
             bins=bins,
             color=color,
             edgecolor=edgecolor)

    # customize the plot labels and colors
    plt.title(f'{variable} Histogram')
    plt.xlabel(f'{variable}')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.ticklabel_format(style='plain', axis='x')
    plt.grid(True)

    # define the Inter Quartile Range (iqr) and outlier bounds
    q1 = df[variable].quantile(0.25)
    q3 = df[variable].quantile(0.75)
    iqr = q3 - q1
    if iqr_on == True:
      lower_bound = q1
      upper_bound = q3
    else:
      lower_bound = q1 - 1.5 * iqr
      upper_bound = q3 + 1.5 * iqr

    # mark the outlier boundson the histogram
    plt.axvline(lower_bound, color='blue', linestyle='dashed', linewidth=2, label='Lower Bound')
    plt.axvline(upper_bound, color='blue', linestyle='dashed', linewidth=2, label='Upper Bound')

    # Show the plot
    plt.legend()
    plt.show()

    # count the outliers
    num_outliers = ((df[variable] < lower_bound) | (df[variable] > upper_bound)).sum()

    # print information about outliers
    if num_outliers > 0:
        print(f"{num_outliers} potential outliers detected in {variable} distribution")
        print(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")
    else:
        print(f"no potential outliers detected in {variable} distribution")
        print(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")

    # print a new line
    print("""
          -----
          """)


# Exploratory Data Analysis (EDA)
sns.countplot(x='Target', data=training_data)
plt.title("Target Variable Distribution")
plt.show()


# Run the histogram function on appropriate numerical features
numerical_features = ["Age at enrollment", "Admission grade", "Unemployment rate", "Inflation rate", "GDP"]
for feature in numerical_features:
    plot_histogram(df=training_data, variable=feature, bins=15)



# create a function to make a bar chart of the count of categorical variables
def count_plot(df: pd.DataFrame,
               variable: str):

    plt.figure(figsize=(4, 2))
    sns.countplot(data = df,
                  x = f"{variable}",
                  color = "grey")
    plt.title(f'Count of {variable}')
    plt.xlabel(f'{variable}')
    plt.ylabel('count')
    plt.show()
    print("""
    -----
    """)


# Run the count plot function on categorical variables
categorical_features = ["Gender", "Scholarship holder", "Daytime/evening attendance", "Previous qualification (grade)"]
for feature in categorical_features:
    count_plot(df=training_data, variable=feature)



import numpy as np
import pandas as pd

# Function to log transform any column
def log_transform_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Log transform a column (if not null) with a log(x+1) transformation.
    """
    df = df.copy()
    df[column] = df[column].apply(lambda x: np.log(x + 1) if pd.notnull(x) else x)
    return df

# Function to fill missing Age values safely
def fill_age_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values in 'Age at enrollment' using the median or default value.
    """
    df = df.copy()
    median_age = df["Age at enrollment"].median()
    df["Age at enrollment"] = df["Age at enrollment"].fillna(median_age)
    return df

# Function to fill missing values with the mode safely
def fill_column_with_mode(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Fill missing values in a column with the most frequent value (mode).
    """
    df = df.copy()
    most_frequent_value = df[column].mode()[0]
    df[column] = df[column].fillna(most_frequent_value)
    return df

# Apply transformations to train_df
training_data = log_transform_column(training_data, 'Admission grade')
training_data = log_transform_column(training_data, 'Previous qualification (grade)')

training_data = fill_age_column(training_data)

training_data = fill_column_with_mode(training_data, 'Course')
training_data = fill_column_with_mode(training_data, 'Application mode')

# Print first few rows to check
print(training_data.head(5))



plotting_df = training_data.copy()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Function to plot the influence of a new feature
def plot_new_feature(df: pd.DataFrame, new_feature_name: str, legend_loc='upper right'):
    # Copy the original dataframe
    plot_df = df.copy()

    # Check if 'Target' column exists and map its values (update if necessary)
    if 'Target' in plot_df.columns:
        plot_df['Target'] = plot_df['Target'].map({'Graduated': 1, 'Enrolled': 0.5, 'Dropout': 0})  # Update this line if necessary

    # Calculate the success rate for each unique value of the new feature
    new_feature_success_rate = plot_df.groupby(new_feature_name)['Target'].mean().reset_index()

    # Sort the data by success rate
    new_feature_success_rate = new_feature_success_rate.sort_values(by='Target', ascending=False)

    # Calculate the average success rate
    average_success_rate = plot_df['Target'].mean()

    # Plot the data
    plt.figure(figsize=(12, 3))
    sns.barplot(x=new_feature_name, y='Target', data=new_feature_success_rate, color='grey', 
                order=new_feature_success_rate[new_feature_name])

    # Add a horizontal line for the average success rate
    plt.axhline(average_success_rate, color='blue', linestyle='--', 
                label=f'Average Success Rate ({average_success_rate:.2f})')

    # Add labels and title
    plt.ylabel('Success Rate')
    plt.title(f'Success Rate by {new_feature_name}')
    
    # Add the legend with loc parameter
    plt.legend(loc=legend_loc)  # You can change 'upper right' to other positions like 'upper left', 'lower right', etc.
    
    # Show the plot
    plt.show()




# Example: Creating a new feature - Overall Grade Average
plotting_df['OverallGrade'] = (plotting_df['Curricular units 1st sem (grade)'] + plotting_df['Curricular units 2nd sem (grade)']) / 2
plotting_df['OverallGrade'] = plotting_df['OverallGrade'].fillna(0)  # Fills NaN in 'OverallGrade' with 0

# Call the function to plot the new feature with the specified legend location
plot_new_feature(df=plotting_df, 
                    new_feature_name='OverallGrade')



# Creating 2nd new feature: Success Ratio (Approved / Enrolled):
plotting_df['SuccessRatio'] = plotting_df['Curricular units 1st sem (approved)'] / plotting_df['Curricular units 1st sem (enrolled)'] + plotting_df['Curricular units 2nd sem (approved)'] / plotting_df['Curricular units 2nd sem (enrolled)']

#plot_new_feature(df = plotting_df,
#plotting_df['SuccessRatio2ndSem'] = plotting_df['Curricular units 2nd sem (approved)'] / plotting_df['Curricular units 2nd sem (enrolled)']

plot_new_feature(df = plotting_df,
                 new_feature_name = 'SuccessRatio')



def create_success_rank(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Creating Success Ratio (Approved / Enrolled)
    df['SuccessRatio'] = (df['Curricular units 1st sem (approved)'] / df['Curricular units 1st sem (enrolled)']) + \
                         (df['Curricular units 2nd sem (approved)'] / df['Curricular units 2nd sem (enrolled)'])

    # Function to rank students based on Success Ratio
    def rank_rows(row):
        if row['SuccessRatio'] <= df['SuccessRatio'].quantile(0.25):
            return 'Low'
        elif row['SuccessRatio'] <= df['SuccessRatio'].quantile(0.50):
            return 'Medium'
        elif row['SuccessRatio'] <= df['SuccessRatio'].quantile(0.75):
            return 'High'
        else:
            return 'Very High'
    
    df['SuccessRate'] = df.apply(rank_rows, axis=1)
    
    return df


plot_new_feature(df = plotting_df,
                 new_feature_name = 'SuccessRatio')


def categorize_unemployment_rate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure Unemployment Rate is numeric, handling missing values
    df["Unemployment rate"] = pd.to_numeric(df["Unemployment rate"], errors='coerce')
    
    # Fill missing values with "Missing Rate"
    df['Unemployed'] = df["Unemployment rate"].fillna("Missing Rate")

    # Define function to categorize unemployment rate
    def categorize(rate):
        if isinstance(rate, str):  # Handles cases where it's 'Missing Rate'
            return rate
        elif rate < 8:
            return 'Low'
        elif 8 <= rate <= 12:
            return 'Moderate'
        else:
            return 'High' 

    # Apply categorization function
    df['Unemployment_category'] = df['Unemployed'].apply(categorize)
    
    return df


plot_new_feature(df = plotting_df,
                 new_feature_name = 'Unemployment rate')


def categorize_inflation_rate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure Inflation Rate is numeric, handling missing values
    df["Inflation rate"] = pd.to_numeric(df["Inflation rate"], errors='coerce')

    # Fill missing values with "Missing Rate"
    df['Inflation_Value'] = df["Inflation rate"].fillna("Missing Rate")

    # Define function to categorize inflation rate
    def categorize(rate):
        if isinstance(rate, str):  # Handles cases where it's 'Missing Rate'
            return rate
        elif rate < 0:
            return 'Low'
        elif 0 <= rate <= 2:
            return 'Moderate'
        else:
            return 'High'

    # Apply categorization function
    df['Inflation_Category'] = df['Inflation_Value'].apply(categorize)

    return df


plot_new_feature(df = plotting_df,
                 new_feature_name = 'Inflation rate')


# run all feature engineering functions
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    new_df = df.copy()
    new_df = log_transform_column(df = new_df, column = "Admission grade")
    new_df = create_success_rank(df = new_df)
    new_df = categorize_unemployment_rate(df = new_df)
    new_df = categorize_inflation_rate(df = new_df)

    return new_df

training_data_new = feature_engineering(df = training_data)

# see the newly created columns
training_data_new.sample(5)


training_data_new.info()


training_data_new.select_dtypes(include=['int64', 'float64']).columns.tolist()


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Select numeric features
numeric_features = training_data.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Compute correlation matrix
corr_matrix = training_data[numeric_features].corr()

# Handle NaN and infinite values
corr_matrix = corr_matrix.replace([np.inf, -np.inf], np.nan).fillna(0)

# Create the upper triangle mask
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Plot the heatmap
plt.figure(figsize=(12, 8))  # Increase figure size
sns.heatmap(upper_triangle, annot=True, cmap="coolwarm", fmt=".1f", linewidths=0.5, annot_kws={"size": 8})
plt.xticks(rotation=45, ha="right")  # Rotate X labels for better visibility
plt.yticks(rotation=0)  # Keep Y labels readable
plt.title('Improved Correlation Heatmap', fontsize=14)
plt.show()



# Check the column names in the dataset
print(training_data_new.columns.tolist())

# Adjust the list of numeric features based on available columns
numeric_features = [
    'Admission grade', 'Previous qualification (grade)', 'Curricular units 1st sem (enrolled)',
    'Curricular units 2nd sem (enrolled)', 'Curricular units 1st sem (approved)',
    'Curricular units 2nd sem (approved)', 'Curricular units 1st sem (grade)',
    'Curricular units 2nd sem (grade)', 'Unemployment rate', 'Inflation rate', 'GDP',
    'Age at enrollment'  # Removed 'Curricular units 1st sem (credited)' as it may not exist
]

# Specify nominal and ordinal features based on your data
nominal_features = [
    'Gender', 'Scholarship holder', 'International', 'Educational special needs', 'Displaced',
    'Debtor', 'Tuition fees up to date'
]

ordinal_features = [
    "Mother's qualification", "Father's qualification", "Mother's occupation", "Father's occupation",
    'Application order', 'Marital status', 'Application mode', 'Course',
    'Daytime/evening attendance', 'Previous qualification', 'Nacionality'
]

# Combine all selected features
all_features = numeric_features + nominal_features + ordinal_features



from sklearn.model_selection import train_test_split

# Prepare features (X) and target variable (y)
X = training_data_new[all_features]
y = training_data_new['Target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Size of training set: {len(X_train)}")
print(f"Size of testing set: {len(X_test)}")



from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Define the feature columns (assuming defined earlier)
numeric_features = ['Admission grade', 'Previous qualification (grade)', 'Curricular units 1st sem (enrolled)',
                    'Curricular units 2nd sem (enrolled)', 'Curricular units 1st sem (approved)',
                    'Curricular units 2nd sem (approved)', 'Curricular units 1st sem (grade)',
                    'Curricular units 2nd sem (grade)', 'Unemployment rate', 'Inflation rate', 'GDP', 'Age at enrollment']

nominal_features = ['Gender', 'Scholarship holder', 'International', 'Educational special needs', 'Displaced',
                    'Debtor', 'Tuition fees up to date']

ordinal_features = ["Mother's qualification", "Father's qualification", "Mother's occupation", "Father's occupation",
                    'Application order', 'Marital status', 'Application mode', 'Course', 'Daytime/evening attendance',
                    'Previous qualification', 'Nacionality']

# Combine all features
all_features = numeric_features + nominal_features + ordinal_features

# Split the data (train/test)
X = training_data_new[all_features]
y = training_data_new['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

ordinal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))  # Handling unseen categories
])

nominal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Handling unseen categories
])

# Column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('ord', ordinal_transformer, ordinal_features),
        ('nom', nominal_transformer, nominal_features)
    ])

# view your pipeline
preprocessor




# Random Forest Pipeline
random_forest_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10, min_samples_split=5))
])

# Train the model
random_forest_model.fit(X_train, y_train)


random_forest_model



# Make predictions on the test set
random_forest_predictions = random_forest_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, random_forest_predictions)
report = classification_report(y_test, random_forest_predictions)

# Print the accuracy and classification report
print(f'Model Accuracy: {round(accuracy, 3)}')
print("="*5)
print(f'Classification Report:\n{report}')


# Get the names of the features after encoding and preprocessing
# Get the numeric feature names (which are directly passed to the transformer)
numeric_feature_names = numeric_features

# Get the one-hot encoded feature names from the 'onehot' transformer
# Access the onehot encoder and extract feature names
onehot_feature_names = random_forest_model.named_steps['preprocessor'].transformers_[2][1].named_steps['onehot'].get_feature_names_out(nominal_features)

# Get the ordinal feature names (since they remain the same, we just use the original list)
ordinal_feature_names = ordinal_features

# Combine all feature names (numeric, one-hot, and ordinal)
feature_names = numeric_feature_names + list(onehot_feature_names) + ordinal_feature_names

# Display the combined feature names
feature_names



import numpy as np
import matplotlib.pyplot as plt

# Get feature importances from the Random Forest model
importances = random_forest_model.named_steps['classifier'].feature_importances_

# Sort the feature importances in descending order
indices = np.argsort(importances)[::-1]

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(range(len(indices)), importances[indices], align="center")
plt.yticks(range(len(indices)), np.array(feature_names)[indices])
plt.xlabel('Feature Importance')
plt.title('Feature Importances from Random Forest')
plt.show()



# Ensure predictions are in the correct format
random_forest_predictions = random_forest_model.predict(X_test)

# Convert predictions to a Pandas Series to check value counts
prediction_counts = pd.Series(random_forest_predictions).value_counts()
print("Value Counts of Predictions:")
print(prediction_counts)




testing_data_new = feature_engineering(df = testing_data)
testing_data_new.sample(5)


# Make predictions on the test dataset
test_predictions = random_forest_model.predict(testing_data_new)

# Prepare submission file without 'id'
submission = pd.DataFrame({"id" : testing_data_new["id"],"Target": test_predictions})

# Save submission file
submission.to_csv("submission.csv", index=False)

# Verify if the file is saved correctly
import os
if "submission.csv" in os.listdir():
    print("submission.csv saved successfully!")
else:
    print("Error: submission.csv was not found.")



submission.head(5)




