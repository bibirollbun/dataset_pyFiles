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
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn modules
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report


# use pandas to load your data to dataframe objects
import pandas as pd

# load the training, testing, and sample submission data
training_data = pd.read_csv(r'/kaggle/input/playground-series-s4e6/train.csv')
testing_data = pd.read_csv(r'/kaggle/input/playground-series-s4e6/test.csv')
sample_submission = pd.read_csv(r'/kaggle/input/playground-series-s4e6/sample_submission.csv')


# Verify the data was loaded correctly
for name, dataset in zip(["Training Data", "Testing Data", "Sample Submission"], 
                         [training_data, testing_data, sample_submission]):
    print(f"{name} shape: {dataset.shape}")


# data profiling function
def create_data_profiling_df(data: pd.DataFrame) -> pd.DataFrame:

    # create an empty dataframe to gather information about each column
    data_profiling_df = pd.DataFrame(columns = ["column_name",
                                                "data_type",
                                                "values",
                                                "null_values",
                                                "percent_null",
                                                "unique_values",
                                                "duplicate_values",
                                                "min",
                                                "max",
                                                "median",
                                                "stdev",
                                                "IQR",
                                                "skewness",
                                                "most_common_value",
                                                "outliers"])

    # loop through each column to add rows to the data_profiling_df dataframe
    for column in data.columns:

        # create an empty dictionary to store the columns data
        column_dict = {}

        try:
            column_dict["column_name"] = [column]
            column_dict["data_type"] = [data[column].dtypes]
            column_dict["values"] = [data[column].notnull().sum()]
            column_dict["null_values"] = [data[column].isna().sum()]
            column_dict["percent_null"] = [round(data[column].isna().sum() / len(data[column]), 2)]
            column_dict["unique_values"] = [len(data[column].unique())]
            column_dict["duplicate_values"] = [(data[column].notnull().sum()) - len(data[column].unique())]
            column_dict["min"] = [data[column].min() if (data[column].dtypes != object) else "NA"]
            column_dict["max"] = [round(data[column].max(), 1) if (data[column].dtypes != object) else "NA"]
            column_dict["mean"] = [round(data[column].mean(), 1) if (data[column].dtypes != object) else "NA"]
            column_dict["median"] = [round(data[column].median(), 1) if (data[column].dtypes != object) else "NA"]
            column_dict["stdev"] = [round(data[column].std(), 1) if (data[column].dtypes != object) else "NA"]
            column_dict["IQR"] = [round(data[column].quantile(.75), 1) - data[column].quantile(.25) if (data[column].dtypes != object) else "NA"]
            column_dict["most_common_value"] = data[column].mode().iloc[0] if not data[column].mode().empty else "NA"
            column_dict["skewness"] = [data[column].skew(skipna=True) if (data[column].dtypes != object) else "NA"]

            # calculate likely outliers
            if data[column].dtypes != object:
                Q1 = data[column].quantile(0.25)
                Q3 = data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)][column]
                column_dict["outliers"] = len(outliers)
            else:
                column_dict["outliers"] = "NA"

        except:
            print(f"unable to read column: {column}, you may want to drop this column")

        # add the information from the columns dict to the final dataframe
        data_profiling_df = pd.concat([data_profiling_df, pd.DataFrame(column_dict)],
                                      ignore_index = True)

    # sort the final dataframe by unique values descending
    data_profiling_df.sort_values(by = ['unique_values'],
                                  ascending = [False],
                                  inplace=True)

    # print the function is complete
    print(f"data profiling complete, dataframe contains {len(data_profiling_df)} columns")
    return data_profiling_df


# run the data profiling function
data_profiling_df = create_data_profiling_df(data = training_data)

# print the dataframe
data_profiling_df_sorted = data_profiling_df.sort_index()

data_profiling_df_sorted


# Define function to plot histogram and identify outliers
def plot_histogram(df: pd.DataFrame,
                   variable: str,
                   bins=10,
                   color='grey',
                   edgecolor='black',
                   figsize=(7, 2),
                   iqr_on=False):
  
    # Set figure size
    plt.figure(figsize=figsize)

    # Plot histogram
    plt.hist(df[variable], bins=bins, color=color, edgecolor=edgecolor)

    # Customize labels and formatting
    plt.title(f'{variable} Histogram')
    plt.xlabel(variable)
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.ticklabel_format(style='plain', axis='x')
    plt.grid(True)

    # Define the Interquartile Range (IQR) and outlier bounds
    q1 = df[variable].quantile(0.25)
    q3 = df[variable].quantile(0.75)
    iqr = q3 - q1

    if iqr_on:
        lower_bound = q1
        upper_bound = q3
    else:
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

    # Mark outlier bounds on histogram
    plt.axvline(lower_bound, color='blue', linestyle='dashed', linewidth=2, label='Lower Bound')
    plt.axvline(upper_bound, color='blue', linestyle='dashed', linewidth=2, label='Upper Bound')

    # Show plot with legend
    plt.legend()
    plt.show()

    # Count outliers
    num_outliers = ((df[variable] < lower_bound) | (df[variable] > upper_bound)).sum()

    # Print outlier details
    if num_outliers > 0:
        print(f"{num_outliers} potential outliers detected in {variable} distribution")
        print(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")
    else:
        print(f"No potential outliers detected in {variable} distribution")
        print(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")

    # Print separator
    print("\n-----\n")

# Function to plot categorical variable distribution
def plot_categorical_distribution(df, column):
    """
    Plots the distribution of a categorical variable.

    Parameters:
        df (pd.DataFrame): The dataframe containing the variable.
        column (str): The name of the categorical variable to plot.
    """
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x=column, order=df[column].value_counts().index, palette="coolwarm")
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

# Function to plot correlation heatmap
def plot_correlation_heatmap(df):
    """
    Plots a heatmap of the correlation matrix for numerical features.

    Parameters:
        df (pd.DataFrame): The dataframe containing numerical variables.
    """
    plt.figure(figsize=(12, 8))
    numeric_cols = df.select_dtypes(include=["number"]).columns
    correlation_matrix = df[numeric_cols].corr()
    
    sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Heatmap of Numerical Features")
    plt.show()

# Function to detect and visualize outliers using boxplots
def plot_boxplot(df, variable):
    """
    Plots a boxplot for a numerical variable to visualize outliers.

    Parameters:
        df (pd.DataFrame): The dataframe containing the variable.
        variable (str): The name of the numerical variable.
    """
    plt.figure(figsize=(6, 4))
    sns.boxplot(y=df[variable], color="royalblue")
    plt.title(f"Boxplot of {variable}")
    plt.show()

# Function to perform full EDA following the given structure
def full_eda(df, target_column):
    """
    Performs full exploratory data analysis on a given dataset.

    Parameters:
        df (pd.DataFrame): The dataset.
        target_column (str): The name of the target variable.
    """
    print("Performing EDA...\n")
    
    # Print dataset structure and summary
    print("\nDataset Info:")
    print(df.info())
    
    print("\nSummary Statistics:")
    print(df.describe().T)
    
    # Visualize target variable distribution
    plot_categorical_distribution(df, target_column)

    # Correlation heatmap
    plot_correlation_heatmap(df)

    # Detect and visualize outliers for numerical features
    numerical_features = df.select_dtypes(include=["number"]).columns
    for feature in numerical_features:
        plot_histogram(df, feature)
        plot_boxplot(df, feature)

    print("EDA Completed.")


# run the histogram function on all numerical features
for feature in ["Age at enrollment", "Admission grade", "Unemployment rate", "Inflation rate", "GDP"]:
    plot_histogram(df = training_data,
                   variable = feature,
                   bins = 15)


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


# Count plots for categorical features
selected_categorical_features = ["Gender", "Marital status", "Scholarship holder", "Daytime/evening attendance"]
for feature in selected_categorical_features:
    count_plot(df=training_data, variable=feature)


# Load the dataset (replace with actual path if necessary)
# Assuming you have a dataset in a CSV file
# training_data = pd.read_csv("path_to_your_dataset.csv")

# Function to clean and transform the data
def clean_and_transform_data(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Convert Gender to numerical encoding
    gender_map = {'M': 1, 'F': 2}  # Ensure consistency
    df['Gender'] = df['Gender'].replace(gender_map).astype(int)
    
    # 2. Feature Engineering: Create 'total_curricular_units_enrolled'
    df["total_curricular_units_enrolled"] = (
        df["Curricular units 1st sem (enrolled)"] +
        df["Curricular units 2nd sem (enrolled)"]
    )
    
    # 3. Handle missing values in 'Age at enrollment' by using the median per grouped total_curricular_units_enrolled
    df["Age at enrollment"] = df.groupby("total_curricular_units_enrolled")["Age at enrollment"].transform("median")
    df["Age at enrollment"].fillna(df["Age at enrollment"].median(), inplace=True)
    
    # 4. Fill missing values for categorical variable 'Daytime/evening attendance' using the mode
    df["Daytime/evening attendance"] = df["Daytime/evening attendance"].fillna(df["Daytime/evening attendance"].mode()[0])
    
    # 5. Log transform highly skewed numerical features (if applicable)
    df["Admission grade"] = np.log1p(df["Admission grade"])
    
    # 6. Fill missing values in other numerical columns with the median (for safety)
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    for column in numerical_columns:
        df[column].fillna(df[column].median(), inplace=True)
    
    # 7. Final check: Verify if any missing values remain
    print("Missing values after cleaning:")
    print(df.isnull().sum())
    
    return df

# Clean the dataset
# Assuming training_data is already loaded
training_data_cleaned = clean_and_transform_data(training_data)

# Display the entire cleaned dataset
print("Cleaned Dataset:")
print(training_data_cleaned)



# Store the original ID column separately before feature engineering
original_id_column = "StudentID"  # Change this to the actual column name in raw data
if original_id_column in testing_data.columns:
    testing_data_id = testing_data[[original_id_column]].copy()
else:
    print(f"Error: {original_id_column} column is missing from the raw dataset!")


from sklearn.preprocessing import StandardScaler, LabelEncoder

# Function to setup transformations for the dataset
def setup_data_transformations(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Apply StandardScaler to numerical features
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    
    # 2. Encode categorical features (if not already done)
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    label_encoder = LabelEncoder()
    for column in categorical_columns:
        if df[column].dtype == 'object':  # Apply encoding only if it is categorical
            df[column] = label_encoder.fit_transform(df[column].astype(str))
    
    return df

# Apply transformations to the cleaned dataset
dataset_transformed = setup_data_transformations(training_data_cleaned)

# Display the transformed dataset
print("Transformed Dataset:")
print(dataset_transformed.head())



from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# Modify the OneHotEncoder and OrdinalEncoder to handle unknown categories
one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
ordinal_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)



# Function for Feature Engineering
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # Example feature engineering
    df['academic_engagement_score'] = (
        df['Curricular units 1st sem (enrolled)'] + df['Curricular units 2nd sem (enrolled)']
    )
    
    # Creating economic hardship rank based on some features
    def economic_rank(row):
        if row["GDP"] > 2 and row["Inflation rate"] < 1:
            return 5  # Most stable economy
        elif row["GDP"] > 1 and row["Inflation rate"] < 2.5:
            return 3  # Moderately stable
        else:
            return 1  # Unstable economy

    df['economic_hardship_rank'] = df.apply(economic_rank, axis=1)
    
    return df


# Apply feature engineering to the transformed dataset
dataset_with_features = feature_engineering(dataset_transformed)

# Display the dataset with new features
print("Dataset with New Features:")
print(dataset_with_features.head())



# Function for Feature Selection (Filtering Method)
def feature_selection(df: pd.DataFrame, threshold=0.85) -> pd.DataFrame:

    # Compute correlation matrix
    corr_matrix = df.corr().abs()  # Compute the absolute value of the correlation matrix
    
    # Create a mask for the upper triangle of the correlation matrix (to avoid duplicate checks)
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Get columns with correlation greater than the threshold
    high_corr_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
    
    # Drop the highly correlated features from the dataset
    df_filtered = df.drop(columns=high_corr_features, errors="ignore")
    
    # Display which features were removed
    print(f"Removed features due to high correlation (above {threshold}):", high_corr_features)
    
    return df_filtered

# Apply feature selection (correlation-based filtering)
dataset_selected_features = feature_selection(dataset_with_features)

# Display the selected dataset with reduced features
print("Dataset After Feature Selection:")
print(dataset_selected_features.head())



# Compute the correlation matrix
corr_matrix = dataset_selected_features.corr()

# Mask the upper triangle to avoid redundancy
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Define the upper triangle of the correlation matrix
upper_triangle = corr_matrix.where(mask)

# Plot the heatmap of the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(upper_triangle, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap of Features')
plt.show()



from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Step 8: Split the dataset
X = dataset_selected_features.drop(columns=['Target'])  # Ensure the target column is removed
y = dataset_selected_features['Target'].astype(int)  # Convert target to integer

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"""size of training set: {len(X_train)}
size of testing set: {len(X_test)}""")


# Define the lists of numeric, nominal, and ordinal features
numeric_features = ['Age at enrollment', 'Admission grade', 'Unemployment rate', 
                    'Inflation rate', 'GDP', 'parent_education_rank']

nominal_features = ['Gender', 'Scholarship holder', 'Displaced', 'Debtor', 'International']

ordinal_features = ['academic_engagement_score', 'economic_hardship_rank', 'family_size_rank']

# Check for column existence before applying transformations
valid_features = X_train.columns.tolist()

# Ensure only valid columns are used in preprocessing
numeric_features = [col for col in numeric_features if col in valid_features]
nominal_features = [col for col in nominal_features if col in valid_features]
ordinal_features = [col for col in ordinal_features if col in valid_features]

print("✅ Final Numeric Features:", numeric_features)
print("✅ Final Nominal Features:", nominal_features)
print("✅ Final Ordinal Features:", ordinal_features)



from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier

# Define transformations for numeric features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Handle missing values
    ('scaler', StandardScaler())  # Normalize numerical data
])

# Define transformations for nominal (categorical) features
nominal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing with most frequent
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Ignore unknown categories
])

# Define transformations for ordinal features
ordinal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing with most frequent
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))  # Handle unknown ordinal categories
])

# Create column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),  # Apply numeric transformations
        ('nom', nominal_transformer, nominal_features),  # Apply nominal transformations
        ('ord', ordinal_transformer, ordinal_features)   # Apply ordinal transformations
    ]
)

# Build the full pipeline including DecisionTreeClassifier
decision_tree_model = Pipeline(steps=[
    ('preprocessor', preprocessor),  # Apply feature transformations
    ('classifier', DecisionTreeClassifier(random_state=42, max_depth=4, min_samples_split=10))  # Train DecisionTreeClassifier
])



# view your pipeline
preprocessor


# Ensure target variable is categorical
y_train = y_train.astype(int)
y_test = y_test.astype(int)

# Train model
decision_tree_model.fit(X_train, y_train)

# Make predictions
decision_tree_predictions = decision_tree_model.predict(X_test)

# Evaluate model
from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test, decision_tree_predictions)
report = classification_report(y_test, decision_tree_predictions)

print(f"Model Accuracy: {round(accuracy, 3)}")
print("\nClassification Report:\n", report)



# Ensure the model has been fitted before extracting feature names
if hasattr(decision_tree_model.named_steps['preprocessor'], 'transformers_'):
    onehot_encoder = decision_tree_model.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot']
    if hasattr(onehot_encoder, 'get_feature_names_out'):
        feature_names = list(onehot_encoder.get_feature_names_out(nominal_features))
    else:
        feature_names = nominal_features
else:
    feature_names = nominal_features

# Append numerical and ordinal features
feature_names = numeric_features + feature_names + ordinal_features

print("Feature Names Used in the Model:", feature_names)



from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))
plot_tree(decision_tree_model.named_steps['classifier'],
          feature_names=feature_names,
          class_names=['Dropout', 'Enrolled', 'Graduate'],
          rounded=True, filled=True)
plt.show()



# Apply feature engineering to the test set (ensure it's preprocessed correctly)
testing_data_new = feature_engineering(df=testing_data)

# Display a sample of the transformed test data to verify
testing_data_new.sample(5)


print("Available columns in testing_data:", testing_data.columns.tolist())


def create_parent_education_rank(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    def rank_education(row):
        if row["Mother's qualification"] >= 4 and row["Father's qualification"] >= 4:
            return 1  # High parental education
        elif row["Mother's qualification"] >= 2 or row["Father's qualification"] >= 2:
            return 2  # Medium parental education
        else:
            return 3  # Low parental education
    df["parent_education_rank"] = df.apply(rank_education, axis=1)
    return df

# Apply feature engineering again
testing_data = create_parent_education_rank(testing_data)

# Check if the feature is now available
print("Updated columns in testing_data:", testing_data.columns.tolist())


# Step 1: Define categories for the Target (e.g., "Graduate", "At Risk")
# You can map your numerical predictions to these categories. Example:
target_mapping = {
    0: 'At Risk',  # Mapping 0 to "At Risk"
    2: 'Graduate'  # Mapping 2 to "Graduate"
}

# Step 2: Make predictions on the test set
decision_tree_predictions = decision_tree_model.predict(testing_data_new)

# Step 3: Convert predictions to a list
decision_tree_predictions = decision_tree_predictions.tolist()

# Step 4: Map the numerical predictions to categorical values (e.g., 'Graduate', 'At Risk')
decision_tree_predictions = [target_mapping[pred] for pred in decision_tree_predictions]

# Step 5: Create the submission DataFrame with 'id' and 'Target' columns
submission_df = pd.DataFrame({
    "id": testing_data_new["id"],  # Use the 'id' column for the unique identifier
    "Target": decision_tree_predictions  # Use mapped categorical predictions for Target
})

# Step 6: Save the submission file as a CSV with header
submission_file_path = "submission.csv"
submission_df.to_csv(submission_file_path, index=False)

# Step 7: Print success message
print(f"✅ Submission file created successfully! File saved as {submission_file_path}.")


