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


print(training_data.columns)


# Convert Gender to numerical encoding
gender_map = {'M': 1, 'F': 2}  # Ensure consistency
training_data['Gender'] = training_data['Gender'].replace(gender_map).astype(int)
testing_data['Gender'] = testing_data['Gender'].replace(gender_map).astype(int)



# Data Cleaning
# Ensure 'total_curricular_units_enrolled' exists before using it
training_data["total_curricular_units_enrolled"] = (
    training_data["Curricular units 1st sem (enrolled)"] +
    training_data["Curricular units 2nd sem (enrolled)"]
)

# Handle missing values in 'Age at enrollment' by using the median per grouped total_curricular_units_enrolled
training_data["Age at enrollment"] = training_data.groupby(
    "total_curricular_units_enrolled"
)["Age at enrollment"].transform("median").fillna(training_data["Age at enrollment"])

# Fill missing values for categorical variable 'Daytime/evening attendance'
training_data["Daytime/evening attendance"] = training_data["Daytime/evening attendance"].fillna(
    training_data["Daytime/evening attendance"].mode()[0]
)

# Log transform highly skewed numerical features
training_data["Admission grade"] = np.log1p(training_data["Admission grade"])

# Final verification of missing values
print("Missing values after cleaning:")
print(training_data.isnull().sum())


# Setup Data Transformations
training_data["fam_size"] = training_data["Curricular units 1st sem (enrolled)"] + training_data["Curricular units 2nd sem (enrolled)"]

# Group by 'fam_size' and compute median 'Age at enrollment'
fam_size_age_median = training_data.groupby("fam_size", as_index=False)["Age at enrollment"].median().sort_values(by="fam_size")

# Display the transformation output
print(fam_size_age_median)



# Function to log transform any column
def log_transform_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df = df.copy()
    df[column] = df[column].apply(lambda x: np.log(x + 1) if pd.notnull(x) else x)
    return df

# Function to fill (assume) null Age values
def fill_age_column(df: pd.DataFrame) -> pd.DataFrame:
    df["fam_size"] = df["Curricular units 1st sem (enrolled)"] + df["Curricular units 2nd sem (enrolled)"]
    
    def assume_age(row):
        if pd.notna(row["Age at enrollment"]):
            return row["Age at enrollment"]
        elif row["fam_size"] == 0 or row["fam_size"] == 1:
            return 29
        elif row["fam_size"] == 2:
            return 27
        elif row["fam_size"] == 3:
            return 23
        elif row["fam_size"] == 4:
            return 18
        elif row["fam_size"] == 5:
            return 12
        elif row["fam_size"] == 6:
            return 9
        elif row["fam_size"] == 7:
            return 12
        else:
            return 20
    
    df["Age at enrollment"] = df.apply(assume_age, axis=1)
    assert df["Age at enrollment"].isna().sum() == 0, "something went wrong in the Age column..."
    return df

# Function to fill the most frequent values in the 'Embarked' column
def fill_embarked_column(df: pd.DataFrame) -> pd.DataFrame:
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
    return df



# Convert 'Gender' to 0 and 1 (instead of 1 and 2) for consistency with training data
gender_map = {'M': 0, 'F': 1}  
training_data['Gender'] = training_data['Gender'].replace(gender_map).astype(int)
testing_data['Gender'] = testing_data['Gender'].replace(gender_map).astype(int)


plotting_df = training_data.copy()


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["study_hours"] = df["Daytime/evening attendance"]
    df["qualification_rank"] = df["Previous qualification (grade)"].apply(lambda x: 1 if x >= 15 else (2 if x >= 10 else 3))
    df["economic_rank"] = df.apply(lambda row: 1 if row['GDP'] > 2 and row['Inflation rate'] < 1.5 else (2 if row['GDP'] > 1 and row['Inflation rate'] < 2.5 else 3), axis=1)
    return df

# Apply feature engineering to both datasets
training_data = feature_engineering(training_data)
testing_data = feature_engineering(testing_data)



# Function to visualize the impact of a new feature
def plot_new_feature(df: pd.DataFrame, new_feature_name: str):
    plot_df = df.copy()
    feature_analysis = plot_df.groupby(new_feature_name)['Target'].mean().reset_index()
    feature_analysis = feature_analysis.sort_values(by='Target', ascending=False)
    avg_target = plot_df['Target'].mean()
    
    plt.figure(figsize=(12, 3))
    sns.barplot(x=new_feature_name, y='Target', data=feature_analysis, color='grey', order=feature_analysis[new_feature_name])
    plt.axhline(avg_target, color='blue', linestyle='--', label=f'Average Target ({avg_target:.2f})')
    plt.ylabel('Average Target')
    plt.title(f'Impact of {new_feature_name} on Target')
    plt.legend()
    plt.show()


# Function to create family size ranking
def create_family_size_rank(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["family_size"] = df['Curricular units 1st sem (enrolled)'] + df['Curricular units 2nd sem (enrolled)']
    
    def rank_rows(row):
        if row['family_size'] in [3, 2, 1]:
            return 1
        elif row['family_size'] in [6, 0, 4]:
            return 2
        else:
            return 3
    
    df['family_size_rank'] = df.apply(rank_rows, axis=1)
    return df


# Function to create enrollment count rank
def create_enrollment_count_rank(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["enrollment_count"] = df["Curricular units 1st sem (enrolled)"] + df["Curricular units 2nd sem (enrolled)"]
    
    def rank_rows(row):
        if row["enrollment_count"] in [3, 2, 4]:
            return 1  # Low workload
        elif row["enrollment_count"] in [1, 7]:
            return 2  # Medium workload
        else:
            return 3  # High workload
    
    df["enrollment_count_rank"] = df.apply(rank_rows, axis=1)
    return df



# Function to create qualification rank
def create_qualification_rank(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    def rank_qualification(row):
        if row["Previous qualification"] in [1, 2]:
            return 1  # Primary or Middle School
        elif row["Previous qualification"] in [3, 4, 5]:
            return 2  # High School
        else:
            return 3  # Higher Education
    df["qualification_rank"] = df.apply(rank_qualification, axis=1)
    return df



# Function to create parental education rank
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


# Apply feature engineering functions
training_data = create_family_size_rank(training_data)
training_data = create_enrollment_count_rank(training_data)
training_data = create_qualification_rank(training_data)
training_data = create_parent_education_rank(training_data)


# Show a random sample of 5 rows with new features
training_data.sample(5)


# Display all column names
print(training_data.columns)


# Check if any of the new columns have missing values
print(training_data[['family_size_rank', 'enrollment_count_rank', 'qualification_rank', 'parent_education_rank']].isnull().sum())


# Display summary statistics for the newly created categorical features
print(training_data[['family_size_rank', 'enrollment_count_rank', 'qualification_rank', 'parent_education_rank']].describe())


training_data.info()


# Select numeric features
numeric_features = training_data.select_dtypes(include=['int64', 'float64']).columns.tolist()


# Create correlation matrix
corr_matrix = training_data[numeric_features].corr().abs()


# Create upper triangle mask
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))


# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(upper_triangle, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap of Features')
plt.show()


# Manually selecting relevant features for modeling
numeric_features = ['Age at enrollment', 'Admission grade', 'Unemployment rate', 'Inflation rate', 'GDP']
nominal_features = ['Gender']
ordinal_features = ['qualification_rank', 'family_size_rank', 'parent_education_rank', 'enrollment_count_rank']

all_features = numeric_features + nominal_features + ordinal_features
print("Selected Features:", all_features)


# Split data into Training and Testing Sets
X = training_data[all_features]
y = training_data['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"""size of training set: {len(X_train)}
size of testing set: {len(X_test)}""")


# Define the transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

nominal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

ordinal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', numeric_transformer, numeric_features),
        ('nominal', nominal_transformer, nominal_features),
        ('ordinal', ordinal_transformer, ordinal_features)
])



# view your pipeline
preprocessor


# Define the pipeline with a DecisionTreeClassifier optimized for recall
decision_tree_model = Pipeline(steps=[('preprocessor', preprocessor),
                                      ('classifier', DecisionTreeClassifier(random_state=42,
                                                                             max_depth=6,  # Increased depth to improve recall
                                                                             min_samples_split=5,  # Allow more splits for better recall
                                                                             class_weight='balanced'))])  # Balance classes to improve recall for minority class

# Train the model
decision_tree_model.fit(X_train, y_train)

# Evaluate the model
y_pred = decision_tree_model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))


# Extract feature names from the pipeline
feature_names = (numeric_features +
                 list(decision_tree_model.named_steps['preprocessor'].transformers_[1][1]
                      .named_steps['onehot'].get_feature_names_out(nominal_features)) +
                 ordinal_features)


from sklearn.tree import plot_tree

# Visualize the decision tree
plt.figure(figsize=(20, 10))
plot_tree(decision_tree_model.named_steps['classifier'],
          feature_names=feature_names,
          class_names=['Dropout', 'Enrolled', 'Graduate'],
          rounded=True, filled=True)
plt.show()


training_data = feature_engineering(df = training_data)
testing_data_new = feature_engineering(df = testing_data)

testing_data = create_family_size_rank(testing_data)
testing_data = create_enrollment_count_rank(testing_data)

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



if "parent_education_rank" in testing_data.columns:
    print("✅ parent_education_rank successfully added!")
else:
    print("❌ parent_education_rank is still missing!")



# Extract the correct features after applying feature engineering
testing_data_final = testing_data[['Age at enrollment', 'Admission grade', 'Unemployment rate', 
                                   'Inflation rate', 'GDP', 'Gender', 'qualification_rank', 
                                   'family_size_rank', 'parent_education_rank', 'enrollment_count_rank']].copy()

# Apply the trained pipeline to preprocess the test dataset
testing_data_transformed = decision_tree_model.named_steps['preprocessor'].transform(testing_data_final)

# Make predictions
decision_tree_test_predictions = decision_tree_model.named_steps['classifier'].predict(testing_data_transformed)

# Create submission DataFrame
submission_df = pd.DataFrame({
    "id": testing_data["id"],
    "Target": decision_tree_test_predictions
})

# Save the submission file
submission_df.to_csv("submission.csv", index=False)
print("✅ Submission file created successfully!")



# Save submission file
submission_df.to_csv("submission.csv", index=False)
print("Submission file saved as 'submission.csv' - Ready for Kaggle!")


