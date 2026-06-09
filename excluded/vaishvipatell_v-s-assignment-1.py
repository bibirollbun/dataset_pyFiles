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


import pandas as pd

# load the training, testing, and sample submission data
training_data = pd.read_csv(r'/kaggle/input/playground-series-s4e6/train.csv')
testing_data = pd.read_csv(r'/kaggle/input/playground-series-s4e6/test.csv')
sample_submission = pd.read_csv(r'/kaggle/input/playground-series-s4e6/sample_submission.csv')


# verify the data was loaded
for dataset in [training_data, testing_data, sample_submission]:
    print(f"dataset shape: {dataset.shape}")


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


data_profiling_df = create_data_profiling_df(data = training_data)
# print the dataframe
data_profiling_df


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming `training_data` is your DataFrame
sns.countplot(x='Target', data=training_data)
plt.title('Distribution of Target Variable')
plt.xticks([0, 1, 2], ['DROPOUT', 'ENROLLED', 'GRADUATE'])
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# List of categorical variables
categorical_vars = ['Application mode', 'Course', 'Daytime/evening attendance','Marital status',
                     'Nacionality',  'Previous qualification',
                     "Mother's qualification","Mother's occupation", "Father's occupation","Father's qualification",
                    'Displaced', 'Educational special needs', 'Debtor', 'Tuition fees up to date',
                    'Gender', 'Scholarship holder', 'International']

# Define the number of subplots
n_vars = len(categorical_vars)
n_cols = 3  
n_rows = (n_vars // n_cols) + (n_vars % n_cols > 0)  

# Create subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 5))


if n_rows > 1:
    axes = axes.flatten()
# Loop through each categorical variable for analysis
for i, var in enumerate(categorical_vars):

    category_counts = training_data[var].value_counts()
    
    # Plot the frequency of each category
    ax = axes[i]
    category_counts.plot(kind='bar', color='purple', ax=ax)
    ax.set_title('Frequency of ' + var)
    ax.set_xlabel(var)
    ax.set_ylabel('Frequency')
    ax.tick_params(axis='x', rotation=45)

# Hide any empty subplots
for j in range(i + 1, n_rows * n_cols):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


warnings.filterwarnings('ignore')

# List of numerical variables
numerical_vars = ['Application order', 'Age at enrollment', 'Curricular units 1st sem (credited)',
                  'Curricular units 1st sem (enrolled)', 'Curricular units 1st sem (evaluations)',
                  'Curricular units 1st sem (approved)','Curricular units 2nd sem (credited)',
                  'Curricular units 2nd sem (enrolled)', 'Curricular units 2nd sem (evaluations)',
                  'Curricular units 2nd sem (approved)']

# Define the number of subplots
n_vars = len(numerical_vars)
n_cols = 2  # Number of columns in the subplot grid
n_rows = (n_vars // n_cols) + (n_vars % n_cols > 0)  # Calculate rows needed

# Create subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 5))
fig.suptitle('Numerical Variable Analysis', fontsize=17)

if n_rows > 1:
    axes = axes.flatten()

for i, var in enumerate(numerical_vars):
    ax_hist = axes[i]

    # Plot histogram with KDE
    sns.histplot(training_data[var], kde=True, ax=ax_hist, color='blue')
    ax_hist.set_title(f'Histogram of {var}')
    ax_hist.set_xlabel(var)
    ax_hist.set_ylabel('Count')
    
    # Calculate outliers using IQR
    Q1 = training_data[var].quantile(0.25)
    Q3 = training_data[var].quantile(0.75)
    IQR = Q3 - Q1
    outliers = training_data[(training_data[var] < (Q1 - 1.5 * IQR)) | (training_data[var] > (Q3 + 1.5 * IQR))]

# Print summary statistics and outliers
    print(f"\n{var}:")
    print(training_data[var].describe())
    print(f"  Outliers count: {outliers[var].count()}")

# Hide any empty subplots
for j in range(i + 1, n_rows * n_cols):
    fig.delaxes(axes[j])

plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust layout to fit title
plt.show()


training_data.info()

# Checking for null values
training_data.isnull().sum()


# function to log transform the any column
def log_transform_column(df: pd.DataFrame,
                        column: str) -> pd.DataFrame:
    df = df.copy()
    df[column] = df[column].apply(lambda x: np.log(x + 1) if pd.notnull(x) else x)
    return df


#Encoding Target Vaiables 
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

# Fit and transform the target variable
training_data['Target'] = label_encoder.fit_transform(training_data['Target'])


import pandas as pd

# Calculate the average grade across both semesters
training_data['Average Curricular Units Grade'] = (training_data['Curricular units 1st sem (grade)'] + training_data['Curricular units 2nd sem (grade)']) / 2


import pandas as pd
# Create new features
df = training_data

df['Average Grade of Curricular Units'] = (df['Curricular units 1st sem (grade)'] + df['Curricular units 2nd sem (grade)']) / 2
df['Total Number of Evaluations'] = df['Curricular units 1st sem (evaluations)'] + df['Curricular units 2nd sem (evaluations)']


import pandas as pd

# Create a new binary feature indicating students who have Scholarship holder but are also at financial risk
training_data['Scholarship holder'] = ((training_data['Scholarship holder'] == 1) & (training_data['Debtor'] == 1)).astype(int)


training_data.info()


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# List of numerical variables including the newly created features
numerical_vars = ['Application order', 'Age at enrollment', 'Previous qualification (grade)',
                  'Curricular units 1st sem (credited)', 'Admission grade',
                  'Curricular units 1st sem (enrolled)', 'Curricular units 1st sem (evaluations)',
                  'Curricular units 1st sem (approved)', 'Curricular units 2nd sem (credited)',
                  'Curricular units 2nd sem (enrolled)', 'Curricular units 2nd sem (evaluations)',
                  'Curricular units 2nd sem (approved)', 'Scholarship holder', 'Target',
                  'Average Grade of Curricular Units', 'Total Number of Evaluations']

# Assuming `training_data` is your DataFrame and it includes the new features
# Compute the correlation matrix
correlation_matrix = training_data[numerical_vars].corr().abs()

# Create a mask for the lower triangle
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Set up the matplotlib figure
plt.figure(figsize=(16, 12))

# Plot the heatmap
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='inferno', fmt=".2f")
plt.title('Correlation Matrix (Upper Triangle)')
plt.show()

# Find the correlation of each feature with the target variable
target_correlation = correlation_matrix['Target'].abs().sort_values(ascending=False)

# Display the correlation values
print(target_correlation)


from sklearn.model_selection import train_test_split

# List of all features including the new features
all_features = ['Application order', 'Age at enrollment', 'Previous qualification (grade)',
                'Curricular units 1st sem (credited)', 'Admission grade',
                'Curricular units 1st sem (enrolled)', 'Curricular units 1st sem (evaluations)',
                'Curricular units 1st sem (approved)', 'Curricular units 2nd sem (credited)',
                'Curricular units 2nd sem (enrolled)', 'Curricular units 2nd sem (evaluations)',
                'Curricular units 2nd sem (approved)', 'Scholarship holder', 
                'Average Grade of Curricular Units', 'Total Number of Evaluations']

# Define the feature set and target variable
features = training_data[all_features]
target = training_data['Target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Display the sizes of the training and testing sets
train_size = len(X_train)
test_size = len(X_test)
print(f"Training set size: {train_size}")
print(f"Testing set size: {test_size}")


import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Define the list of feature types
numeric_features = ['Application order', 'Age at enrollment', 'Previous qualification (grade)', 
                    'Curricular units 1st sem (credited)', 'Admission grade',
                    'Curricular units 1st sem (enrolled)', 'Curricular units 1st sem (evaluations)', 
                    'Curricular units 1st sem (approved)', 'Curricular units 2nd sem (credited)', 
                    'Curricular units 2nd sem (enrolled)', 'Curricular units 2nd sem (evaluations)', 
                    'Curricular units 2nd sem (approved)', 'Average Grade of Curricular Units', 
                    'Total Number of Evaluations']
# Example lists for nominal and ordinal features
# Replace these with actual feature names as necessary
nominal_features = ['Marital status', 'Gender', 'Nacionality']
ordinal_features = ['Father\'s qualification', 'Mother\'s qualification', 'Father\'s occupation', 'Mother\'s occupation']

# Define preprocessing for numerical data
numeric_preprocessor = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Handle missing values by imputing the median
    ('scaler', StandardScaler())  # Scale features to have zero mean and unit variance
])

# Define preprocessing for ordinal data
ordinal_preprocessor = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent'))  # Handle missing values by imputing the most frequent value
])

# Define preprocessing for nominal data
nominal_preprocessor = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing values by imputing the most frequent value
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Encode categorical features as a one-hot numeric array
])

# Combine preprocessing steps for all feature types
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_preprocessor, numeric_features),  # Apply numerical preprocessing to numerical features
        ('nom', nominal_preprocessor, nominal_features),  # Apply nominal preprocessing to nominal features
        ('ord', ordinal_preprocessor, ordinal_features)  # Apply ordinal preprocessing to ordinal features
    ])

# Define the target and feature variables
X = training_data[numeric_features + nominal_features + ordinal_features]
y = training_data['Target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Display the sizes of the training and testing sets
print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

# Build the complete pipeline with preprocessing and classifier
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Build the complete pipeline with preprocessing and classifier
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Fit the model
model_pipeline.fit(X_train, y_train)

# Make predictions
y_pred = model_pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)

print(f"Model accuracy: {accuracy:.2f}")
print("Classification report:")
print('classification_report_str')


# VIEW  PIPELINE
preprocessor


import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier

# Construct the machine learning pipeline
# This pipeline preprocesses the data and then applies a DecisionTreeClassifier
pipeline = Pipeline(steps=[
    ('data_preprocessing', preprocessor),  # Apply the preprocessor defined earlier
    ('decision_tree', DecisionTreeClassifier(random_state=42, max_depth=3))  # Use a DecisionTreeClassifier with specified parameters
])

# Train the model using the training data
pipeline.fit(X_train, y_train)

# Generate predictions on the test data
predictions = pipeline.predict(X_test)

# Display a message indicating completion of the training and prediction steps
print("Model training and prediction completed successfully.")

# Visualize the decision tree (optional)
plt.figure(figsize=(21, 9))
plot_tree(pipeline.named_steps['decision_tree'], filled=True, feature_names=X_train.columns, class_names=str(list(y_train.unique())))
plt.title('Decision Tree Visualization')
plt.show()


from sklearn.metrics import accuracy_score, classification_report

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f'Model Accuracy: {accuracy}')
print("="*5)
print(f'Classification Report:\n{report}')


testing_data['Total Curricular Units Enrolled'] = testing_data['Curricular units 1st sem (enrolled)'] + testing_data['Curricular units 2nd sem (enrolled)']
testing_data['Total Curricular Units Credited'] = testing_data['Curricular units 1st sem (credited)'] + testing_data['Curricular units 2nd sem (credited)']
testing_data['Total Curricular Units Evaluated'] = testing_data['Curricular units 1st sem (evaluations)'] + testing_data['Curricular units 2nd sem (evaluations)']
testing_data['Total Curricular Units Approved'] = testing_data['Curricular units 1st sem (approved)'] + testing_data['Curricular units 2nd sem (approved)']
testing_data['Total Curricular Units Grade'] = testing_data['Curricular units 1st sem (grade)'] + testing_data['Curricular units 2nd sem (grade)'] / 2
# Add the additional columns
testing_data['Average Grade of Curricular Units'] = (testing_data['Curricular units 1st sem (grade)'] + testing_data['Curricular units 2nd sem (grade)']) / 2
testing_data['Total Number of Evaluations'] = testing_data['Curricular units 1st sem (evaluations)'] + testing_data['Curricular units 2nd sem (evaluations)']
testing_data['First Semester Performance'] = testing_data['Curricular units 1st sem (approved)'] / testing_data['Curricular units 1st sem (enrolled)']
testing_data['Second Semester Performance'] = testing_data['Curricular units 2nd sem (approved)'] / testing_data['Curricular units 2nd sem (enrolled)']


final_predictions =model_pipeline.predict(testing_data)


sample_submission.head(6)


label_mapping = {0: 'Graduate', 1: 'Dropout', 2: 'Enrolled'}

# Map the numerical predictions back to the original labels
final_predictions = [label_mapping[pred] for pred in final_predictions]

submission_df = pd.DataFrame({"id": testing_data["id"], "Target": final_predictions})
submission_df.head()


# Save the DataFrame to a CSV file
submission_df.to_csv("submission.csv")


# Save to CSV (required format)
submission_df.to_csv("submission.csv", index=False)

