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

# load the sample submission, training and testing data
sample_submission = pd.read_csv(r'/kaggle/input/playground-series-s4e6/sample_submission.csv')
training_data = pd.read_csv(r'/kaggle/input/playground-series-s4e6/train.csv')
testing_data = pd.read_csv(r'/kaggle/input/playground-series-s4e6/test.csv')

# display the data frame information
display(training_data)
training_data.info()
training_data.describe()

display(testing_data)
testing_data.info()
testing_data.describe()


# verify the data was loaded
for dataset in [sample_submission, training_data, testing_data]:
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


# run the data profiling function
data_profiling_df = create_data_profiling_df(data = training_data)

# print the dataframe
data_profiling_df


 #Check the list of columns available in the train dataset
import pandas as pd
trainset_columns_list=pd.DataFrame({"Name of each column in Train Set": training_data.columns})
display(trainset_columns_list)

 #Strip column names to remove unexpected spaces
training_data.columns = training_data.columns.str.strip()

 #Check the numbers of rows and columns:
print(training_data.shape)

 #Check missing values in each column:
print(training_data.isnull().sum())
print(training_data.isnull().sum().sum())

 #Check duplicated lines:
print(training_data.duplicated().sum())

 #Erase duplicated lines:
training_data = training_data.drop_duplicates()


#Check if the 'Target' column is available in the dataset:
Target = 'Target'  # Adjust according to dataset
if Target not in training_data.columns:
    raise KeyError(f"Column '{Target}' not found in dataset. Available columns: {train_df.columns.tolist()}")


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


# looping through each numerical feature to look at its distribution
for feature in ["Previous qualification (grade)", "Admission grade", "Age at enrollment", "Curricular units 1st sem (grade)", "Curricular units 2nd sem (grade)"]:
    plot_histogram(df = training_data,
                  variable = feature,
                  bins = 12)


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


for feature in ["Course", "Daytime/evening attendance", "Admission grade", "Age at enrollment"]:
    count_plot(df = training_data,
              variable = feature) 


# Check for missing values
print(training_data.isnull().sum())


training_data["Parental Education Level"] = training_data["Mother's qualification"] + training_data["Father's qualification"]

training_data.groupby("Parental Education Level", as_index=False)["Admission grade"].median().sort_values(by="Parental Education Level")


training_data["First Semester Success Rate"] = training_data["Curricular units 1st sem (approved)"] + training_data["Curricular units 1st sem (enrolled)"]

training_data.groupby("First Semester Success Rate", as_index=False)["Admission grade"].median().sort_values(by="First Semester Success Rate")


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# create and plot the first new feature
new_feature_name = ('Parental Education Level')
def plot_new_feature(df: pd.DataFrame, new_feature_name: str):
    plot_df = df.copy()
    
    # Fixed column name (space vs underscore consistency)
    # Group by new feature and calculate mean admission grade
    grouped_data = plot_df.groupby(new_feature_name)['Admission grade'].mean().reset_index()
    
    # Sort by admission grade for better visualization
    grouped_data = grouped_data.sort_values(by='Admission grade', ascending=False)
    
    # Calculate overall average admission grade
    average_admission_grade = plot_df['Admission grade'].mean()
    
    # Create plot
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        x=new_feature_name,
        y='Admission grade',
        data=grouped_data,
        color='grey',
        order=grouped_data[new_feature_name]  # Maintain sorted order
    )
    
    # Add reference line and improve labeling
    plt.axhline(average_admission_grade, 
                color='blue', 
                linestyle='--', 
                label=f'Overall Average ({average_admission_grade:.1f})')
    
    # Improve formatting
    plt.xlabel('Parental Education Level')
    plt.ylabel('Average Admission Grade')
    plt.title(f'Admission Grades by {new_feature_name.replace("_", " ").title()}')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


# Create parental education feature (ensure numerical qualifications)
plotting_df = training_data.copy()
plotting_df['parental_education'] = (
    plotting_df["Mother's qualification"] + 
    plotting_df["Father's qualification"]
) / 2

# Run the analysis
plot_new_feature(
    df=plotting_df,
    new_feature_name='parental_education'
)


#create and plot the second new feature
new_feature_name = ('First Semester Success Rate Groups')

#To figure out how the First Semester Success Rate influence the admission grade of students
#create a function to look into possible influcence of new features - admission grade

def plot_new_feature(df: pd.DataFrame, new_feature_name: str):
    plot_df = df.copy()
    
    # Group by new feature and calculate mean admission grade
    new_feature_admission_grade = plot_df.groupby(new_feature_name)['Admission grade'].mean().reset_index()
    
    # Sort by admission grade for better visualization
    new_feature_admission_grade = new_feature_admission_grade.sort_values(by='Admission grade', ascending=False)
    
    # Calculate overall average admission grade
    average_admission_grade = plot_df['Admission grade'].mean()
    
    # Create plot
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        x=new_feature_name,
        y='Admission grade',
        data=new_feature_admission_grade,
        color='skyblue',
        order=new_feature_admission_grade[new_feature_name]
    )
    
    # Add reference line and labels
    plt.axhline(average_admission_grade, 
                color='red', 
                linestyle='--', 
                label=f'Overall Average ({average_admission_grade:.1f})')
    
    # Formatting
    feature_label = new_feature_name.replace("_", " ").title()
    plt.xlabel(f'First Semester Success Rate Groups')
    plt.ylabel('Average Admission Grade')
    plt.title(f'Admission Grades by {feature_label}')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show()




# Calculate success rate with error handling
plotting_df['first_sem_success_rate'] = (
    plotting_df['Curricular units 1st sem (approved)'] / 
    plotting_df['Curricular units 1st sem (enrolled)']
).replace([np.inf, -np.inf], np.nan).round(2)

# Create performance categories
plotting_df['success_group'] = pd.cut(plotting_df['first_sem_success_rate'],
                                     bins=[0, 0.3, 0.6, 0.9, 1],
                                     labels=['Low (0-30%)', 'Moderate (31-60%)', 
                                             'High (61-90%)', 'Excellent (91-100%)'])

# Run the analysis
plot_new_feature(
    df=plotting_df.dropna(subset=['success_group']),
    new_feature_name='success_group'
)



def create_family_size_rank(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["parental_educational_level"] = df['Mother\'s qualification'] + df['Father\'s qualification']

    def rank_rows(row):

        if row['Mother\'s qualification'] in [1,15]:
            return 1
        elif row['Father\'s qualification'] in [15,30]:
            return 2
        else:
            return 3

    df['first_sem_success_rate'] = df.apply(rank_rows, axis=1)
    
    return df



def create_first_sem_success_rate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["first_sem_success_rate"] = df['Curricular units 1st sem (approved)'] + df['Curricular units 1st sem (enrolled)']


    def rank_rows(row):

        if row['first_sem_success_rate'] in [91,100]:
            return 1
        elif row['first_sem_success_rate'] in [61,90]:
            return 2
        elif row['first_sem_success_rate'] in [31,60]:
            return 3
        else:
            return 4

    df['first_sem_success_rate'] = df.apply(rank_rows, axis=1)
    
    return df



# Function to combine new features into the original dataset
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    new_df = df.copy()
    new_df = create_family_size_rank(df=new_df)  # Add Parental Education Level
    new_df = create_first_sem_success_rate(df=new_df)  # Add First Semester Success Rate
    return df

# Apply the function to training_data
training_data_new = feature_engineering(df=training_data)

#See newly created columns
training_data_new.sample(5)



# see the newly created columns
training_data.sample(5)


training_data.info()


training_data.select_dtypes(include=['int64', 'float64']).columns.tolist()


# use a correlation coefficient to determine which features to filter out
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

numeric_features = training_data.select_dtypes(include=['int64', 'float64']).columns.tolist()

# create correlation matrix
corr_matrix = training_data[numeric_features].corr().abs()

# the upper triangle of correlation matrix
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# plot the heatmap of the upper triangle
plt.figure(figsize=(8, 6))
sns.heatmap(upper_triangle, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap of Features')
plt.show()


# choose features to use in classification model
numeric_features = ['Previous qualification (grade)', 'Unemployment rate', 'Inflation rate', 'GDP']
nominal_features = ['Marital status', 'Nacionality', 'Scholarship holder']
ordinal_features = ['Father\'s occupation']

all_features = numeric_features + nominal_features + ordinal_features
all_features


from sklearn.model_selection import train_test_split

X = training_data[all_features]

y = training_data['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"""size of training set: {len(X_train)}
size of testing set: {len(X_test)}""")


# Import necessary libraries
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Define the transformers for different data types

# Transformer for numerical features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Fill missing values with median
    ('scaler', StandardScaler())  # Standardize numerical values
])

# Transformer for categorical nominal features (One-Hot Encoding)
nominal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing values with most frequent category
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Encode categorical variables
])

# Transformer for categorical ordinal features (Ordinal Encoding)
ordinal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing values with most frequent category
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))  # Assuming Target is ordinal (Adjust if needed)
    ])


# Combine all transformers into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('numeric_transformer', numeric_transformer, numeric_features),
        ('nominal_transformer', nominal_transformer, nominal_features),
        ('ordinal_transformer', ordinal_transformer, ordinal_features)
    ]
)

# Check transformation on a sample dataset before fitting the full pipeline
X_train_transformed = preprocessor.fit_transform(X_train)
print(X_train_transformed.shape)


# View the preprocessing pipeline

preprocessor


# Create a pipeline that includes preprocessing and model training
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))  # Using RandomForestClassifier as an example
])

# View the pipeline structure
pipeline


# Train the model using the pipeline
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model
from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Model Accuracy: {round(accuracy, 3)}")
print("-" * 5)
print(f"Classification Report:\n{report}")


#Feature Importance Analysis for Random Forest model:

# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np

# Extract feature importance scores from the trained Random Forest model
feature_importance = pipeline.named_steps['classifier'].feature_importances_

# Retrieve feature names from the pipeline
feature_names = (numeric_features +
                 list(pipeline.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names_out(nominal_features)) +
                 ordinal_features)

# Sort features by importance in descending order
sorted_idx = np.argsort(feature_importance)[::-1]
sorted_features = np.array(feature_names)[sorted_idx]
sorted_importance = feature_importance[sorted_idx]

# Plot the top 10 most important features
plt.figure(figsize=(12, 6))
plt.barh(sorted_features[:10], sorted_importance[:10], color="pink")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Top 10 Most Important Features in Random Forest")
plt.gca().invert_yaxis()  # Invert y-axis for better visualization
plt.show()


print(testing_data.columns)


testing_data['parental_education'] = (
    testing_data["Mother's qualification"] + 
    testing_data["Father's qualification"]
) / 2


print("Missing values in 'parental_education':", testing_data["parental_education"].isna().sum())


print("Data type of 'parental_education' in testing_data:", testing_data["parental_education"].dtype)
print("Unique values in 'parental_education':", testing_data["parental_education"].unique())


#Submit predictions:

# Apply feature engineering on the test dataset
testing_data_new = feature_engineering(df=testing_data)
testing_data_new.sample(5)

# Make predictions using the trained Random Forest model
predictions = pipeline.predict(testing_data_new)

# Convert predictions to a list
predictions = predictions.tolist()

# Create a DataFrame for submission
submission_df = pd.DataFrame({
    "id": testing_data_new["id"],  # Ensure correct ID column from the Academic Success dataset
    "Prediction": predictions  # Predictions from the Random Forest model
})

# Print value counts for predictions
submission_df["Prediction"].value_counts()

# Save submission file as CSV
submission_df.to_csv("submission.csv", index=False)

print("Your submission file has been created")
display(submission_df)


# Check the quality of data in the train set:
#Import scikit-learn libraries to build the Machine Learning model:

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score

#Load data from Kaggle
training_data = pd.read_csv('/kaggle/input/playground-series-s4e6/train.csv')
testing_data = pd.read_csv('/kaggle/input/playground-series-s4e6/test.csv')


#Display Data Frame information:
display(training_data)
training_data.info()
training_data.describe()

display(testing_data)
testing_data.info()
testing_data.describe()


#Check the list of columns available in the train dataset
import pandas as pd
trainset_columns_list=pd.DataFrame({"Name of each column in Train Set":
                                   training_data.columns})
display(trainset_columns_list)

#Strip column names to remove unexpected spaces
training_data.columns = training_data.columns.str.strip()

#check the numbers of rows and columns:
print(training_data.shape)

#check missing value in each column:
print(training_data.isnull().sum())
print(training_data.isnull().sum().sum())

#check duplicate lines:
print(training_data.duplicated().sum)

#Erase duplicated lines:
training_data = training_data.drop_duplicates


#Check if the 'Target' column is available in the dataset:
Target = 'Target'  # Adjust according to dataset
if Target not in train_df.columns:
    raise KeyError(f"Column '{Target}' not found in dataset. Available columns: {train_df.columns.tolist()}")


# Define features and target variable
X = train_df.drop(columns=[Target]) #Available Features #Input
Y = train_df[Target] #Target #Output


#Feature Engineering: Create 2 new features
X_numerical = X.select_dtypes(include=['int64', 'float64'])  #Select only numerical features in X
testdf_numerical = test_df.select_dtypes(include=['int64', 'float64']) #Select only numerical features in the test set

 #Create a feature that contains the sum of all numerical columns for each row in X
X['feature_sum'] = X_numerical.sum(axis=1)

 #Create a feature that contains the mean of all numerical columns for each row in X
X['feature_mean'] = X_numerical.mean(axis=1) 

 #Create a feature that contains the sum of all numerical columns for each row in test set
test_df['feature_sum'] = testdf_numerical.sum(axis=1)

 #Create a feature that contains the mean of all numerical columns for each row in X
test_df['feature_mean'] = testdf_numerical.mean(axis=1)

display(X)
display(test_df)


# Categorize columns of features in X by data type:

 #1_To preprocess data properly - Numerical datatype and Categorical datatype requires different ways of Processing before being fed into the model for training.
 #2_Machine Learning Model only works with Numerical datatype -> Need to process Categorical datatype into numerical datatype.
 #3_To automate the data preprocessing process by Pipeline.


 #Numerical features
num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist() #1 #For Pipeline preprocessing

num_features_display = pd.DataFrame({"Numerical Features": X.select_dtypes(include=['int64', 'float64']).columns}) #2 For better display when running this code

  #Categorical features #Include both Nominal Features & Ordinal Features
cat_features = X.select_dtypes(include=['object']).columns.tolist() #1 #For Pipeline preprocessing

cat_features_display = pd.DataFrame({"Categorical Features": X.select_dtypes(include=['object']).columns}) #2 For better display when running this code

 #To check if ['object'] dtypes is missed or not:
if 'object' in X.dtypes.values:
    cat_features = X.select_dtypes(include=['object']).columns.tolist()
else:
    cat_features = []

#Check if columns is also available in X:
num_features = [col for col in num_features if col in X.columns]
cat_features = [col for col in cat_features if col in X.columns]

X_columns_display = pd.DataFrame({"List of Columns in X":X.columns})

combine_for_comparison = pd.concat([num_features_display, X_columns_display, cat_features_display], axis=1)

print("Categorical Features:", cat_features)
display(combine_for_comparison)


# Data Preprocessing Pipelines: With ColumnTransformer #Sci-kit learn Pipeline

 #Create 2 Preprocessing Pipelines:
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # Fill missing values with mean
    ('scaler', StandardScaler())  # Standardize numerical data
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing values with most frequent
    ('encoder', OneHotEncoder(handle_unknown='ignore'))  # Encode categorical features
])

 #Use ColumnTransformer to combine Pipelines and apply them to categorized features:
Applying_Preprocessing_Pipelines_on_categorized_features = ColumnTransformer([  
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])

#Set up Preprocessing Pipeline for your training data (X):
X_final = Applying_Preprocessing_Pipelines_on_categorized_features.fit_transform(X) 
print("Congratulations! You have preprocessed the training data by Pipeline successfully")


#Choose model (Decision Tree or Random Forest)
Final_Model = RandomForestClassifier(n_estimators=100, random_state=1)

#Random Forest used because the training data has a lot of features -> need a strong model for more precise predictions.


# Create End-to-End Machine Learning Pipeline - Preprocessing Pipeline & Training & Prediction Pipeline
clf = Pipeline([
    ('Applying_Preprocessing_Pipelines_on_categorized_features', Applying_Preprocessing_Pipelines_on_categorized_features),
    ('ML classifier', Final_Model)
])


#Split train and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=1)


#Apply End-to-End Machine Learning Pipeline to train the model:
clf.fit(X_train, Y_train)


# Evaluate model by Cross-Validation (CV) 
cross_val_scores = cross_val_score(clf, X_train, Y_train, cv=5, scoring='accuracy')
cross_val_scores_display = pd.DataFrame(cross_val_scores, columns=["Cross Validation Scores"])  #for display
display(cross_val_scores_display)
print("Cross-validation accuracy:", np.mean(cross_val_scores))


# Evaluate model on validation set
validation_accuracy = clf.score(X_val, Y_val)
print(f"Validation Accuracy: {validation_accuracy:.4f}")


# Make predictions on the test data (CSV) #Data run through the same Pipeline set up for the training data

Predictions = clf.predict(test_df)

#Create a copy for test_df to add a column of Predictions results for comparison:
test_df_copy = test_df.copy()
test_df_copy['Predictions'] = Predictions
display(test_df_copy)


# Create submission file
submission_file = pd.DataFrame({'id': test_df['id'], 'Target': Predictions})  # Adjust 'target' according to Kaggle competition
submission_file.to_csv('submission.csv', index=False)
print("Your submission file has been created")
display(submission_file)

