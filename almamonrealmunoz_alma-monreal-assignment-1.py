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


#importing pandas
import pandas as pd

#loading the training, testing, and sample submission data
trainingData = pd.read_csv(r'/kaggle/input/playground-series-s4e6/train.csv')
testingData = pd.read_csv(r'/kaggle/input/playground-series-s4e6/test.csv')
sampleSubmission = pd.read_csv(r'/kaggle/input/playground-series-s4e6/sample_submission.csv')


#verifying the data was loaded
for dataset in [trainingData, testingData, sampleSubmission]:
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

            column_dict = column_dict.fillna(NN)  # Replace NaN values with NN

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
data_profiling_df = create_data_profiling_df(data = trainingData)

# print the dataframe
data_profiling_df


#importing matplot library
import matplotlib.pyplot as plt

# define function to plot histogram and identify outliers
def plotHistogram(df: pd.DataFrame,
                   variable: str,
                   bins=10,
                   color='pink',
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
for feature in ["Gender", "Age at enrollment", "Marital status", "Daytime/evening attendance", "Scholarship holder", "Admission grade"]:
    plotHistogram(df = trainingData,
                   variable = feature,
                   bins = 15)


import seaborn as sns

plt.figure(figsize=(7, 2))

sns.countplot(x='Target', data=trainingData, color='pink') 

#adding title
plt.title('Target Categories')

# Show the plot
plt.show()


# Feature 1: grade overview
trainingData['Grade Overview'] = (
    trainingData['Admission grade'] + 
    trainingData['Curricular units 1st sem (grade)'] + 
    trainingData['Curricular units 2nd sem (grade)']
)

# Feature 2: Parental Qualification Level (combining both parents' qualifications)
trainingData['Parental qualification'] = trainingData['Mother\'s qualification'].astype(str) + "_" + trainingData['Father\'s qualification'].astype(str)

# Print the first few rows of the new features
print(trainingData[['Grade Overview', 'Parental qualification']].head())

# Feature 1: grade overview
testingData['Grade Overview'] = (
    testingData['Admission grade'] + 
    testingData['Curricular units 1st sem (grade)'] + 
    testingData['Curricular units 2nd sem (grade)']
)

# Feature 2: Parental Qualification Level (combining both parents' qualifications) in the testing data
testingData['Parental qualification'] = testingData['Mother\'s qualification'].astype(str) + "_" + testingData['Father\'s qualification'].astype(str)

# Print the first few rows of the new features in the testing data
print(testingData[['Grade Overview', 'Parental qualification']].head())


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report

threshold_grade = trainingData['Grade Overview'].nlargest(22).min()
trainingData['Target'] = (trainingData['Grade Overview'] <= threshold_grade) & \
                          (trainingData['Parental qualification'].isin(trainingData['Parental qualification'].value_counts().nsmallest(22).index))

#creating the target variable for dropout prediction
threshold_grade_test = testingData['Grade Overview'].nlargest(22).min()
testingData['Target'] = (testingData['Grade Overview'] <= threshold_grade_test) & \
                        (testingData['Parental qualification'].isin(testingData['Parental qualification'].value_counts().nsmallest(22).index))

#define features and target
features = ['Grade Overview', 'Parental qualification']
target = 'Target'

#define the preprocessing for the features
numeric_features = ['Grade Overview']
categorical_features = ['Parental qualification']

#imputing missing values and scale in numerical data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

#imputting missing values and encode labels in categorical
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
        transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])


#pipeline with a Random Forest Classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])
                    
#test and train split
X = trainingData[features]
y = trainingData[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#training the Random Forest Model
pipeline.fit(X_train, y_train)

#predictions
y_pred = pipeline.predict(X_test)

#evaluation
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))





test_features = testingData[features]
test_predictions = pipeline.predict(test_features)
testingData['Predicted Target'] = test_predictions




#predictions on the test data using the trained model
test_features = testingData[features]  
test_predictions = pipeline.predict(test_features)  

#adding predictions to the testingData DataFrame
testingData['Predicted Target'] = test_predictions 

#printing the first few rows of the testingData with predictions
print(testingData[['Grade Overview', 'Parental qualification', 'Predicted Target']].head())



submission = pd.DataFrame({
    'Grade Overview': testingData['Grade Overview'],
    'Parental qualification': testingData['Parental qualification'],
    'Predicted Target': testingData['Predicted Target']
})

#saving the submission DataFrame to a CSV file
submission.to_csv('submission.csv', index=False)

#printing a message confirming the save
print("Submission file saved as 'submission.csv'")

