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

sample = pd.read_csv('/kaggle/input/playground-series-s4e6/sample_submission.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e6/test.csv')
train = pd.read_csv('/kaggle/input/playground-series-s4e6/train.csv')


for dataset in [sample,test,train]:
    print(f"dataset shape: {dataset.shape}")


train.head()


test.info()


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
data_profiling_df = create_data_profiling_df(data = train)

# print the dataframe
data_profiling_df


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
train['Target_encoded'] = encoder.fit_transform(train['Target'])

# Mapping of categories to numbers
label_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
label_mapping



# CONVETING CATEGORICAL DATA TO NUMERIC DATA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Encode the categorical column
encoder = LabelEncoder()
train['Target_encoded'] = encoder.fit_transform(train['Target'])

# Create a mapping dictionary
label_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

# Compute mean count
mean_value = train['Target_encoded'].value_counts().mean()

# Plot histogram with better styling
plt.figure(figsize=(8, 5))
ax = sns.barplot(
    x=train['Target_encoded'].value_counts().index,
    y=train['Target_encoded'].value_counts().values,
    palette="coolwarm"
)

# Set labels and title
plt.xlabel("Encoded Target Values", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.title("Distribution of Encoded Target Categories", fontsize=14)

# Add labels on bars
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=12, fontweight='bold')

# Show category names on x-axis
plt.xticks(ticks=list(label_mapping.values()), labels=label_mapping.keys(), fontsize=12)

# Add a horizontal mean line
plt.axhline(y=mean_value, color='black', linestyle='--', linewidth=2, label=f'Mean Count: {mean_value:.1f}')
plt.legend()

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()



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


train.columns


# run the histogram function on all numerical features
for feature in ["Curricular units 1st sem (approved)","Curricular units 2nd sem (approved)","Curricular units 1st sem (grade)","Curricular units 2nd sem (grade)"]:
    plot_histogram(df = train,
                   variable = feature,
                   bins = 15)


import matplotlib.pyplot as plt
import seaborn as sns

numeric_features = train.select_dtypes(include=['int64', 'float64']).columns.tolist()

# create correlation matrix
corr_matrix = train[numeric_features].corr().abs()

# the upper triangle of correlation matrix
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# plot the heatmap of the upper triangle
plt.figure(figsize=(30, 24))
sns.heatmap(upper_triangle, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap of Features')
plt.show()


# AVERAGE GRADE OF THE STUDENT
train['Average Grade'] = (train['Curricular units 2nd sem (grade)'] + train['Curricular units 1st sem (grade)']) / 2


# AVERAGE CIRICULAR UNITS A STUDENT PERSUE
train['Average Curricular Units'] = (train['Curricular units 1st sem (grade)'] + train['Curricular units 2nd sem (grade)']) / 2


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Calculate Q1 (25th percentile) and Q3 (75th percentile) for IQR method
Q1 = train['Average Grade'].quantile(0.25)
Q3 = train['Average Grade'].quantile(0.75)
IQR = Q3 - Q1

# Define bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Find outliers using IQR method
outliers_IQR = train[(train['Average Grade'] < lower_bound) | (train['Average Grade'] > upper_bound)]

# Plotting the distribution of Average Grade with outliers marked
plt.figure(figsize=(10, 6))
sns.histplot(train['Average Grade'], kde=True, color='blue', label='Average Grade Distribution', bins=30)

# Plot the outliers as red dots
plt.scatter(outliers_IQR['Average Grade'], [0] * len(outliers_IQR), color='red', label='Outliers', zorder=5)

# Add title and labels
plt.title('Distribution of Average Grade with Outliers', fontsize=16)
plt.xlabel('Average Grade', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

# Add legend
plt.legend()

# Display the plot
plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

numeric_features = train.select_dtypes(include=['int64', 'float64']).columns.tolist()

# create correlation matrix
corr_matrix = train[numeric_features].corr().abs()

# the upper triangle of correlation matrix
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# plot the heatmap of the upper triangle
plt.figure(figsize=(30, 24))
sns.heatmap(upper_triangle, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap of Features')
plt.show()


numeric_features = ['Average Grade', 'Average Curricular Units','Admission grade',
                    'Curricular units 1st sem (approved)', 'Curricular units 2nd sem (approved)',
                    'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)',
                    'Unemployment rate', 'Inflation rate', 'GDP']

nominal_features = ['Marital status', 'Application mode', 'Application order', 'Course','Previous qualification', "Mother's qualification",
                    "Father's qualification", "Mother's occupation","Father's occupation",'Displaced', 'Debtor', 'Scholarship holder']



all_features = numeric_features + nominal_features 
all_features


from sklearn.model_selection import train_test_split

X = train[all_features]

y = train['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"""size of training set: {len(X_train)}
size of testing set: {len(X_test)}""")


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier

# Define the transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

nominal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', numeric_transformer, numeric_features),
        ('nominal', nominal_transformer, nominal_features)
    ])


# define the pipeline with the preprocessor and the DecisionTreeClassifier
decision_tree_model = Pipeline(steps=[('preprocessor', preprocessor),
                                      ('classifier', DecisionTreeClassifier(random_state = 42,
                                                             max_depth = 6,
                                                             min_samples_split = 10))])

# train the model
decision_tree_model.fit(X_train, y_train)

# print the model
decision_tree_model


from sklearn.metrics import accuracy_score, classification_report

# Predict the labels for the test set
y_pred = decision_tree_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print accuracy
print(f'Accuracy: {accuracy}')

# Print classification report
print(classification_report(y_test, y_pred))



from sklearn.model_selection import GridSearchCV
grid_search_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                       ('classifier', DecisionTreeClassifier(random_state=42))])

# define GridSearchCV parameters (don't start with too many)
param_grid = {
    'classifier__max_depth': [3, 5, 7, 10, None],
    'classifier__min_samples_split': [2, 5, 10, 20],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__max_features': ['auto', 'sqrt', 'log2', None],
    'classifier__criterion': ['gini', 'entropy']

}

# create the model
grid_search_model = GridSearchCV(grid_search_pipeline,
                                 param_grid,
                                 cv = 3,
                                 scoring='accuracy',
                                 n_jobs=-1)

# train the model using GridSearchCV
grid_search_model.fit(X_train, y_train)

# Print the best parameters and best score
print(f"Best parameters: {grid_search_model.best_params_}")
print(f"Best cross-validation score: {grid_search_model.best_score_}")

grid_search_model


grid_search_pred = grid_search_model.predict(X_test)

# evaluate the model
accuracy = accuracy_score(y_test, grid_search_pred)
report = classification_report(y_test, grid_search_pred)
print(f'Model Accuracy: {round(accuracy, 3)}')
print("="*5)
print(f'Classification Report:\n{report}')


test['Average Grade'] = (test['Curricular units 2nd sem (grade)'] + test['Curricular units 1st sem (grade)']) / 2
test['Average Curricular Units'] = (test['Curricular units 1st sem (grade)'] + test['Curricular units 2nd sem (grade)']) / 2


# make predictions on the test set
grid_search_test_predictions = grid_search_model.predict(test)

# turn your predictions into a list
grid_search_test_predictions = grid_search_test_predictions.tolist()

# make your predictions into a dataframe
submission_df = pd.DataFrame({"id": test["id"], "Target": grid_search_test_predictions})

# print a value count from the predictions
submission_df["Target"].value_counts()


submission_df.to_csv("submission.csv", index=False)


submission_df.head()

