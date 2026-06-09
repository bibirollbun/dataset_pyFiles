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
training_data = pd.read_csv(r'/kaggle/input/playground-series-s5e2/train.csv')
training_extra = pd.read_csv(r'/kaggle/input/playground-series-s5e2/training_extra.csv')
testing_data = pd.read_csv(r'/kaggle/input/playground-series-s5e2/test.csv')
sample_submission = pd.read_csv(r'/kaggle/input/playground-series-s5e2/sample_submission.csv')

# verify the data was loaded
for dataset in [training_data, training_extra, testing_data, sample_submission]:
    print(f"dataset shape: {dataset.shape}")


training_data.head(2)


training_extra.head(2)


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
data_profiling_extra_df = create_data_profiling_df(data = training_extra)

# print the dataframe
data_profiling_df


# concat training data into one dataframe
training_data_combined = pd.concat([training_data, training_extra], ignore_index=True)

data_profiling_combined_df = create_data_profiling_df(data = training_data_combined)
data_profiling_combined_df


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


training_data_combined.select_dtypes(include=['int64', 'float64']).columns.tolist()


# run the histogram function on all numerical features
for feature in training_data_combined.select_dtypes(include=['int64', 'float64']).columns.tolist():
    plot_histogram(df = training_data_combined,
                   variable = feature,
                   bins = 10)


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


training_data_combined.select_dtypes(include=['object', 'bool']).columns.tolist()


for feature in training_data_combined.select_dtypes(include=['object', 'bool']).columns.tolist():
    count_plot(df = training_data_combined,
               variable = feature)


# function to create a violinplot of categorical features against price

sns.violinplot(data=training_data_combined,
                y="Brand",
                x="Price",
               fill=False,
               orient = "h",
               color = "grey"
              )


sns.boxplot(data=training_data_combined,
                x="Size",
                y="Price"
              )


# function to create a scatter plot of numeric features against price
import matplotlib.pyplot as plt
import seaborn as sns

sns.scatterplot(data=training_data_combined,
                x="Weight Capacity (kg)",
                y="Price",
                color = "grey",
                #hue="Brand",
                alpha = 0.002)


# let's use a correlation coefficient to determine which features to filter out
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

numeric_features = training_data_combined.select_dtypes(include=['int64','int32','float64']).columns.tolist()

# create correlation matrix
corr_matrix = training_data_combined[numeric_features].corr().abs()

# the upper triangle of correlation matrix
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# plot the heatmap of the upper triangle
plt.figure(figsize=(8, 6))
sns.heatmap(upper_triangle, annot=True, cmap='coolwarm', fmt=".5f", linewidths=.5)
plt.title('Correlation Heatmap of Features')
plt.show()


# ...


#df["work_backpack"] = df["Brand"].apply(lambda x: 1 if x in ["PremiumBrandA", "PremiumBrandB"] else 0)
#df["Durability_Score"] = df["Waterproof"].map({"Yes": 1, "No": 0}) + df["Laptop Compartment"].map({"Yes": 1, "No": 0})
#df["Price_Per_Capacity"] = df["Price"] / df["Weight Capacity (kg)"]



training_data_combined.info()


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

cluster_features = ["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style"]

# Convert categorical features into numerical using One-Hot Encoding
training_data_encoded = pd.get_dummies(training_data_combined[cluster_features], drop_first=True)

# Standardize the data for clustering
scaler = StandardScaler()
training_data_encoded_scaled = scaler.fit_transform(training_data_encoded)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)  # Try different k values
training_data_combined["cluster"] = kmeans.fit_predict(training_data_encoded_scaled)



training_data_combined['cluster'].value_counts()


sns.violinplot(data=training_data_combined,
                y="cluster",
                x="Price",
               fill=False,
               orient = "h",
               color = "grey"
              )


training_data.info()


# choose features to use in classification model
numeric_features = ['Compartments', 'Weight Capacity (kg)']
nominal_features = ['cluster']
ordinal_features = ['Size']

all_features = numeric_features + nominal_features + ordinal_features
all_features


from sklearn.model_selection import train_test_split

X = training_data_combined[all_features]

y = training_data_combined['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"""size of training set: {len(X_train)}
size of testing set: {len(X_test)}""")


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Define the transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    #('scaler', StandardScaler())
                               ])

nominal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

ordinal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder(categories=[['Small','Medium', 'Large']]))])

preprocessor = ColumnTransformer(
    transformers=[
        ('numeric_transformer', numeric_transformer, numeric_features),
        ('nominal_transformer', nominal_transformer, nominal_features),
        ('ordinal_transformer', ordinal_transformer, ordinal_features)])


# view your pipeline
preprocessor


from sklearn.linear_model import LinearRegression

# define the pipeline with the preprocessor and the Linear Regression
LR_model = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', LinearRegression())])

# train the model
LR_model.fit(X_train, y_train)

# print the model
LR_model


from sklearn.metrics import mean_squared_error
import numpy as np

# make prdictions on the training data
LR_predictions = LR_model.predict(X_test)
rmse = mean_squared_error(y_test, LR_predictions, squared=False)
print(f"Root Mean Square Error (RMSE): {rmse}")


# Convert categorical features into numerical using One-Hot Encoding
testing_data_encoded = pd.get_dummies(testing_data[cluster_features], drop_first=True)

# Standardize the data for clustering
testing_data_encoded_scaled = scaler.fit_transform(testing_data_encoded)

# Apply K-Means clustering
#kmeans = KMeans(n_clusters=5, random_state=42)  # Try different k values
testing_data["cluster"] = kmeans.fit_predict(testing_data_encoded_scaled)
testing_data.head(3)


sample_submission.head(2)


# make predictions on the test set
final_predictions = LR_model.predict(testing_data)

# turn your predictions into a list
final_predictions = final_predictions.tolist()

# make your predictions into a dataframe
submission_df = pd.DataFrame({"id" : testing_data["id"],
                              "Price" : final_predictions})

# print a value count from the predictions
submission_df["Price"].describe()


submission_df.to_csv("submission.csv", index = False)




