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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
df.drop(columns=['id'], inplace=True)
df.head()


df.info()


df.shape


df.isnull().sum()


numerical_features = df.select_dtypes(include=[np.number])
df.describe(include=[np.number]).transpose()


def detail_extractor_categorical_cols(col_name):
    print(f"\n\nDetails of {col_name}:")
    print(f"Value Counts: {df[col_name].value_counts()}\n")
    print(f"Missing Values: {df[col_name].isnull().sum()}\n")
    print(f"Unique Values: {df[col_name].nunique()}\n")

    # plot of the column using plotly
    fig = px.bar(df[col_name].value_counts(), title=f'{col_name} Distribution')
    fig.show()

    # plot of the column using plotly
    fig = px.line(df.groupby(col_name)['Price'].mean(), 
             title=f'Average Price of the {col_name}',
             markers=True)  # Add markers at data points
    # Customize the layout
    fig.update_layout(
    xaxis_title=col_name,
    yaxis_title="Average Price",
    xaxis={'side': 'bottom'},  # Move x-axis to bottom
    yaxis={'side': 'left'}     # Move y-axis to left
    )
    fig.show()



for col in df.select_dtypes(include=['object']).columns:
    detail_extractor_categorical_cols(col)


# Create a box plot for the Price column
fig = px.box(df, y='Price', title='Distribution of Prices')

# Customize the layout
fig.update_layout(
    yaxis_title="Price",
    showlegend=False,
    # Add some margin to make the plot more readable
    margin=dict(t=50, b=50)
)

# Add a more detailed hover template
fig.update_traces(
    boxpoints='outliers',  # Show outliers as individual points
    hovertemplate="<br>".join([
        "Price: %{y:.2f}",
        "<extra></extra>"  # This removes the trace name from hover
    ])
)

fig.show()


# Create a line chart of null values
null_counts = df.isnull().sum()

fig = px.line(x=null_counts.index, y=null_counts.values, 
              title='Number of Null Values per Column',
              markers=True)  # Add markers at data points

# Customize the layout
fig.update_layout(
    xaxis_title="Columns",
    yaxis_title="Number of Null Values",
    xaxis={'tickangle': 45},  # Rotate x-axis labels for better readability
    showlegend=False
)

fig.show()


# First let us convert the categorical columns to numerical columns
# We will mark Null Values as "Not Defined"

def convert_categorical_to_numerical(df, col_name):
    df[col_name] = df[col_name].fillna("Not Defined")

    # Now we will convert the column to numerical
    df[col_name] = df[col_name].astype('category')
    df[col_name] = df[col_name].cat.codes
    return df


for col in df.select_dtypes(include=['object']).columns:
    convert_categorical_to_numerical(df, col)




df.dtypes


# df['Weight Capacity (kg)'].fillna(df['Weight Capacity (kg)'].mean(), inplace=True)
# df['Weight Capacity (kg)'].isnull().sum()


# Initialize KNN Imputer with 3 neighbors
imputer = KNNImputer(n_neighbors=3)

# Apply imputation (assuming df is your DataFrame)
df[['Weight Capacity (kg)']] = imputer.fit_transform(df[['Weight Capacity (kg)']])

# Check if null values are imputed
print(f"Missing values after imputation: {df['Weight Capacity (kg)'].isnull().sum()}")


df.isnull().sum()


df.dtypes


def get_test_df(path):
    test_df = pd.read_csv(path)

    # print(test_df.columns)
   
   # Drop id column if present
    if 'id' in test_df.columns:
        test_df.drop(columns=['id'], inplace=True)

    # print(test_df.columns)
   
   # Preprocess categorical columns using existing function
    for col in test_df.select_dtypes(include=['object']).columns:
        # print(col)
        test_df = convert_categorical_to_numerical(test_df, col)

    # print(test_df.columns)
   
   # Preprocess numerical columns
    # test_df['Weight Capacity (kg)'] = test_df['Weight Capacity (kg)'].fillna(
    #     test_df['Weight Capacity (kg)'].mean()
    # )
    test_df[['Weight Capacity (kg)']] = imputer.transform(test_df[['Weight Capacity (kg)']])

    print(test_df.columns)
    
    return test_df


test = get_test_df("/kaggle/input/student-bag-price-prediction-dataset/Noisy_Student_Bag_Price_Prediction_Dataset.csv")


test.shape


test.isnull().sum()


test = test.dropna()


test.isnull().sum()


test.shape


df.isnull().sum()



X_train = df.drop(columns=['Price'])
y_train = df['Price']
X_test = test.drop(columns=['Price'])
y_test = test['Price']


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Initialize StandardScaler
scaler = StandardScaler()

# Fit on training data and transform both train & test data
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


X_train


model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Root Mean Squared Error: {rmse}")



def get_predictions(model, test_df):
    # Make predictions
    predictions = model.predict(test_df)
    
    # Create submission dataframe
    submission_df = pd.DataFrame()
    submission_df['id'] = range(300000, 300000 + len(predictions))
    submission_df['Price'] = predictions
    
    return submission_df


def get_test_df(path="/kaggle/input/playground-series-s5e2/test.csv"):
    test_df = pd.read_csv(path)

    # print(test_df.columns)
   
   # Drop id column if present
    if 'id' in test_df.columns:
        test_df.drop(columns=['id'], inplace=True)

    # print(test_df.columns)
   
   # Preprocess categorical columns using existing function
    for col in test_df.select_dtypes(include=['object']).columns:
        print(col)
        test_df = convert_categorical_to_numerical(test_df, col)

    # print(test_df.columns)
   
   # Preprocess numerical columns
    test_df['Weight Capacity (kg)'] = test_df['Weight Capacity (kg)'].fillna(
        test_df['Weight Capacity (kg)'].mean()
    )

    print(test_df.columns)
    
    return test_df


test_df = get_test_df("/kaggle/input/playground-series-s5e2/test.csv")


test_df.head()


submission_csv = get_predictions(model, test_df)


submission_csv.head()


submission_csv.to_csv("submission.csv", index=False)




