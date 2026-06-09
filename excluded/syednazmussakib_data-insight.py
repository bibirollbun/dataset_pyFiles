import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



!pip install edapipeline


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from edapipeline import EDAPipeline


train_df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


print("Shape of Train Data", train_df.shape)
print("Shape of Test Data", test_df.shape)


num_cols = train_df.select_dtypes(include=['number']).columns.to_list()
cat_cols = train_df.select_dtypes(include=['object']).columns.to_list()


num_cols


cat_cols


def get_column_type(col):
    if pd.api.types.is_integer_dtype(col):
        return 'Integer';
    elif pd.api.types.is_float_dtype(col):
        return 'Float';
    elif pd.api.types.is_object_dtype(col):
        return 'Object';
    elif pd.api.types.is_bool_dtype(col):
        return 'Boolean';
    elif pd.api.types.is_categorical_dtype(col):
        return 'Categorical';
    elif pd.api.types.is_datetime64_any_dtype(col):
        return 'DateTime';
    else:
        return 'Other';

def create_dataset_info(df, name):
    info = [];

    # Define the order of column types
    type_order = ['Integer', 'Float', 'Object', 'Boolean', 'Categorical', 'DateTime', 'Other'];

    # Group columns by their types
    columns_by_type = {type_name: [] for type_name in type_order};

    for col in df.columns:
        col_type = get_column_type(df[col]);
        columns_by_type[col_type].append(col);

    # Process columns in the specified order
    for type_name in type_order:
        for col in columns_by_type[type_name]:
            dtype = str(df[col].dtype);
            missing = df[col].isnull().sum();
            unique = df[col].nunique();
            missing_percentage = (missing / len(df)) * 100;

            is_numeric = pd.api.types.is_numeric_dtype(df[col]);
            mean = df[col].mean() if is_numeric else None;
            median = df[col].median() if is_numeric else None;
            std_dev = df[col].std() if is_numeric else None;
            min_value = df[col].min() if is_numeric else None;
            max_value = df[col].max() if is_numeric else None;

            most_common = df[col].mode()[0] if unique > 0 else None;

            info.append({
                'Column': col,
                'Data Type': dtype,
                'Column Type': type_name,
                'Missing Values': missing,
                'Unique Values': unique,
                'Missing Percentage': missing_percentage,
                'Mean': mean,
                'Median': median,
                'Standard Deviation': std_dev,
                'Min Value': min_value,
                'Max Value': max_value,
                'Most Common Value': most_common
            });

    return pd.DataFrame(info);


train_info = create_dataset_info(train_df, "Train")
test_info = create_dataset_info(test_df, "Test")


train_info


test_info


eda = EDAPipeline(df=train_df, target_col='Calories')


# Overview of dataset
eda.data_overview()


# Numerical feature analysis
eda.analyze_numerical_features()


# Categorical feature analysis
eda.analyze_categorical_features()


# Correlation analysis
eda.correlation_analysis()


eda.categorical_bivariate_analysis()


eda.numerical_bivariate_analysis()


eda.detect_outliers(method='iqr')


eda.detect_outliers(method='zscore')




