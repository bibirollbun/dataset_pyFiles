import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, skew, kurtosis

from prettytable import PrettyTable
import warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        FILE_PATH = os.path.join(dirname, filename)

        if 'train' in filename:
            TRAIN_PATH = FILE_PATH

        elif 'test' in filename:
            TEST_PATH = FILE_PATH

        else:
            SUBMISSION_PATH = FILE_PATH

pd.set_option('display.max_columns', None)

TARGET_FEATURE = 'Premium Amount'
SAMPLE_SIZE = 600000


train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
submission = pd.read_csv(SUBMISSION_PATH)


def reduce_memory_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage of dataframe is {start_mem:.2f} MB")
    
    for col in df.columns:
        col_type = df[col].dtype
        
        
        if col_type != object:
            if pd.api.types.is_float_dtype(col_type):
                df[col] = pd.to_numeric(df[col], downcast='float')
            elif pd.api.types.is_integer_dtype(col_type):
                df[col] = pd.to_numeric(df[col], downcast='integer')
        else:
            num_unique_values = df[col].nunique()
            num_total_values = len(df[col])
            if num_unique_values / num_total_values < 0.5:
                df[col] = df[col].astype('category')
    
    end_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage after optimization is: {end_mem:.2f} MB")
    reduction_percentage = ((start_mem - end_mem) / start_mem) * 100
    print(f"Reduction in memory usage: {reduction_percentage:.1f}%")
    
    return df


train = reduce_memory_usage(train)


train.drop(columns = 'id', inplace = True)


train.info()


def get_info(dataframe: pd.core.frame.DataFrame):
    
    """
    This function takes a dataframe as input and
    returns a short summary.
    """
    print(f"Total Records: {dataframe.shape[0]}")
    print(f"Total Features: {dataframe.shape[1]}")
    print(f"Total Duplicate Records: {dataframe.duplicated().sum()}")
    info = PrettyTable()
    info.field_names = ['Column', 
                        'Data Type', 
                        'Missing Values', 
                        'Missing Percentage', 
                        'Unique Values', 
                       'Percentage Unique']
    
    for column in dataframe.columns:
        
        data_type = dataframe[column].dtypes
        missing_values = dataframe[column].isnull().sum()
        missing_percentage = np.round(100 * dataframe[column].isnull().sum() / len(dataframe), 2)
        unique_values = dataframe[column].nunique()
        percentage_unique = np.round(100 * dataframe[column].nunique() / len(dataframe), 2)
        
        info.add_row([column, data_type, missing_values, missing_percentage, unique_values, percentage_unique])
    
    print(info)


get_info(train)


train['Policy Start Date'] = pd.to_datetime(train['Policy Start Date'])


train.describe().T


train['Annual Income'].min(), train['Annual Income'].max()


train = train[train['Annual Income'] >= 5000]


print(f"Total Records: {len(train)}")


train_sample = train.sample(n = SAMPLE_SIZE, random_state = 4)


plt.figure(figsize = (12, 6))
sb.kdeplot(train_sample['Premium Amount'])

plt.title(f'Distribution of {TARGET_FEATURE}')


from scipy.stats import skew, kurtosis

skewness = skew(train_sample[TARGET_FEATURE]).round(2)
kurtosis = kurtosis(train_sample[TARGET_FEATURE]).round(2)

print(f"Skewness: {skewness}")
print(f"Kurtosis: {kurtosis}")


def plot_feature_distributions(df, kind: str) -> None:

    """
    This function plots the feature distribution from the
    given dataframe

    Args:
        df(pd.core.frame.dataframe): The dataframe
        kind(str): The type of plot

    Raises: 
        Exception: When provided plot is not among the options
    """
    
    if kind in ['kde', 'box']:
        features = df.select_dtypes(include = np.number).columns
        
    elif kind in ['count']:
        features = df.select_dtypes(include = ['object', 'category']).columns

    else:
        raise Exception("Invalid plot type! Expected values are 'kind', 'box', 'count'")
    
    num_features = len(features)
    num_rows = (num_features // 3) + 1
    plt.figure(figsize=(15, num_rows * 5))

    for i, column in enumerate(features):
        plt.subplot(num_rows, 3, i + 1)
        
        if kind == 'kde':
            sb.kdeplot(x = df[column])
            
        elif kind == 'box':
            sb.boxplot(x = df[column])
            
        elif kind == 'count':
            sb.countplot(x = df[column])
            plt.xticks(rotation = 60)
            
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()


plot_feature_distributions(train_sample, kind = 'kde')


plot_feature_distributions(train_sample, kind = 'count')


plt.figure(figsize = (10, 6))

sb.heatmap(train_sample.isnull())

plt.title("Heatmap for missing values")
plt.xlabel("Features")
plt.show()


def handle_missing_values(df):

    data_copy = df.copy()
    for feature in data_copy.columns:
        if data_copy[feature].dtype in ['category', 'object']:
            data_copy[feature] = data_copy[feature].fillna(data_copy[feature].mode()[0])

        else:
            data_copy[feature] = data_copy[feature].fillna(data_copy[feature].median())

    return data_copy


for dataset in [train_sample, train, test]:
    dataset = handle_missing_values(dataset)


for dataset in [train, train_sample]:
    dataset['Year'] = dataset['Policy Start Date'].dt.year
    dataset['Month'] = dataset['Policy Start Date'].dt.month
    dataset['Day'] = dataset['Policy Start Date'].dt.day
    dataset.drop(columns = ['Policy Start Date'], inplace = True)


train_sample


categorical_features = train_sample.select_dtypes(include=['category', 'object']).columns

for feature in categorical_features:
    mean_encoding = train_sample.groupby(feature)[TARGET_FEATURE].mean()
    train_sample[feature] = train_sample[feature].map(mean_encoding)


correlation_df = train_sample.corr()

sb.heatmap(correlation_df)
plt.title("Correlation Heatmap")
plt.show()


# Top 5 correlated features
correlation_df['Premium Amount'].abs().nlargest(6)


def cap_all_numerical_features_iqr(df, factor=1.5) -> pd.core.frame.DataFrame:
    """
    Identifies all numerical columns in a DataFrame and performs 
    outlier capping on each one using the IQR method.

    Args:
        df (pd.DataFrame): The input DataFrame (e.g., train_sample).
        factor (float): The multiplication factor for the IQR (default is 1.5,
                        which typically defines outliers).

    Returns:
        pd.DataFrame: A new DataFrame with all numerical features capped.
    """
    df_capped = df.copy()
    numerical_features = df.select_dtypes(include=np.number).columns
    
    capping_summary = {}

    for column_name in numerical_features:
        data = df_capped[column_name].dropna()
        if len(data.unique()) < 4:
            print(f"Skipping '{column_name}': Too few unique values.")
            continue

        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - (factor * IQR)
        upper_bound = Q3 + (factor * IQR)

        original_outlier_count = (df_capped[column_name] < lower_bound).sum() + (df_capped[column_name] > upper_bound).sum()
        
        if original_outlier_count > 0:
            df_capped[column_name] = np.clip(
                df_capped[column_name], 
                lower_bound, 
                upper_bound
            )

            
        capping_summary[column_name] = {
            "Lower Bound": round(lower_bound, 2),
            "Upper Bound": round(upper_bound, 2),
            "Outliers Capped": original_outlier_count
        }
    
    return df_capped, pd.DataFrame.from_dict(capping_summary, orient='index')


train_sample, capping_report = cap_all_numerical_features_iqr(train_sample)


plt.figure(figsize = (12, 5))

sb.barplot(
    data = capping_report.sort_values(by = 'Outliers Capped', ascending = False), 
    x = capping_report.index, 
    y = 'Outliers Capped'
)
plt.title("Outliers by features")
plt.xticks(rotation = 60)
plt.show()


train['IsCovidYear'] = np.where(
    train['Year'] == 2020,
    1,
    0
)


def target_guided_encoding(train, test, target_feature, categorical_features=None, fillna_value=None):
    """
    Apply target-guided encoding to categorical features in train and test datasets.
    
    Parameters:
        train (pd.DataFrame): Training dataset
        test (pd.DataFrame): Test dataset
        target_feature (str): Name of the target column
        categorical_features (list, optional): List of categorical feature names.
        fillna_value (float, optional): Value to fill NaNs in test data. If None, uses mean of target_feature from train
    
    Returns:
        train (pd.DataFrame): Training data with encoded features
        test (pd.DataFrame): Test data with encoded features
    """
    if categorical_features is None:
        categorical_features = train.select_dtypes(include=['category', 'object']).columns

    if fillna_value is None:
        fillna_value = train[target_feature].mean()
    
    encoding_maps = {}
    
    for feature in categorical_features:
        mean_encoding = train.groupby(feature)[target_feature].mean()
        encoding_maps[feature] = mean_encoding
        train[feature] = train[feature].map(mean_encoding)
    
    for feature in categorical_features:
        test[feature] = test[feature].map(encoding_maps[feature]).fillna(fillna_value)
    
    return train, test


train, test = target_guided_encoding(train, test, target_feature = TARGET_FEATURE)


from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(train, target_feature=None):

    numeric_features = train.select_dtypes(include=[np.number]).columns
    if target_feature and target_feature in numeric_features:
        numeric_features = numeric_features.drop(target_feature)
    
    vif_data = pd.DataFrame()
    vif_data['Feature'] = numeric_features
    X = train[numeric_features].copy()

    X = X.fillna(0)
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    return vif_data


vif_report = calculate_vif(train, TARGET_FEATURE)
vif_report


useful_features = vif_report[vif_report['VIF'] < 10]['Feature']
train = train[useful_features]


train

