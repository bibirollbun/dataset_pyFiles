!python -m pip install -qq --no-index --find-links=/kaggle/input/library-for-cibmtr \
lifelines


import math
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from lifelines import KaplanMeierFitter


import warnings
warnings.filterwarnings('ignore')


data_dict = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/data_dictionary.csv')
ss = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv')
train_df = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv', index_col='ID')
test_df = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv', index_col='ID')


ss


train_cols = train_df.columns
test_cols = test_df.columns

train_only = [col for col in train_cols if col not in test_cols]
train_only


train_df.describe()


def transform_survival_probability(df, time_col='efs_time', event_col='efs'):
    """
    Transform using survival probability estimates
    """
    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col], df[event_col])
    
    # Get survival probabilities at each time point
    y = kmf.survival_function_at_times(df[time_col]).values
    
    # Adjust for censoring
    # censored_mask = df[event_col] == 0
    #y[censored_mask] = y[censored_mask] * 1.2  # Increase survival prob for censored
    
    return y

train_df["y"] = transform_survival_probability(train_df, time_col='efs_time', event_col='efs')


data_dict['type'].unique()


def categorize_columns(df, columns_to_exclude = None):
    """
    Separate columns into categories.

    Params:      
        df - Input DataFrame to categorize columns.

    Returns:
        A dictionary with keys:
            - 'discrete': Numerical columns with int data types and low cardinality.
            - 'continuous': Numerical columns with float data types or high cardinality integers.
            - 'categorical': Object/string columns with low cardinality.
            - 'non_categorical': Object/string columns with high cardinality.
            - 'other': Columns that don't fit into the above categories.
    """
    discrete = []
    continuous = []
    categorical = []
    non_categorical = []
    other = []

    columns = list(df.columns)
    if columns_to_exclude:
        for col in columns_to_exclude:
            columns.remove(col)
            
    for col in columns:
        col_type = df[col].dtype

        if col_type in ['int64', 'int32', 'int', 'float32', 'float64', 'float']:  # Integer columns
            if df[col].nunique() < 20:  # Low cardinality
                discrete.append(col)
            else:  # High cardinality
                continuous.append(col)

        elif col_type in ['object', 'str']:  # Categorical columns
            if df[col].nunique() < 20:  # Low cardinality
                categorical.append(col)
            else:  # High cardinality
                non_categorical.append(col)

        else:  # Columns with any other type
            other.append(col)

    # Create the resulting categories dictionary
    categories = {
        'discrete': discrete,
        'continuous': continuous,
        'categorical': categorical,
        'non_categorical': non_categorical,
        'other': other
    }

    return categories


categorized_columns = categorize_columns(train_df, columns_to_exclude=['efs', 'efs_time'])

for key in categorized_columns:
    print('Column Type: ', key)
    print('Columns: ', categorized_columns[key])
    print('\n\n')


# For low cardinality features
def plot_pie_charts(df, columns):
    """
    Plots pie charts for the given categorical or discrete columns.

    Args:
        df (DataFrame): The dataset containing the columns.
        columns (list): List of categorical or discrete columns.
    """
    nrows = len(columns)
    nrows, ncols = math.ceil(nrows / 3), 3  # Ensure at least 1 row

    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))

    # Flatten axs for easier iteration
    axs = axs.flat  

    for i, col in enumerate(columns):  
        if i >= len(axs):  # Avoid exceeding available subplots
            break  

        value_counts = df[col].value_counts()
        keys = value_counts.keys()
        values = value_counts.values

        axs[i].pie(values, labels=keys, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
        axs[i].set_title(f'Distribution of {col}')  

    # Hide unused subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()


# For continuous features
def violin_plots(df, columns):
    """
        Ignores NaN values
    """
    nrows = len(columns)
    nrows, ncols = math.ceil(nrows/2), 2
    
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*4, nrows*3.5))
    axs = axs.flat
    
    for i, col in enumerate(columns):
        if i >= len(axs):
            break

        data = df[col].dropna()

        if data.empty:
            axs[i].set_tile(f'{col} (No data)')
            axs[i].axis('off')
            continue
        
        axs[i].violinplot(data, showmeans=False, showmedians=True)
        axs[i].set_title(f'Distribution of {col}')

    # Hide unused subplots
    for j in range(i+1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.show()


categorical = categorized_columns['categorical']
plot_pie_charts(train_df, categorical)


discrete = categorized_columns['discrete']
plot_pie_charts(train_df, discrete)


violin_plots(train_df, categorized_columns['continuous'])


def missing_data_summary(df):
    """Summarizes missing data in a structured and readable format."""
    n_samples = len(df)
    missing_data = {
        '0%': [], '10%': [], '40%': [], '80%': [], '100%': []
    }
    
    for col in df.columns:
        missing_percentage = df[col].isna().sum() / n_samples * 100
        if missing_percentage == 0:
            missing_data['0%'].append(col)
        elif missing_percentage < 10:
            missing_data['10%'].append(col)
        elif missing_percentage < 40:
            missing_data['40%'].append(col)
        elif missing_percentage < 80:
            missing_data['80%'].append(col)
        else:
            missing_data['100%'].append(col)

    print("\nğŸ”� **Missing Data Summary (upto and excluding)**\n")
    
    for category, columns in missing_data.items():
        print(f"ğŸ“Œ {category} missing: ({len(columns)} columns)")
        if columns:
            formatted_columns = ", ".join(columns)
            print(f"   {formatted_columns}\n\n")
        else:
            print("   None\n")


missing_data_summary(train_df)


plt.scatter(train_df['efs_time'], train_df['y'], marker='o', linestyle='-')

plt.xlabel('EFS Time (seconds)')
plt.ylabel('y - Survival Score')
plt.title('Survival Score over time')

plt.grid(True)
plt.show()


plt.scatter(train_df['efs_time'], train_df['efs'], marker='o', linestyle='-')

plt.xlabel('EFS_Time (seconds)')
plt.ylabel('EFS')
plt.title('EFS over time')

plt.grid(True)
plt.show()

