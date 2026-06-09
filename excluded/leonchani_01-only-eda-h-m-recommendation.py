# Table Manipulation, CalculatinTg
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 100) # increase the maximum number of columns

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import missingno as msno
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

# Ignore all warnings
import warnings
warnings.simplefilter("ignore")

# Set seed for reproducibility
np.random.seed(42)


df_articles  = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/articles.csv')
df_customers = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/customers.csv')
df_train     = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/transactions_train.csv')


total_memory_bytes = df_articles.memory_usage(deep=True).sum()
print(f"Total memory usage for the entire DataFrame: {total_memory_bytes} byte")
print(f"Total memory usage for the entire DataFrame: {total_memory_bytes / (1024**2):.2f} MB") # megabyte display
print(f"Total memory usage for the entire DataFrame: {total_memory_bytes / (1024**3):.2f} GB")  # Gigabyte display
display(df_articles.info(memory_usage='deep'))

display(df_articles)


total_memory_bytes = df_customers.memory_usage(deep=True).sum()
print(f"Total memory usage for the entire DataFrame: {total_memory_bytes} byte")
print(f"Total memory usage for the entire DataFrame: {total_memory_bytes / (1024**2):.2f} MB") # megabyte display
print(f"Total memory usage for the entire DataFrame: {total_memory_bytes / (1024**3):.2f} GB")  # Gigabyte display
display(df_customers.info(memory_usage='deep'))

display(df_customers)


total_memory_bytes = df_train.memory_usage(deep=True).sum()
print(f"Total memory usage for the entire DataFrame: {total_memory_bytes} byte")
print(f"Total memory usage for the entire DataFrame: {total_memory_bytes / (1024**2):.2f} MB") # megabyte display
print(f"Total memory usage for the entire DataFrame: {total_memory_bytes / (1024**3):.2f} GB")  # Gigabyte display
display(df_train.info(memory_usage='deep'))

display(df_train)


# Check if each column has a unique value of 0 and 1, and assign 1 or 0
def check_if_binary(column):
    unique_values = column.unique()
    if len(unique_values) == 2 and set(unique_values) == {0, 1}:
        return 1
    else:
        return 0


def check_if_outliers(df):

    outliers_rate_dict = {}

    for column_name in df.columns:
        column = df[column_name]

        # Skip if not a numeric column
        if not pd.api.types.is_numeric_dtype(column):
            print(f"'{column_name}' is skipped because it is not a numeric type.")
            continue

        # Calculate the mean and standard deviation of the data
        mean = column.mean()
        std = column.std()

        # Set outlier threshold
        threshold = 2  # Adjust this value to change the outlier criteria

        # Set conditions for detecting outliers
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std

        # Detect outliers
        outliers = (column < lower_bound) | (column > upper_bound)

        # Calculate the percentage of outliers
        outliers_rate = outliers.sum() / len(column)

        # Save results to dictionary
        outliers_rate_dict[column_name] = outliers_rate

    return outliers_rate_dict


def data_inspection(df):
    """A function that generates data inspection information for a data frame"""

    # process_one:Create a DataFrame with basic statistics
    data_inspection = pd.DataFrame({
        'column_name'         : df.columns,
        'data_type'           : df.dtypes,
        'cnt_rows'            : len(df),
        'cnt_unique_rows'     : df.nunique(),
        'cnt_duplicated_rows' : len(df) - df.nunique(),
        'cnt_non_null_rows'   : df.count().values,
        'cnt_null_rows'       : df.isnull().sum(),
        'rate_null_rows'      : (df.isnull().sum() / len(df)),
        # 'date min': df['t_dat'].min(),
        # 'date max': df['t_dat'].max()
    })

    # process_two:Calculate descriptive statistics, modes, and percentages of modes for numerical data:

    # The below code is no longer needed as 'custom_description' provides the stats
    # description = df.describe(include=np.number).T.reset_index().rename(columns={'index': 'column_name'})
    # median = df.median(numeric_only=True).reset_index().rename(columns={'index': 'column_name', 0: 'median'})

    # Calculate the most frequent value for each column
    mode_values = {}
    for column in df.columns:
        try:
            mode_values[column] = df[column].mode().iloc[0]
        except IndexError:
            mode_values[column] = None  # Handle cases with no mode

    # Calculate the percentage of most frequent values
    mode_rates = {}
    for column in df.columns:
        if column in mode_values and mode_values[column] is not None:
            mode_value = mode_values[column]
            mode_rate = (df[column] == mode_value).sum() / len(df)
            mode_rates[column] = mode_rate
        else:
            mode_rates[column] = None

    # Convert the most common value and its percentage into a data frame
    df_mode = pd.DataFrame(list(mode_values.items()), columns=['column_name', 'mode'])
    df_mode['rate mode'] = df_mode['column_name'].map(mode_rates)

    # Custom statistics calculation without using df.describe for basic stats
    def calculate_custom_stats(series):
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        return pd.Series({
            'count': series.count(),
            'mean': series.mean(),
            'std': series.std(),
            # 'q1-1.5*QTR': q1 - 1.5 * iqr,
            '0%': series.quantile(0),
            '25%': q1,
            '50%': series.median(),
            '75%': q3,
            '100%': series.quantile(1),
            'lower_bound': series.mean() - 2 * series.std(),
            'upper_bound': series.mean() + 2 * series.std()
            # 'q3+1.5*QTR': q3 + 1.5 * iqr
        })

    df_numeric = df.select_dtypes(include=np.number)
    custom_description = df_numeric.apply(calculate_custom_stats).T.reset_index(names='column_name')

    # process_three:Combining DataFrames and adding outlier rates, skewness, and kurtosis:
    data_inspection = pd.merge(data_inspection, custom_description, how='left', on='column_name')
    data_inspection = pd.merge(data_inspection, df_mode, how='left', on='column_name')

    # # The 'description' merge is no longer needed as 'custom_description' provides the stats
    # data_inspection = pd.merge(data_inspection, description, how='left', on='column_name')

    outliers_rate = check_if_outliers(df)
    outliers_rate = pd.DataFrame(list(outliers_rate.items()), columns=['column_name', 'outliers_rate']).reset_index(drop=True)
    data_inspection = pd.merge(data_inspection, outliers_rate, how='left', on='column_name')

    # skewness and kurtosis
    skew = df.select_dtypes(include=np.number).skew().reset_index().rename(columns={'index': 'column_name', 0: 'skewness'})
    kurt = df.select_dtypes(include=np.number).kurt().reset_index().rename(columns={'index': 'column_name', 0: 'kurtosis'})

    data_inspection = pd.merge(data_inspection, skew, how='left', on='column_name')
    data_inspection = pd.merge(data_inspection, kurt, how='left', on='column_name')

    # Calculating correlation coefficients (numeric columns only)
    df_numeric = df.select_dtypes(include=np.number)
    if not df_numeric.empty and df_numeric.shape[1] > 1:
        try:
            correlation_matrix = df_numeric.corr(numeric_only=True)

            # Assuming the last column is the target variable for correlation
            target_column = df_numeric.columns[-1]
            if target_column in correlation_matrix.index:
                target_corr = pd.DataFrame({'column_name': correlation_matrix.index, 'target correlation': correlation_matrix[target_column]})
                data_inspection = pd.merge(data_inspection, target_corr, how='left', on='column_name')
            else:
                data_inspection['target correlation'] = np.nan
        except Exception as e:
            print(f"Error calculating correlation: {e}")
            data_inspection['target correlation'] = np.nan
    else:
        data_inspection['target correlation'] = np.nan

    # process_four:Adding more column information and example data
    # --- MODIFIED PART START ---
    data_examples = {}
    for col in df.columns:
        non_null_values = df[col].dropna()
        if not non_null_values.empty:
            # Randomly select a non-null value
            example_value = non_null_values.sample(n=1).iloc[0]
            data_examples[col] = str(example_value).replace('\n', '<br>')
        else:
            data_examples[col] = None # Or an empty string, depending on preference

    # process_four:Adding more column information and example data
    data_inspection_else = pd.DataFrame({
        'column_name': df.columns,
        'flag_or_not': df.apply(check_if_binary),
        'columns_details': None,
        'remarks': None,
        'trigger': None,
        'dataset_name': None,
        'existence_of_table_definition': None,
        'data_exmaple': pd.Series(data_examples) # Assign the generated examples
    })

    data_inspection = pd.merge(data_inspection, data_inspection_else, how='left', on='column_name')

    # visualization `data_inspection`
    # blue â†’ green â†’ yellow
    styled_columns = data_inspection.select_dtypes(include=np.number).columns
    if not styled_columns.empty:
        data_inspection_styled = data_inspection.style.background_gradient(cmap='viridis', subset=pd.IndexSlice[:, styled_columns])
    else:
        data_inspection_styled = data_inspection

    return data_inspection_styled



data_inspection_styled = data_inspection(df_articles)
display(data_inspection_styled)
# data_inspection.to_csv('data_inspection.csv', index = 'false') # Saving the dataframe if necessary


data_inspection_styled = data_inspection(df_customers)
display(data_inspection_styled)
# data_inspection.to_csv('data_inspection.csv', index = 'false') # Saving the dataframe if necessary


data_inspection_styled = data_inspection(df_train)
display(data_inspection_styled)
# data_inspection.to_csv('data_inspection.csv', index = 'false') # Saving the dataframe if necessary


display(len(df_articles) == len(df_articles.drop_duplicates()))
display(len(df_customers) == len(df_customers.drop_duplicates()))
display(len(df_train) == len(df_train.drop_duplicates()))


df_articles.head()


# List of columns to visualize
columns_to_visualize = [
    'prod_name',
    'product_type_name',
    'product_group_name',
    'department_name',
    'index_name',
    'index_group_name',
    'section_name',
    'garment_group_name',
    'detail_desc'
]

# Loop through each column and create a bar plot
for col_name in columns_to_visualize:
    # Get the top 50 values for the current column
    # value_counts()ã�®å®Ÿè¡Œå‰�ã�«ã‚«ãƒ†ã‚´ãƒªã‚«ãƒ«å�‹ã�«å¤‰æ�›ã�™ã‚‹ã�“ã�¨ã�§ã€�
    # æ¬ æ��å€¤ï¼ˆNaNï¼‰ã�Œã�‚ã‚‹å ´å�ˆã�«ã‚¨ãƒ©ãƒ¼ã�Œç™ºç”Ÿã�™ã‚‹ã�®ã‚’é˜²ã��ã€�
    # æ¬ æ��å€¤ã‚’ã‚«ã‚¦ãƒ³ãƒˆã�‹ã‚‰é™¤å¤–ã�§ã��ã�¾ã�™ã€‚
    top_50_values = df_articles[col_name].astype(str).value_counts().head(50)

    # Create a new figure for each plot
    plt.figure(figsize=(12, max(6, len(top_50_values) * 0.4))) # Adjust height based on number of bars

    # Create the bar plot
    sns.barplot(x=top_50_values.values, y=top_50_values.index, palette='viridis')

    # Set title and labels dynamically
    plt.title(f'Top 50 {col_name.replace("_", " ").title()} by Count', fontsize=16)
    plt.xlabel('Count', fontsize=12)
    plt.ylabel(col_name.replace("_", " ").title(), fontsize=12)

    # Adjust layout
    plt.tight_layout()

    # Display the plot
    plt.show()

print("All plots displayed successfully.")





df_customers.head()


# List of columns to visualize
columns_to_visualize = [
    'FN',
    'Active',
    'club_member_status',
    'fashion_news_frequency',
    'age'
]

# Loop through each column and create a bar plot
for col_name in columns_to_visualize:
    # Get the top 50 values for the current column
    # value_counts()ã�®å®Ÿè¡Œå‰�ã�«ã‚«ãƒ†ã‚´ãƒªã‚«ãƒ«å�‹ã�«å¤‰æ�›ã�™ã‚‹ã�“ã�¨ã�§ã€�
    # æ¬ æ��å€¤ï¼ˆNaNï¼‰ã�Œã�‚ã‚‹å ´å�ˆã�«ã‚¨ãƒ©ãƒ¼ã�Œç™ºç”Ÿã�™ã‚‹ã�®ã‚’é˜²ã��ã€�
    # æ¬ æ��å€¤ã‚’ã‚«ã‚¦ãƒ³ãƒˆã�‹ã‚‰é™¤å¤–ã�§ã��ã�¾ã�™ã€‚
    top_50_values = df_customers[col_name].astype(str).value_counts().head(50)

    # Create a new figure for each plot
    plt.figure(figsize=(12, max(6, len(top_50_values) * 0.4))) # Adjust height based on number of bars

    # Create the bar plot
    sns.barplot(x=top_50_values.values, y=top_50_values.index, palette='viridis')

    # Set title and labels dynamically
    plt.title(f'Top 50 {col_name.replace("_", " ").title()} by Count', fontsize=16)
    plt.xlabel('Count', fontsize=12)
    plt.ylabel(col_name.replace("_", " ").title(), fontsize=12)

    # Adjust layout
    plt.tight_layout()

    # Display the plot
    plt.show()

print("All plots displayed successfully.")





display(len(df_train))
df_train['t_dat'] = pd.to_datetime(df_train['t_dat'], errors='coerce')
df_train.head()


top_50_detail_desc = df_train['t_dat'].value_counts().head(50)

plt.figure(figsize=(12, 10))
sns.barplot(x=top_50_detail_desc.values, y=top_50_detail_desc.index, palette='viridis')

plt.title('Top 50 t_dat by Count', fontsize=16)
plt.xlabel('Count', fontsize=12)
plt.ylabel('t_dat', fontsize=12)
plt.tight_layout()
plt.show()


df_train['t_dat'].max()


# æœ€æ–°ã�®æ—¥ä»˜ã‚’å�–å¾—
max_date = df_train['t_dat'].max()
three_months_ago = max_date - pd.DateOffset(months=3)
df_last_3_months = df_train[df_train['t_dat'] >= three_months_ago]

display(three_months_ago), display(max_date)
df_last_3_months


top_50_detail_desc = df_last_3_months['t_dat'].value_counts().head(50)

plt.figure(figsize=(12, 10))
sns.barplot(x=top_50_detail_desc.values, y=top_50_detail_desc.index, palette='viridis')

plt.title('Top 50 t_dat by Count', fontsize=16)
plt.xlabel('Count', fontsize=12)
plt.ylabel('t_dat', fontsize=12)
plt.tight_layout()
plt.show()


df_merge_trainsaction_customer = pd.merge(df_last_3_months, df_customers, how = 'inner', on  = 'customer_id')
df_merge_trainsaction_customer


# List of columns to visualize
columns_to_visualize = [
    'FN',
    'Active',
    'club_member_status',
    'fashion_news_frequency',
    'age'
]

# Loop through each column and create a bar plot
for col_name in columns_to_visualize:
    # Get the top 50 values for the current column
    # value_counts()ã�®å®Ÿè¡Œå‰�ã�«ã‚«ãƒ†ã‚´ãƒªã‚«ãƒ«å�‹ã�«å¤‰æ�›ã�™ã‚‹ã�“ã�¨ã�§ã€�
    # æ¬ æ��å€¤ï¼ˆNaNï¼‰ã�Œã�‚ã‚‹å ´å�ˆã�«ã‚¨ãƒ©ãƒ¼ã�Œç™ºç”Ÿã�™ã‚‹ã�®ã‚’é˜²ã��ã€�
    # æ¬ æ��å€¤ã‚’ã‚«ã‚¦ãƒ³ãƒˆã�‹ã‚‰é™¤å¤–ã�§ã��ã�¾ã�™ã€‚
    top_50_values = df_merge_trainsaction_customer[col_name].astype(str).value_counts().head(50)

    # Create a new figure for each plot
    plt.figure(figsize=(12, max(6, len(top_50_values) * 0.4))) # Adjust height based on number of bars

    # Create the bar plot
    sns.barplot(x=top_50_values.values, y=top_50_values.index, palette='viridis')

    # Set title and labels dynamically
    plt.title(f'Top 50 {col_name.replace("_", " ").title()} by Count', fontsize=16)
    plt.xlabel('Count', fontsize=12)
    plt.ylabel(col_name.replace("_", " ").title(), fontsize=12)

    # Adjust layout
    plt.tight_layout()

    # Display the plot
    plt.show()

print("All plots displayed successfully.")


df_merge_trainsaction_article = pd.merge(df_last_3_months, df_articles, how = 'inner', on  = 'article_id')
df_merge_trainsaction_article


# List of columns to visualize
columns_to_visualize = [
    'prod_name',
    'product_type_name',
    'product_group_name',
    'department_name',
    'index_name',
    'index_group_name',
    'section_name',
    'garment_group_name',
    'detail_desc'
]

# Loop through each column and create a bar plot
for col_name in columns_to_visualize:
    # Get the top 50 values for the current column
    # value_counts()ã�®å®Ÿè¡Œå‰�ã�«ã‚«ãƒ†ã‚´ãƒªã‚«ãƒ«å�‹ã�«å¤‰æ�›ã�™ã‚‹ã�“ã�¨ã�§ã€�
    # æ¬ æ��å€¤ï¼ˆNaNï¼‰ã�Œã�‚ã‚‹å ´å�ˆã�«ã‚¨ãƒ©ãƒ¼ã�Œç™ºç”Ÿã�™ã‚‹ã�®ã‚’é˜²ã��ã€�
    # æ¬ æ��å€¤ã‚’ã‚«ã‚¦ãƒ³ãƒˆã�‹ã‚‰é™¤å¤–ã�§ã��ã�¾ã�™ã€‚
    top_50_values = df_merge_trainsaction_article[col_name].astype(str).value_counts().head(50)

    # Create a new figure for each plot
    plt.figure(figsize=(12, max(6, len(top_50_values) * 0.4))) # Adjust height based on number of bars

    # Create the bar plot
    sns.barplot(x=top_50_values.values, y=top_50_values.index, palette='viridis')

    # Set title and labels dynamically
    plt.title(f'Top 50 {col_name.replace("_", " ").title()} by Count', fontsize=16)
    plt.xlabel('Count', fontsize=12)
    plt.ylabel(col_name.replace("_", " ").title(), fontsize=12)

    # Adjust layout
    plt.tight_layout()

    # Display the plot
    plt.show()

print("All plots displayed successfully.")




