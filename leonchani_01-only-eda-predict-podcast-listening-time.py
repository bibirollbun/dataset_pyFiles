# table manipulation, calculating
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 100) # increase the maximum number of columns

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Ignore all warnings
import warnings
warnings.simplefilter("ignore")


df_train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


df_train


df_test





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
    data_inspection_else = pd.DataFrame({
        'column_name': df.columns,
        'flag_or_not': df.apply(check_if_binary),
        'columns_details': None,
        'remarks': None,
        'trigger': None,
        'dataset_name': None,
        'existence_of_table_definition': None,
        'data_exmaple': df.head(1).T.iloc[:, 0].astype(str).replace('\n', '<br>')
    })

    data_inspection = pd.merge(data_inspection, data_inspection_else, how='left', on='column_name')

    # visualization `data_inspection`
    # blue → green → yellow
    styled_columns = data_inspection.select_dtypes(include=np.number).columns
    if not styled_columns.empty:
        data_inspection_styled = data_inspection.style.background_gradient(cmap='viridis', subset=pd.IndexSlice[:, styled_columns])
    else:
        data_inspection_styled = data_inspection

    return data_inspection_styled


def clip_upper_to_quantile_keep_null(series, quantile=0.95):
    """
    Replaces values above the specified percentile with that percentile value and leaves null values alone.

    Args:
        series (pd.Series): The Series to process.
        quantile (float): The percentile to use as upper bound (range 0 to 1). Default is 0.95.

    Returns:
        pd.Series: A Series in which values outside the upper bound are replaced by the specified percentile value, leaving null values as is.
    """
    null_mask = series.isnull()        # Preserve the index of null values
    not_null_series = series.dropna()  # Series excluding null values

    if not not_null_series.empty:
        upper_bound = not_null_series.quantile(quantile)
        clipped_not_null_series = not_null_series.where(not_null_series <= upper_bound, upper_bound)
    else:
        clipped_not_null_series = pd.Series()  # If all original Series are null

    # Return null values to their original positions
    result_series = pd.Series(index=series.index)
    result_series[~null_mask] = clipped_not_null_series.reindex(series.index[~null_mask])
    result_series[null_mask] = np.nan

    return result_series


# preprocessing outlier
df_train['Episode_Length_minutes'] = clip_upper_to_quantile_keep_null(df_train['Episode_Length_minutes'], quantile=0.99)
df_train['Number_of_Ads'] = clip_upper_to_quantile_keep_null(df_train['Number_of_Ads'], quantile=0.99)


df_train_tmp = df_train.drop('id', axis=1)
data_inspection_styled = data_inspection(df_train_tmp)
display(data_inspection_styled)
# data_inspection.to_csv('data_inspection.csv', index = 'false')





df_test['Episode_Length_minutes'] = clip_upper_to_quantile_keep_null(df_test['Episode_Length_minutes'], quantile=0.99)
df_test['Number_of_Ads'] = clip_upper_to_quantile_keep_null(df_test['Number_of_Ads'], quantile=0.99)


df_test_tmp = df_test.drop('id', axis=1)
data_inspection_styled = data_inspection(df_test_tmp)
display(data_inspection_styled)
# data_inspection.to_csv('data_inspection.csv', index = 'false')





# # EDA for both of numeric and category
# # custom palette of colors
# custom_palette = ['#3498db', '#e74c3c','#2ecc71']

# # Add 'Dataset' column to distinguish between train and test data
# df_train['dataset'] = 'train'
# df_test['dataset'] = 'test'

# # Create a list of variables (both numerical and categorical data)
# numerical_variables = df_train.select_dtypes(include=['number']).columns
# categorical_variables = df_train.select_dtypes(include=['object']).columns

# # A function to create plots for each variable
# def create_variable_plots(variable, data_type='numerical'):

#     sns.set_style('whitegrid')

#     # For numeric data
#     if data_type == 'numerical':
#         fig, axes = plt.subplots(1, 2, figsize=(15, 4))

#         # Box plot
#         plt.subplot(1, 2, 1)
#         sns.boxplot(data=pd.concat([df_train, df_test]), x=variable, y="dataset", palette=custom_palette)
#         plt.xlabel(variable)
#         plt.title(f"Box plot for {variable}")

#         # histgram
#         plt.subplot(1, 2, 2)
#         sns.histplot(data=df_train, x=variable, color=custom_palette[0], kde=True, bins=30, label="train")
#         if variable in df_test.columns:
#             sns.histplot(data=df_test, x=variable, color=custom_palette[1], kde=True, bins=30, label="test")
        
#         plt.xlabel(variable)
#         plt.ylabel("Frequency")
#         plt.title(f"Histogram for {variable} [train & test]" if variable in df_test.columns else f"Histogram for {variable} [train]")
#         plt.legend()

#     # For categorical data
#     elif data_type == 'categorical':
#         fig, axes = plt.subplots(1, 2, figsize=(15, 4))

#         # pie chart
#         plt.subplot(1, 2, 1)
#         catogory_counts = pd.concat([df_train[variable], df_test[variable]]).value_counts(normalize=True)
#         catogory_counts.plot(kind='pie', autopct='%1.1f%%', colors=custom_palette, startangle=90, ax=plt.gca())
#         plt.title(f"Pie chart for {variable}")

#         # countplat
#         plt.subplot(1, 2, 2)
#         sns.countplot(data=pd.concat([df_train, df_test]), x=variable, hue="dataset", palette=custom_palette)
#         plt.title(f"Count plot for {variable}")

#     plt.tight_layout()
#     plt.show()

# # Create plots for numerical data
# for variable in numerical_variables:
#     create_variable_plots(variable, data_type='numerical')

# # Create plots for categorical data
# for variable in categorical_variables:
#     create_variable_plots(variable, data_type='categorical')

# # remove unnecessary columns
# del df_train['dataset']
# del df_test['dataset']


# custom palette of colors
custom_palette = ['#3498db', '#e74c3c','#2ecc71']

# Add 'Dataset' column to distinguish between train and test data
df_train['dataset'] = 'train'
df_test['dataset'] = 'test'

# Create a list of variables (both numerical and categorical data)
numerical_variables = df_train.select_dtypes(include=['number']).columns
categorical_variables = df_train.select_dtypes(include=['object']).columns

# A function to create plots for each variable
def create_variable_plots(variable, data_type='numerical'):

    sns.set_style('whitegrid')

    # For numeric data
    if data_type == 'numerical':
        fig, axes = plt.subplots(1, 2, figsize=(15, 4))

        # Box plot
        plt.subplot(1, 2, 1)
        sns.boxplot(data=pd.concat([df_train, df_test]), x=variable, y="dataset", palette=custom_palette)
        plt.xlabel(variable)
        plt.title(f"Box plot for {variable}")

        # histgram
        plt.subplot(1, 2, 2)
        sns.histplot(data=df_train, x=variable, color=custom_palette[0], kde=True, bins=30, label="train")
        if variable in df_test.columns:
            sns.histplot(data=df_test, x=variable, color=custom_palette[1], kde=True, bins=30, label="test")

        plt.xlabel(variable)
        plt.ylabel("Frequency")
        plt.title(f"Histogram for {variable} [train & test]" if variable in df_test.columns else f"Histogram for {variable} [train]")
        plt.legend()

    # For categorical data
    elif data_type == 'categorical':
        fig, axes = plt.subplots(1, 2, figsize=(15, 6)) # figsizeを少し大きくしました

        # pie chart for train data
        plt.subplot(1, 2, 1)
        train_counts = df_train[variable].value_counts(normalize=True)
        train_counts.plot(kind='pie', autopct='%1.1f%%', colors=[custom_palette[0]] * len(train_counts), startangle=90, ax=plt.gca())
        plt.title(f"Pie chart for {variable} [train]")

        # pie chart for test data
        plt.subplot(1, 2, 2)
        if variable in df_test.columns:
            test_counts = df_test[variable].value_counts(normalize=True)
            test_counts.plot(kind='pie', autopct='%1.1f%%', colors=[custom_palette[1]] * len(test_counts), startangle=90, ax=plt.gca())
            plt.title(f"Pie chart for {variable} [test]")
        else:
            plt.gca().axis('off') # Empty plot if no columns in test data

        # countplot
        fig_countplot, ax_countplot = plt.subplots(figsize=(10, 5))
        sns.countplot(data=df_train, x=variable, color=custom_palette[0], alpha=0.7, label="train", ax=ax_countplot)
        if variable in df_test.columns:
            sns.countplot(data=df_test, x=variable, color=custom_palette[1], alpha=0.7, label="test", ax=ax_countplot)
        ax_countplot.set_title(f"Count plot for {variable} [train & test]")
        ax_countplot.legend()
        plt.tight_layout()
        plt.show()

    plt.tight_layout()
    plt.show()

# Create plots for numerical data
for variable in numerical_variables:
    create_variable_plots(variable, data_type='numerical')

# Create plots for categorical data
for variable in categorical_variables:
    create_variable_plots(variable, data_type='categorical')

# # remove unnecessary columns
del df_train['dataset']
del df_test['dataset']


# Create a subplot (1 column, 2 rows)
fig, axes = plt.subplots(2, 1, figsize=(10, 12))

numerical_variables_tmp = ['Episode_Length_minutes', 'Host_Popularity_percentage','Guest_Popularity_percentage', 'Number_of_Ads']

# Correlation matrix of df_train
# (if necessary, extract only highly correlated variables.ex：mask=(corr < 0.8)))
sns.heatmap(df_train[numerical_variables].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=axes[0])
axes[0].set_title('Train Data Feature Correlation')

# # Correlation matrix of df_test
# (if necessary, extract only highly correlated variables.ex：mask=(corr < 0.8)))
sns.heatmap(df_test[numerical_variables_tmp].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=axes[1])
axes[1].set_title('Test Data Feature Correlation')

# Layout adjustment
plt.tight_layout()
plt.show()


# making pairplot
print('Pairplot of Train Data')
grid_train = sns.pairplot(df_train[numerical_variables])
grid_train.fig.set_size_inches(12, 8)
plt.show()

print('------------------------------------------------------------------------------------------------------')

print('Pairplot of Test Data')
grid_test = sns.pairplot(df_test[numerical_variables_tmp])
grid_test.fig.set_size_inches(12, 8)
plt.show()


RMV = df_train.select_dtypes(include=['number']).columns.tolist()
FEATURES = [c for c in list( df_train.columns ) if not c in RMV]
print(f"We have {len(FEATURES)} basic features:")
print( FEATURES )


for feature in FEATURES:
    plt.figure(figsize=(14, 6))
    avg_listening = df_train.groupby(feature)['Listening_Time_minutes'].mean().sort_values()
    plt.plot(avg_listening.index, avg_listening.values, marker='o', linestyle='-', color='skyblue')
    plt.xlabel(feature)
    plt.ylabel('Average Listening Time (minutes)')
    plt.title(f'Average Listening Time per {feature} ')
    plt.xticks(rotation=90, ha='center')
    plt.grid(True)
    plt.tight_layout()
    plt.show()





def extract_episode_number(df, episode_title_col='Episode_Title',
                           output_episode_number_col='Episode_Number',
                           temp_episode_number_str_col='Episode_Number_str'):
  """
  A function that extracts the episode number from the episode title in the DataFrame, converts it to a numeric type, 
  adds it to a new column, and deletes the temporary string column.

  Args:
    df (pd.DataFrame): The DataFrame to process.
    Episode_Title_col (str): Episode title column name (default: 'Episode_Title').
    output_episode_number_col (str): Output episode number column name (default: 'Episode_Number').
    temp_episode_number_str_col (str): Temporary string episode number column name (default: 'Episode_Number_str').
  """
  # Split the episode title on the space and extract the second element
  df[temp_episode_number_str_col] = df[episode_title_col].str.split(' ').str[1]

  # Convert extracted string to number (with error handling)
  df[output_episode_number_col] = pd.to_numeric(df[temp_episode_number_str_col], errors='coerce')

  # Delete the temporary string column
  del df[temp_episode_number_str_col]


extract_episode_number(df_train)
extract_episode_number(df_test)


def display_crosstab_with_row_percentages(df, target_col, comparision_col):
    """
    Creates and displays a cross-tabulation table and its row percentages
    with background gradient styling.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_col (str): The column to use as the index (rows) of the cross-tabulation.
        comparision_col (str): The column to use for the values (columns) of the cross-tabulation.
    """
    # Create a cross-tabulation table
    cross_table = pd.crosstab(df[target_col], df[comparision_col])

    # Apply gradient background color to the counts
    styled_columns = cross_table.columns
    styled_cross_table = cross_table.style.background_gradient(cmap='viridis', subset=pd.IndexSlice[:, styled_columns])
    print(f"Cross-tabulation of {target_col} vs {comparision_col}:")
    display(styled_cross_table)

    # Calculate the percentage across the row
    row_totals = cross_table.sum(axis=1)
    row_percentages = cross_table.div(row_totals, axis=0)

    # Percentage crosstabs with styles applied
    styled_row_percentages = row_percentages.style.background_gradient(cmap='viridis')
    print(f"\nRow Percentages of {target_col} vs {comparision_col}:")
    display(styled_row_percentages)


def display_combined_crosstab_with_row_percentages(df, target_col, comparision_col):
    """
    Creates and displays a cross-tabulation table combined with its row percentages
    with background gradient styling.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_col (str): The column to use as the index (rows) of the cross-tabulation.
        comparision_col (str): The column to use for the values (columns) of the cross-tabulation.
    """
    # Create a cross-tabulation table
    cross_table = pd.crosstab(df[target_col], df[comparision_col], dropna=False)

    # Calculate the percentage across the row
    row_totals = cross_table.sum(axis=1)
    row_percentages = cross_table.div(row_totals, axis=0)

    # Create percentage column names
    percentage_cols = [f'{col} (%)' for col in row_percentages.columns]
    row_percentages.columns = percentage_cols

    # Combine the count and percentage tables
    combined_table = pd.concat([cross_table, row_percentages], axis=1)

    # Apply gradient background color to the combined table
    styled_combined_table = combined_table.style.background_gradient(cmap='viridis')
    print(f"Combined Cross-tabulation of {target_col} vs {comparision_col} (Counts and Row Percentages):")
    display(styled_combined_table)


 display_crosstab_with_row_percentages(df_train, 'Podcast_Name', 'Episode_Number')


display_combined_crosstab_with_row_percentages(df_train, 'Podcast_Name', 'Genre')


display_combined_crosstab_with_row_percentages(df_train, 'Podcast_Name', 'Episode_Sentiment')


display_combined_crosstab_with_row_percentages(df_train, 'Genre', 'Publication_Day')


display_combined_crosstab_with_row_percentages(df_train, 'Genre', 'Publication_Time')


display_combined_crosstab_with_row_percentages(df_train, 'Genre', 'Episode_Sentiment')


display_combined_crosstab_with_row_percentages(df_train, 'Publication_Time', 'Episode_Sentiment')


plt.figure(figsize=(20, 6))
plt.scatter(df_train['Episode_Length_minutes'], df_train['Listening_Time_minutes'])
plt.xlabel('Episode_Length_minutes')
plt.ylabel('Listening_Time_minutes')
plt.title('pairplot')
plt.grid(True)
plt.show()





# Create a 1-by-2 subplot
fig, axes = plt.subplots(1, 2, figsize=(15, 10))

# Missing value heatmap of training data
sns.heatmap(df_train.isnull(), cbar=False, cmap='viridis', ax=axes[0])
axes[0].set_title('Missing Data in Train')

# Missing value heatmap for test data
sns.heatmap(df_test.isnull(), cbar=False, cmap='viridis', ax=axes[1])
axes[1].set_title('Missing Data in Test')

# Adjust layout
plt.tight_layout()
plt.show()


# def analyze_episode_data_flexible_columns(df, episode_num_col='Episode_Number',
#                                           podcast_name_col='Podcast_Name',
#                                           episode_sentiment_col='Episode_Sentiment',
#                                           episode_length_col='Episode_Length_minutes'):
#   """
#   A function that returns the number of unique combinations related to Episodes in a DataFrame, groups by a specified column,
#   and aggregates and displays the column for a specified length.

#   Args:
#     df (pd.DataFrame): The DataFrame to analyze.
#     episode_num_col (str): Episode number column name (default: 'Episode_Number').
#     podcast_name_col (str): Podcast name column name (default: 'Podcast_Name').
#     episode_sentiment_col (str): Episode sentiment column name (default: 'Episode_Sentiment').
#     episode_length_col (str): Episode length column name (default: 'Episode_Length_minutes').
#   """

#   # Calculate and display the number of unique combinations
#   unique_combinations = (df[episode_num_col].nunique() * df[podcast_name_col].nunique() * df[episode_sentiment_col].nunique())
#   print(f"unique ({episode_num_col}, {podcast_name_col}, {episode_sentiment_col}) combination: {unique_combinations}")

#   # Group by a specified column and display a count of columns of a specified length.
#   tmp_count = df.groupby([podcast_name_col, episode_num_col, episode_sentiment_col],as_index=False, dropna=False)[episode_length_col].count()
#   tmp_count = tmp_count.rename(columns={episode_length_col: f'{episode_length_col}_count'})
#   print(f"\n cnt:{episode_length_col}")
#   display(tmp_count.sort_values(f'{episode_length_col}_count', ascending=True))

#   # Group by a specified column and calculate and display the average of the column for a specified length.
#   tmp_mean = df.groupby([podcast_name_col, episode_num_col, episode_sentiment_col],as_index=False, dropna=False)[episode_length_col].mean()
#   tmp_mean = tmp_mean.rename(columns={episode_length_col: f'{episode_length_col}_mean'}) # Change the column name of the average result
#   print(f"\n avg:{episode_length_col} ")
#   display(tmp_mean)


def analyze_episode_data_flexible_columns(df, episode_num_col='Episode_Number',
                                          podcast_name_col='Podcast_Name',
                                          episode_length_col='Episode_Length_minutes'):
  """
  A function that returns the number of unique combinations related to Episodes in a DataFrame, groups by a specified column,
  and aggregates and displays the column for a specified length.

  Args:
    df (pd.DataFrame): The DataFrame to analyze.
    episode_num_col (str): Episode number column name (default: 'Episode_Number').
    podcast_name_col (str): Podcast name column name (default: 'Podcast_Name').
    episode_length_col (str): Episode length column name (default: 'Episode_Length_minutes').
  """

  # Calculate and display the number of unique combinations
  unique_combinations = (df[episode_num_col].nunique() * df[podcast_name_col].nunique())
  print(f"unique ({episode_num_col}, {podcast_name_col}) combination: {unique_combinations}")

  # Group by a specified column and display a count of columns of a specified length.
  tmp_count = df.groupby([podcast_name_col, episode_num_col],as_index=False, dropna=False)[episode_length_col].count()
  tmp_count = tmp_count.rename(columns={episode_length_col: f'{episode_length_col}_count'})
  print(f"\n cnt:{episode_length_col}")
  display(tmp_count.sort_values(f'{episode_length_col}_count', ascending=True))

  # Group by a specified column and calculate and display the average of the column for a specified length.
  tmp_mean = df.groupby([podcast_name_col, episode_num_col],as_index=False, dropna=False)[episode_length_col].mean()
  tmp_mean = tmp_mean.rename(columns={episode_length_col: f'{episode_length_col}_mean'}) # Change the column name of the average result
  print(f"\n avg:{episode_length_col} ")
  display(tmp_mean)


analyze_episode_data_flexible_columns(df_train,
                                        episode_num_col='Episode_Number',
                                        podcast_name_col='Podcast_Name',
                                        episode_length_col='Episode_Length_minutes')


analyze_episode_data_flexible_columns(df_train,
                                        episode_num_col='Episode_Number',
                                        podcast_name_col='Podcast_Name',
                                        episode_length_col='Guest_Popularity_percentage')


analyze_episode_data_flexible_columns(df_test,
                                        episode_num_col='Episode_Number',
                                        podcast_name_col='Podcast_Name',
                                        episode_length_col='Episode_Length_minutes')


analyze_episode_data_flexible_columns(df_test,
                                        episode_num_col='Episode_Number',
                                        podcast_name_col='Podcast_Name',
                                        episode_length_col='Guest_Popularity_percentage')


# Display missing rows
df_train_nan_index = df_train[df_train.isnull().any(axis=1)].index # Save the index of the row that contains the missing value
df_train_nan_rows = df_train.loc[df_train_nan_index] # Recall the row later using the index
display(df_train_nan_index)
display(df_train_nan_rows)


# Display missing rows
df_test_nan_index = df_test[df_test.isnull().any(axis=1)].index # Save the index of the row that contains the missing value
df_test_nan_rows = df_train.loc[df_test_nan_index] # Recall the row later using the index
display(df_test_nan_index)
display(df_test_nan_rows)


df_train['minutes'] = df_train['Episode_Length_minutes'] - df_train['Listening_Time_minutes']
tmp = pd.DataFrame(round(df_train['minutes'], 0)).sort_values(by = 'minutes')
tmp[tmp['minutes'] < 0].value_counts()




