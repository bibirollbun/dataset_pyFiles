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


df_train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv') # importing 'train' data
df_test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')   # importing 'test' data


total_memory_bytes = df_train.memory_usage(deep=True).sum()
print(f"Total memory usage for the entire DataFrame: {total_memory_bytes} byte")
print(f"Total memory usage for the entire DataFrame: {total_memory_bytes / (1024**2):.2f} MB") # megabyte display
print(f"Total memory usage for the entire DataFrame: {total_memory_bytes / (1024**3):.2f} GB")  # Gigabyte display
display(df_train.info(memory_usage='deep'))

# 'train' data
display(df_train)


total_memory_bytes = df_test.memory_usage(deep=True).sum()
print(f"Total memory usage for the entire DataFrame: {total_memory_bytes} byte")
print(f"Total memory usage for the entire DataFrame: {total_memory_bytes / (1024**2):.2f} MB") # megabyte display
print(f"Total memory usage for the entire DataFrame: {total_memory_bytes / (1024**3):.2f} GB")  # Gigabyte display
display(df_test.info(memory_usage='deep'))

# 'test' data
display(df_test)


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



df_train_tmp = df_train.drop('id', axis=1)
data_inspection_styled = data_inspection(df_train_tmp)
display(data_inspection_styled)
# data_inspection.to_csv('data_inspection.csv', index = 'false') # Saving the dataframe if necessary


df_test_tmp = df_test.drop('id', axis=1)
data_inspection_styled = data_inspection(df_test_tmp)
display(data_inspection_styled)
# data_inspection.to_csv('data_inspection.csv', index = 'false')  # Saving the dataframe if necessary


def create_missing_value_plots(df: pd.DataFrame):
    """
    It generates four types of plots (Matrix, Bar, Heatmap, Dendrogram) to visualize missing values â€‹â€‹in a data frame.

    Args:
        df (pd.DataFrame): The DataFrame to perform missing value analysis on.
    """
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))

    # Missing Value Matrix
    msno.matrix(df, sparkline=True, ax=axes[0, 0], color=(0.2, 0.2, 0.2))
    axes[0, 0].set_title("Missing Value Matrix")

    # Missing Value Bar Plot
    msno.bar(df, ax=axes[0, 1], color='skyblue')
    axes[0, 1].set_title("Missing Value Bar Plot")

    # Missing Value Correlation Heatmap
    msno.heatmap(df, ax=axes[1, 0], cmap='viridis')
    axes[1, 0].set_title("Missing Value Correlation Heatmap")

    # Missing Value Dendrogram
    msno.dendrogram(df, ax=axes[1, 1])
    axes[1, 1].set_title("Missing Value Dendrogram")

    # Adjusting the layout
    plt.tight_layout()
    plt.show()


# df_train
create_missing_value_plots(df_train)


# df_test
create_missing_value_plots(df_test)


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
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # pie chart for train data
        plt.subplot(1, 2, 1)
        train_counts = df_train[variable].value_counts(normalize=True)
        train_counts.plot(kind='pie', autopct='%1.1f%%', colors=[custom_palette[0]] * len(train_counts), startangle=90, ax=plt.gca())
        plt.title(f"Pie chart for {variable} [train]")
        plt.tight_layout()

        # pie chart for test data
        plt.subplot(1, 2, 2)
        if variable in df_test.columns:
            test_counts = df_test[variable].value_counts(normalize=True)
            test_counts.plot(kind='pie', autopct='%1.1f%%', colors=[custom_palette[1]] * len(test_counts), startangle=90, ax=plt.gca())
            plt.title(f"Pie chart for {variable} [test]")
        else:
            plt.gca().axis('off') # Empty plot if no columns in test data

        # countplot
        fig_countplot, ax_countplot = plt.subplots(figsize=(15, 6))
        sns.countplot(data=df_train, x=variable, color=custom_palette[0], alpha=0.7, label="train", ax=ax_countplot)
        if variable in df_test.columns:
            sns.countplot(data=df_test, x=variable, color=custom_palette[1], alpha=0.7, label="test", ax=ax_countplot)
        ax_countplot.set_title(f"Count plot for {variable} [train & test]")
        ax_countplot.legend()
        plt.tight_layout()
        plt.show()

    plt.tight_layout()
    plt.show()


# custom palette of colors
custom_palette = ['#3498db', '#e74c3c','#2ecc71']

# Add 'Dataset' column to distinguish between train and test data
df_train['dataset'] = 'train'
df_test['dataset'] = 'test'

# Create a list of variables (both numerical and categorical data)
numerical_variables = df_train.select_dtypes(include=['number']).columns
categorical_variables = df_train.select_dtypes(include=['object']).columns

# Create plots for numerical data
for variable in numerical_variables:
    create_variable_plots(variable, data_type='numerical')

# Create plots for categorical data
for variable in categorical_variables:
    create_variable_plots(variable, data_type='categorical')

# remove unnecessary columns
del df_train['dataset']
del df_test['dataset']


target_variable = 'Fertilizer Name'

# Automatically distinguish between numerical and categorical variables
numerical_variables = df_train.select_dtypes(include=np.number).columns.tolist()
all_categorical_variables = df_train.select_dtypes(include='object').columns.tolist()

# Exclude the target variable from the categorical variables list
categorical_variables_filtered = [col for col in all_categorical_variables if col != target_variable]

# Combine columns to be compared
compare_cols = numerical_variables + categorical_variables_filtered

# Add and merge train/test tags
train_tag = df_train.assign(dataset="train")[compare_cols + ["dataset"]]
test_tag = df_test.assign(dataset="test")[compare_cols + ["dataset"]]
combo = pd.concat([train_tag, test_tag], axis=0)

# Visualize the distribution for each column
for col in compare_cols:
    if col in categorical_variables_filtered:
        # For categorical variables
        ct = pd.crosstab(combo[col], combo["dataset"], normalize="columns") * 100
        ct.plot.barh(figsize=(6, 4), stacked=False, title=f"{col} â€“ train vs test %")
        plt.tight_layout()
        plt.show()
    else:
        # For numeric variables
        sns.kdeplot(data=combo, x=col, hue="dataset", fill=True, common_norm=False, alpha=0.4)
        plt.title(f"{col} â€“ train vs test distribution")
        plt.tight_layout()
        plt.show()


numerical_variables = df_train.select_dtypes(include=['number']).columns
numerical_variables_tmp = df_test.select_dtypes(include=['number']).columns

# Create a subplot (1 column, 2 rows)
fig, axes = plt.subplots(2, 1, figsize=(10, 12))

# Correlation matrix of df_train
# (if necessary, extract only highly correlated variables.exï¼šmask=(corr < 0.8)))
sns.heatmap(df_train[numerical_variables].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=axes[0])
axes[0].set_title('Train Data Feature Correlation')

# # Correlation matrix of df_test
# (if necessary, extract only highly correlated variables.exï¼šmask=(corr < 0.8)))
sns.heatmap(df_test[numerical_variables_tmp].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=axes[1])
axes[1].set_title('Test Data Feature Correlation')

# Layout adjustment
plt.tight_layout()
plt.show()



# Making Pairplot
print('Pairplot of Train Data')
grid_train = sns.pairplot(df_train[numerical_variables], corner = True, diag_kind = "kde")
grid_train.fig.set_size_inches(12, 8)
plt.show()

print('------------------------------------------------------------------------------------------------------')

print('Pairplot of Test Data')
grid_test = sns.pairplot(df_test[numerical_variables_tmp], corner = True, diag_kind = "kde")
grid_test.fig.set_size_inches(12, 8)
plt.show()


# target_col = df_train.columns[-1]  #  object variable

# # Get numeric and categorical columns
# numeric_cols = df_train.select_dtypes(include=['number']).columns.tolist()
# categorical_cols = df_train.select_dtypes(include=['object', 'category']).columns.tolist()

# # Exclude the response variable from the numeric column
# if target_col in numeric_cols:
#     numeric_cols.remove(target_col)

# # Visualization settings
# num_numeric = len(numeric_cols)
# num_categorical = len(categorical_cols)


# # Visualization of Numeric Variables 
# if num_numeric > 0:
#     fig_numeric, axes_numeric = plt.subplots(num_numeric, 2, figsize=(12, 4 * num_numeric))
#     if num_numeric == 1:
#         axes_numeric = axes_numeric.reshape(1, -1) # Treat even a single array as a 2D array
#     for i, col in enumerate(numeric_cols):
        
#         # Histograms and kernel density estimation
#         sns.histplot(df_train[col], stat='count', kde=True, ax=axes_numeric[i, 0], label='train')
#         # sns.histplot(df_train[col], stat='density', kde=True, ax=axes_numeric[i, 0], label='train')
#         axes_numeric[i, 0].set_title(f'Distribution of {col} (train)')
#         axes_numeric[i, 0].set_xlabel(col)
#         axes_numeric[i, 0].set_ylabel('Count')
#         # axes_numeric[i, 0].set_ylabel('Density')
#         axes_numeric[i, 0].legend()

#         # Line graph of univariate and dependent variable
#         # df_grouped_numeric = df_train.groupby(col)[target_col].mean().sort_index()
#         df_grouped_numeric = df_train.groupby(col)[target_col].count().sort_index()
#         axes_numeric[i, 1].plot(df_grouped_numeric.index, df_grouped_numeric.values, marker='o', linestyle='-', label='train')
#         # axes_numeric[i, 1].set_title(f'Mean {target_col} per {col} (train)')
#         axes_numeric[i, 1].set_title(f'Count {target_col} per {col} (train)')
#         axes_numeric[i, 1].set_xlabel(col)
#         # axes_numeric[i, 1].set_ylabel(f'Mean {target_col}')
#         axes_numeric[i, 1].set_ylabel(f'Count {target_col}')
#         axes_numeric[i, 1].grid(True)
#         axes_numeric[i, 1].legend()
#     plt.tight_layout()
#     plt.show()

# # Visualization of categorical variables
# if num_categorical > 0:
#     fig_categorical, axes_categorical = plt.subplots(num_categorical, 2, figsize=(12, 4 * num_categorical))
#     if num_categorical == 1:
#         axes_categorical = axes_categorical.reshape(1, -1) # Treat even a single array as a 2D array
#     for i, col in enumerate(categorical_cols):
        
#         # Frequency of categories
#         sns.countplot(data=df_train, x=col, ax=axes_categorical[i, 0], order=df_train[col].value_counts().index)
#         axes_categorical[i, 0].set_title(f'Distribution of {col} (train)')
#         axes_categorical[i, 0].set_xlabel(col)
#         axes_categorical[i, 0].set_ylabel('Count')
#         axes_categorical[i, 0].tick_params(axis='x', rotation=45)
#         axes_categorical[i, 0].tick_params(axis='x', labelsize='small')

#         # Relationship between categories and objective variables (Line Plot)
#         df_grouped_categorical = df_train.groupby(col)[target_col].mean().sort_index() # sort_index() ã‚’è¿½åŠ 
#         sns.lineplot(x=df_grouped_categorical.index, y=df_grouped_categorical.values, marker='o', linestyle='-', ax=axes_categorical[i, 1])
#         axes_categorical[i, 1].set_title(f'Mean {target_col} per {col} (train)')
#         axes_categorical[i, 1].set_xlabel(col)
#         axes_categorical[i, 1].set_ylabel(f'Mean {target_col}')
#         axes_categorical[i, 1].grid(True)
#         axes_categorical[i, 1].tick_params(axis='x', rotation=45)
#         axes_categorical[i, 1].tick_params(axis='x', labelsize='small')

#     plt.tight_layout()
#     plt.show()


def calculate_vif(df: pd.DataFrame, numerical_cols: list):
    """
    Calculates and displays the VIF (Variance Inflation Factor) for a specified numeric column in a data frame.

    Args:
        df (pd.DataFrame): The dataframe over which to compute the VIF.
        numerical_cols (list): A list of the names of the numeric columns you want to calculate the VIF for.
                               Do not include the target variable in this list.
    Returns:
        pd.DataFrame: A data frame containing each feature and its VIF value.
    """
    # Create a data frame with only selected numeric columns
    X = df[numerical_cols]
    X_vif = add_constant(X)

    # Initialize a data frame to store the VIFs.
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_vif.columns
    vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]

    # Display the VIF excluding the constant term (const)
    vif_data = vif_data[vif_data['Feature'] != 'const'].sort_values("VIF", ascending=False)

    return vif_data


# df_train for VIF
df_train_vif = calculate_vif(df_train, numerical_variables_tmp)
df_train_vif = df_train_vif.rename(columns={'VIF': 'VIF_train'})

# df_test for VIF
df_test_vif = calculate_vif(df_test, numerical_variables_tmp)
df_test_vif = df_test_vif.rename(columns={'VIF': 'VIF_test'})

# Merge df_train_vif and df_test_vif
df_vif = pd.merge(df_train_vif, df_test_vif, how = 'inner', on = 'Feature').style.background_gradient(cmap="viridis")
df_vif


# # value for mapping 
# mapping = {'male': 1, 'female': 0}

# # df_train_tmp
# df_train_tmp = df_train.drop(['id'], axis=1)
# df_train_tmp['Sex'] = df_train_tmp['Sex'].replace(mapping)

# selecting numeric columns
numerical_variables_for_pca = df_train_tmp.select_dtypes(include=['number']).columns
numerical_variables_for_pca = pd.Index(numerical_variables_for_pca[:-1], dtype='object').tolist()


# Sampling
n_samples_total = df_train_tmp.shape[0] # Total number of data
sample_percentage = 0.05                # Setting the sampling rate
n_samples_for_analysis = int(n_samples_total * sample_percentage)

# Randomly select n_samples_for_analysis rows from df_train_tmp
df_train_tmp = df_train_tmp.sample(n=n_samples_for_analysis, random_state=42).copy()

print(f"Number of original data: {n_samples_total} records")
print(f"Number of data points after sampling: {df_train_tmp.shape[0]} records")


df_train_pca_target = df_train_tmp[numerical_variables_for_pca + [df_train_tmp.columns[-1]]].dropna()

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_train_pca_target[numerical_variables_for_pca])

# Running PCA
pca = PCA(n_components=5, random_state=42)
pca_result = pca.fit_transform(scaled_data)

# Calculating quantiles of the response variable
# target_quintiles = pd.qcut(df_train_pca_target[df_train.columns[-1]], 5, labels=False, duplicates='drop')


# Scree plot
# The proportion of variance each principal component explains
explained_variance_ratio = pca.explained_variance_ratio_
display(f"The proportion of variance each principal component explains: {explained_variance_ratio}")
display(f"Cumulative explained variance ratio: {explained_variance_ratio.sum():.2f}")

plt.figure(figsize=(8, 5))
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.7, align='center', label='Individual explained variance')
plt.step(range(1, len(explained_variance_ratio) + 1), np.cumsum(explained_variance_ratio), where='mid', label='Cumulative explained variance', color='red')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal Component Index')
plt.title('Scree Plot')
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# Principal Component Loadings
loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(pca.n_components)], index=numerical_variables_for_pca)

plt.figure(figsize=(10, len(numerical_variables_for_pca) * 0.5))
sns.heatmap(loadings, cmap='vlag', annot=True, fmt=".2f", linewidths=.5, center=0)
plt.title('Principal Component Loadings')
plt.xlabel('Principal Component')
plt.ylabel('Original Feature')
plt.tight_layout()
plt.show()


# # Drawing a scatter plot
# plt.figure(figsize=(10, 5))
# sns.scatterplot(x=pca_result[:, 0],
#                 y=pca_result[:, 1],
#                 hue=target_quintiles,
#                 palette="viridis",
#                 alpha=0.3,
#                 s=10)

# # Add explanatory variance contributions to axis labels
# plt.xlabel(f'Principal Component 1 ({explained_variance_ratio[0]*100:.1f}% explained)')
# plt.ylabel(f'Principal Component 2 ({explained_variance_ratio[1]*100:.1f}% explained)')
# plt.title(f"PCA â€“ coloured by {df_train.columns[-1]} quintile")

# # Adjust the legend position
# plt.legend(title="Quintile", bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.tight_layout()
# plt.show()


# Determining the optimal number of clusters K (elbow method and silhouette score)
# Specify the range of clusters to try
k_range = range(2, 6)
inertia = []           # SSB (Sum of Squared Distances to Centroids) for each K
silhouette_scores = [] # # Silhouette score for each K

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    # kmeans.fit(pca_result)
    kmeans.fit(pca_result[:,0:3])
    inertia.append(kmeans.inertia_) # WCSS (Within-Cluster Sum of Squares)

    # Since silhouette scores cannot be calculated when K=1, they are calculated only when K > 1.
    if k > 1:
        silhouette_scores.append(silhouette_score(pca_result, kmeans.labels_))

# Drawing Elbow Plots
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia (Within-cluster sum of squares)')
plt.grid(True, linestyle='--', alpha=0.6)

# Drawing silhouette score plots
plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, marker='o')
plt.title('Silhouette Score for Optimal K')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Silhouette Score')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

print("\nK-Means: Elbow Method (Inertia):", inertia)
print("K-Means: silhouette score:", silhouette_scores)


# Silhouette diagram
# Define the range of K you want to evaluate
# Since the silhouette score is not defined for one cluster, it is common to start with K at 2.
range_n_clusters = range(2, 6)

silhouette_scores = {} # A dictionary to store the silhouette scores for each K

for n_clusters in range_n_clusters:
    # Rerun K-Means with the optimal K (change n_clusters to the loop variable)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    # cluster_labels = kmeans.fit_predict(pca_result)
    cluster_labels = kmeans.fit_predict(pca_result[:,0:3])

    # Calculation of silhouette score and silhouette value for each sample
    silhouette_avg = silhouette_score(pca_result, cluster_labels)
    sample_silhouette_values = silhouette_samples(pca_result, cluster_labels)

    silhouette_scores[n_clusters] = silhouette_avg # save score

    print(f"\nFor n_clusters = {n_clusters}")
    print(f"The average silhouette_score is : {silhouette_avg}")

    # Silhouette figure drawing
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))

    ax1.set_xlim([-0.1, 1])
    # Adjust Y-axis range: depending on the number of data points and clusters
    ax1.set_ylim([0, len(pca_result) + (n_clusters + 1) * 10])

    y_lower = 10
    for i in range(n_clusters):
        # Sort the silhouette values of each cluster
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Cluster number is displayed in the center of the silhouette diagram.
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Calculate the new y_lower for the next plot
        y_lower = y_upper + 10 # 10 for the 0 samples

    ax1.set_title(f"Silhouette plot for n_clusters = {n_clusters}")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The average silhouette score is shown as a vertical line.
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear Y-Axis Labels and Ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.show()

# Show silhouette scores for all K
print("\n--- Summary of Silhouette Scores ---")
for k, score in silhouette_scores.items():
    print(f"n_clusters = {k}: Silhouette Score = {score:.4f}")

# Find the K with the highest silhouette score
best_k = max(silhouette_scores, key=silhouette_scores.get)
print(f"\nOptimal K based on highest silhouette score: {best_k} (Score: {silhouette_scores[best_k]:.4f})")


# Run K-Means clustering (choose K)
# Look at the elbow plot and silhouette score to choose the optimal K.
optimal_k = best_k
kmeans_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans_model.fit_predict(pca_result[:,0:3])
# clusters = kmeans_model.fit_predict(pca_result)

# Add the clustering results to a DataFrame (for later visualization)
pca_df_for_plotting = pd.DataFrame(data=pca_result, columns=['Principal Component 1', 'Principal Component 2', 'Principal Component 3', 'Principal Component 4', 'Principal Component 5'])
pca_df_for_plotting['KMeans_Cluster'] = clusters
# pca_df_for_plotting['Target_Quintile'] = target_quintiles.reset_index(drop=True)

# Visualization of clustering results
plt.figure(figsize=(18, 7))

# Plot colored by K-Means cluster
plt.subplot(1, 2, 1)
sns.scatterplot(x='Principal Component 1',
                y='Principal Component 2',
                hue='KMeans_Cluster',
                palette='viridis',
                alpha=0.7,
                s=20,
                data=pca_df_for_plotting)
plt.title(f'PCA (PC1 vs PC2) - Coloured by K-Means Cluster (K={optimal_k})')
plt.xlabel(f'Principal Component 1')
plt.ylabel(f'Principal Component 2')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')

# # Plot colored by target quantile (for comparison)
# plt.subplot(1, 2, 2)
# sns.scatterplot(x='Principal Component 1',
#                 y='Principal Component 2',
#                 hue='Target_Quintile',
#                 palette='viridis',
#                 alpha=0.7,
#                 s=20,
#                 data=pca_df_for_plotting)
# plt.title(f'PCA (PC1 vs PC2) - Coloured by {df_train.columns[-1]} Quintile (for comparison)')
# plt.xlabel(f'Principal Component 1')
# plt.ylabel(f'Principal Component 2')
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.legend(title="Quintile", bbox_to_anchor=(1.05, 1), loc='upper left')

# plt.tight_layout()
# plt.show()

# Check the average value of the target variable for each cluster
print("Distribution of the target variable ('Calories') in each K-Means cluster:")
display(df_train_pca_target.groupby(clusters)[df_train.columns[-1]].describe())


!pip install watermark


%load_ext watermark
%watermark -n -u -v -iv -w -p pytensor,aeppl,xarray

