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


import os  # Operating system interactions

import pandas as pd  # Data manipulation and analysis
import numpy as np  # Numerical operations
import matplotlib.pyplot as plt  # Data visualization
import seaborn as sns  # High-level data visualization based on matplotlib
from scipy import stats

from sklearn.impute import SimpleImputer  # Handling missing values
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder  # Encoding categorical features
from sklearn.compose import ColumnTransformer  # Applying transformers to columns
from sklearn.pipeline import Pipeline  # Assembling steps for cross-validation
from sklearn.model_selection import cross_val_score  # Cross-validation for evaluating scores

pd.set_option('display.max_rows', None)  # Display all rows in pandas DataFrame
pd.set_option('display.max_columns', None)  # Display all columns in the DataFrame

# Ignore all warnings
import warnings
warnings.filterwarnings('ignore')


# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
test4test=test.copy()


print(f"Shape of Train {train.shape}")
print(f"Shape of Test {test.shape}")


def analyze_missing_data(dataframe):
    """
    Analyzes missing data in the provided DataFrame.
    Parameters:
        dataframe (pd.DataFrame): The input DataFrame to analyze.
    Returns:
        pd.DataFrame: A DataFrame containing missing percentages, data types, and null counts.
    """
    # Calculate missing percentages for each column
    missing_percent = (dataframe.isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_percent = missing_percent.apply(lambda x: f"{x:.2f}%")

    # Extract DataFrame info (data types and null counts)
    info_df = pd.DataFrame({
        'DataType': dataframe.dtypes,  # Get the data type of each column
        'Null Count': dataframe.isnull().sum()  # Count the number of null values in each column
    })
    
    # Combine missing percentage and column information into a single DataFrame
    combined_df = pd.concat([missing_percent, info_df], axis=1)
    # Rename the columns of the resulting DataFrame
    combined_df.columns = ['Missing Percent', 'DataType', 'Null Count']

    # Display the total number of rows in the dataset
    print(f'Number of rows: {dataframe.shape[0]}')
    # Return the DataFrame sorted by missing percentage in descending order
    return combined_df.sort_values(by='Missing Percent', ascending=False)


analyze_missing_data(train)


analyze_missing_data(test)


def correlation_analysis(df, method='table', target_column=None, 
                         figsize=(12, 10), cmap='coolwarm', annot=True):
    """
    Perform correlation analysis either as a sorted table or a heatmap.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    method (str): 'table' for sorted correlation table, 'heatmap' for heatmap plot.
    target_column (str): Optional. For 'table' mode, show correlation with a specific column.
    figsize (tuple): Figure size for heatmap.
    cmap (str): Colormap for heatmap.
    annot (bool): Annotate values in heatmap.
    
    Returns:
    pd.DataFrame (if method='table') or displays a heatmap plot (if method='heatmap')
    """
    
    corr = df.corr(numeric_only=True)

    if method == 'table':
        if target_column:
            if target_column not in corr.columns:
                raise ValueError(f"'{target_column}' is not a numeric column in the DataFrame.")
            corr_target = corr[[target_column]].drop(index=target_column)
            corr_target['abs_correlation'] = corr_target[target_column].abs()
            return corr_target.sort_values(by='abs_correlation', ascending=False)

        # Full pairwise correlation table (excluding self and duplicates)
        corr_pairs = corr.unstack().reset_index()
        corr_pairs.columns = ['Feature1', 'Feature2', 'Correlation']
        corr_pairs = corr_pairs[corr_pairs['Feature1'] != corr_pairs['Feature2']]
        corr_pairs['abs_correlation'] = corr_pairs['Correlation'].abs()
        corr_pairs = corr_pairs.drop_duplicates(subset=['abs_correlation'])
        return corr_pairs.sort_values(by='abs_correlation', ascending=False).reset_index(drop=True)

    elif method == 'heatmap':
        plt.figure(figsize=figsize)
        sns.heatmap(corr, annot=annot, fmt=".2f", cmap=cmap, square=True,
                    linewidths=0.5, cbar_kws={"shrink": .8}, annot_kws={"size": 8})
        plt.title("Correlation Heatmap", fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        plt.show()
    else:
        raise ValueError("Invalid method. Use 'table' or 'heatmap'.")



correlation_analysis(train, method='table',)


def plot_numeric_distributions(train, test):
    """
    Fast plotting of numeric distributions - shows all train columns, 
    matching test columns when available.
    """
    # Get numeric columns
    train_num_cols = train.select_dtypes(include=['number']).columns
    
    plt.figure(figsize=(14, len(train_num_cols) * 2))
    
    for i, col in enumerate(train_num_cols, 1):
        # Always plot train data
        plt.subplot(len(train_num_cols), 2, i*2-1)
        sns.histplot(train[col], color='blue', bins=15, kde=False)
        plt.title(f"Train: {col}")
        
        # Only plot test if column exists
        if col in test.columns:
            plt.subplot(len(train_num_cols), 2, i*2)
            sns.histplot(test[col], color='green', bins=15, kde=False)
            plt.title(f"Test: {col}")
        else:
            # Empty subplot if no test data
            plt.subplot(len(train_num_cols), 2, i*2)
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()


plot_numeric_distributions(train, test)


def plot_categorical_pie_charts(train, test):
    """
    Plots pie chart comparisons of categorical variables in train, test, and train_ex datasets.
    """
    obj_cols = train.select_dtypes(include=['object']).columns  # Get categorical columns

    for variable in obj_cols:
        sns.set_style('whitegrid')

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        plt.subplots_adjust(wspace=0.3)

        # Pie Chart for Train
        train[variable].value_counts().plot.pie(ax=axes[0], autopct='%1.1f%%', startangle=90)
        axes[0].set_ylabel('')
        axes[0].set_title(f"Train [{variable}]")

        # Pie Chart for Test
        test[variable].value_counts().plot.pie(ax=axes[1], autopct='%1.1f%%', startangle=90)
        axes[1].set_ylabel('')
        axes[1].set_title(f"Test [{variable}]")

        plt.show()


plot_categorical_pie_charts(train, test)


def plot_missing_data(*datasets, names=None):
    """
    Plots comparative charts (pie charts and bar plots) of missing values 
    for multiple datasets.

    Parameters:
        *datasets: Multiple Pandas DataFrames.
        names (list, optional): List of dataset names for reference.
    
    Displays:
        Pie charts and bar plots for missing values in each dataset.
    """
    if names is None:
        names = [f"Dataset {i+1}" for i in range(len(datasets))]

    missing_values = {
        name: df.isnull().sum()[df.isnull().sum() > 0]  # Only columns with missing values
        for df, name in zip(datasets, names)
    }

    # Filter datasets that have missing values
    missing_values = {name: values for name, values in missing_values.items() if not values.empty}

    if not missing_values:
        print("No missing values in the provided datasets.")
        return

    fig, axes = plt.subplots(len(missing_values), 2, figsize=(12, len(missing_values) * 5))

    if len(missing_values) == 1:  # Ensure axes is iterable when only one dataset
        axes = [axes]

    for ax, (name, missing_data) in zip(axes, missing_values.items()):
        # Pie chart
        ax[0].pie(missing_data, labels=missing_data.index, autopct='%1.1f%%', startangle=90)
        ax[0].set_title(f'Missing Values in {name} Dataset')

        # Bar plot
        ax[1].barh(missing_data.index, missing_data.values, color='skyblue')
        ax[1].set_title(f'Missing Values in {name} Dataset')
        ax[1].set_xlabel('Count')
        ax[1].invert_yaxis()

    plt.tight_layout()
    plt.show()

# Example usage:
# plot_missing_data(train, test, train_ex, names=["Train", "Test", "Train_ex"]


plot_missing_data(train, test)


import missingno as msno


def plot_missing_data_matrix(*datasets, names=None):
    """
    Plots missing data locations using the missingno matrix for multiple datasets.

    Parameters:
        *datasets: Multiple Pandas DataFrames.
        names (list, optional): List of dataset names for reference.
    
    Displays:
        Missing data matrices for each dataset.
    """
    if names is None:
        names = [f"Dataset {i+1}" for i in range(len(datasets))]

    colors = [(0.0, 0.2, 0.4), (0.0, 0.4, 0.2), (0.6, 0.2, 0.0), (0.4, 0.0, 0.6), (0.2, 0.6, 0.2)]  # Different color options

    for i, (df, name) in enumerate(zip(datasets, names)):
        plt.figure(figsize=(12, 6))
        msno.matrix(df, color=colors[i % len(colors)])  # Cycle through colors
        plt.title(f"Missing Data Locations in {name} Dataset", fontsize=24)
        plt.xlabel("Columns", fontsize=20)
        plt.show()

# Example usage:
# plot_missing_data_matrix(train, test, train_ex, names=["Train", "Test", "Train_ex"]


plot_missing_data_matrix(train, test)


def scatter_plot(a, b, train_df, sample_size=None, show_table=False, table_size=5):
    """
    Optimized scatter plot for large DataFrames with optional display of extreme points.
    
    Args:
        a, b (str): Columns to plot (x, y)
        train_df (pd.DataFrame): Full DataFrame
        sample_size (int): If None, uses full data (slower). If int, samples data.
        show_table (bool): If True, prints extreme points information in the console.
        table_size (int): Number of extreme points to display.
    """
    # Sample data if specified (for very large datasets)
    plot_data = train_df if sample_size is None else train_df.sample(min(sample_size, len(train_df)))
    
    # Create scatter plot using seaborn
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=plot_data[a], 
        y=plot_data[b], 
        hue=plot_data.index,  # Color by index (or another numeric column)
        palette='viridis',  # Use a color palette
        alpha=0.6,  # Transparency
        edgecolor=None,  # No edges
        s=50  # Size of the scatter points
    )
    
    plt.title(f"Scatter Plot of {a} vs {b}", fontsize=15)
    plt.xlabel(a, fontsize=12)
    plt.ylabel(b, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Show extreme points in the console (if enabled)
    if show_table:
        # Find points with highest/lowest x and y values
        extremes = pd.concat([ 
            train_df.nlargest(table_size, a),
            train_df.nsmallest(table_size, a),
            train_df.nlargest(table_size, b),
            train_df.nsmallest(table_size, b)
        ]).drop_duplicates()
        
        print(f"\nExtreme Points (Top/Bottom {table_size} in {a} and {b}):")
        print(extremes[[a, b]])  # Print extreme points without table formatting
    
    plt.show()



scatter_plot('Genre', 'Listening_Time_minutes', train)


import matplotlib.pyplot as plt

def scatter_plot1(a, b, train_df, sample_size=None):
    """
    Simple scatter plot for large DataFrames.
    
    Args:
        a, b (str): Columns to plot (x, y)
        train_df (pd.DataFrame): Full DataFrame
        sample_size (int): If None, uses full data. If int, samples data.
    """
    # Sample data if specified (for very large datasets)
    plot_data = train_df if sample_size is None else train_df.sample(min(sample_size, len(train_df)))

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(
        plot_data[a], 
        plot_data[b], 
        alpha=0.6,  # Transparency
        edgecolor=None,  # No edges
        s=10  # Size of the scatter points
    )
    
    plt.title(f"Scatter Plot of {a} vs {b}", fontsize=15)
    plt.xlabel(a, fontsize=12)
    plt.ylabel(b, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



scatter_plot1('Genre', 'Listening_Time_minutes', train)


from tabulate import tabulate
def find_outliers(df, column, method='frequency', threshold=0.01, show_table=True, top_n=10):
    """
    Detect and display outliers in a DataFrame column (works for both numeric and categorical columns).
    
    For numeric columns:
    - 'zscore': Uses Z-score (default threshold=3)
    - 'iqr': Uses IQR (default threshold=1.5)
    
    For categorical columns:
    - 'frequency': Finds rare categories (default threshold=1% frequency)
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column to analyze
        method (str): 'zscore', 'iqr', or 'frequency'
        threshold (float): Cutoff for outliers
        show_table (bool): Print a formatted table of outliers
        top_n (int): Number of top outliers to display
    
    Returns:
        pd.DataFrame: Outliers with original index, value, and outlier score (frequency for categorical)
    """
    # Check if column exists
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found")
    
    # Handle numeric columns
    if np.issubdtype(df[column].dtype, np.number):
        if method == 'zscore':
            z_scores = (df[column] - df[column].mean()) / df[column].std()
            outliers = df[abs(z_scores) > threshold].copy()
            outliers['Outlier_Score'] = z_scores[outliers.index]
            
        elif method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)].copy()
            outliers['Outlier_Score'] = np.where(
                df[column] < lower_bound,
                (df[column] - Q1) / IQR,
                (df[column] - Q3) / IQR
            )[outliers.index]
            
        else:
            raise ValueError("For numeric columns, method must be 'zscore' or 'iqr'")
    
    # Handle categorical columns
    else:
        if method != 'frequency':
            print("âš ï¸� For categorical columns, only 'frequency' method is supported. Using 'frequency'.")
        
        # Calculate frequency of each category
        freq = df[column].value_counts(normalize=True)
        rare_categories = freq[freq < threshold].index
        
        # Get all rows with rare categories
        outliers = df[df[column].isin(rare_categories)].copy()
        outliers['Outlier_Score'] = outliers[column].map(freq)
    
    # Print table if requested
    if show_table:
        if outliers.empty:
            print(f"No outliers detected in '{column}'")
        else:
            print(f"ğŸ”� Outliers in '{column}' (Method: {method}, Threshold: {threshold})")
            
            if np.issubdtype(df[column].dtype, np.number):
                if method == 'zscore':
                    sorted_outliers = outliers.sort_values('Outlier_Score', ascending=False)
                    print(f"\nğŸ”¥ Top {top_n} Positive Outliers:")
                    print(tabulate(
                        sorted_outliers.head(top_n)[[column, 'Outlier_Score']],
                        headers=['Value', 'Z-Score'],
                        showindex=True,
                        tablefmt='grid',
                        floatfmt=".2f"
                    ))
                    
                    print(f"\nâ�„ï¸� Top {top_n} Negative Outliers:")
                    print(tabulate(
                        sorted_outliers.tail(top_n)[::-1][[column, 'Outlier_Score']],
                        headers=['Value', 'Z-Score'],
                        showindex=True,
                        tablefmt='grid',
                        floatfmt=".2f"
                    ))
                else:
                    print(tabulate(
                        outliers[[column, 'Outlier_Score']].sort_values('Outlier_Score', key=abs, ascending=False),
                        headers=['Value', 'Outlier Score'],
                        showindex=True,
                        tablefmt='grid',
                        floatfmt=".2f"
                    ))
            else:
                # For categorical outliers, show frequency
                print(f"\nğŸ“Š Rare Categories (Frequency < {threshold:.1%}):")
                print(tabulate(
                    outliers[[column, 'Outlier_Score']].sort_values('Outlier_Score'),
                    headers=['Category', 'Frequency'],
                    showindex=True,
                    tablefmt='grid',
                    floatfmt=".2%"
                ))
    
    return outliers


find_outliers(train, 'Guest_Popularity_percentage', method='zscore', threshold=2)


find_outliers(train, 'Genre', method='frequency', threshold=0.01, show_table=True, top_n=10)


def line_plot(df, y_column, x_column=None, sample_size=None, title=None):
    """
    Creates a seaborn line plot for a specified column in a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        y_column (str): The column name for the y-axis.
        x_column (str or None): The column name for the x-axis. Defaults to index if None.
        sample_size (int or None): Optional number of rows to sample from df.
        title (str or None): Optional plot title.
    """
    data = df if sample_size is None else df.sample(n=min(sample_size, len(df))).sort_index()

    x_data = data[x_column] if x_column else data.index

    plt.figure(figsize=(12, 6))
    sns.lineplot(
        x=x_data,
        y=data[y_column],
        data=data,
        color='crimson',
        marker='o',
        markersize=5
    )

    plt.title(title if title else f'{y_column} Over Time', fontsize=14)
    plt.xlabel(x_column if x_column else 'Index')
    plt.ylabel(y_column)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



line_plot(train, y_column='Guest_Popularity_percentage', x_column='Listening_Time_minutes',sample_size=100)


sns.histplot(data=train, x="Episode_Length_minutes", kde=True, bins=30)


sns.histplot(
    data=train[train['Episode_Length_minutes'] < 122],
    x="Episode_Length_minutes",
    kde=True,
    bins=30
)


sns.histplot(
    data=train,
    x="Guest_Popularity_percentage",
    kde=True,
    bins=30
)


train.query('Episode_Length_minutes > 122')


train.query('Guest_Popularity_percentage> 105').shape[0]


# Delete rows where Episode_Length_minutes > 122
train = train[(train['Episode_Length_minutes'] <= 122) | (train['Episode_Length_minutes'].isna())]


# Delete rows where Guest_Popularity_percentage > 105
train = train[(train['Guest_Popularity_percentage'] <= 105) | (train['Guest_Popularity_percentage'].isna())]


analyze_missing_data(train)


sns.histplot(
    data=train,
    x="Guest_Popularity_percentage",
    kde=True,
    bins=30
)


train_df=train.copy()
test_df=test.copy()


def fast_preserving_impute(df, target_col):
    """Faster version that still preserves distribution"""
    df = df.copy()
    
    # 1. Group by similar podcasts
    df['Genre_group'] = df['Genre'] + df['Episode_Sentiment']
    
    # 2. Calculate distribution parameters per group
    group_stats = df.groupby('Genre_group')[target_col].agg(['mean', 'std', 'count'])
    
    # 3. Impute missing values from group distribution
    for group in group_stats.index:
        mask = (df['Genre_group'] == group) & (df[target_col].isnull())
        n_missing = mask.sum()
        
        if n_missing > 0:
            mean = group_stats.loc[group, 'mean']
            std = group_stats.loc[group, 'std']
            
            # Sample from normal distribution with group parameters
            imputed_values = np.random.normal(mean, std, n_missing)
            imputed_values = np.clip(imputed_values, 0, 100)
            df.loc[mask, target_col] = imputed_values
    
    # Fallback to global distribution if any remain
    if df[target_col].isnull().any():
        global_mean = df[target_col].mean()
        global_std = df[target_col].std()
        n_missing = df[target_col].isnull().sum()
        df.loc[df[target_col].isnull(), target_col] = np.clip(
            np.random.normal(global_mean, global_std, n_missing),
            0, 100
        )
    
    return df.drop('Genre_group', axis=1)



# Apply faster version
train_df = fast_preserving_impute(train_df, 'Episode_Length_minutes')
test_df = fast_preserving_impute(test_df, 'Episode_Length_minutes')

train_df = fast_preserving_impute(train_df, 'Guest_Popularity_percentage')
test_df = fast_preserving_impute(test_df, 'Guest_Popularity_percentage')


analyze_missing_data(train_df)


sns.histplot(data=train_df, x="Episode_Length_minutes", kde=True, bins=30)


sns.histplot(data=train_df, x="Guest_Popularity_percentage", kde=True, bins=30)


train_df['Number_of_Ads'].fillna(train_df['Number_of_Ads'].median(), inplace=True)


analyze_missing_data(test_df)


train_df['Episode_Number'] = train_df['Episode_Title'].str.extract('(\d+)').astype(float)
test_df['Episode_Number'] = test_df['Episode_Title'].str.extract('(\d+)').astype(float)


column=train_df.columns.to_list()
for i in column:
    print(f"{i}: {train_df[i].nunique()} : {train_df[i].dtype}")


train_df = train_df.drop(['id','Episode_Title'], axis=1)
test_df = test_df.drop(['id','Episode_Title'], axis=1)


# Separate features and target for training data
X_train = train_df.drop('Listening_Time_minutes', axis=1)
y_train = train_df['Listening_Time_minutes']

# Test data (no target column)
X_test = test_df.copy()  # Or test if it's already loaded without target


# Define numerical and categorical features (same as before)
numerical_features = ['Episode_Length_minutes', 'Host_Popularity_percentage', 
                     'Guest_Popularity_percentage', 'Number_of_Ads', 'Episode_Number']

categorical_features = ['Podcast_Name', 'Genre', 'Publication_Day', 
                       'Publication_Time', 'Episode_Sentiment']


# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)
    ])

# Fit on training data only
preprocessor.fit(X_train)

# Transform both datasets
X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)


X_test_processed.shape


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
import time

## 1. Fast Preprocessing (assuming X_train_processed exists)
# If you need to recreate preprocessing:
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_train_processed = scaler.fit_transform(X_train)
# X_test_processed = scaler.transform(X_test)

## 2. Model Selection - Focus on 3 Most Efficient Algorithms
models = {
    'LightGBM': LGBMRegressor(
        random_state=42,
        verbose=-1,
        n_jobs=1,  # Use 1 core to prevent overheating
        n_estimators=150,
        learning_rate=0.1,
        max_depth=5
    ),
    'XGBoost': XGBRegressor(
        random_state=42,
        n_jobs=1,
        n_estimators=150,
        learning_rate=0.1,
        max_depth=5,
        tree_method='hist'  # Faster training
    ),
    'HistGradientBoosting': HistGradientBoostingRegressor(
        random_state=42,
        max_iter=150,
        learning_rate=0.1,
        max_depth=5
    )
}

## 3. Quick Evaluation (Single Train-Test Split)
X_train_fast, X_val, y_train_fast, y_val = train_test_split(
    X_train_processed, y_train, 
    test_size=0.2, 
    random_state=42
)

results = {}
for name, model in models.items():
    start_time = time.time()
    
    print(f"\nTraining {name}...")
    model.fit(X_train_fast, y_train_fast)
    
    train_time = time.time() - start_time
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    results[name] = {
        'RMSE': rmse,
        'Training Time (s)': train_time
    }
    
    print(f"{name} | RMSE: {rmse:.4f} | Time: {train_time:.1f}s")

## 4. Select and Train Best Model
best_model_name = min(results, key=lambda x: results[x]['RMSE'])
print(f"\nBest model: {best_model_name}")

final_model = models[best_model_name]
print("Training final model on full data...")
final_model.fit(X_train_processed, y_train)

## 5. Generate Predictions
test_pred = final_model.predict(X_test_processed)


submission = pd.DataFrame({
    'id': test4test['id'],
    'Listening_Time_minutes': test_pred
})
submission.to_csv('submission6.csv', index=False)
print("Submission file created!")

