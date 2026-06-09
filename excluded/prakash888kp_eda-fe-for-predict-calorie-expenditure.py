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








import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
warnings.filterwarnings('ignore', category=FutureWarning) 


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


train.head()


train.isnull().sum()


def load_data(train_path, test_path):
    """Loads the training and testing data from CSV files."""
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        print(f"Training data loaded successfully: {train_df.shape}")
        print(f"Testing data loaded successfully: {test_df.shape}")
        return train_df, test_df
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None, None


try:
    train_path = '/kaggle/input/playground-series-s5e5/train.csv'
    test_path = '/kaggle/input/playground-series-s5e5/test.csv'
    train_df, test_df = load_data(train_path, test_path)
except FileNotFoundError:
     print("Kaggle paths not found. Please provide the correct paths to train.csv and test.csv")
     train_df, test_df = None, None


def explore_data(df, df_name="DataFrame"):
    """Performs initial exploration of a DataFrame."""
    if df is None:
        print(f"{df_name} not loaded. Skipping exploration.")
        return

    print(f"\n--- Exploring {df_name} ---")
    print("\nFirst 5 rows:")
    display(df.head()) # Use display for better rendering in notebooks

    print("\nDataFrame Info:")
    df.info()

    print("\nMissing Values:")
    missing_values = df.isnull().sum()
    if missing_values.sum() == 0:
        print("No missing values found.")
    else:
        print(missing_values[missing_values > 0])

    print("\nSummary Statistics (Numerical Columns):")
    # Use display for better rendering in notebooks
    display(df.describe(include=np.number))

    print("\nSummary Statistics (Object Columns):")
    # Use display for better rendering in notebooks
    display(df.describe(include='object'))


if train_df is not None:
    explore_data(train_df, "Training Data")


def encode_sex(df):
    """Encodes the 'Sex' column to numeric and converts to category."""
    if 'Sex' not in df.columns:
        print("'Sex' column not found.")
        return df

    print("\nEncoding 'Sex' column...")
    df['Sex_numeric'] = df['Sex'].map({'male': 1, 'female': 0})
    # Convert original 'Sex' and new numeric to category type for efficiency
    df['Sex'] = df['Sex'].astype('category')
    df['Sex_numeric'] = df['Sex_numeric'].astype('category') # Or int if preferred for modeling later
    print("Unique values in 'Sex':", df['Sex'].unique())
    print("Unique values in 'Sex_numeric':", df['Sex_numeric'].unique())
    return df


if train_df is not None:
    train_df = encode_sex(train_df.copy()) # Use copy to avoid SettingWithCopyWarning
    print("\nTraining Data info after encoding 'Sex':")
    train_df.info()


def engineer_features(df):
    """Creates new features: BMI, Age_Group, Intensity."""
    if not all(col in df.columns for col in ['Weight', 'Height', 'Age', 'Heart_Rate']):
        print("Required columns for feature engineering not found.")
        return df

    print("\nEngineering new features...")
    # Calculate BMI
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    print("BMI calculated.")

    # Create Age Groups
    bins = [19, 29, 39, 49, 59, 69, 79]
    labels = ['20-29', '30-39', '40-49', '50-59', '60-69', '70-79']
    df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True)
    df['Age_Group'] = df['Age_Group'].astype('category')
    print("Age groups created.")

    # Create Workout Intensity Levels
    df['Intensity'] = pd.cut(df['Heart_Rate'], bins=[0, 90, 110, 200], labels=['Low', 'Moderate', 'High'])
    df['Intensity'] = df['Intensity'].astype('category')
    print("Workout intensity levels created.")

    print("\nDataFrame with new features (first 5 rows):")
    display(df[['BMI', 'Age_Group', 'Intensity']].head())
    return df


if train_df is not None:
    train_df = engineer_features(train_df.copy())


def plot_correlation(df_numeric, title="Correlation Matrix"):
    """Plots the correlation matrix for numerical features."""
    if df_numeric is None or df_numeric.empty:
        print("No numerical data to plot correlation for.")
        return

    plt.figure(figsize=(12, 8))
    corr_matrix = df_numeric.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(title)
    plt.show()


if train_df is not None:
    numeric_cols_for_corr = train_df.select_dtypes(include=np.number).columns.tolist()
    # Remove ID and potentially Sex_numeric if it's category; keep BMI
    cols_to_exclude = ['id']
    if 'Sex_numeric' in numeric_cols_for_corr and train_df['Sex_numeric'].dtype.name == 'category':
         cols_to_exclude.append('Sex_numeric')

    numeric_cols_for_corr = [col for col in numeric_cols_for_corr if col not in cols_to_exclude]

    plot_correlation(train_df[numeric_cols_for_corr], title="Correlation Matrix (Original Features + BMI)")



def plot_distributions(df, columns, title_suffix=""):
    """Plots histograms and KDE for specified numerical columns."""
    if df is None or df.empty:
        print("DataFrame is empty. Cannot plot distributions.")
        return
    num_cols = len(columns)
    if num_cols == 0:
        print("No numerical columns specified for distribution plots.")
        return

    # Calculate grid size (e.g., 3 plots per row)
    n_rows = (num_cols + 2) // 3
    n_cols = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = axes.flatten() # Flatten to easily iterate

    print(f"\nPlotting Distributions{title_suffix}...")
    for i, col in enumerate(columns):
        if pd.api.types.is_numeric_dtype(df[col]):
            sns.histplot(df[col], kde=True, bins=30, ax=axes[i], color='skyblue')
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
        else:
            print(f"Skipping non-numerical column for distribution plot: {col}")
            axes[i].set_visible(False) # Hide axis if not numeric

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"Distribution of Numerical Features{title_suffix}", y=1.02, fontsize=16)
    plt.tight_layout()
    plt.show()

# Select relevant numeric columns for plotting
if train_df is not None:
    numeric_cols_for_plot = train_df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols_for_plot = [col for col in numeric_cols_for_plot if col != 'id'] # Exclude ID
    plot_distributions(train_df, numeric_cols_for_plot)


def plot_boxplots(df, columns, title_suffix=""):
    """Plots boxplots for specified numerical columns."""
    if df is None or df.empty:
        print("DataFrame is empty. Cannot plot boxplots.")
        return
    num_cols = len(columns)
    if num_cols == 0:
        print("No numerical columns specified for boxplots.")
        return

    n_rows = (num_cols + 2) // 3
    n_cols = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = axes.flatten() # Flatten

    print(f"\nPlotting Boxplots{title_suffix}...")
    for i, col in enumerate(columns):
         if pd.api.types.is_numeric_dtype(df[col]):
            sns.boxplot(x=df[col], ax=axes[i], color='lightblue')
            axes[i].set_title(f'Boxplot of {col}')
            axes[i].set_xlabel(col)
         else:
            print(f"Skipping non-numerical column for boxplot: {col}")
            axes[i].set_visible(False)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"Box Plots of Numerical Features{title_suffix}", y=1.02, fontsize=16)
    plt.tight_layout()
    plt.show()

if train_df is not None:
    plot_boxplots(train_df, numeric_cols_for_plot, title_suffix=" (Before Outlier Removal)")



def handle_outliers_iqr(df, columns):
    """Removes outliers from specified columns using the IQR method."""
    if df is None or df.empty:
        print("DataFrame is empty. Cannot handle outliers.")
        return df
    if not columns:
        print("No columns specified for outlier handling.")
        return df

    df_clean = df.copy()
    outliers_removed_count = 0
    initial_rows = len(df_clean)

    print("\nHandling outliers using IQR method...")
    for col in columns:
         if pd.api.types.is_numeric_dtype(df_clean[col]):
            q1 = df_clean[col].quantile(0.25)
            q3 = df_clean[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            original_col_len = len(df_clean)
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
            removed_in_col = original_col_len - len(df_clean)
            if removed_in_col > 0:
                 print(f"Removed {removed_in_col} outliers from '{col}' (bounds: [{lower_bound:.2f}, {upper_bound:.2f}])")
         else:
             print(f"Skipping non-numerical column for outlier handling: {col}")


    final_rows = len(df_clean)
    total_removed = initial_rows - final_rows
    print(f"\nTotal rows removed: {total_removed}")
    print(f"Original shape: {df.shape}")
    print(f"Shape after outlier removal: {df_clean.shape}")
    return df_clean

if train_df is not None:
    cols_for_outlier_handling = [col for col in numeric_cols_for_corr if col != 'BMI']
    train_no_outliers = handle_outliers_iqr(train_df, cols_for_outlier_handling)



if train_no_outliers is not None:
    plot_distributions(train_no_outliers, cols_for_outlier_handling, title_suffix=" (After Outlier Removal)")



if train_no_outliers is not None:
    plot_boxplots(train_no_outliers, cols_for_outlier_handling, title_suffix=" (After Outlier Removal)")



def plot_categorical_relationships(df, target_col, cat_cols):
    """Plots boxplots of target vs categorical features."""
    if df is None or df.empty:
        print("DataFrame is empty. Cannot plot relationships.")
        return
    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found.")
        return

    valid_cat_cols = [col for col in cat_cols if col in df.columns]
    if not valid_cat_cols:
        print("No valid categorical columns found or specified.")
        return

    n_cols = len(valid_cat_cols)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))
    if n_cols == 1: # Handle case with only one subplot
        axes = [axes]

    print(f"\nPlotting {target_col} vs Categorical Features...")
    for i, col in enumerate(valid_cat_cols):
        sns.boxplot(data=df, x=col, y=target_col, ax=axes[i], palette='viridis')
        axes[i].set_title(f'{target_col} by {col}')
        axes[i].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()



if train_df is not None:
    categorical_features = ['Sex', 'Age_Group', 'Intensity']
    plot_categorical_relationships(train_df, 'Calories', categorical_features)



def plot_scatter(df, x_col, y_col, hue_col=None, title=None, alpha=0.5):
    """Creates a scatter plot."""
    if df is None or df.empty:
        print("DataFrame is empty. Cannot plot scatter.")
        return
    if x_col not in df.columns or y_col not in df.columns:
        print(f"Columns '{x_col}' or '{y_col}' not found.")
        return
    if hue_col and hue_col not in df.columns:
        print(f"Hue column '{hue_col}' not found. Plotting without hue.")
        hue_col = None

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, alpha=alpha, palette='viridis')
    plot_title = title if title else f'{x_col} vs {y_col}' + (f' by {hue_col}' if hue_col else '')
    plt.title(plot_title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.show()



if train_df is not None:
    plot_scatter(train_df, 'Duration', 'Calories', hue_col='Sex', title='Duration vs Calories Burned by Gender')



if train_df is not None:
    plot_scatter(train_df, 'Heart_Rate', 'Calories', hue_col='Sex', title='Heart Rate vs Calories Burned by Gender')



def plot_pairplot(df, columns, hue_col=None, title="Pair Plot of Key Features"):
    """Creates a pair plot for specified columns."""
    if df is None or df.empty:
        print("DataFrame is empty. Cannot create pair plot.")
        return
    if not columns:
         print("No columns specified for pair plot.")
         return

    print("\nGenerating Pair Plot (this might take a moment)...")
    sns.pairplot(df[columns], hue=hue_col, palette='viridis', plot_kws={'alpha': 0.5})
    plt.suptitle(title, y=1.02)
    plt.show()


if train_no_outliers is not None:
    pairplot_cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'BMI', 'Calories', 'Sex_numeric'] # Use Sex_numeric for hue
    plot_pairplot(train_no_outliers, pairplot_cols, hue_col='Sex_numeric')




