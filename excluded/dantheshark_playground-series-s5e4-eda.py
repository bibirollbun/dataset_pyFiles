import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import math
from IPython.display import display, Markdown
import scipy.stats as stats

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Decide between local or kaggle cloud storage         
KAGGLE_ENV = 'kaggle' in os.listdir('/')
data_path = '/kaggle/input' if KAGGLE_ENV else '../kaggle/input'
    
    
for dirname, _, filenames in os.walk(data_path):
    for filename in filenames:
        print(os.path.join(dirname, filename)) 


def merge_rare_categories(df, feature, threshold):
    value_counts = df[feature].value_counts()

    to_remove = value_counts[value_counts < threshold].index
    print(to_remove)

    # Replace the data cell with NaN
    df[feature] = df[feature].apply(lambda x: x if x not in to_remove else None)

    print(df.head(25))
    return df


def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))


def plot_heatmap(df, feature, target):
    # Absolute values calculation (number of cases per feature and target)
    degree_dep_table = pd.crosstab(df[feature], df[target])

    # Check if target has only two categories (e.g., binary classification)
    if len(degree_dep_table.columns) == 2:
        # Sort rows by the number of cases where target = 1 (depression cases)
        degree_dep_table = degree_dep_table.sort_values(by=1, ascending=False)

    # Relative values calculation (row-wise normalization to get percentage values)
    degree_dep_table_rel = degree_dep_table.div(degree_dep_table.sum(axis=1), axis=0) * 100

    # Ensure that the sorting is applied to both tables
    degree_dep_table_rel = degree_dep_table_rel.loc[degree_dep_table.index]

    # Combined display: Absolute values + percentage values in one cell
    combined_table = degree_dep_table.astype(str) + " (" + degree_dep_table_rel.round(2).astype(str) + "%)"

    # Plot heatmap
    plt.figure(figsize=(35, 12))
    sns.heatmap(degree_dep_table_rel, annot=combined_table, fmt="", cmap="coolwarm")
    plt.title(f"{target} Distribution by {feature} (Sorted by Highest {target} Cases)")
    plt.show()


# Load the data
train_original = pd.read_csv(data_path + '/playground-series-s5e4/train.csv')
test_original = pd.read_csv(data_path + '/playground-series-s5e4/test.csv')
sample_submission = pd.read_csv(data_path + '/playground-series-s5e4/sample_submission.csv')

original_data = pd.read_csv(data_path + '/podcast-listening-time-prediction-dataset/podcast_dataset.csv')


train_original.head(50)


test_original.head()


original_data.head()


sample_submission.head()


def data_overview(data, target):
    # Overview
    display(Markdown("## Data Overview"))
    
    display(Markdown("### General Information"))
    display(Markdown(f"- Number of rows and columns: {data.shape[0]} x {data.shape[1]}"))
    display(Markdown("- Column names:"))
    display(list(data.columns))

    display(Markdown("### Data Types & Missing Values"))
    missing = data.isnull().sum()
    dtypes = pd.DataFrame(data.dtypes, columns=["Data Type"])
    missing_df = pd.DataFrame(missing, columns=["Missing Values"])
    overview_df = dtypes.join(missing_df)
    display(overview_df.style.background_gradient(cmap="coolwarm"))

    display(Markdown("### Classic head of Data"))
    display(data.head().style.set_properties(**{"background-color": "#f5f5f5"}))

    display(Markdown("### Statistical Summary (describe)"))
    display(data.describe().T.style.background_gradient(cmap="viridis"))

    # Target variable analysis
    display(Markdown(f"## Target Variable: `{target}`"))
    sns.set_style("whitegrid")  
    sns.set_palette("viridis")   

    if target in data.columns:
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))

        sns.histplot(data[target], bins=30, kde=True, ax=ax[0])
        ax[0].set_title("Absolute Frequency", fontsize=12, fontweight="bold")
        ax[0].set_ylabel("Count")
        ax[0].set_xlabel(target)
        ax[0].grid(axis="y", linestyle="--", alpha=0.5)


        # # Percentage distribution barplot
        # # Prepare percentage distribution as DataFrame
        # percentages = data[target].value_counts(normalize=True).reset_index()
        # percentages.columns = [target, "percentage"]

        # sns.barplot(x=target, y="percentage", data=percentages, ax=ax[1])
    


        # ax[1].set_title("Percentage Distribution", fontsize=12, fontweight="bold")
        # ax[1].set_ylabel("Percentage")
        # ax[1].set_xlabel(target)
        # ax[1].grid(axis="y", linestyle="--", alpha=0.5)

        

        for spine in ["top", "right"]:
            ax[0].spines[spine].set_visible(False)
            ax[1].spines[spine].set_visible(False)

    plt.tight_layout()
    plt.show()


data_overview(original_data, 'Listening_Time_minutes')


data_overview(train_original, 'Listening_Time_minutes')


data_overview(test_original, 'Listening_Time_minutes')


train_original.drop('id', axis=1, inplace=True) #id is not needed for training
original_data = original_data[train_original.columns]
original_data.head(100)


train_original.head()


#Concat train and the original data set
train = train_original.copy()
test = test_original.copy()
test.drop('id', axis=1, inplace=True) #id is not needed for testing
train = pd.concat([train, original_data],ignore_index=True)


# just make sure to concat worked,check if the objecte type is the same
train.iloc[train_original.shape[0]-5:train_original.shape[0]+5].head(10)


def visualize_feature_attributes(df, target=None):
    """ Visualizes numeric and categorical features """

    # Get Numeric & Categorical Features
    numeric_features, categorical_features =get_categorical_numerical_features(df)

    # Numeric Features
    if numeric_features:
        display(Markdown("## Numeric Feature Attributes"))
        for col in numeric_features:
            if col != target:
                plot_numeric_feature(df, col, target)
    else:
        print("No numeric features found.")

    # Categorical Features
    if categorical_features:
        display(Markdown("## Categorical Feature Attributes"))
        for col in categorical_features:
            if col != target:
                if df[col].nunique() > 10:
                    df[col] = reduce_categories(df[col], top_n=15)
                plot_categorical_feature(df, col, target)
    else:
        print("No categorical features found.")


def plot_numeric_feature(df, col, target):
    """ Plots Histogram, Boxplot, and Violinplot for a numeric feature """
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    sns.histplot(df[col], ax=axes[0], kde=True)
    axes[0].set_title(f"Distribution of {col}", fontweight="bold")

    sns.boxplot(x=df[col], ax=axes[1])
    axes[1].set_title(f"Boxplot of {col}", fontweight="bold")

    # if target and target in df.columns and df[target].nunique() == 2:
    #     sns.violinplot(x=df[target], y=df[col], ax=axes[2], split=True)
    # elif target and target in df.columns:
    #     sns.violinplot(x=df[target], y=df[col], ax=axes[2], split=False)
    # else:
    #     sns.violinplot(y=df[col], ax=axes[2])

    axes[2].set_title(f"Violinplot of {col} by {target}", fontweight="bold")

    plt.tight_layout()
    plt.show()


def plot_categorical_feature(df, col, target):
    """ Plots Countplot, Hue-Countplot, and Barplot (if target is numeric) for a categorical feature """
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # sns.countplot(x=df[col], ax=axes[0])
    sns.histplot(x=df[col], bins=30, ax=axes[0])
    # axes[0].set_title(f"Countplot of {col}", fontweight="bold")
    axes[0].set_title(f"Distribution of Episode Numbers {col}", fontweight="bold")
    axes[0].tick_params(axis='x', rotation=45)

    # Plot 2: Countplot with hue (only if hue has few unique values)
    if target in df.columns and df[target].nunique() <= 10:
        sns.countplot(x=df[col], hue=df[target], ax=axes[1])
        axes[1].set_title(f"Countplot of {col} by {target}", fontweight="bold")
        axes[1].tick_params(axis='x', rotation=45)
    else:
        axes[1].remove()


    # Plot 3: Barplot of mean target per category
    if target in df.columns and df[target].dtype in [np.float64, np.int64]:
        sns.barplot(x=df[col], y=df[target], ax=axes[2], estimator=np.mean, errorbar='sd')
        axes[2].set_title(f"Mean {target} by {col}", fontweight="bold")
        axes[2].tick_params(axis='x', rotation=45)
    else:
        axes[2].remove()

    plt.tight_layout()
    plt.show()
    

# def reduce_categories(df, col, top_n):
#     """ Shows only the categories with highes numbers, seldoms are shown with "others" """
#     top_categories = df[col].value_counts().nlargest(top_n).index
#     df[col] = df[col].apply(lambda x: x if x in top_categories else 'Other')
#     return df

def reduce_categories(col_series, top_n):
    top_categories = col_series.value_counts().nlargest(top_n).index
    return col_series.apply(lambda x: x if x in top_categories else 'Other')


def get_categorical_numerical_features(df):
    # Get Numeric & Categorical Features
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return numeric_features, categorical_features

visualize_feature_attributes(train_original, target="Listening_Time_minutes")


# # Get Numeric & Categorical Features
numeric_features, categorical_features = get_categorical_numerical_features(train)
sns.heatmap(train[numeric_features].corr(), annot=True, cmap='coolwarm')


# for col in categorical_features:
#     plot_heatmap(train, col, 'Listening_Time_minutes')


# for col in train[categorical_features]:
#     print(col)
#     print(cramers_v(train[col], train["Listening_Time_minutes"]))


# merge_features = ['Working Professional or Student', 'Degree', 'Have you ever had suicidal thoughts ?']
# for col in merge_features:
#     train = merge_rare_categories(train, col, 15)
#     test = merge_rare_categories(test, col, 15)
#     plot_heatmap(train, col, 'Depression')
#     #plot_heatmap(test, col, 'Depression')


# if KAGGLE_ENV:
#     train.to_csv('/kaggle/input/s5-e4-train-concat/s5-e4-train-concat.csv', index=False)
# else:
#     train.to_csv('../kaggle/input/' + '/s5-e4-train-concat/s5-e4-train-concat.csv', index=False)


# # save test data set
# if KAGGLE_ENV:
#     test.to_csv('/kaggle/input/s5-e4-test-concat/s5-e4-test-concat.csv', index=False)
# else:
#     test.to_csv( '../kaggle/input/' + '/s5-e4-test-concat/s5-e4-test-concat.csv', index=False)


# train.head()


# test.head()

