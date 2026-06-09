# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# NOTEBOOK SETTINGS
import warnings
warnings.filterwarnings('ignore')

# Import The Datasets
train_data = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv").drop("id", axis=1)
test_data = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


train_data.head()


# Convert discrete variables into categorical
# train_data['Number_of_Ads'] = train_data['Number_of_Ads'].astype('category') 


# Dataset details
train_data.info()


# SUMMARY STATISTICS
train_data.describe()


# DISTRIBUTION ANALYSIS
def plot_dist(df):
    num_var_list = df.select_dtypes(include=np.number).columns.tolist()
    n_rows = int(np.ceil(len(num_var_list)))
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 15))
    # axes = axes.flatten()

    for i, col in enumerate(num_var_list):
        sns.histplot(df, x=col, ax=axes[i, 0], kde=True, stat='frequency', palette="Set2")
        axes[i, 0].set_title(f"Histogram of {col}")
        sns.boxplot(df, x=col, ax=axes[i, 1], palette="Spectral")
        axes[i, 1].set_title(f"Boxplot of {col}")
        sns.violinplot(df, x=col, ax=axes[i, 2], palette="Spectral")
        axes[i, 2].set_title(f"ViolinPlot of {col}")
        
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5, wspace=0.2)
    plt.show()
 
plot_dist(train_data)


# Correlation Analysis
sns.heatmap(
    train_data.select_dtypes(include=np.number).corr(),
    annot=True,
    fmt=".2f",
    cmap="Blues"
)


# Helper Function 1
def cat_plots(data, target):
    data = data.drop(["Podcast_Name", "Episode_Title"], axis=1)
    cat_cols = data.select_dtypes(exclude=np.number).columns.tolist()
    
    
    # Plot layout
    fig, axes = plt.subplots(int(np.ceil(len(cat_cols)/2)), 2, figsize=(30, 18))
    axes = axes.flatten() # For easy iteration due to the use of axes[i]

    for i, col in enumerate(cat_cols):
        sns.barplot(data=data, x=col, y=target, ax=axes[i], palette='Spectral', ci=None)
        axes[i].set_title(f"{col} against {target}", fontsize=15)
        axes[i].grid(axis='x', linestyle='--', alpha=0.6)
        axes[i].grid(axis='y', linestyle='--', alpha=0.6)
        axes[i].set_ylabel(f"{target}", fontsize=15)
        axes[i].set_yticks(axes[i].get_yticks(), axes[i].get_yticklabels(),fontsize=15)
        axes[i].set_xlabel(col, fontsize=15)
        axes[i].set_xticks(axes[i].get_xticks(), axes[i].get_xticklabels(), rotation=45, fontsize=20)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5, wspace=0.2)
    plt.show()


# Helper Function 2
def cat_plot_podname_ep(data, target):
    cat_cols = ["Podcast_Name", "Episode_Title"]

    # Plot layout
    fig, axes = plt.subplots(2, 1, figsize=(30, 18))
    axes = axes.flatten() # For easy iteration due to the use of axes[i]

    for i, col in enumerate(cat_cols):
        sns.barplot(data=data, x=col, y=target, ax=axes[i], palette='Spectral', ci=None)
        axes[i].set_title(f"{col} against {target}", fontsize=15)
        axes[i].grid(axis='x', linestyle='--', alpha=0.6)
        axes[i].grid(axis='y', linestyle='--', alpha=0.6)
        axes[i].set_ylabel(f"{cat_cols[i]}", fontsize=15)
        axes[i].set_yticks(axes[i].get_yticks(), axes[i].get_yticklabels(),fontsize=15)
        axes[i].set_xlabel(col, fontsize=15)
        axes[i].set_xticks(axes[i].get_xticks(), axes[i].get_xticklabels(), rotation=45, fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5, wspace=0.2)
    plt.show()


######################################################################
cat_plot_podname_ep(data=train_data, target="Listening_Time_minutes")
cat_plots(train_data, target="Listening_Time_minutes")


# Display all missing values from all variables
plt.figure(figsize=(8, 4))
sns.heatmap(train_data.isnull(), cmap='viridis')
plt.title("Plot showing missing values")
plt.show()


# A SIMPLE PIPLINE TO IMPUTE MISSING VALUES FROM CAT AND NUM VARIABLES
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_selector
from sklearn.compose import make_column_transformer

num_cat_imputer = make_column_transformer(
    # Replace missing vals in num varaibles with their median
    (SimpleImputer(strategy="median"), make_column_selector(dtype_include=np.number)),
    # Replace missing vals in cat varaibles with their most_frequent cats
    (SimpleImputer(strategy="most_frequent"), make_column_selector(dtype_include=object)),
).set_output(transform="pandas")

train_data = num_cat_imputer.fit_transform(train_data)


plt.figure(figsize=(8, 4))
sns.heatmap(train_data.isnull(), cmap='viridis')
plt.title("Plot showing missing values (After Imputation)")
plt.show()




