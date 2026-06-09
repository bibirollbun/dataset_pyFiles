# MAP Competition - Starter Notebook for Dataset Exploration

# -------------------------
# Section 1: Environment Setup
# -------------------------
import os                                       # Operating system interactions
import random                                   # Python built-in random utilities

# Numerical and data manipulation libraries
import numpy as np                              # Array operations and numerical computing
import pandas as pd                             # Data manipulation and analysis

# Scikit-learn modules for modeling
from sklearn.model_selection import train_test_split  # Splitting data into train/test sets
from sklearn.feature_extraction.text import TfidfVectorizer  # Converting text data to TF-IDF features
from sklearn.linear_model import LogisticRegression   # Logistic Regression classifier
from sklearn.metrics import accuracy_score            # Performance evaluation metric




# Enable inline plots
%matplotlib inline

# Show all columns, do not truncate
pd.set_option('display.max_columns', None)

# Optional: set a large width to avoid ugly line breaks (still relevant for HTML tables)
pd.set_option('display.width', 1000)

# Show full text in each cell, up to a large limit
pd.set_option('display.max_colwidth', None)


# Define data directory
DATA_DIR = "/kaggle/input/map-charting-student-math-misunderstandings"

# Construct file paths
train_path = os.path.join(DATA_DIR, 'train.csv')
test_path = os.path.join(DATA_DIR, 'test.csv')
sample_submission_path = os.path.join(DATA_DIR, 'sample_submission.csv')

# Load datasets into DataFrames
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_submission_df = pd.read_csv(sample_submission_path)

# Display basic information to verify successful loading
print("\nData Loading Summary:")
print(f"  - Training set: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
print(f"  - Test set:     {test_df.shape[0]} rows, {test_df.shape[1]} columns")
print(f"  - Submission:   {sample_submission_df.shape[0]} rows, {sample_submission_df.shape[1]} columns")


train_df.head(10)


# Compute the frequency of each unique question in the training set
train_df['QuestionText'].value_counts().reset_index()



# Calculate the count of each category in the training set
train_df['Category'].value_counts().reset_index()



from IPython.display import display, Math, Latex

# GET ANSWER CHOICES
tmp = train_df.groupby(['QuestionId','MC_Answer']).size().reset_index(name='count')
tmp['rank'] = tmp.groupby('QuestionId')['count'].rank(method='dense', ascending=False).astype(int) - 1
tmp = tmp.drop('count',axis=1)
tmp = tmp.sort_values(['QuestionId','rank'])

# DISPLAY QUESTION AND ANSWER CHOICES
Q = tmp.QuestionId.unique()
for q in Q:
    question = train_df.loc[train_df.QuestionId==q].iloc[0].QuestionText
    choices = tmp.loc[tmp.QuestionId==q].MC_Answer.values
    labels="ABCD"
    choice_str = " ".join([f"({labels[i]}) {choice}" for i, choice in enumerate(choices)])
    
    print()
    display(Latex(f"QuestionId {q}: {question}") )
    display(Latex(f"MC Answers: {choice_str}"))



import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_categorical_bar(
    series,
    categories=None,
    category_labels=None,
    title="Bar Chart",
    xlabel="Category",
    ylabel="Count",
    base_cmap="Blues",
    figsize=(8, 5)
):
    """
    Plots a styled bar chart for a categorical pandas Series with a descending gradient using Matplotlib.

    Args:
        series (pd.Series): The categorical data to plot.
        categories (list, optional): List of category names (order & inclusion).
                                     If None, inferred from series.value_counts().
        category_labels (dict, optional): Mapping of category -> display name.
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        base_cmap (str): Name of a Matplotlib colormap for gradient.
        figsize (tuple): Figure size.
    """
    # Compute counts
    counts = series.value_counts()
    if categories:
        counts = counts.reindex(categories, fill_value=0)

    df_plot = counts.reset_index()
    df_plot.columns = ["category", "count"]

    # Map display labels
    if category_labels:
        df_plot["display_label"] = df_plot["category"].map(lambda x: category_labels.get(x, x))
    else:
        df_plot["display_label"] = df_plot["category"]

    # Enforce order if provided
    if categories:
        ordered = [category_labels.get(c, c) if category_labels else c for c in categories]
        df_plot["display_label"] = pd.Categorical(df_plot["display_label"], categories=ordered, ordered=True)
        df_plot = df_plot.sort_values("display_label")

    # Prepare colors using a colormap
    cmap = plt.get_cmap(base_cmap)
    colors = cmap(np.linspace(0.6, 0.2, len(df_plot)))

    # Plot bars
    x_positions = np.arange(len(df_plot))
    plt.figure(figsize=figsize)
    plt.bar(x_positions, df_plot['count'], color=colors)

    # Labels and title
    plt.xticks(x_positions, df_plot['display_label'], rotation=45, ha='right', fontsize=10)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)

    # Style axes
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.grid(False)

    plt.tight_layout()
    plt.show()



import matplotlib.pyplot as plt

# Define the labels and category order once
category_labels = {
    "True_Correct":        "Correct + Correct Reasoning",
    "True_Misconception":  "Correct + Misconception",
    "True_Neither":        "Correct + Uninformative Explanation",
    "False_Correct":       "Wrong + Correct Reasoning",
    "False_Misconception": "Wrong + Misconception",
    "False_Neither":       "Wrong + Uninformative Explanation"
}
categories = [
    "True_Correct", "True_Misconception", "True_Neither",
    "False_Correct", "False_Misconception", "False_Neither"
]

# Loop through each unique QuestionId and generate a plot
for qid in train_df['QuestionId'].unique():
    subset = train_df[train_df['QuestionId'] == qid]['Category']
    plot_categorical_bar(
        series=subset,
        category_labels=category_labels,
        categories=categories,
        title=f"Student Answer-Explanation Categories for Question {qid}",
        xlabel="Category",
        ylabel="Count",
        figsize=(8, 5)
    )



plot_categorical_bar(
    series=train_df['Misconception'],
    category_labels={},  # no mapping if you just want raw names
    title="Math Misconceptions in Explanations (Train Set)",
    xlabel="Misconception",
    ylabel="Count",
    figsize=(20, 6)
)



for qid in train_df['QuestionId'].unique():
    subset = train_df[train_df['QuestionId'] == qid]['Misconception']
    plot_categorical_bar(
        series=subset,
        title=f"Student Misconceptions for Question {qid}",
        xlabel="Misconception Type",
        ylabel="Count",
        figsize=(8, 5)
    )

