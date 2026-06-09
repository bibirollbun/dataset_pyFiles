# Import libs
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from collections import Counter
import plotly.express as px


# Read data
train_df = pd.read_csv('/kaggle/input/eedi-mining-misconceptions-in-mathematics/train.csv')
test_df = pd.read_csv('/kaggle/input/eedi-mining-misconceptions-in-mathematics/test.csv')
misconceptions = pd.read_csv('/kaggle/input/eedi-mining-misconceptions-in-mathematics/misconception_mapping.csv')

# Transform misconceptions of each option into rows
stacked_misconceptions = train_df[['QuestionId', 'SubjectName', 'ConstructName', 'CorrectAnswer', 'QuestionText', 
                                   'MisconceptionAId', 'MisconceptionBId', 'MisconceptionCId', 'MisconceptionDId']].melt(
    id_vars=['QuestionId', 'SubjectName', 'ConstructName', 'CorrectAnswer', 'QuestionText'], 
    var_name='Choice', 
    value_name='Misconception'
).reset_index()
stacked_misconceptions['Choice'] = stacked_misconceptions['Choice'].str.replace('Misconception', '').str.replace('Id', '')

# Transform answer of each option into rows
stacked_answers = train_df[['QuestionId', 'SubjectName', 'ConstructName', 'CorrectAnswer', 'QuestionText', 
                            'AnswerAText', 'AnswerBText', 'AnswerCText', 'AnswerDText']].melt(
    id_vars=['QuestionId', 'SubjectName', 'ConstructName', 'CorrectAnswer', 'QuestionText'], 
    var_name='Choice', 
    value_name='Answer'
).reset_index()
stacked_answers['Choice'] = stacked_answers['Choice'].str.replace('Answer', '').str.replace('Text', '')


# Combine misconceptions and answers
stacked_df = pd.merge(left = stacked_misconceptions, right=stacked_answers).drop(columns=['index'])
stacked_df = stacked_df.dropna().reset_index(drop=True)

# Add Misconception description
misconception_df = pd.merge(left = stacked_df, right=misconceptions, left_on=['Misconception'], right_on='MisconceptionId')


# Some more pre-processing
misconception_df['SubjectName'] = misconception_df['SubjectName'].apply(lambda x: x.replace('BIDMAS', 'Brackets, Indices, Division, Multiplication, Addition, Subtraction'))
misconception_df['SubjectName'] = misconception_df['SubjectName'].apply(lambda x: str.lower(x))


misconception_df.head()


misconception_df['Misconception'].value_counts()


print("Number of unique subjects:", misconception_df['SubjectName'].nunique())
print("Max length (in characters) of subjects:", max(misconception_df['SubjectName'].str.len()))
print("List of unique subjects:", misconception_df['SubjectName'].unique()[:10])


# I made some viz for this. Pushed the images to a dataset but couldn't render in md 
# If anyone have any idea how to fix, please comment

from IPython.display import Image
Image(filename='/kaggle/input/eedi-semantic-clustering/SubjectCluster.png')


# The categorized list of subject
math_categories = {
    "Number Operations": {
        "Basic Operations": [
            "counting",
            "mental addition and subtraction",
            "mental multiplication and division",
            "written addition",
            "written subtraction",
            "written multiplication",
            "written division",
            "combining operations",
            "basic calculator use",
            "brackets, indices, division, multiplication, addition, subtraction",
            "place value"
        ],
        "Negative Numbers": [
            "ordering negative numbers",
            "adding and subtracting negative numbers",
            "multiplying and dividing negative numbers"
        ],
        "Decimals": [
            "adding and subtracting with decimals",
            "multiplying and dividing with decimals",
            "ordering decimals",
            "rounding to decimal places",
            "rounding to the nearest whole (10, 100, etc)",
            "rounding to significant figures",
            "converting between decimals and percentages"
        ],
        "Fractions": [
            "simplifying fractions",
            "equivalent fractions",
            "ordering fractions",
            "fractions of an amount",
            "adding and subtracting fractions",
            "multiplying fractions",
            "dividing fractions",
            "converting mixed number and improper fractions",
            "converting between fractions and decimals",
            "converting between fractions and percentages",
            "recurring decimals to fractions"
        ],
        "Percentages": [
            "percentages of an amount",
            "percentage increase and decrease"
        ],
        "Factors and Multiples": [
            "factors and highest common factor",
            "multiples and lowest common multiple"
        ],
        "Estimation": [
            "types, naming and estimating",
            "estimation"
        ]
    },
    "Algebra": {
        "Basic Algebra": [
            "writing expressions",
            "simplifying expressions by collecting like terms",
            "writing formula",
            "substitution into formula",
            "multiplying terms",
            "function machines"
        ],
        "Equations": [
            "linear equations",
            "rearranging formula and equations",
            "simultaneous equations",
            "quadratic equations",
            "solving linear inequalities",
            "solving quadratic inequalities",
            "trial and improvement and iterative methods"
        ],
        "Algebraic Fractions": [
            "simplifying algebraic fractions",
            "adding and subtracting algebraic fractions",
            "multiplying and dividing algebraic fractions"
        ],
        "Brackets and Factoring": [
            "expanding single brackets",
            "expanding double brackets",
            "expanding triple brackets and more",
            "factorising into a single bracket",
            "factorising into a double bracket",
            "difference of two squares"
        ]
    },
    "Geometry": {
        "2D Shapes": [
            "properties of triangles",
            "properties of quadrilaterals",
            "properties of polygons",
            "2d names and properties of shapes-others",
            "parts of a circle",
            "construct triangle"
        ],
        "3D Shapes": [
            "names and properties of 3d shapes",
            "nets"
        ],
        "Angles": [
            "measuring angles",
            "basic angle facts (straight line, opposite, around a point, etc)",
            "angles in triangles",
            "angles in polygons",
            "angle facts with parallel lines",
            "construct angle"
        ],
        "Lines": [
            "parallel lines",
            "perpendicular lines",
            "horizontal and vertical lines"
        ],
        "Transformations": [
            "translation and vectors",
            "reflection",
            "rotation",
            "enlargement",
            "line symmetry",
            "rotational symmetry"
        ]
    },
    "Measurement": {
        "Length and Area": [
            "length units",
            "perimeter",
            "area units",
            "area of simple shapes",
            "compound area",
            "missing lengths"
        ],
        "Volume": [
            "volume of prisms",
            "volume of non-prisms",
            "volume and capacity units",
            "surface area of prisms"
        ],
        "Scale and Proportion": [
            "length scale factors in similar shapes",
            "length, area and volume scale factors"
        ],
        "Other Units": [
            "time",
            "weight units",
            "temperature units",
            "basic money",
            "currency conversions"
        ]
    },
    "Statistics and Data": {
        "Data Representation": [
            "frequency tables",
            "pictogram",
            "block graphs and bar charts",
            "pie chart",
            "time series and line graphs",
            "types of data and questionnaires"
        ],
        "Data Analysis": [
            "averages (mean, median, mode) from a list of data",
            "range and interquartile range from a list of data",
            "averages and range from frequency table",
            "averages and range from grouped data"
        ],
        "Probability": [
            "probability of single events",
            "experimental probability and relative frequency",
            "combined events",
            "tree diagrams with dependent events",
            "systematic listing strategies"
        ]
    },
    "Functions and Graphs": {
        "Coordinate Geometry": [
            "naming co-ordinates in 2d",
            "distance between two co-ordinates",
            "midpoint between two co-ordinates",
            "co-ordinate geometry with straight lines"
        ],
        "Linear Functions": [
            "plotting lines from tables of values",
            "finding the equation of a line",
            "finding the gradient and intercept of a line from the equation",
            "gradient as change in y over change in x",
            "gradient between two co-ordinates",
            "straight line graphs-others"
        ],
        "Non-Linear Functions": [
            "plotting quadratics from tables of values",
            "quadratic graphs-others",
            "sketching from factorised form",
            "sketching from completing the square form",
            "graphs of exponentials and other powers of x",
            "cubics and reciprocals",
            "transformations of functions in the form f(x)",
            "equation of a circle",
            "real life graphs",
            "other graphs-others"
        ]
    },
    "Sequences and Pattern": {
        "Number Sequences": [
            "linear sequences (nth term)",
            "quadratic sequences",
            "other sequences",
            "sequences-others"
        ]
    },
    "Advanced Topics": {
        "Trigonometry": [
            "right-angled triangles (sohcahtoa)",
            "exact values of trigonometric ratios"
        ],
        "Pythagoras": [
            "2d pythagoras"
        ],
        "Indices and Surds": [
            "laws of indices",
            "square roots, cube roots, etc",
            "squares, cubes, etc",
            "simplifying surds",
            "operations with surds"
        ],
        "Proportion": [
            "direct proportion",
            "indirect (inverse) proportion",
            "sharing in a ratio",
            "writing ratios"
        ],
        "Other Advanced": [
            "standard form",
            "upper and lower bounds",
            "algebraic proof",
            "completing the square",
            "graphical solution of simultaneous equations",
            "graphing linear inequalities (shading regions)",
            "quadratic inequalities on number lines",
            "inequalities on number lines",
            "venn diagrams",
            "bearings",
            "congruency in other shapes",
            "speed, distance, time"
        ]
    }
}
topic_to_category = {}
topic_to_subcategory = {}

# Function to find the corresponding category for a topic
for category, subcats in math_categories.items():
    for subcat, topics in subcats.items():
        for topic in topics:
            topic_to_category[topic] = category
            topic_to_subcategory[topic] = subcat

# Then use with pandas
misconception_df['SubjectMainCategory'] = misconception_df['SubjectName'].map(topic_to_category)
misconception_df['SubjectSubCategory'] = misconception_df['SubjectName'].map(topic_to_subcategory)


misconception_df.head()


# Group and calculate the distribution
subject_theme_dist = misconception_df.groupby(['SubjectMainCategory'])['QuestionId'].nunique().sort_values(ascending=False)

# Calculate the mean
mean_value = subject_theme_dist.mean()

# Plot the bar chart
plt.figure(figsize=(8, 5))
plt.title("Distribution of Subject Main Categories by Number of Questions")
plt.bar(subject_theme_dist.index, subject_theme_dist, color='skyblue')

# Add a mean line
plt.axhline(mean_value, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_value:.2f}')

# Add labels and legend
plt.xticks(rotation=45, ha='right')
plt.ylabel("Number of Questions")
plt.legend()
plt.tight_layout()



# Import required libraries if not already imported
import seaborn as sns
import matplotlib.pyplot as plt

# Group the data to get counts of QuestionId within each main and subcategory
category_counts = (
    misconception_df.groupby(['SubjectMainCategory', 'SubjectSubCategory'])['QuestionId']
    .count()
    .reset_index(name='Count')
)

# Get unique main categories
main_categories = misconception_df['SubjectMainCategory'].unique()

# Calculate number of rows and columns for subplots
n_categories = len(main_categories)
n_cols = 2
n_rows = (n_categories + 1) // 2  # Ceiling division to ensure enough rows

# Create figure and subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
axes = axes.flatten()  # Flatten axes array for easier indexing
palette = sns.color_palette("husl", n_colors=n_categories)

# Create a bar plot for each main category
for idx, main_cat in enumerate(main_categories):
    # Get data for this category
    cat_data = category_counts[category_counts['SubjectMainCategory'] == main_cat]
    # Sort the data by 'Count' in descending order
    cat_data = cat_data.sort_values(by='Count', ascending=False)

    # Create bar plot
    sns.barplot(
        data=cat_data,
        x='SubjectSubCategory',
        y='Count',
        ax=axes[idx],
        color=palette[idx]  # Use the generated palette
    )
    
    # Customize subplot
    axes[idx].set_title(f'Phân bố chủ đề phụ của: {main_cat}')  # Translated title
    axes[idx].set_xlabel('Chủ đề phụ')  # Translated x-axis label
    axes[idx].set_ylabel('Số lượng')  # Translated y-axis label
    
    # Rotate x-axis labels for better readability
    axes[idx].tick_params(axis='x', rotation=45)
    axes[idx].set_xticklabels(axes[idx].get_xticklabels(), ha='right')
    
    # Add value labels on top of bars
    for i, v in enumerate(cat_data['Count']):
        axes[idx].text(i, v, str(v), ha='center', va='bottom')
    
    # Add grid
    # axes[idx].grid(axis='y', linestyle='--', alpha=0.7)

# Remove any empty subplots
for idx in range(len(main_categories), len(axes)):
    fig.delaxes(axes[idx])

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()


# Group the data and calculate the unique counts
grouped_data = misconception_df.groupby(['SubjectMainCategory', 'SubjectSubCategory']).nunique()[['QuestionId', 'MisconceptionId', 'ConstructName']]

# Sort values by 'QuestionId' in descending order
sorted_data = grouped_data.sort_values(by='QuestionId', ascending=False)

# Reset index to include 'SubjectMainCategory' in the plot
sorted_data = sorted_data.reset_index()

# Extract columns for the scatter plot
x = sorted_data['MisconceptionId']
y = sorted_data['QuestionId']
categories = sorted_data['SubjectMainCategory']

# Calculate means for the x and y axes
mean_x = x.mean()
mean_y = y.mean()

# Set color palette for SubjectMainCategory
palette = sns.color_palette("tab10", n_colors=len(categories.unique()))

# Create scatter plot with color coding by SubjectMainCategory
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x=x,
    y=y,
    hue=categories,
    palette=palette,  # Use color palette for hue
    alpha=0.7,
    s=100  # Size of the scatter points
)

# Add regression line (use Seaborn's regplot without hue, overlayed)
sns.regplot(
    x=x,
    y=y,
    scatter=False,  # No scatter points, just the regression line
    line_kws={'color': 'grey', 'lw': 2},  # Customize the regression line
    ci=None  # Disable the confidence interval shading
)

# Add mean lines
plt.axvline(x=mean_x, color='black', linestyle='--')
plt.axhline(y=mean_y, color='black', linestyle='--')

# Add labels and title in English
plt.xlabel('Number of Misconceptions', fontsize=12)  # X-axis label
plt.ylabel('Number of Questions', fontsize=12)  # Y-axis label
plt.title('Distribution of Mathematical Topics by Number of Questions and Misconceptions', fontsize=14)  # Title

# Adjust legend
plt.legend(title='Main Category', bbox_to_anchor=(1.05, 1), loc='upper left')

# Add grid
plt.grid(True, linestyle='--', alpha=0.7)

# Show plot
plt.tight_layout()
plt.show()



# Group by 'MisconceptionName' and count unique 'SubjectMainCategory' values
main_category_counts = misconception_df.groupby(['MisconceptionName'])['SubjectMainCategory'].nunique()

# Count the number of misconceptions for each unique 'SubjectMainCategory' count
main_category_distribution = main_category_counts.value_counts().sort_index()

# Plot the results
plt.figure(figsize=(6, 4))
main_category_distribution.plot(kind='bar', color='skyblue')

# Add labels and title in English
plt.xlabel('Number of SubjectMainCategories containing each Misconception', fontsize=12)  # X-axis label
plt.ylabel('Number of Misconceptions', fontsize=12)  # Y-axis label
plt.title('Number of Misconceptions Appearing in One/Multiple SubjectMainCategories', fontsize=14)  # Title
plt.xticks(rotation=0)  # Rotate x-axis labels for better readability

# Display the plot
plt.tight_layout()
plt.show()



# Step 1: Group by 'MisconceptionName' and 'SubjectMainCategory', and count unique 'SubjectSubCategory' values
misconception_subcategory_count = misconception_df.groupby(['MisconceptionName', 'SubjectMainCategory'])['SubjectSubCategory'].nunique().reset_index()

# Step 2: Create a pivot table where rows are 'SubjectMainCategory', columns are the number of unique 'SubjectSubCategory' values
subcategory_counts_pivot = misconception_subcategory_count.groupby(['SubjectMainCategory', 'SubjectSubCategory']).size().unstack(fill_value=0)

# Step 3: Plot the heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(subcategory_counts_pivot, annot=True, cmap='YlGnBu', fmt='d', cbar_kws={'label': 'Number of Misconceptions'}, linewidths=0.5)

# Add labels and title in English
plt.xlabel('Number of SubjectSubCategories containing each Misconception', fontsize=12)  # X-axis label
plt.ylabel('Type of SubjectMainCategory', fontsize=12)  # Y-axis label
plt.title('Distribution of Misconceptions Appearing in One SubjectMainCategory\n for Each SubjectSubCategory', fontsize=14)  # Title

# Show plot
plt.tight_layout()
plt.show()



import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from collections import Counter

# Step 1: Group by 'MisconceptionName' and count unique 'SubjectMainCategory' values
misconception_category_count = misconception_df.groupby(['MisconceptionName', 'SubjectMainCategory']).size().reset_index()

# Step 2: Find misconceptions that fall into more than one 'SubjectMainCategory'
misconception_multiple_categories = misconception_category_count.groupby('MisconceptionName')['SubjectMainCategory'].nunique()
misconceptions_multiple_categories = misconception_multiple_categories[misconception_multiple_categories > 1]

# Step 3: Get all combinations of 'SubjectMainCategory' for misconceptions that fall into multiple categories
# Create a dictionary of misconceptions and their corresponding main category combinations
combinations_dict = {}
for misconception in misconceptions_multiple_categories.index:
    categories = misconception_df[misconception_df['MisconceptionName'] == misconception]['SubjectMainCategory'].unique()
    combinations_dict[misconception] = combinations(sorted(categories), 2)

# Step 4: Flatten the combinations and count the occurrences of each pair
pair_counts = Counter()
for combo_list in combinations_dict.values():
    for combo in combo_list:
        pair_counts[combo] += 1

# Step 5: Plot the most common combinations
# Get the most common combinations
common_combinations = pair_counts.most_common(10)

# Prepare data for plotting
combos, counts = zip(*common_combinations)
combo_labels = [f"{combo[0]} - {combo[1]}" for combo in combos]

# Create a DataFrame for sorting and plotting
misc_combo_pairs = pd.DataFrame({'labels': combo_labels, 'count': counts}).sort_values(by='count')

# Step 6: Plot
plt.figure(figsize=(8, 6))
plt.barh(misc_combo_pairs['labels'], misc_combo_pairs['count'], color='skyblue')

# Translated labels and title
plt.xlabel('Number of Misconceptions', fontsize=12)  # X-axis: Count of Misconceptions
plt.ylabel('SubjectMainCategory Combinations', fontsize=12)  # Y-axis: Main Category Combinations
plt.title('Most Common Combinations of SubjectMainCategory \nfor Misconceptions Belonging to Multiple Categories', fontsize=14)  # Title

plt.tight_layout()
plt.show()



# misconception_df['ConstructName'].apply(lambda x: ' '.join(x.split()[:3])).unique()


# I made some viz for this. Pushed the images to a dataset but couldn't render in md 
# If anyone have any idea how to fix, please comment

from IPython.display import Image
Image(filename='/kaggle/input/eedi-semantic-clustering/ConstructCluster.png')


# Get the list of 3-grams of ConstructName 
# misconception_df['ConstructName'].apply(lambda x : ' '.join(x.split()[:3])).unique()


# First, define the dictionary and function
MATH_ACTION_CATEGORIES = {
    # Direct Actions (Simple, One-Step Commands)
    'calculate': 'Direct Actions',
    'find': 'Direct Actions',
    'write': 'Direct Actions',
    'draw': 'Direct Actions',
    'read': 'Direct Actions',
    'add': 'Direct Actions',
    'subtract': 'Direct Actions',
    'multiply': 'Direct Actions',
    'divide': 'Direct Actions',
    'count': 'Direct Actions',
    'state': 'Direct Actions',
    'tell': 'Direct Actions',
    'mark': 'Direct Actions',
    'work': 'Direct Actions',
    
    # Transformative Actions
    'convert': 'Transformative Actions',
    'change': 'Transformative Actions',
    'express': 'Transformative Actions',
    'translate': 'Transformative Actions',
    'rotate': 'Transformative Actions',
    'reflect': 'Transformative Actions',
    'enlarge': 'Transformative Actions',
    'round': 'Transformative Actions',
    'raise': 'Transformative Actions',
    'substitute': 'Transformative Actions',
    
    # Analytical Actions
    'identify': 'Analytical Actions',
    'recognise': 'Analytical Actions',
    'recognize': 'Analytical Actions',
    'compare': 'Analytical Actions',
    'estimate': 'Analytical Actions',
    'order': 'Analytical Actions',
    'determine': 'Analytical Actions',
    'distinguish': 'Analytical Actions',
    'match': 'Analytical Actions',
    'prove': 'Analytical Actions',
    
    # Complex Actions
    'solve': 'Complex Actions',
    'construct': 'Complex Actions',
    'rearrange': 'Complex Actions',
    'simplify': 'Complex Actions',
    'expand': 'Complex Actions',
    'factorise': 'Complex Actions',
    'factorize': 'Complex Actions',
    'manipulate': 'Complex Actions',
    
    # Interpretive Actions
    'interpret': 'Interpretive Actions',
    'understand': 'Interpretive Actions',
    'describe': 'Interpretive Actions',
    'explain': 'Interpretive Actions',
    'know': 'Interpretive Actions',
    'recall': 'Interpretive Actions',
    
    # Process Actions
    'use': 'Process Actions',
    'follow': 'Process Actions',
    'perform': 'Process Actions',
    'carry': 'Process Actions',
    'complete': 'Process Actions',
    'continue': 'Process Actions',
    
    # Generative Actions
    'create': 'Generative Actions',
    'generate': 'Generative Actions',
    'plot': 'Generative Actions',
    'label': 'Generative Actions',
    'shade': 'Generative Actions',
    'construct': 'Generative Actions',
}

def categorize_math_action(text):
    # Handle empty or non-string input
    if not isinstance(text, str) or not text.strip():
        return 'Unknown'
    
    # Get the first word (assumed to be the action verb)
    first_word = text.lower().split()[0]
    
    # Special handling for multi-word verbs
    if first_word == 'carry' and 'out' in text.lower():
        return 'Process Actions'
    if first_word == 'work' and 'out' in text.lower():
        return 'Direct Actions'
    
    # Return the category if found, otherwise return 'Other'
    return MATH_ACTION_CATEGORIES.get(first_word, 'Other')


# Import the functions and dictionary from previous code
misconception_df['ConstructActionType'] = misconception_df['ConstructName'].apply(categorize_math_action)


import seaborn as sns
import matplotlib.pyplot as plt

# Count unique QuestionId values for each ConstructActionType
action_type_counts = (
    misconception_df.groupby('ConstructActionType')['QuestionId']
    .nunique()
    .reset_index()
    .rename(columns={'QuestionId': 'UniqueQuestionCount'})
)

# Sort the data by unique question count in descending order
action_type_counts = action_type_counts.sort_values(by='UniqueQuestionCount', ascending=False)

# Calculate the mean UniqueQuestionCount
mean_value = action_type_counts['UniqueQuestionCount'].mean()

# Set figure size
plt.figure(figsize=(6, 5))

# Create the bar plot
sns.barplot(
    data=action_type_counts,
    x='ConstructActionType',
    y='UniqueQuestionCount',
    color='skyblue'
)

# Add the mean line
plt.axhline(y=mean_value, color='red', linestyle='--', label=f'Mean: {mean_value:.2f}')

# Customize the plot with English titles and labels
plt.title('Distribution of ConstructActionType by Number of Unique Questions', fontsize=16)
plt.xlabel('ConstructActionType', fontsize=12)
plt.ylabel('Number of Unique Questions', fontsize=12)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Add value labels on top of bars
for i, row in enumerate(action_type_counts.itertuples()):
    plt.text(i, row.UniqueQuestionCount, str(row.UniqueQuestionCount), ha='center', va='bottom', fontsize=10)

# Add grid lines
# plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add legend for the mean line
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()



# Create a pivot table aggregating counts of QuestionId
pivot_table = misconception_df.pivot_table(
    values='QuestionId',  # Column to aggregate
    columns='ConstructActionType',
    index='SubjectMainCategory',
    aggfunc='count',  # Count non-null QuestionId entries
    fill_value=0
)

# Create a heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(pivot_table, annot=True, fmt='d', cmap='YlGnBu', cbar=True)
plt.title('Distribution of ConstructActionType and SubjectMainCategory\nBased on Number of Questions')
plt.xticks(rotation=45, ha='right')
plt.xlabel('ConstructActionType')
plt.ylabel('SubjectMainCategory')
plt.show()



# I made some viz for this. Pushed the images to a dataset but couldn't render in md 
# If anyone have any idea how to fix, please comment

from IPython.display import Image
Image(filename='/kaggle/input/update-misconception-cluster/MisconceptionCluster.png')


# Input information for LLM segmentation
# misconception_df['MisconceptionName'].apply(lambda x: ' '.join(x.split()[:3])).unique()


id_check = 9
misconception_df['MisconceptionName'][id_check]


import pandas as pd
import numpy as np

def classify_error_action(text):
    """
    Classifies mathematical errors into streamlined categories based on the primary action verb.
    
    Parameters:
    text (str): Description of the mathematical misconception
    
    Returns:
    str: Action-based classification
    """
    text = text.lower()
    
    # Conceptual Understanding Issues
    if any(phrase in text for phrase in [
        'does not know', 'does not understand', 'does not think', 
        'does not realise', 'does not see', 'does not fully', 
        'does not connect', 'does not recall', 'does not link',
        'does not recognise', "doesn't", 'does not', 'misunderstands',
        'confusion', 'confused', 'mixes', 'mistakes', 'misinterprets',
        'cannot', 'can not', 'not able', 'struggles', 'unable', 
        'difficulty', 'thinks', 'assumes', 'believes',
        'identifies', 'recognises', 'chooses', 'selects', 'names'  # Added from Visual/Spatial
    ]):
        return 'Conceptual Misunderstanding'
    
    # Procedural Errors
    elif any(word in text for word in [
        'multiplied', 'divided', 'adds', 'subtracts', 'doubles', 
        'halves', 'incorrectly', 'incorrect', 'orders', 'lines up',
        'starts', 'carries out', 'stops', 'order', 'before',
        'converts', 'translates', 'way around',
        'rotates', 'reflects', 'enlarges'  # Added from Visual/Spatial
    ]):
        return 'Procedural Error'
    
    # Computation Errors
    elif any(word in text for word in [
        'counts', 'miscounts', 'counting', 'estimates', 
        'approximates', 'rounds', 'has used'
    ]):
        return 'Computation Error'
    
    # Memory/Attention Errors
    elif any(word in text for word in [
        'forgets', 'forgotten', 'forgot', 'without', 'leaves',
        'omits', "hasn't", 'ignores', 'fails', 'misread',
        'misremember', 'not realised', 'not noticed'
    ]):
        return 'Memory/Attention Error'
    
    # Symbol/Notation & Representation Errors
    elif any(word in text for word in [
        'writes', 'repeats', 'includes', 'puts', 'describes',
        'uses', 'substitutes', 'changes', 'switches', 'applies',
        'instead of', 'rather than'
    ]):
        return 'Symbol/Notation Error'
    
    return 'Other Error'

def add_action_categories(df, text_column):
    """
    Adds action-based error classifications to a DataFrame.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing error descriptions
    text_column (str): Name of the column containing error descriptions
    
    Returns:
    pandas.DataFrame: Original DataFrame with new 'misconception_category' column
    """
    df['MisconceptionType'] = df[text_column].apply(classify_error_action)
    return df


# Apply segmentayion
misconception_df = add_action_categories(misconception_df, 'MisconceptionName')
misconception_df.groupby('MisconceptionType')['Misconception'].nunique()


import seaborn as sns
import matplotlib.pyplot as plt

# Count unique Misconception values for each misconception_category
category_counts = (
    misconception_df.groupby('MisconceptionType')['Misconception']
    .nunique()
    .reset_index()
    .rename(columns={'Misconception': 'UniqueMisconceptionCount'})
)

# Sort the data by unique misconception count in descending order
category_counts = category_counts.sort_values(by='UniqueMisconceptionCount', ascending=False)

# Set figure size
plt.figure(figsize=(6, 5))

# Create the bar plot
sns.barplot(
    data=category_counts,
    x='MisconceptionType',
    y='UniqueMisconceptionCount',
    color='skyblue'  # Single consistent color
)

# Customize the plot with English titles and labels
plt.title('Distribution of Misconception Types by Number of Misconceptions', fontsize=16)
plt.xlabel('Misconception Type', fontsize=12)
plt.ylabel('Number of Misconceptions', fontsize=12)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Add value labels on top of bars
for i, row in enumerate(category_counts.itertuples()):
    plt.text(i, row.UniqueMisconceptionCount, str(row.UniqueMisconceptionCount), ha='center', va='bottom', fontsize=10)

# Add grid lines
# plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()



# Create a crosstab between misconception_category and SubjectMainCategory
heatmap_data = pd.crosstab(
    misconception_df['MisconceptionType'],
    misconception_df['SubjectMainCategory']
)

# Normalize the crosstab to percentages by column
heatmap_data_percentage = heatmap_data.div(heatmap_data.sum(axis=0), axis=1) * 100

# Set figure size
plt.figure(figsize=(8, 6))

# Plot the heatmap
sns.heatmap(
    heatmap_data_percentage,
    annot=True,  # Annotate cells with the numeric value
    fmt='.1f',   # Display values as percentages with one decimal point
    cmap='YlGnBu',  # Color map
    linewidths=0.5,  # Add gridlines
    cbar_kws={'label': 'Percentage (%)'}  # Color bar label
)

# Customize the plot with English titles and labels
plt.title('Percentage Distribution of MisconceptionType and SubjectMainCategory', fontsize=16)
plt.xlabel('SubjectMainCategory Type', fontsize=12)
plt.ylabel('MisconceptionType Category', fontsize=12)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Show the plot
plt.tight_layout()
plt.show()



# Create a crosstab between misconception_category and ConstructActionType
heatmap_data = pd.crosstab(
    misconception_df['MisconceptionType'],
    misconception_df['ConstructActionType']
)

# Normalize the crosstab to percentages by column
heatmap_data_percentage = heatmap_data.div(heatmap_data.sum(axis=0), axis=1) * 100

# Set figure size
plt.figure(figsize=(8, 6))

# Plot the heatmap
sns.heatmap(
    heatmap_data_percentage,
    annot=True,  # Annotate cells with the numeric value
    fmt='.1f',   # Display values as percentages with one decimal point
    cmap='YlGnBu',  # Color map
    linewidths=0.5,  # Add gridlines
    cbar_kws={'label': 'Percentage (%)'}  # Color bar label
)

# Customize the plot with English titles and labels
plt.title('Percentage Distribution of MisconceptionType and ConstructActionType', fontsize=16)
plt.xlabel('ConstructActionType Type', fontsize=12)
plt.ylabel('MisconceptionType Category', fontsize=12)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Show the plot
plt.tight_layout()
plt.show()


