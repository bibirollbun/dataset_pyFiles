# import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import display, HTML
import io
from itertools import combinations
import gc
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
# ignore warnings
import warnings
warnings.filterwarnings('ignore')
from tqdm.notebook import tqdm
warnings.filterwarnings('ignore', category=UserWarning)


# Vibrant color palette with lighter shades
color_palette = ["#FF5733", "#FF8A66",  "#33FF57", "#66FF88", "#3357FF", "#6690FF",  "#FF33A6", "#FF66C2",  
"#FFC733", "#FFE066", "#33FFF2", "#66FFF7",  "#8D33FF", "#A566FF",  "#FF8D33", "#FFA966",  "#33FF8D", "#66FFAA",  
"#FF3333", "#FF6666",  "#338DFF", "#66A6FF"]

# Main heading style
def styled_main_heading(text):
    return f"""
    <div style="
        text-align: center;
        background-image: linear-gradient(to right, {color_palette[0]}, {color_palette[2]});
        color: #FFFFFF;
        padding: 20px;
        font-family: 'Montserrat', sans-serif;
        font-size: 28px;
        font-weight: 800;
        border-radius: 12px;
        margin: 20px 0;
        box-shadow: 0 8px 15px rgba(0,0,0,0.3);
        letter-spacing: 1.5px;
        border-bottom: 5px solid {color_palette[4]};
        border-top: 5px solid {color_palette[6]};
    ">
        {text}
    </div>
    """

# Sub-heading style
sub_heading_colors = [color_palette[4], color_palette[6], color_palette[8], color_palette[10]]
def styled_sub_heading(text, idx):
    color_accent = sub_heading_colors[idx % len(sub_heading_colors)]
    return f"""
    <h3 style="
        font-size: 20px;
        color: #000000;
        background-color: {color_palette[1]};
        padding: 10px 20px;
        border-radius: 10px;
        margin: 20px 0 10px;
        text-align: left;
        font-family: 'Montserrat', sans-serif;
        border-left: 6px solid {color_accent};
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        display: flex;
        align-items: center;
        font-weight: 600;
    ">
        <span style="font-size: 1.2em; margin-right: 10px; color: {color_accent};">&#9679;</span> {text}
    </h3>
    """

# Table styling
def style_table(df):
    styled_df = df.style.set_table_styles([
        {"selector": "th", "props": [
            ("background-color", color_palette[2]),
            ("color", "#FFFFFF"),
            ("padding", "8px"),
            ("font-family", "'Open Sans', sans-serif"),
            ("font-size", "14px"),
            ("text-align", "center"),
            ("border-bottom", f"2px solid {color_palette[4]}")
        ]},
        {"selector": "td", "props": [
            ("background-color", color_palette[3]),
            ("color", "#000000"),
            ("padding", "8px"),
            ("text-align", "center"),
            ("font-family", "'Open Sans', sans-serif"),
            ("font-size", "12px"),
            ("border-bottom", "1px solid #C0C0C0")
        ]},
        {"selector": "tr:nth-child(even)", "props": [("background-color", color_palette[5])]},
        {"selector": "table", "props": [
            ("width", "80%"),
            ("border-collapse", "collapse"),
            ("border-radius", "8px"),
            ("overflow", "hidden"),
            ("box-shadow", "0 4px 15px rgba(0,0,0,0.1)")
        ]},
        {"selector": "tbody tr:hover", "props": [("background-color", color_palette[7])]}
    ]).hide(axis="index")
    return styled_df.to_html()

# Dataset analysis function
def print_dataset_analysis(dataset, dataset_name):
    display(HTML(styled_main_heading(f"📊 {dataset_name} Overview")))
    
    def show_subsection(title, content_html=None, idx=0):
        display(HTML(styled_sub_heading(title, idx)))
        if content_html:
            display(HTML(f"<div style='font-family: \"Open Sans\", sans-serif; color: #000000; margin-bottom: 15px; line-height: 1.5;'>{content_html}</div>"))
    
    show_subsection("🔍 Shape of the Dataset", f"<p>This dataset contains <strong>{dataset.shape[0]} rows</strong> and <strong>{dataset.shape[1]} columns</strong>.</p>", 0)
    show_subsection("👀 First 5 Rows", style_table(dataset.head()), 1)
    show_subsection("📈 Summary Statistics", style_table(dataset.describe()), 2)
    
    # Null values
    show_subsection("🚨 Null Values", idx=3)
    null_counts = dataset.isnull().sum()
    if null_counts.sum() == 0:
        display(HTML(f"<p style='color: #000000; font-family: \"Open Sans\", sans-serif;'>No null values found!</p>"))
    else:
        null_df = null_counts[null_counts > 0].to_frame(name='Null Values')
        null_df['Column'] = null_df.index
        display(HTML(style_table(null_df)))
    
    # Duplicates
    show_subsection("🔍 Duplicate Rows", f"<p>A total of <strong>{dataset.duplicated().sum()} duplicate rows</strong> were identified.</p>", 0)
    
    # Data types
    show_subsection("📝 Data Types", style_table(pd.DataFrame({
        'Column Name': dataset.columns,
        'Data Type': [dataset[col].dtype for col in dataset.columns]
    })), 1)
    
    # Column names
    show_subsection("📋 Column Names", f"<p style='word-break: break-all;'>{', '.join([f'<code>{col}</code>' for col in dataset.columns])}</p>", 2)
    
    # Unique values
    show_subsection("🔢 Unique Values per Column", style_table(pd.DataFrame({
        'Column Name': dataset.columns,
        'Data Type': [dataset[col].dtype for col in dataset.columns],
        'Unique Values Count': [dataset[col].nunique() for col in dataset.columns]
    })), 3)
    
    # Dataset info
    show_subsection("ℹ️ Detailed Dataset Information", idx=0)
    buffer = io.StringIO()
    dataset.info(buf=buffer)
    display(HTML(f"<pre style='background-color: {color_palette[0]}; color: #FFFFFF; padding: 15px; border-radius: 8px; font-family: \"Fira Code\", monospace; font-size: 13px; overflow-x: auto; box-shadow: 0 3px 12px rgba(0,0,0,0.2); line-height: 1.4;'>{buffer.getvalue()}</pre>"))

# Load datasets
print("🚀 Initializing Dataset Visual Analysis...")
df_train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv', index_col='id')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv', index_col='id')
df_original = pd.read_csv('/kaggle/input/bank-marketing-dataset-full/bank-full.csv', delimiter=';')
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")

# Display analysis
print_dataset_analysis(df_train, "Training Data")
print_dataset_analysis(df_test, "Test Data")
print_dataset_analysis(sample_submission, "Sample Submission")
print_dataset_analysis(df_original, "Original Bank Marketing Dataset")



age = df_train['age']
print(f"Average (mean) age according to training Data is: {age.mean():.2f} years")
print(f"Median age according to training Data is: {age.median():.2f} years")
print(f"Minimum age of people according to training Data is: {age.min()} years")
print(f"Maximum age of people according to training Data is: {age.max()} years")
print(f"Age Range of people Training Data is: {age.max() - age.min()} years\n")


# Frequency Distribution of Jobs
print("Frequency Distribution of the Jobs")
job_counts = df_train['job'].value_counts().rename_axis('Job').reset_index(name='Count')
display(job_counts)

print("========================================")
most_frequent_job = df_train['job'].mode()[0]
least_frequent_jobs = df_train['job'].value_counts(ascending=True).rename_axis('Job').reset_index(name='Count')
print(f"The most common job category is {most_frequent_job}")
print("========================================")

# Univariate Analysis: Job Percentage
job_percent = df_train['job'].value_counts(normalize=True).rename_axis('Job').reset_index(name='Percentage')
job_percent['Percentage'] = job_percent['Percentage'] * 100
print("\nJob Percentage:\n")
display(job_percent)



# Data
job_counts = df_train['job'].value_counts()
n_jobs = len(job_counts)

# Custom prominent and attractive color palette
custom_colors = ["#FF5733","#33FF57","#3357FF","#FF33A6","#FFC733","#33FFF2","#8D33FF","#FF8D33","#33FF8D","#FF3333","#338DFF"]

# Row-wise layout with increased figure height
fig, axes = plt.subplots(2, 1, figsize=(22, 26))  # taller figure

# Bar Chart
colors_bar = [custom_colors[i % len(custom_colors)] for i in range(n_jobs)]
bars = axes[0].bar(job_counts.index, job_counts.values, color=colors_bar, edgecolor='black', linewidth=1.5)

axes[0].set_title("Frequency Distribution of Jobs", fontsize=22, fontweight='bold')
axes[0].set_xlabel("Job Categories", fontsize=16, fontweight='bold')
axes[0].set_ylabel("Count", fontsize=16, fontweight='bold')
axes[0].tick_params(axis='x', rotation=90, labelsize=14)
axes[0].tick_params(axis='y', labelsize=14)
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

# Add bar value labels
for bar in bars:
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2, height + max(job_counts.values)*0.01,
                 f'{height}', ha='center', fontsize=12, fontweight='bold')

# Pie Chart
top_n = 10
top_jobs = job_counts.head(top_n)
others_count = job_counts[top_n:].sum()
pie_counts = list(top_jobs) + [others_count]
pie_labels = list(top_jobs.index) + ['Others']
colors_pie = [custom_colors[i % len(custom_colors)] for i in range(top_n)] + ['#CCCCCC']  # Gray for Others

explode = [0.05]*len(pie_counts)  # small gap between slices

axes[1].pie(
    pie_counts,
    labels=pie_labels,
    colors=colors_pie,
    autopct=lambda p: '{:.1f}%'.format(p) if p > 1 else '',  # show only >1%
    startangle=140,
    wedgeprops={'edgecolor':'black', 'linewidth':2},
    explode=explode,
    shadow=True,
    radius=1.4,  # larger pie
    textprops={'fontsize': 13, 'fontweight':'bold'}  # prominent inside values
)

# Move pie chart title much higher above chart
axes[1].set_title("Job Category Proportion", fontsize=22, fontweight='bold', y=1.2)

# Increase vertical gap between bar and pie chart
plt.subplots_adjust(hspace=0.6)

plt.show()



avg_age_by_job = df_train.groupby('job')['age'].mean().sort_values()
print("Average age by job category (sorted by age):")
print(avg_age_by_job.to_string())
print("\n")

age_range_by_job = df_train.groupby('job')['age'].apply(lambda x: x.max() - x.min()).sort_values(ascending=False)
print("Age range within job categories (sorted by range, largest first):")
print(age_range_by_job)


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Data
avg_age_by_job = df_train.groupby('job')['age'].mean().sort_values()
age_range_by_job = df_train.groupby('job')['age'].apply(lambda x: x.max() - x.min()).sort_values(ascending=False)

# Custom prominent color palette
custom_colors = ["#FF5733","#33FF57","#3357FF","#FF33A6","#FFC733","#33FFF2","#8D33FF","#FF8D33","#33FF8D","#FF3333","#338DFF"]

# Create Subplots
fig, axes = plt.subplots(2, 1, figsize=(22, 24))

# Bar Chart 1: Average Age by Job
colors_avg = [custom_colors[i % len(custom_colors)] for i in range(len(avg_age_by_job))]
bars1 = axes[0].bar(avg_age_by_job.index, avg_age_by_job.values, color=colors_avg, edgecolor='black', linewidth=1.5)

axes[0].set_title("Average Age by Job Category", fontsize=22, fontweight='bold')
axes[0].set_xlabel("Job Categories", fontsize=16, fontweight='bold')
axes[0].set_ylabel("Average Age", fontsize=16, fontweight='bold')
axes[0].tick_params(axis='x', rotation=90, labelsize=14)
axes[0].tick_params(axis='y', labelsize=14)
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

# Annotate bar values
for bar in bars1:
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2, height + 0.3,
                 f'{height:.1f}', ha='center', fontsize=12, fontweight='bold')

# Bar Chart 2: Age Range by Job
colors_range = [custom_colors[i % len(custom_colors)] for i in range(len(age_range_by_job))]
bars2 = axes[1].bar(age_range_by_job.index, age_range_by_job.values, color=colors_range, edgecolor='black', linewidth=1.5)

axes[1].set_title("Age Range within Job Categories", fontsize=22, fontweight='bold')
axes[1].set_xlabel("Job Categories", fontsize=16, fontweight='bold')
axes[1].set_ylabel("Age Range (Max - Min)", fontsize=16, fontweight='bold')
axes[1].tick_params(axis='x', rotation=90, labelsize=14)
axes[1].tick_params(axis='y', labelsize=14)
axes[1].grid(axis='y', linestyle='--', alpha=0.7)

# Annotate bar values
for bar in bars2:
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2, height + 0.3,
                 f'{height:.1f}', ha='center', fontsize=12, fontweight='bold')

# Adjust spacing
plt.subplots_adjust(hspace=0.5)

plt.show()



# Counts with column names
marital_counts_df = df_train['marital'].value_counts().rename_axis('Marital Status').reset_index(name='Count')
print("Marital Status Counts:")
display(marital_counts_df)

# Percentage with column names
marital_percent_df = df_train['marital'].value_counts(normalize=True).rename_axis('Marital Status').reset_index(name='Percentage')
marital_percent_df['Percentage'] = marital_percent_df['Percentage'] * 100
print("\nMarital Status Percentage:")
display(marital_percent_df)



# Counts and Percentages
marital_counts = df_train['marital'].value_counts()
marital_percent = df_train['marital'].value_counts(normalize=True) * 100

# Custom prominent color palette
custom_colors = ["#FF5733","#33FF57","#3357FF","#FF33A6","#FFC733","#33FFF2","#8D33FF","#FF8D33","#33FF8D","#FF3333","#338DFF"]
colors = [custom_colors[i % len(custom_colors)] for i in range(len(marital_counts))]
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Left: Bar chart counts
bars1 = axes[0].bar(marital_counts.index, marital_counts.values, color=colors, edgecolor='black', linewidth=1.5)
axes[0].set_title("Marital Status Counts", fontsize=18, fontweight='bold', pad=20)
axes[0].set_xlabel("Marital Status", fontsize=14, fontweight='bold')
axes[0].set_ylabel("Count", fontsize=14, fontweight='bold')
axes[0].tick_params(axis='x', rotation=0, labelsize=12)
axes[0].tick_params(axis='y', labelsize=12)
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

# Annotate bar values
for bar in bars1:
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2, height + max(marital_counts.values)*0.01,
                 f'{height}', ha='center', fontsize=12, fontweight='bold')

# Right: Pie chart percentages
explode = [0.05]*len(marital_percent)
axes[1].pie(
    marital_percent.values,
    labels=marital_percent.index,
    colors=colors,
    autopct=lambda p: '{:.1f}%'.format(p) if p > 0 else '',
    startangle=140,
    wedgeprops={'edgecolor':'black', 'linewidth':2},
    explode=explode,
    shadow=True,
    radius=1.2,
    textprops={'fontsize': 14, 'fontweight':'bold'}
)
axes[1].set_title("Marital Status Proportion", fontsize=18, fontweight='bold', y=1.15)

# Adjust spacing between plots
plt.subplots_adjust(wspace=0.4)
plt.show()



# Job vs Marital Status analysis
job_marital_counts = pd.crosstab(df_train['job'], df_train['marital']).reset_index()
job_marital_counts.rename_axis(None, axis=1, inplace=True)  # Remove index name for cleaner display
print("Job vs Marital Status Counts:")
display(job_marital_counts)

# Percentages (percentage distribution within each job)
job_marital_percent = pd.crosstab(df_train['job'], df_train['marital'], normalize='index') * 100
job_marital_percent = job_marital_percent.reset_index()
job_marital_percent.rename_axis(None, axis=1, inplace=True)
print("\nJob vs Marital Status Percentage (% of each Job category):")
display(job_marital_percent)



# Custom prominent color palette
custom_colors = ["#FF5733","#33FF57","#3357FF","#FF33A6","#FFC733","#33FFF2","#8D33FF","#FF8D33","#33FF8D","#FF3333","#338DFF"]

plt.figure(figsize=(28, 15))
ax = sns.countplot(
    data=df_train,
    x='job',
    hue='marital',
    palette=custom_colors[:df_train['marital'].nunique()],
    edgecolor='black'
)

# Title and labels
plt.title("Job vs Marital Status (Counts)", fontsize=20, fontweight='bold', pad=20)
plt.xlabel("Job", fontsize=16, fontweight='bold')
plt.ylabel("Count", fontsize=16, fontweight='bold')

# Tick labels
plt.xticks(rotation=45, ha='right', fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')

# Legend
plt.legend(title="Marital Status", fontsize=12, title_fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Annotate count values above bars with a point and smaller font
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f'{height}.',
                    (p.get_x() + p.get_width() / 2, height),
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()



# Education vs Job Analysis
edu_job = pd.crosstab(df_train['education'], df_train['job'], normalize='index') * 100
print("Job Distribution per Education Level (%):")
display(edu_job)



# Compute percentage distribution of jobs per education level
edu_job = pd.crosstab(df_train['education'], df_train['job'], normalize='index') * 100
edu_levels = edu_job.index
job_categories = edu_job.columns
n_edu = len(edu_levels)
n_jobs = len(job_categories)

# Custom prominent color palette
custom_colors = ["#FF5733","#33FF57","#3357FF","#FF33A6","#FFC733","#33FFF2","#8D33FF","#FF8D33","#33FF8D","#FF3333","#338DFF"]
colors = [custom_colors[i % len(custom_colors)] for i in range(n_jobs)]
fig, axes = plt.subplots(n_edu, 1, figsize=(26, 6*n_edu))  # Increased figure size

# Ensure 'axes' is an array even if n_edu is 1
if n_edu == 1:
    axes = [axes]

for i, edu in enumerate(edu_levels):
    values = edu_job.loc[edu].values
    bars = axes[i].bar(job_categories, values, color=colors, edgecolor='black', linewidth=1.5)
    
    # Annotate percentage above bars
    for j, bar in enumerate(bars):
        height = bar.get_height()
        if height > 0:
            axes[i].text(bar.get_x() + bar.get_width()/2, height + 0.5,
                         f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    axes[i].set_title(f"Job Distribution for Education Level: {edu}", fontsize=18, fontweight='bold', pad=14)
    axes[i].set_ylabel("Percentage (%)", fontsize=14, fontweight='bold')
    axes[i].set_ylim(0, max(values)*1.25 if len(values) > 0 else 10)  # add more space above
    
    axes[i].tick_params(axis='x', rotation=45, labelsize=13) 
    axes[i].tick_params(axis='y', labelsize=13)
    axes[i].grid(axis='y', linestyle='--', alpha=0.7)

# Set common xlabel for the entire figure
fig.text(0.5, 0.02, "Job Categories", ha='center', va='center', fontsize=16, fontweight='bold')

plt.tight_layout(rect=[0, 0.04, 1, 1])  # Adjust layout to make space for xlabel
plt.show()



# Counts Crosstab
from itertools import combinations

categorical_cols = ['education', 'default', 'housing', 'loan']
cat_pairs = list(combinations(categorical_cols, 2))

for pair in cat_pairs:
    ctab_counts = pd.crosstab(df_train[pair[0]], df_train[pair[1]]).reset_index()
    print(f"\nCounts Crosstab: {pair[0]} vs {pair[1]}")
    display(ctab_counts)
    print("===========================================================")



# Custom prominent color palette
custom_colors = ["#FF5733","#33FF57","#3357FF","#FF33A6","#FFC733","#33FFF2","#8D33FF","#FF8D33","#33FF8D","#FF3333","#338DFF"]

categorical_cols = ['education', 'default', 'housing', 'loan']
cat_pairs = list(combinations(categorical_cols, 2))

for pair in cat_pairs:
    ctab_counts = pd.crosstab(df_train[pair[0]], df_train[pair[1]])
    
    # Column-wise layout: bar chart + pie chart
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # --- Left: Grouped bar chart ---
    ctab_counts.plot(kind='bar', ax=axes[0], color=custom_colors[:len(ctab_counts.columns)], 
                     edgecolor='black', linewidth=1.5, legend=True)
    axes[0].set_title(f"{pair[0].title()} vs {pair[1].title()} (Counts)", fontsize=18, fontweight='bold', pad=20)
    axes[0].set_xlabel(pair[0].title(), fontsize=14, fontweight='bold')
    axes[0].set_ylabel("Count", fontsize=14, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=0, labelsize=12)
    axes[0].tick_params(axis='y', labelsize=12)
    axes[0].legend(title=pair[1].title(), fontsize=12, title_fontsize=14)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Annotate bar values
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt='%d', label_type='edge', fontsize=10, fontweight='bold')
    
    # --- Right: Pie chart for the first category ---
    pie_counts = ctab_counts.sum(axis=1)
    explode = [0.05]*len(pie_counts)
    axes[1].pie(
        pie_counts.values,
        labels=pie_counts.index,
        colors=custom_colors[:len(pie_counts)],
        autopct=lambda p: '{:.1f}%'.format(p) if p > 0 else '',
        startangle=140,
        wedgeprops={'edgecolor':'black', 'linewidth':2},
        explode=explode,
        shadow=True,
        radius=1.2,
        textprops={'fontsize':12, 'fontweight':'bold'}
    )
    axes[1].set_title(f"{pair[0].title()} Distribution Across {pair[1].title()}", 
                      fontsize=18, fontweight='bold', y=1.1)
    
    plt.tight_layout()
    plt.show()



for pair in cat_pairs:
    ctab_percent = pd.crosstab(df_train[pair[0]], df_train[pair[1]], normalize='index') * 100
    ctab_percent = ctab_percent.reset_index()
    print(f"\nPercentage Crosstab: {pair[0]} vs {pair[1]} (% of {pair[0]} category)")
    display(ctab_percent)
    print("===========================================================")



# Custom prominent color palette
custom_colors = ["#FF5733","#33FF57","#3357FF","#FF33A6","#FFC733","#33FFF2","#8D33FF","#FF8D33","#33FF8D","#FF3333","#338DFF"]

categorical_cols = ['education', 'default', 'housing', 'loan']
cat_pairs = list(combinations(categorical_cols, 2))

for pair in cat_pairs:
    # percentage crosstab
    ctab_percent = pd.crosstab(df_train[pair[0]], df_train[pair[1]], normalize='index') * 100

    # Bar chart & Pie chart
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Grouped bar chart of percentages
    ctab_percent.plot(kind='bar', ax=axes[0], color=custom_colors[:len(ctab_percent.columns)], 
                      edgecolor='black', linewidth=1.5, legend=True)
    axes[0].set_title(f"{pair[0].title()} vs {pair[1].title()} (% within {pair[0]})", 
                      fontsize=18, fontweight='bold', pad=20)
    axes[0].set_xlabel(pair[0].title(), fontsize=14, fontweight='bold')
    axes[0].set_ylabel("Percentage (%)", fontsize=14, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=0, labelsize=12)
    axes[0].tick_params(axis='y', labelsize=12)
    axes[0].legend(title=pair[1].title(), fontsize=12, title_fontsize=14)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Annotate bars with percentage values
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt='%.1f%%', label_type='edge', fontsize=10, fontweight='bold')
    
    # Pie chart for overall percentage of first category
    pie_counts = ctab_percent.mean(axis=1)  # mean percentage across second category
    explode = [0.05]*len(pie_counts)
    axes[1].pie(
        pie_counts.values,
        labels=pie_counts.index,
        colors=custom_colors[:len(pie_counts)],
        autopct=lambda p: '{:.1f}%'.format(p) if p > 0 else '',
        startangle=140,
        wedgeprops={'edgecolor':'black', 'linewidth':2},
        explode=explode,
        shadow=True,
        radius=1.2,
        textprops={'fontsize':12, 'fontweight':'bold'}
    )
    axes[1].set_title(f"Average % Distribution of {pair[0].title()} Across {pair[1].title()}", 
                      fontsize=18, fontweight='bold', y=1.05)
    
    plt.tight_layout()
    plt.show()



# Numerical columns to summarize
numerical_cols = ['age', 'balance']

categorical_cols = ['education', 'default', 'housing', 'loan']

for col in categorical_cols:
    avg_df = df_train.groupby(col)[numerical_cols].mean().reset_index()
    print(f"\nAverage values of numerical columns by {col}:")
    display(avg_df)
    print("===========================================================")



# Custom prominent color palette
custom_colors = ["#FF5733","#33FF57","#3357FF","#FF33A6","#FFC733","#33FFF2","#8D33FF","#FF8D33","#33FF8D","#FF3333","#338DFF"]

numerical_cols = ['age', 'balance']
categorical_cols = ['education', 'default', 'housing', 'loan']

# Create subplots grid
fig, axes = plt.subplots(len(categorical_cols), len(numerical_cols), figsize=(16, 16))
fig.suptitle("Average Age & Balance by Categorical Variables", fontsize=18, fontweight='bold')

for i, cat_col in enumerate(categorical_cols):
    avg_df = df_train.groupby(cat_col)[numerical_cols].mean().reset_index()
    
    for j, num_col in enumerate(numerical_cols):
        ax = axes[i, j]
        
        # Barplot
        sns.barplot(
            data=avg_df,
            x=cat_col, y=num_col,
            palette=custom_colors,
            ax=ax
        )
        
        # Add values above bars
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.1f}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom',
                        fontsize=9, fontweight='bold', color="black")
        
        # Formatting
        ax.set_title(f"{num_col} by {cat_col}", fontsize=14, fontweight="bold")
        ax.set_xlabel(cat_col.capitalize(), fontsize=12, fontweight="bold")
        ax.set_ylabel(f"Avg {num_col}", fontsize=12, fontweight="bold")
        ax.tick_params(axis='x', rotation=25, labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.grid(axis='y', linestyle="--", alpha=0.6)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()



numeric_cols = ['day', 'duration', 'campaign', 'pdays', 'previous']

for col in numeric_cols:
    data = df_train[col]
    print(f"{col} Summary:")
    print("Mean:", round(data.mean(),2))
    print("Median:", round(data.median(),2))
    print("Min:", data.min())
    print("Max:", data.max())
    print("======================================")



categorical_cols = ['education', 'default', 'housing', 'loan']

for cat in categorical_cols:
    avg_df = df_train.groupby(cat)[['duration', 'campaign', 'previous']].mean().reset_index()
    print(f"Average duration, campaign, previous by {cat}:")
    display(avg_df)
    print("===================================================================")



import matplotlib.pyplot as plt
import seaborn as sns

# Custom prominent color palette
custom_colors = ["#FF5733","#33FF57","#3357FF","#FF33A6","#FFC733","#33FFF2","#8D33FF","#FF8D33","#33FF8D","#FF3333","#338DFF"]

categorical_cols = ['education', 'default', 'housing', 'loan']
numerical_cols = ['duration', 'campaign', 'previous']

# Set style
sns.set(style="whitegrid")

# BARPLOTS
fig, axes = plt.subplots(len(categorical_cols), len(numerical_cols), 
                         figsize=(22, 20), constrained_layout=True)

for i, cat in enumerate(categorical_cols):
    for j, num in enumerate(numerical_cols):
        
        # --- Barplot of mean values ---
        avg_df = df_train.groupby(cat)[num].mean().reset_index()
        sns.barplot(data=avg_df, x=cat, y=num, palette=custom_colors, ax=axes[i, j])
        
        # Annotate bars
        for p in axes[i, j].patches:
            axes[i, j].annotate(format(p.get_height(), '.1f'),
                                (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='bottom', xytext=(0, 6), 
                                textcoords='offset points', fontsize=10, fontweight='bold')
        
        # Titles and labels
        axes[i, j].set_title(f"Avg {num} by {cat}", fontsize=16, fontweight='bold', pad=15)
        axes[i, j].set_xlabel(cat, fontsize=13, fontweight='bold')
        axes[i, j].set_ylabel(f"Avg {num}", fontsize=13, fontweight='bold')
        axes[i, j].tick_params(axis='x', rotation=25, labelsize=11)
        axes[i, j].tick_params(axis='y', labelsize=11)

plt.suptitle("Categorical vs Numerical Insights (Barplots)", fontsize=20, fontweight='bold', y=1.02)
plt.show()


# BOXPlots for duration
fig, axes = plt.subplots(1, len(categorical_cols), figsize=(24, 7), constrained_layout=True)

for i, cat in enumerate(categorical_cols):
    sns.boxplot(data=df_train, x=cat, y='duration', palette=custom_colors, ax=axes[i])
    
    axes[i].set_title(f"Distribution of Duration by {cat}", fontsize=16, fontweight='bold', pad=15)
    axes[i].set_xlabel(cat, fontsize=13, fontweight='bold')
    axes[i].set_ylabel("Duration", fontsize=13, fontweight='bold')
    axes[i].tick_params(axis='x', rotation=25, labelsize=11)
    axes[i].tick_params(axis='y', labelsize=11)

plt.suptitle("Distribution of Duration Across Categories (Boxplots)", fontsize=20, fontweight='bold', y=1.05)
plt.show()



# Contact type vs loan
print("Contact type vs Loan status:")
display(pd.crosstab(df_train['contact'], df_train['loan']))
print("===================================================================")
# Month vs Previous outcome
print("Month vs Previous Campaign Outcome (poutcome):")
display(pd.crosstab(df_train['month'], df_train['poutcome']))



import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Custom colors
custom_colors = ["#FF5733","#33FF57","#3357FF","#FF33A6","#FFC733","#33FFF2","#8D33FF","#FF8D33","#33FF8D","#FF3333","#338DFF"]

print("Month vs Previous Campaign Outcome (poutcome):")

# Subplots for Countplots & Boxplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

# Countplot: Contact vs Loan
sns.countplot(data=df_train, x="contact", hue="loan", palette=custom_colors, ax=axes[0])
axes[0].set_title("Contact Type vs Loan Status (Countplot)", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Contact Type")
axes[0].set_ylabel("Count")

# Boxplot: Contact vs Duration
sns.boxplot(data=df_train, x="contact", y="duration", palette=custom_colors, ax=axes[1])
axes[1].set_title("Contact Type vs Duration (Boxplot)", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Contact Type")
axes[1].set_ylabel("Duration")

# Countplot: Month vs Poutcome
sns.countplot(data=df_train, x="month", hue="poutcome", palette=custom_colors, ax=axes[2])
axes[2].set_title("Month vs Previous Campaign Outcome (Countplot)", fontsize=12, fontweight="bold")
axes[2].set_xlabel("Month")
axes[2].set_ylabel("Count")
axes[2].tick_params(axis='x', rotation=45)

# Boxplot: Month vs Campaign
sns.boxplot(data=df_train, x="month", y="campaign", palette=custom_colors, ax=axes[3])
axes[3].set_title("Month vs Campaign (Boxplot)", fontsize=12, fontweight="bold")
axes[3].set_xlabel("Month")
axes[3].set_ylabel("Campaign")
axes[3].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Scatter Plots
scatter_features = ["age", "balance", "day"]
target_col = "duration"

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes = axes.flatten()

for i, col in enumerate(scatter_features):
    sns.scatterplot(
        data=df_train,
        x=col,
        y=target_col,
        ax=axes[i],
        color=custom_colors[i % len(custom_colors)],
        alpha=0.7,
        edgecolor="k"
    )
    axes[i].set_title(f"{col} vs {target_col}", fontsize=12, fontweight="bold")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel(target_col)

plt.tight_layout()
plt.show()



# Average duration for clients with loan vs without
print("Average call duration by loan status:")
display(df_train.groupby('loan')['duration'].mean())
print("===================================================================")
# Days since last contact for previous successes vs failures
print("Average pdays by previous outcome:")
display(df_train.groupby('poutcome')['pdays'].mean())



# Counts and percentages together for target column 'y'
target_counts = df_train['y'].value_counts().reset_index()
target_counts.columns = ['y', 'Count']

target_percent = df_train['y'].value_counts(normalize=True).reset_index(drop=True) * 100
target_counts['Percentage'] = target_percent

print("Counts and Percentage of target column y:")
display(target_counts)



# Custom color palette
custom_colors = ["#FF5733","#33FF57","#3357FF","#FF33A6","#FFC733","#33FFF2","#8D33FF","#FF8D33","#33FF8D","#FF3333","#338DFF"]

# Create figure
fig, axes = plt.subplots(3, 2, figsize=(16, 15))
fig.suptitle("Bank Dataset Insights", fontsize=18, fontweight="bold", color="#333")

# Average call duration by loan status
duration_means = df_train.groupby('loan')['duration'].mean().reset_index()

# Bar plot
sns.barplot(data=duration_means, x='loan', y='duration', ax=axes[0,0],
            palette=custom_colors[:2])
axes[0,0].set_title("Average Call Duration by Loan Status", fontsize=14, fontweight="bold")

# Box plot
sns.boxplot(data=df_train, x='loan', y='duration', ax=axes[0,1],
            palette=custom_colors[:2])
axes[0,1].set_title("Distribution of Call Duration by Loan Status", fontsize=14, fontweight="bold")

# Average pdays by previous outcome
pdays_means = df_train.groupby('poutcome')['pdays'].mean().reset_index()

# Bar plot
sns.barplot(data=pdays_means, x='poutcome', y='pdays', ax=axes[1,0],
            palette=custom_colors[:4])
axes[1,0].set_title("Average Pdays by Previous Outcome", fontsize=14, fontweight="bold")

sns.boxplot(data=df_train, x='poutcome', y='pdays', ax=axes[1,1],
            palette=custom_colors[:4])
axes[1,1].set_title("Distribution of Pdays by Previous Outcome", fontsize=14, fontweight="bold")

# Counts & percentages for target column 'y'
target_counts = df_train['y'].value_counts().reset_index()
target_counts.columns = ['y', 'Count']
target_counts['Percentage'] = (target_counts['Count'] / target_counts['Count'].sum()) * 100

# Count plot
sns.barplot(data=target_counts, x='y', y='Count', ax=axes[2,0],
            palette=custom_colors[:2])
axes[2,0].set_title("Counts of Target Column 'y'", fontsize=14, fontweight="bold")

# Pie chart
axes[2,1].pie(target_counts['Count'], labels=target_counts['y'],
              autopct='%1.1f%%', startangle=90,
              colors=custom_colors[:2], textprops={'fontsize': 12})
axes[2,1].set_title("Target 'y' Percentage Distribution", fontsize=14, fontweight="bold")

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()



print("Mean campaign features by target y:")
display(df_train.groupby('y')[['duration','campaign','pdays','previous']].mean().reset_index())


# Features of interest
features = ['campaign', 'pdays', 'previous']  # skip 'duration' itself

# Custom palette for binary target 'y'
custom_colors = ["#FF5733","#33FF57","#3357FF"]  # Red and Blue for distinction

# Create subplots
fig, axes = plt.subplots(1, len(features), figsize=(18, 6))  # wider for prominence

# ================= Scatterplots =================
for i, col in enumerate(features):
    sns.scatterplot(
        data=df_train, 
        x=col, 
        y="duration", 
        hue="y", 
        palette=custom_colors, 
        alpha=0.8, 
        s=80,  # bigger marker size for prominence
        edgecolor="k",  # black edge around points
        ax=axes[i]
    )
    axes[i].set_title(f"{col} vs Duration", fontsize=16, fontweight="bold")
    axes[i].set_xlabel(col, fontsize=14, fontweight="bold")
    axes[i].set_ylabel("Duration", fontsize=14, fontweight="bold")
    axes[i].tick_params(axis='x', labelsize=12)
    axes[i].tick_params(axis='y', labelsize=12)
    axes[i].legend(title="y", fontsize=12, title_fontsize=12)
    axes[i].grid(alpha=0.3)  # subtle grid for better readability

plt.suptitle("Scatterplots of Duration vs Other Features", fontsize=20, fontweight="bold", y=1.05)
plt.tight_layout()
plt.show()



# Configuration
TARGET_COL = 'y'
NUMERIC_FEATURES = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
CATEGORICAL_FEATURES = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

# Model Parameters
N_FOLDS = 10
N_SEEDS = 3
PSEUDO_LABEL_THRESHOLD_HIGH = 0.99
PSEUDO_LABEL_THRESHOLD_LOW = 0.01


# Map target
df_original[TARGET_COL] = df_original[TARGET_COL].map({'yes': 1, 'no': 0})

# Convert categorical
for col in CATEGORICAL_FEATURES:
    if col in df_original.columns and df_original[col].dtype == 'object':
        df_original[col] = df_original[col].astype('category')

# Combined DataFrame for feature engineering
df_combined_fe = pd.concat([
    df_train.drop(columns=[TARGET_COL]),
    df_test,
    df_original.drop(columns=[TARGET_COL])
], axis=0)

df_combined_fe[TARGET_COL] = pd.concat([
    df_train[TARGET_COL],
    pd.Series([-1]*len(df_test), index=df_test.index),
    df_original[TARGET_COL]
], axis=0)

print(f"Train: {df_train.shape}, Test: {df_test.shape}, Original: {df_original.shape}, Combined: {df_combined_fe.shape}")


print("Feature Engineering")

for col in NUMERIC_FEATURES:
    df_combined_fe[f'{col}_log1p'] = np.log1p(df_combined_fe[col])
    df_combined_fe[f'{col}_sqrt'] = np.sqrt(df_combined_fe[col].clip(lower=0))
    df_combined_fe[f'{col}_squared'] = df_combined_fe[col] ** 2
    
    if col == 'balance':
        df_combined_fe['balance_per_duration'] = np.clip(df_combined_fe['balance'] / (df_combined_fe['duration'] + 1e-6), -1e6, 1e6)
    if col == 'campaign':
        df_combined_fe['campaign_per_duration'] = np.clip(df_combined_fe['campaign'] / (df_combined_fe['duration'] + 1e-6), -1e6, 1e6)
    if col == 'pdays':
        df_combined_fe['pdays_contacted_ratio'] = np.clip(df_combined_fe['pdays'] / (df_combined_fe['previous'].replace(0,1) + 1e-6), -1e6, 1e6)
        df_combined_fe['pdays_missing'] = (df_combined_fe['pdays'] == -1).astype(int)

df_combined_fe['age_balance_interaction'] = np.clip(df_combined_fe['age'] * df_combined_fe['balance'], -1e9, 1e9)
df_combined_fe['duration_campaign_ratio'] = np.clip(df_combined_fe['duration'] / (df_combined_fe['campaign'] + 1e-6), -1e6, 1e6)
df_combined_fe['housing_loan_interaction'] = df_combined_fe['housing'].astype(str) + '_' + df_combined_fe['loan'].astype(str)
df_combined_fe['day_of_month_sin'] = np.sin(2 * np.pi * df_combined_fe['day'] / 31)
df_combined_fe['day_of_month_cos'] = np.cos(2 * np.pi * df_combined_fe['day'] / 31)
df_combined_fe['poutcome_encoded'] = df_combined_fe['poutcome'].map({'failure': -1, 'other': 0, 'success': 1, 'unknown': 0}).fillna(0)

CATEGORICAL_FEATURES_UPDATED = CATEGORICAL_FEATURES + ['housing_loan_interaction']

for col in CATEGORICAL_FEATURES_UPDATED:
    if col in df_combined_fe.columns:
        df_combined_fe[col] = df_combined_fe[col].astype('category').cat.codes
        if -1 in df_combined_fe[col].unique():
            df_combined_fe[col] = df_combined_fe[col].replace(-1, df_combined_fe[col].mode()[0])

for col in df_combined_fe.columns:
    if df_combined_fe[col].dtype in ['float64','int64']:
        df_combined_fe[col] = df_combined_fe[col].replace([np.inf,-np.inf], np.nan)
        if df_combined_fe[col].isnull().any():
            df_combined_fe[col] = df_combined_fe[col].fillna(df_combined_fe[col].median())

FINAL_FEATURES = [col for col in df_combined_fe.columns if col != TARGET_COL]
print(f"Total features: {len(FINAL_FEATURES)}, Example: {FINAL_FEATURES[:10]}")



print("Count & Target Encoding")

def get_target_encoded_features(df_train_te, df_val_te, df_test_te, df_org_te, feature_col, target_col, agg_type='mean', smooth=200):
    global_mean = df_train_te[target_col].mean()
    df_combined_for_stats = pd.concat([df_train_te[[feature_col,target_col]], df_org_te[[feature_col,target_col]]], axis=0)
    agg_stats = df_combined_for_stats.groupby(feature_col)[target_col].agg([agg_type,'count'])
    agg_stats.columns = ['agg_val','count']
    agg_stats[f'TE_{agg_type}_{feature_col}'] = ((agg_stats['agg_val']*agg_stats['count']) + (global_mean*smooth)) / (agg_stats['count']+smooth)
    te_map = agg_stats[f'TE_{agg_type}_{feature_col}'].to_dict()
    df_train_te[f'TE_{agg_type}_{feature_col}'] = df_train_te[feature_col].map(te_map).fillna(global_mean)
    df_val_te[f'TE_{agg_type}_{feature_col}'] = df_val_te[feature_col].map(te_map).fillna(global_mean)
    df_test_te[f'TE_{agg_type}_{feature_col}'] = df_test_te[feature_col].map(te_map).fillna(global_mean)
    df_org_te[f'TE_{agg_type}_{feature_col}'] = df_org_te[feature_col].map(te_map).fillna(global_mean)
    return df_train_te, df_val_te, df_test_te, df_org_te

def get_count_encoded_features(df_input, feature_col):
    counts = df_input[feature_col].value_counts()
    df_input[f'CE_{feature_col}'] = df_input[feature_col].map(counts).fillna(0).astype('int32')
    rank_map = counts.rank(method='dense', ascending=False).astype('int32').to_dict()
    df_input[f'RANK_{feature_col}'] = df_input[feature_col].map(rank_map).fillna(counts.max()+1).astype('int32')
    return df_input

df_tra_proc = df_combined_fe.iloc[:len(df_train)].copy()
df_test_proc = df_combined_fe.iloc[len(df_train):len(df_train)+len(df_test)].drop(columns=[TARGET_COL]).copy()
df_org_proc = df_combined_fe.iloc[len(df_train)+len(df_test):].copy()
df_org_proc[TARGET_COL] = df_original[TARGET_COL].values

print("Applying Count and Rank Encoding...")
df_temp_combined_features = df_combined_fe[[col for col in df_combined_fe.columns if col != TARGET_COL]].copy()
for col in tqdm(CATEGORICAL_FEATURES_UPDATED):
    if col in df_temp_combined_features.columns:
        df_temp_combined_features = get_count_encoded_features(df_temp_combined_features, col)

new_ce_features = [col for col in df_temp_combined_features.columns if col not in FINAL_FEATURES]

df_tra_proc = pd.concat([df_tra_proc, df_temp_combined_features.iloc[:len(df_train)][new_ce_features]], axis=1)
df_test_proc = pd.concat([df_test_proc, df_temp_combined_features.iloc[len(df_train):len(df_train)+len(df_test)][new_ce_features]], axis=1)
df_org_proc = pd.concat([df_org_proc, df_temp_combined_features.iloc[len(df_train)+len(df_test):][new_ce_features]], axis=1)

del df_temp_combined_features
gc.collect()
FINAL_FEATURES.extend(new_ce_features)
print(f"Final feature count: {len(FINAL_FEATURES)}")



print("Training Ensemble")

oof_predictions_xgb = np.zeros(len(df_train))
oof_predictions_lgbm = np.zeros(len(df_train))
test_predictions_xgb = np.zeros(len(df_test))
test_predictions_lgbm = np.zeros(len(df_test))

for seed_idx in range(N_SEEDS):
    print(f"\n--- Seed {seed_idx+1} ---")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42+seed_idx)
    fold_test_preds_xgb = np.zeros(len(df_test))
    fold_test_preds_lgbm = np.zeros(len(df_test))

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df_tra_proc, df_tra_proc[TARGET_COL])):
        print(f"\n=== Fold {fold_idx+1}/{N_FOLDS} ===")

        X_train_fold_te = df_tra_proc.iloc[train_idx].copy()
        X_val_fold_te = df_tra_proc.iloc[val_idx].copy()
        df_test_te = df_test_proc.copy()
        df_org_te = df_org_proc.copy()
        y_val_fold = X_val_fold_te[TARGET_COL]

        te_features_added_this_fold = []
        for col in tqdm(CATEGORICAL_FEATURES_UPDATED, desc="Target Encoding"):
            if col in X_train_fold_te.columns:
                X_train_fold_te, X_val_fold_te, df_test_te, df_org_te = get_target_encoded_features(
                    X_train_fold_te, X_val_fold_te, df_test_te, df_org_te, col, TARGET_COL)
                te_features_added_this_fold.append(f'TE_mean_{col}')

        X_train_fold = X_train_fold_te.drop(columns=[TARGET_COL])
        y_train_fold = X_train_fold_te[TARGET_COL]
        X_val_fold = X_val_fold_te.drop(columns=[TARGET_COL])
        X_original_fold = df_org_te.drop(columns=[TARGET_COL])
        y_original_fold = df_org_te[TARGET_COL]

        current_features = [f for f in FINAL_FEATURES if f in X_train_fold.columns]
        current_features.extend([f for f in te_features_added_this_fold if f not in current_features])

        X_train_combined = pd.concat([X_train_fold, X_original_fold], axis=0)
        y_train_combined = pd.concat([y_train_fold, y_original_fold], axis=0)

        # --- Pseudo-labeling ---
        pl_model_lgbm = lgb.LGBMClassifier(
            objective='binary', metric='auc', n_estimators=1000, learning_rate=0.03,
            num_leaves=63, max_depth=7, min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            random_state=42+seed_idx, n_jobs=-1, device='gpu'
        )
        pl_model_lgbm.fit(
            X_train_combined[current_features], y_train_combined,
            eval_set=[(X_val_fold[current_features], y_val_fold)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(150, verbose=False)]
        )
        test_pred_pl = pl_model_lgbm.predict_proba(df_test_te[current_features])[:,1]
        high_conf_idx = np.where((test_pred_pl>PSEUDO_LABEL_THRESHOLD_HIGH)|(test_pred_pl<PSEUDO_LABEL_THRESHOLD_LOW))[0]
        df_pseudo_labels = df_test_te.iloc[high_conf_idx].copy()
        df_pseudo_labels[TARGET_COL] = (test_pred_pl[high_conf_idx]>0.5).astype(int)
        X_train_final = pd.concat([X_train_combined, df_pseudo_labels.drop(columns=[TARGET_COL])], axis=0)
        y_train_final = pd.concat([y_train_combined, df_pseudo_labels[TARGET_COL]], axis=0)

        # XGBoost
        parameters_xgboost = {'n_estimators':3000,'max_leaves':63,'min_child_weight':3.5,'learning_rate':0.015,
                              'subsample':0.8,'colsample_bylevel':0.7,'colsample_bytree':0.7,'reg_alpha':6.0,
                              'reg_lambda':3.0,'max_depth':0,'grow_policy':'lossguide','tree_method':'hist',
                              'enable_categorical':True,'device':'cuda','n_jobs':-1,'random_state':42+seed_idx,
                              'objective':'binary:logistic','eval_metric':'auc','early_stopping_rounds':150}
        model_xgb = xgb.XGBClassifier(**parameters_xgboost)
        model_xgb.fit(X_train_final[current_features], y_train_final,
                      eval_set=[(X_val_fold[current_features], y_val_fold)], verbose=False)
        oof_predictions_xgb[val_idx] += model_xgb.predict_proba(X_val_fold[current_features])[:,1]/N_SEEDS
        fold_test_preds_xgb += model_xgb.predict_proba(df_test_te[current_features])[:,1]/N_FOLDS

        # LightGBM
        parameters_lgbm = {'objective':'binary','metric':'auc','n_estimators':3000,'learning_rate':0.01,
                           'num_leaves':63,'max_depth':-1,'min_child_samples':20,'subsample':0.8,'colsample_bytree':0.7,
                           'reg_alpha':0.1,'reg_lambda':0.1,'random_state':42+seed_idx,'n_jobs':-1,'device':'gpu'}
        model_lgbm = lgb.LGBMClassifier(**parameters_lgbm)
        model_lgbm.fit(X_train_final[current_features], y_train_final,
                       eval_set=[(X_val_fold[current_features], y_val_fold)],
                       eval_metric='auc', callbacks=[lgb.early_stopping(150,verbose=False)],
                       categorical_feature=[col for col in CATEGORICAL_FEATURES_UPDATED if col in current_features])
        oof_predictions_lgbm[val_idx] += model_lgbm.predict_proba(X_val_fold[current_features])[:,1]/N_SEEDS
        fold_test_preds_lgbm += model_lgbm.predict_proba(df_test_te[current_features])[:,1]/N_FOLDS

        del model_xgb, model_lgbm, X_train_fold_te, X_val_fold_te, df_org_te, df_pseudo_labels, pl_model_lgbm
        gc.collect()
    
    test_predictions_xgb += fold_test_preds_xgb / N_SEEDS
    test_predictions_lgbm += fold_test_preds_lgbm / N_SEEDS



cv_score_xgb = roc_auc_score(df_tra_proc[TARGET_COL], oof_predictions_xgb)
cv_score_lgbm = roc_auc_score(df_tra_proc[TARGET_COL], oof_predictions_lgbm)
ensemble_oof_preds = 0.5*oof_predictions_xgb + 0.5*oof_predictions_lgbm
ensemble_test_preds = 0.5*test_predictions_xgb + 0.5*test_predictions_lgbm
cv_score_ensemble = roc_auc_score(df_tra_proc[TARGET_COL], ensemble_oof_preds)

print(f'XGBoost OOF AUC: {cv_score_xgb:.6f}')
print(f'LightGBM OOF AUC: {cv_score_lgbm:.6f}')
print(f'Ensemble OOF AUC: {cv_score_ensemble:.6f}')


print("Creating Submission")

def simple_post_process_predictions(predictions):
    return np.clip(predictions, 1e-6, 1-1e-6)

final_test_predictions_processed = simple_post_process_predictions(ensemble_test_preds)
final_oof_predictions_processed = simple_post_process_predictions(ensemble_oof_preds)

sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')
sample_submission['y'] = final_test_predictions_processed
sample_submission.to_csv('xgb_lgbm_ensemble_submission.csv', index=False)

print("\nSubmission file created:")
sample_submission.head()


