import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from matplotlib.ticker import FuncFormatter
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_log_error
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)


# Load the datasets
df_train = pd.read_csv('/kaggle/input/playground-series-s4e12/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s4e12/test.csv')
sample_sub = pd.read_csv('/kaggle/input/playground-series-s4e12/sample_submission.csv')
display("Display First Few Rows of Training Data", df_train.head())
display("Display First Few Rows of Testing Data", df_test.head())
display("Display First Few Rows of Sample Submission", sample_sub.head())


display("Info of the Training Data", df_train.info())
print("=====================================================")
display("Info of Testing Data", df_test.info())


print("Columns of Training Data:\n", df_train.columns)
print("==========================================================================")
print("Columns of Testing:\n", df_test.columns)
print("==========================================================================")
print("Columns of Sample Submisison:\n", sample_sub.columns)


print("Missing values of Training Data:\n", df_train.isnull().sum()/len(df_train)*100)
print("==========================================================================")
print("Missing values of Testing:\n", df_test.isnull().sum()/len(df_test)*100)
print("==========================================================================")
print("Missing values of Sample Submisison:", sample_sub.isnull().sum()/len(sample_sub)*100)


display("Descriptive Summary of the Training Data", df_train.describe())
print("=====================================================")
display("Descriptive Summary of Testing Data", df_test.describe())


# Analyze value counts for 'Gender'
gender_counts = df_train['Gender'].value_counts()

print("Gender Counts:")
print(gender_counts)

# Analyze the range of 'Age'
age_min = df_train['Age'].min()
age_max = df_train['Age'].max()
print("====================================")
print("\nAge Range:")
print(f"Minimum Age: {age_min}")
print(f"Maximum Age: {age_max}")



# Create a figure with two subplots: one for 'Gender' and one for 'Age'
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# Refined color palette for Gender (Male: Vivid Amber, Female: Charcoal Blue)
pie_colors = ['#F4A300', '#003B49']  # Vivid Amber and Charcoal Blue colors

# Plot for 'Gender' value counts with refined colors
gender_counts = df_train['Gender'].value_counts()
sns.barplot(x=gender_counts.index, y=gender_counts.values, 
            palette=pie_colors, ax=axes[0])  # Applying the custom color palette

# Customize the 'Gender' plot
axes[0].set_title('Gender Distribution', fontsize=16, fontweight='bold', color='darkblue')
axes[0].set_xlabel('Gender', fontsize=12, color='black')
axes[0].set_ylabel('Count', fontsize=12, color='black')
axes[0].tick_params(axis='x', rotation=0, labelcolor='black')
axes[0].tick_params(axis='y', labelcolor='black')

# Add count annotations on the bars with white text
for p in axes[0].patches:
    axes[0].annotate(f'{p.get_height():,.0f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', fontsize=12, color='black', fontweight='bold')

# Custom colors for the Age plot
age_color = '#6b0a42'  # Lime Green color for the histogram
age_annotation_color = 'darkred'  # White text for annotations

# Plot for 'Age' range (min and max)
sns.histplot(df_train['Age'], bins=15, kde=False, color=age_color, ax=axes[1])

# Customize the 'Age' plot
axes[1].set_title('Age Range Distribution', fontsize=16, fontweight='bold', color='darkblue')
axes[1].set_xlabel('Age', fontsize=12, color='black')
axes[1].set_ylabel('Count', fontsize=12, color='black')
axes[1].tick_params(axis='x', labelcolor='black')
axes[1].tick_params(axis='y', labelcolor='black')

# Annotate Age Range with custom text color
age_min = df_train['Age'].min()
age_max = df_train['Age'].max()
axes[1].annotate(f'Min: {age_min}\nMax: {age_max}', xy=(0.5, 0.9), xycoords='axes fraction', 
                 ha='center', va='center', fontsize=14, fontweight='bold', color=age_annotation_color)

# Adjust layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()



# Proportion of each gender in the dataset
gender_proportion = df_train['Gender'].value_counts(normalize=True) * 100

print("Proportion of Each Gender in the Dataset:")
print(gender_proportion)
# Check if the dataset has a gender imbalance
most_common_gender = gender_proportion.idxmax()
imbalance_percentage = gender_proportion.max() - gender_proportion.min()

print(f"The most common gender is '{most_common_gender}' with a {gender_proportion.max():.2f}% share.")



# Calculate gender proportions
gender_proportion = df_train['Gender'].value_counts(normalize=True) * 100

# Create a figure with two subplots: one for Pie chart and one for Bar plot
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Define a refined color palette for both plots
pie_colors = ['#F4A300', '#003B49']  # Vivid Amber and Charcoal Blue colors

# Plot 1: Pie chart for gender proportions with enhanced visuals
wedges, texts, autotexts = axes[0].pie(gender_proportion, labels=gender_proportion.index, autopct='%1.1f%%', 
                                      colors=pie_colors, startangle=90, 
                                      wedgeprops={'edgecolor': 'white', 'linewidth': 2, 'linestyle': 'solid'}, 
                                      shadow=True, textprops={'color': 'white'})  # Set text inside pie chart to white

# Add legend for the Pie chart
axes[0].legend(wedges, gender_proportion.index, title="Gender", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=12)

# Customize Pie chart
axes[0].set_title('Gender Proportion', fontsize=18, fontweight='bold', color='darkblue', pad=20)
axes[0].axis('equal')  # Equal aspect ratio ensures that pie chart is circular.

# Plot 2: Bar plot for gender proportions with improved style
sns.barplot(x=gender_proportion.index, y=gender_proportion.values, 
            palette=pie_colors, ax=axes[1])

# Customize Bar plot
axes[1].set_title('Gender Proportion Bar Plot', fontsize=18, fontweight='bold', color='darkblue', pad=20)
axes[1].set_xlabel('Gender', fontsize=14, color='darkblue')
axes[1].set_ylabel('Proportion (%)', fontsize=14, color='darkblue')
axes[1].tick_params(axis='x', rotation=0, labelcolor='black', labelsize=12)
axes[1].tick_params(axis='y', labelcolor='black', labelsize=12)

# Add percentage annotations on the bars with improved styling
for p in axes[1].patches:
    height = p.get_height()
    axes[1].annotate(f'{height:.1f}%', 
                     (p.get_x() + p.get_width() / 2., height),
                     ha='center', va='center', fontsize=14, color='black', fontweight='bold')

# Add gridlines to the bar plot for better readability
axes[1].grid(axis='y', linestyle='--', alpha=0.7)

# Adjust layout for better spacing and make the plot more cohesive
plt.tight_layout(pad=5)

# Show the plot
plt.show()



# Calculate mean and median age by gender
mean_age_by_gender = df_train.groupby('Gender')['Age'].mean()
median_age_by_gender = df_train.groupby('Gender')['Age'].median()

print("Mean Age by Gender:")
print(mean_age_by_gender)
print("\nMedian Age by Gender:")
print(median_age_by_gender)



import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Assuming df_train is your DataFrame

# Calculate mean and median age by gender
mean_age_by_gender = df_train.groupby('Gender')['Age'].mean()
median_age_by_gender = df_train.groupby('Gender')['Age'].median()

# Create a figure with two rows: the first row for bar plots and the second row for pie charts
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Define a refined color palette for both plots
bar_colors = ['#F4A300', '#003B49']  # Vivid Amber and Charcoal Blue colors for bar plots
pie_colors = ['#F4A300', '#003B49']  # Vivid Amber and Charcoal Blue colors for pie charts

# Plot 1: Bar plot for mean age by gender
sns.barplot(x=mean_age_by_gender.index, y=mean_age_by_gender.values, 
            palette=bar_colors, ax=axes[0, 0])

# Customize Mean Age plot
axes[0, 0].set_title('Mean Age by Gender', fontsize=18, fontweight='bold', color='darkblue', pad=20)
axes[0, 0].set_xlabel('Gender', fontsize=14, color='darkblue')
axes[0, 0].set_ylabel('Mean Age', fontsize=14, color='darkblue')
axes[0, 0].tick_params(axis='x', labelcolor='black', labelsize=12)
axes[0, 0].tick_params(axis='y', labelcolor='black', labelsize=12)

# Add value annotations on the bars (three digits after the decimal point)
for p in axes[0, 0].patches:
    height = p.get_height()
    axes[0, 0].annotate(f'{height:.3f}',  # Format to show 3 digits after the decimal point
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='center', fontsize=14, color='black', fontweight='bold')

# Plot 2: Bar plot for median age by gender
sns.barplot(x=median_age_by_gender.index, y=median_age_by_gender.values, 
            palette=bar_colors, ax=axes[0, 1])

# Customize Median Age plot
axes[0, 1].set_title('Median Age by Gender', fontsize=18, fontweight='bold', color='darkblue', pad=20)
axes[0, 1].set_xlabel('Gender', fontsize=14, color='darkblue')
axes[0, 1].set_ylabel('Median Age', fontsize=14, color='darkblue')
axes[0, 1].tick_params(axis='x', labelcolor='black', labelsize=12)
axes[0, 1].tick_params(axis='y', labelcolor='black', labelsize=12)

# Add value annotations on the bars (three digits after the decimal point)
for p in axes[0, 1].patches:
    height = p.get_height()
    axes[0, 1].annotate(f'{height:.3f}',  # Format to show 3 digits after the decimal point
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='center', fontsize=14, color='black', fontweight='bold')

# Plot 3: Pie chart for mean age by gender
axes[1, 0].pie(mean_age_by_gender, labels=mean_age_by_gender.index, autopct='%1.3f%%', 
               colors=pie_colors, startangle=90, 
               wedgeprops={'edgecolor': 'white', 'linewidth': 2, 'linestyle': 'solid'}, 
               shadow=True, textprops={'color': 'white'})  # Set text inside pie chart to white

# Customize Pie chart for Mean Age
axes[1, 0].set_title('Mean Age by Gender (Pie)', fontsize=18, fontweight='bold', color='darkblue', pad=20)
axes[1, 0].axis('equal')  # Equal aspect ratio ensures that pie chart is circular.
axes[1, 0].legend(mean_age_by_gender.index, title="Gender", loc='upper right', fontsize=12)

# Plot 4: Pie chart for median age by gender
axes[1, 1].pie(median_age_by_gender, labels=median_age_by_gender.index, autopct='%1.3f%%', 
               colors=pie_colors, startangle=90, 
               wedgeprops={'edgecolor': 'white', 'linewidth': 2, 'linestyle': 'solid'}, 
               shadow=True, textprops={'color': 'white'})  # Set text inside pie chart to white

# Customize Pie chart for Median Age
axes[1, 1].set_title('Median Age by Gender (Pie)', fontsize=18, fontweight='bold', color='darkblue', pad=20)
axes[1, 1].axis('equal')  # Equal aspect ratio ensures that pie chart is circular.
axes[1, 1].legend(median_age_by_gender.index, title="Gender", loc='upper right', fontsize=12)

# Adjust layout for better spacing and make the plot more cohesive
plt.tight_layout(pad=5)

# Show the plot
plt.show()



# Calculate the range of age for each gender
age_range_by_gender = df_train.groupby('Gender').agg({'Age': lambda x: x.max() - x.min()})

print("Age Range by Gender:")
print(age_range_by_gender)



# Define age bins
bins = [18, 25, 35, 45, 55, 64]
labels = ['18-25', '26-35', '36-45', '46-55', '56-64']

# Create an Age Range column
df_train['Age Range'] = pd.cut(df_train['Age'], bins=bins, labels=labels, right=True)

# Distribution of Gender within Age Ranges
age_range_gender_dist = df_train.groupby('Age Range')['Gender'].value_counts(normalize=True)

# Output the distribution
print("Gender distribution within each Age Range:\n", age_range_gender_dist)



# Define updated age bins and labels
bins = [18, 22, 25, 30, 35, 40, 45, 50, 55, 60, 64]
labels = ['18-22', '23-25', '26-30', '31-35', '36-40', '41-45', '46-50', '51-55', '56-60', '61-64']

# Create an Age Range column
df_train['Age Range'] = pd.cut(df_train['Age'], bins=bins, labels=labels, right=True)

# Define the color palette for gender (darker shades)
gender_palette = ['#B77A00', '#001F2D']  # Darker shades of Vivid Amber and Charcoal Blue

# Create the figure with 2 subplots (2 rows, 1 column)
fig, axes = plt.subplots(2, 1, figsize=(10, 14))

# Histogram plot: Gender distribution within age ranges
sns.histplot(data=df_train, x='Age Range', hue='Gender', stat='probability', common_norm=False, multiple="stack", palette=gender_palette, ax=axes[0])

# Add the proportion values inside the histogram bars for each gender
for p in axes[0].patches:
    height = p.get_height()
    x = p.get_x() + p.get_width() / 2  # x position of the bar
    y = p.get_y() + height / 2  # y position of the bar
    axes[0].annotate(f"{height:.5f}", (x, y), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8, color='black')

axes[0].set_title('Proportion of Gender Distribution within Age Ranges', fontsize=14, fontweight='bold', color='darkblue', pad=20)
axes[0].set_xlabel('Age Range', fontsize=10, color='darkblue')
axes[0].set_ylabel('Proportion', fontsize=10, color='darkblue')
axes[0].tick_params(axis='x', rotation=45, labelsize=8)  # Decreased tick label size for x-axis
axes[0].tick_params(axis='y', labelsize=8)  # Decreased tick label size for y-axis
axes[0].legend(title='Gender', labels=['Female', 'Male'], loc='upper right', fontsize=12)

# Line plot: Gender proportion trend across age ranges
# Calculate the proportion of males and females for each age range
gender_counts = df_train.groupby(['Age Range', 'Gender']).size().unstack(fill_value=0)
gender_proportions = gender_counts.div(gender_counts.sum(axis=1), axis=0)

# Plot the gender proportions as lines
gender_proportions.plot(ax=axes[1], color=gender_palette, marker='o', linewidth=2)
axes[1].set_title('Gender Proportions Across Age Ranges', fontsize=14, fontweight='bold', color='darkblue', pad=20)
axes[1].set_xlabel('Age Range', fontsize=10, color='darkblue')
axes[1].set_ylabel('Proportion', fontsize=10, color='darkblue')
axes[1].tick_params(axis='x', rotation=45, labelsize=8)  # Decreased tick label size for x-axis
axes[1].tick_params(axis='y', labelsize=8)  # Decreased tick label size for y-axis
axes[1].legend(title='Gender', labels=['Female', 'Male'], loc='upper right', fontsize=12)

# Add the proportion values above the lines in black
for i, age_range in enumerate(gender_proportions.index):
    for j, gender in enumerate(gender_proportions.columns):
        axes[1].annotate(f"{gender_proportions.loc[age_range, gender]:.5f}",
                         (i, gender_proportions.loc[age_range, gender]),
                         textcoords="offset points",
                         xytext=(0, 10),  # 10 points vertical offset
                         ha='center', fontsize=7, color=gender_palette[j])

# Adjust layout to avoid overlap
plt.tight_layout(pad=5)

# Show the plot
plt.show()



# Mode of Age and Gender
age_mode = df_train['Age'].mode()[0]
gender_mode = df_train['Gender'].mode()[0]

# Unique pair combinations of Age and Gender
unique_age_gender_pairs = df_train[['Age', 'Gender']]

# Output the results
print("Mode of 'Age':", age_mode)
display("Unique combinations of 'Age' and 'Gender':", unique_age_gender_pairs.head(20))



# Mode of Age and Gender
age_mode = df_train['Age'].mode()[0]
gender_mode = df_train['Gender'].mode()[0]

# Unique pair combinations of Age and Gender
unique_age_gender_pairs = df_train[['Age', 'Gender']]

# Filter top 20 'Age' values by frequency
top_20_ages = df_train['Age'].value_counts().head(20).index
# Group by Age and Gender, then get the counts
age_gender_counts = df_train[df_train['Age'].isin(top_20_ages)].groupby(['Age', 'Gender']).size().reset_index(name='Count')

# Display the dataframe with top 20 Age and Gender combinations along with their counts
age_gender_counts_sorted = age_gender_counts.sort_values(by='Count', ascending=False)
age_gender_counts_sorted



# Mode of Age and Gender
age_mode = df_train['Age'].mode()[0]
gender_mode = df_train['Gender'].mode()[0]

# Unique pair combinations of Age and Gender
unique_age_gender_pairs = df_train[['Age', 'Gender']]

# Filter top 20 'Age' values by frequency
top_20_ages = df_train['Age'].value_counts().head(20).index

# Create a barplot and pie chart subplot with increased figure size
fig, axes = plt.subplots(1, 2, figsize=(25, 10))  # Increased figure size

# Barplot: Mode of Age by Gender (only top 20 ages)
sns.countplot(x='Age', hue='Gender', data=df_train[df_train['Age'].isin(top_20_ages)], ax=axes[0], palette=['#F4A300', '#003B49'])
axes[0].set_title('Count of Mode Age by Gender', fontsize=20, fontweight='bold')  # Make title bold and larger
axes[0].set_xlabel('Age', fontsize=16, fontweight='bold')  # Larger and bold x-label
axes[0].set_ylabel('Count', fontsize=16, fontweight='bold')  # Larger and bold y-label

# Increase font size and bold tick labels
axes[0].tick_params(axis='x', labelsize=18, labelrotation=45)  # Larger x-tick labels
axes[0].tick_params(axis='y', labelsize=18)  # Larger y-tick labels

# Move the legend outside the plot
axes[0].legend(title='Gender', loc='upper left', bbox_to_anchor=(1, 1), labels=['Female', 'Male'], fontsize=20)

# Pie chart: Mode values for Gender (based on the mode of 'Gender' in df_train)
gender_counts = df_train['Gender'].value_counts()

# Increase font size for the pie chart labels and percentage values, and set color to white
axes[1].pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90, colors=['#F4A300', '#003B49'], 
            textprops={'fontsize': 18, 'fontweight': 'bold', 'color': 'white'})  # Set text color to white

axes[1].set_title('Gender Distribution (Mode Values)', fontsize=25, fontweight='bold')  # Make title bold and larger

# Display the results
plt.tight_layout()
plt.show()



# Get the frequency counts of each category in Gender and Marital Status
gender_counts = df_train['Gender'].value_counts().reset_index()
gender_counts.columns = ['Gender', 'Gender Count']

marital_status_counts = df_train['Marital Status'].value_counts().reset_index()
marital_status_counts.columns = ['Marital Status', 'Marital Status Count']

# Merge the two dataframes into one
combined_analysis = pd.merge(gender_counts, marital_status_counts, how='cross')

# Display the combined analysis
display(combined_analysis)



# Get summary statistics for Age and Annual Income
age_summary = df_train['Age'].describe()
income_summary = df_train['Annual Income'].describe()

# Display the summaries
print("Age Summary:\n", age_summary)
print("\nAnnual Income Summary:\n", income_summary)


# Calculate the statistics for Age and Annual Income
age_mean = df_train['Age'].mean()
age_max = df_train['Age'].max()
age_min = df_train['Age'].min()

income_mean = df_train['Annual Income'].mean()
income_max = df_train['Annual Income'].max()
income_min = df_train['Annual Income'].min()


# Calculate the statistics for Age and Annual Income grouped by Gender
gender_stats = df_train.groupby('Gender').agg({
    'Age': ['mean', 'max', 'min'],
    'Annual Income': ['mean', 'max', 'min']
})

# Reset the index and flatten the multi-level columns
gender_stats_reset = gender_stats.reset_index()
gender_stats_reset.columns = ['Gender', 'Age_mean', 'Age_max', 'Age_min', 'Annual_Income_mean', 'Annual_Income_max', 'Annual_Income_min']

# Display the summary
display(gender_stats_reset)




# Define the color palette for gender (same colors for both bar plots and box plots)
gender_palette = ['#B77A00', '#001F2D']  # Darker shades of Vivid Amber and Charcoal Blue

# Create a figure with subplots (2 rows, 2 columns)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Bar plot for Age statistics by Gender (Mean)
sns.barplot(
    x='Gender', 
    y='Age_mean', 
    data=gender_stats_reset, 
    ax=axes[0, 0], 
    palette=gender_palette  # Use the same palette for the bar plot
)
axes[0, 0].set_title('Age Statistics by Gender (Mean)', fontsize=18, fontweight='bold')
axes[0, 0].set_ylabel('Age', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Gender', fontsize=14, fontweight='bold')

# Add values above bars
for container in axes[0, 0].containers:
    axes[0, 0].bar_label(container, fmt='%.3f', fontsize=12, fontweight='bold', label_type='edge', padding=3)

# Bar plot for Annual Income statistics by Gender (Mean)
sns.barplot(
    x='Gender', 
    y='Annual_Income_mean', 
    data=gender_stats_reset, 
    ax=axes[0, 1], 
    palette=gender_palette  # Use the same palette for the bar plot
)
axes[0, 1].set_title('Annual Income Statistics by Gender (Mean)', fontsize=18, fontweight='bold')
axes[0, 1].set_ylabel('Annual Income', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Gender', fontsize=14, fontweight='bold')

# Add values above bars
for container in axes[0, 1].containers:
    axes[0, 1].bar_label(container, fmt='%.1f', fontsize=12, fontweight='bold', label_type='edge', padding=3)

# Box plot for Age by Gender (Distribution)
sns.boxplot(
    x='Gender', 
    y='Age', 
    data=df_train, 
    ax=axes[1, 0], 
    palette=gender_palette  # Use the same palette for the box plot
)
axes[1, 0].set_title('Age Distribution by Gender', fontsize=18, fontweight='bold')
axes[1, 0].set_ylabel('Age', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Gender', fontsize=14, fontweight='bold')

# Box plot for Annual Income by Gender (Distribution)
sns.boxplot(
    x='Gender', 
    y='Annual Income', 
    data=df_train, 
    ax=axes[1, 1], 
    palette=gender_palette  # Use the same palette for the box plot
)
axes[1, 1].set_title('Annual Income Distribution by Gender', fontsize=18, fontweight='bold')
axes[1, 1].set_ylabel('Annual Income', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Gender', fontsize=14, fontweight='bold')

# Add spacing between subplots
plt.tight_layout()

# Display the visualization
plt.show()



# Binning and combining age and income for df_train only
age_bins = [18, 22, 25, 30, 35, 40, 45, 50, 55, 60, 64]
age_labels = ['18-22', '23-25', '26-30', '31-35', '36-40', '41-45', '46-50', '51-55', '56-60', '61-64']
income_bins = [0, 30000, 50000, 70000, 100000, 150000]
income_labels = ['Low', 'Medium', 'High', 'Very High', 'Top']

# Assuming df_train is your training dataset
df_train['Age Range'] = pd.cut(df_train['Age'], bins=age_bins, labels=age_labels)
df_train['Income Group'] = pd.cut(df_train['Annual Income'], bins=income_bins, labels=income_labels)

# Count the occurrences of each combination of Age Range and Income Group
df_train['Count'] = df_train.groupby(['Age Range', 'Income Group'])['Age Range'].transform('count')

# Find the maximum values for Age and Income within each Age Range and Income Group
df_train['Max Age'] = df_train.groupby(['Age Range', 'Income Group'])['Age'].transform('max')
df_train['Max Income'] = df_train.groupby(['Age Range', 'Income Group'])['Annual Income'].transform('max')

# Display only the new columns (Age Range, Income Group, Count, Max Age, Max Income)
df_train[['Age Range', 'Income Group', 'Count', 'Max Age', 'Max Income']].head()



# Define custom color palettes for each plot
age_range_palette = ['#7f278f', '#6B5B95', '#8f2727', '#F7B7A3', '#36b0d1', '#32a852', '#a83281', '#7ba832', '#ebcf52', '#27598f']  # Soft and warm colors
income_group_palette = ['#A6D608', '#A4B1B4', '#F4D03F', '#F39C12', '#2f278f']  # Earthy tones with pops of yellow
gender_palette = ['#D54F39', '#34A853']  # Bright red and green for gender
age_income_palette = ['#888f27', '#278f8f', '#8f273a', '#278f3d', '#8f2a27']  # Bright pastels for the count plot

# Set up the figure and axes for multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Barplot for Age Range vs Count with custom colors
sns.barplot(x='Age Range', y='Count', data=df_train, ax=axes[0, 0], palette=age_range_palette)
axes[0, 0].set_title('Age Range vs Count', fontsize=16, fontweight='bold')
axes[0, 0].set_xlabel('Age Range', fontsize=12)
axes[0, 0].set_ylabel('Count', fontsize=12)

# Add values above bars in first plot
for p in axes[0, 0].patches:
    axes[0, 0].annotate(f'{p.get_height():.0f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=7, color='black', xytext=(0, 10),
                        textcoords='offset points')

# Plot 2: Barplot for Income Group vs Count with custom colors
sns.barplot(x='Income Group', y='Count', data=df_train, ax=axes[0, 1], palette=income_group_palette)
axes[0, 1].set_title('Income Group vs Count', fontsize=16, fontweight='bold')
axes[0, 1].set_xlabel('Income Group', fontsize=12)
axes[0, 1].set_ylabel('Count', fontsize=12)

# Add values above bars in second plot
for p in axes[0, 1].patches:
    axes[0, 1].annotate(f'{p.get_height():.0f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=7, color='black', xytext=(0, 10),
                        textcoords='offset points')

# Plot 3: Scatterplot for Max Age vs Max Income based on Gender with custom colors
sns.scatterplot(x='Max Age', y='Max Income', hue='Gender', data=df_train, ax=axes[1, 0], palette=gender_palette, style='Gender', markers=["o", "X"], edgecolor='darkred')
axes[1, 0].set_title('Max Age vs Max Income (Gender)', fontsize=16, fontweight='bold')
axes[1, 0].set_xlabel('Max Age', fontsize=12)
axes[1, 0].set_ylabel('Max Income', fontsize=12)

# Place legend outside the scatter plot
axes[1, 0].legend(title='Gender', bbox_to_anchor=(1.05, 1), loc='upper left')

# Plot 4: Countplot for combinations of Age Range and Income Group with custom colors
sns.countplot(x='Age Range', hue='Income Group', data=df_train, ax=axes[1, 1], palette=age_income_palette)
axes[1, 1].set_title('Age Range and Income Group visualization', fontsize=16, fontweight='bold')
axes[1, 1].set_xlabel('Age Range', fontsize=12)
axes[1, 1].set_ylabel('Count', fontsize=12)

# Place legend outside the countplot
axes[1, 1].legend(title='Income Group', bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()



# Count the occurrences of each Number of Dependents for each Education Level
dependents_distribution = df_train.groupby('Education Level')['Number of Dependents'].value_counts(normalize=True).unstack()

# Rename columns for clarity
dependents_distribution.columns.name = "Number of Dependents"
dependents_distribution.fillna(0, inplace=True)

# Output the distribution
display("Distribution of Number of Dependents by Education Level:\n", dependents_distribution)



import matplotlib.pyplot as plt
import seaborn as sns

# Set up a figure with 2 rows and 2 columns of subplots, with same figsize for clarity
fig, axes = plt.subplots(2, 2, figsize=(35, 19))  # Increased figure size for better prominence

# Plot 1: Grouped bar plot for Number of Dependents by Education Level
dependents_distribution.plot(kind='bar', ax=axes[0, 0], color=sns.color_palette('Set2', len(dependents_distribution.columns)))
axes[0, 0].set_title('Number of Dependents by Education Level', fontsize=28, fontweight='bold', color='darkblue')
axes[0, 0].set_xlabel('Education Level', fontsize=22, fontweight='bold', color='darkblue')
axes[0, 0].set_ylabel('Proportion', fontsize=22, fontweight='bold', color='darkblue')
axes[0, 0].legend(title='Number of Dependents', fontsize=18, title_fontsize=20, loc='upper left', bbox_to_anchor=(1.05, 1),
                  frameon=True, shadow=True)
axes[0, 0].tick_params(axis='x', labelsize=16, labelrotation=45, labelcolor='black')
axes[0, 0].tick_params(axis='y', labelsize=16, labelcolor='black')

# Annotate values above the bars
for p in axes[0, 0].patches:
    height = p.get_height()
    axes[0, 0].annotate(f'{height:.2f}',  # Value above the bar
                        (p.get_x() + p.get_width() / 2., height),  # Position above the bar
                        ha='center', va='center', fontsize=10, color='black', fontweight='bold', 
                        xytext=(0, 5), textcoords='offset points')

# Plot 2: Pie chart for proportions of Education Levels
education_level_proportions = dependents_distribution.sum(axis=1) / dependents_distribution.sum().sum()  # Calculate the proportions
axes[0, 1].pie(education_level_proportions, labels=education_level_proportions.index, 
               autopct='%1.1f%%', startangle=90, colors=sns.color_palette('tab20', len(education_level_proportions)),
               wedgeprops={'edgecolor': 'black', 'linewidth': 1, 'linestyle': 'solid'}, radius=0.8, labeldistance=1.05)
axes[0, 1].set_title('Proportion of Education Levels', fontsize=28, fontweight='bold', color='darkred')
axes[0, 1].legend(title='Education Level', fontsize=18, title_fontsize=20, loc='upper left', bbox_to_anchor=(1.05, 1),
                  frameon=True, shadow=True)

# Plot 3: Violin plot to visualize the distribution of proportions
sns.violinplot(data=dependents_distribution, ax=axes[1, 0], palette='coolwarm', linewidth=2)
axes[1, 0].set_title('Distribution of Dependents Proportions', fontsize=28, fontweight='bold', color='green')
axes[1, 0].set_xlabel('Number of Dependents', fontsize=22, fontweight='bold', color='green')
axes[1, 0].set_ylabel('Proportion', fontsize=22, fontweight='bold', color='green')
axes[1, 0].set_xticks(range(len(dependents_distribution.columns)))
axes[1, 0].set_xticklabels(dependents_distribution.columns, fontsize=18, rotation=45, fontweight='bold', color='black')
axes[1, 0].tick_params(axis='y', labelsize=18, labelcolor='black')
axes[1, 0].grid(True, linestyle='--', alpha=0.7)

# Plot 4: Scatter plot for proportions with Education Levels as categories
for level in dependents_distribution.index:
    sns.scatterplot(x=dependents_distribution.columns, y=dependents_distribution.loc[level],
                    label=level, ax=axes[1, 1], s=200, alpha=0.8, marker='o', edgecolor='black', linewidth=2)
axes[1, 1].set_title('Scatter Plot of Dependents Distribution', fontsize=28, fontweight='bold', color='purple')
axes[1, 1].set_xlabel('Number of Dependents', fontsize=22, fontweight='bold', color='purple')
axes[1, 1].set_ylabel('Proportion', fontsize=22, fontweight='bold', color='purple')
axes[1, 1].legend(title='Education Level', fontsize=18, title_fontsize=20, loc='upper left', bbox_to_anchor=(1.05, 1),
                  frameon=True, shadow=True)
axes[1, 1].tick_params(axis='x', labelsize=18, labelrotation=45, labelcolor='black')
axes[1, 1].tick_params(axis='y', labelsize=18, labelcolor='black')

# Add borders to the plots for emphasis
for ax in axes.flat:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

# Adjust layout for better spacing
plt.tight_layout(pad=4.0)  # Increased padding to avoid overlap

# Show the plots
plt.show()



# Calculate the average and most common number of dependents for each education level
dependents_summary = df_train.groupby('Education Level')['Number of Dependents'].agg(
    Average='mean',
    Most_Common=lambda x: x.value_counts().idxmax()
).reset_index()

# Display the summarized DataFrame
display("Summary of Dependents by Education Level:", dependents_summary)



# Define the color palette for gender (same colors for both bar plots and pie charts)
gender_palette = ['#B77A00', '#001F2D', '#29002d', '#b85712']  # Darker shades of Vivid Amber and Charcoal Blue

# Create a figure with 1 row and 2 columns for the pie chart and bar plot
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Plot 1: Pie chart showing the average number of dependents by Education Level
dependents_summary.set_index('Education Level')['Average'].plot(kind='pie', autopct='%1.1f%%', ax=axes[0], 
                                                              colors=gender_palette[:len(dependents_summary)],
                                                              legend=False, textprops={'color': 'white'})  # Set text color to white
axes[0].set_title('Average Number of Dependents by Education Level', fontsize=16, fontweight='bold')
axes[0].set_ylabel('')  # Remove y-axis label for pie chart

# Plot 2: Bar plot showing the most common number of dependents by Education Level
bar_plot = sns.barplot(x='Education Level', y='Most_Common', data=dependents_summary, ax=axes[1], 
                       palette=gender_palette[:len(dependents_summary)])

# Add values above bars with black text
for p in bar_plot.patches:
    bar_plot.annotate(f'{p.get_height():.0f}', 
                      (p.get_x() + p.get_width() / 2., p.get_height()), 
                      ha='center', va='center', fontsize=12, color='black', 
                      xytext=(0, 10), textcoords='offset points')

axes[1].set_title('Most Common Number of Dependents by Education Level', fontsize=16, fontweight='bold')
axes[1].set_xlabel('Education Level', fontsize=12)
axes[1].set_ylabel('Most Common Number of Dependents', fontsize=12)

# Create a custom legend to match the color palette
custom_legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=name)
                         for color, name in zip(gender_palette[:len(dependents_summary)], dependents_summary['Education Level'])]
fig.legend(handles=custom_legend_handles, title="Education Level", fontsize=12, title_fontsize=14, loc='lower center', ncol=4)

# Adjust layout for better spacing
plt.tight_layout(rect=[0, 0.1, 1, 1])  # Leave space for legend at the bottom

# Show the plots
plt.show()



# Calculate mean and median income for each combination of Number of Dependents and Education Level
income_analysis = df_train.groupby(['Education Level', 'Number of Dependents'])['Annual Income'].agg(['mean', 'max']).reset_index()

# Output income analysis
print("Income Analysis by Education Level and Number of Dependents:\n", income_analysis)



# Calculate mean and max income for each combination of Number of Dependents and Education Level
income_analysis = df_train.groupby(['Education Level', 'Number of Dependents'])['Annual Income'].agg(['mean', 'max']).reset_index()

# Define the color palette for gender (same colors for both bar plots and pie charts)
gender_palette = ['#B77A00', '#001F2D', '#29002d', '#b85712']  # Darker shades of Vivid Amber and Charcoal Blue

# Create a figure with 2 rows and 1 column for the bar plot and pie chart (row-wise arrangement)
fig, axes = plt.subplots(2, 1, figsize=(20, 18))  # Increased figsize for better prominence

# Plot 1: Bar plot showing mean and max income for each combination of Education Level and Number of Dependents
sns.barplot(x='Number of Dependents', y='mean', hue='Education Level', data=income_analysis, ax=axes[0], palette=gender_palette)
axes[0].set_title('Mean Annual Income by Education Level and Number of Dependents', fontsize=22, fontweight='bold')
axes[0].set_xlabel('Number of Dependents', fontsize=18)
axes[0].set_ylabel('Mean Annual Income', fontsize=18)

# Add values above the bars with black color and increase font size
for p in axes[0].patches:
    axes[0].annotate(f'{p.get_height():,.0f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', fontsize=9, color='black', fontweight='bold', xytext=(0, 5),
                     textcoords='offset points')

# Move the legend outside of the bar plot with larger font size
axes[0].legend(title='Education Level', fontsize=14, title_fontsize=16, bbox_to_anchor=(1.05, 1), loc='upper left')

# Plot 2: Pie chart showing the distribution of mean income by Education Level
education_income = income_analysis.groupby('Education Level')['mean'].mean()
education_income.plot(kind='pie', autopct='%1.1f%%', ax=axes[1], colors=gender_palette,
                      legend=False, textprops={'color': 'white', 'fontsize': 14}, radius=1.2)  # Increased radius for larger pie chart
axes[1].set_title('Mean Annual Income Distribution by Education Level', fontsize=22, fontweight='bold')
axes[1].set_ylabel('')  # Remove y-axis label for pie chart

# Adjust layout for better spacing
plt.tight_layout(pad=5.0)  # Increased padding for better spacing between plots

# Show the plots
plt.show()



# Define a threshold for high-dependency (e.g., 3 or more dependents)
high_dependency_threshold = 3

# Calculate the proportion of individuals with high dependency for each Education Level
high_dependency_profile = df_train[df_train['Number of Dependents'] >= high_dependency_threshold].groupby('Education Level').size() / df_train.groupby('Education Level').size()

# Output high-dependency profile
print("Proportion of High-Dependency Individuals by Education Level:\n", high_dependency_profile)



# Define a threshold for high dependency
high_dependency_threshold = 3

# Calculate the proportion of individuals with high dependency for each Education Level
high_dependency_profile = df_train[df_train['Number of Dependents'] >= high_dependency_threshold].groupby('Education Level').size() / df_train.groupby('Education Level').size()

# Define the color palette for gender (same colors for both bar plots and pie charts)
gender_palette = ['#B77A00', '#001F2D', '#29002d', '#b85712']  # Darker shades of Vivid Amber and Charcoal Blue

# Create a figure with 2 subplots (bar plot and pie chart)
fig, axes = plt.subplots(1, 2, figsize=(30, 15))  # Increased width for side-by-side arrangement

# Plot 1: Bar plot showing the proportion of high-dependency individuals by Education Level
sns.barplot(x=high_dependency_profile.index, y=high_dependency_profile.values, ax=axes[0], palette=gender_palette)
axes[0].set_title('Proportion of High-Dependency Individuals by Education Level', fontsize=24, fontweight='bold')
axes[0].set_xlabel('Education Level', fontsize=20)
axes[0].set_ylabel('Proportion of High-Dependency', fontsize=20)

# Make ticks more prominent
axes[0].tick_params(axis='x', labelsize=18, labelrotation=45, width=3, colors='black')  # x-axis ticks
axes[0].tick_params(axis='y', labelsize=18, width=3, colors='black')  # y-axis ticks

# Add values above the bars with black color and increase font size
for p in axes[0].patches:
    axes[0].annotate(f'{p.get_height():.4f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', fontsize=20, color='black', fontweight='bold', xytext=(0, 5),
                     textcoords='offset points')

# Plot 2: Pie chart showing the proportion of high-dependency individuals by Education Level
high_dependency_profile.plot(kind='pie', autopct='%1.1f%%', ax=axes[1], colors=gender_palette,
                             legend=False, textprops={'color': 'white', 'fontsize': 18}, radius=1.2)  # Increased radius for larger pie chart
axes[1].set_title('Proportion of High-Dependency Individuals by Education Level', fontsize=24, fontweight='bold')
axes[1].set_ylabel('')  # Remove y-axis label for pie chart

# Create a custom legend to match the color palette
custom_legend_handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=15, label=name)
    for color, name in zip(gender_palette, high_dependency_profile.index)
]
fig.legend(handles=custom_legend_handles, title="Education Level", fontsize=18, title_fontsize=20, loc='lower center', ncol=4)

# Adjust layout for better spacing
plt.tight_layout(rect=[0, 0.1, 1, 1])  # Leave space for legend at the bottom

# Show the plots
plt.show()



# Calculate the mean, max, min, and count of Health Score for each Occupation
health_score_by_occupation = df_train.groupby('Occupation')['Health Score'].agg(['mean', 'max', 'min', 'count'])

# Output the selected statistics (mean, max, min, count) of Health Score by Occupation
print("Health Score by Occupation:\n", health_score_by_occupation)



import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.impute import SimpleImputer

# Impute missing values in 'Occupation' with the mode (most frequent value)
occupation_imputer = SimpleImputer(strategy='most_frequent')

# Ensure the input to fit_transform is a 2D array by selecting the column as DataFrame
df_train['Occupation'] = occupation_imputer.fit_transform(df_train[['Occupation']]).flatten()
df_test['Occupation'] = occupation_imputer.transform(df_test[['Occupation']]).flatten()

# Define the custom color palette
gender_palette = ['#B77A00', '#001F2D', '#29002d']

# Ensure the palette has enough colors for the number of unique occupations by cycling the colors
occupation_palette = gender_palette * (len(df_train['Occupation'].unique()) // len(gender_palette)) + gender_palette[:len(df_train['Occupation'].unique()) % len(gender_palette)]

# Create a dictionary that maps occupations to the colors in the palette
occupation_color_map = dict(zip(df_train['Occupation'].unique(), occupation_palette))

# Assign navy blue to "Unemployed"
occupation_color_map['Unemployed'] = '#001F2D'  # Navy blue color

# Calculate the mean, max, min, and count of Health Score for each Occupation
health_score_by_occupation = df_train.groupby('Occupation')['Health Score'].agg(['mean', 'max', 'min', 'count'])

# Set up the figure and axes for subplots with a larger size
fig, axes = plt.subplots(2, 2, figsize=(22, 18))  # Increased figsize for better prominence

# Plot 1: Bar plot of mean Health Score by Occupation with the consistent color palette
barplot = sns.barplot(x=health_score_by_occupation.index, 
                      y=health_score_by_occupation['mean'], 
                      ax=axes[0, 0], 
                      palette=occupation_color_map)
axes[0, 0].set_title('Mean Health Score by Occupation', fontsize=24, fontweight='bold')
axes[0, 0].set_xlabel('Occupation', fontsize=18)
axes[0, 0].set_ylabel('Mean Health Score', fontsize=18)
axes[0, 0].tick_params(axis='x', rotation=45, labelsize=16)

# Add values above bars
for container in barplot.containers:
    barplot.bar_label(container, fmt='%.2f', fontsize=16, fontweight='bold', label_type='edge', padding=6)

# Create custom legend for bar plot
legend_labels = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=12) 
                 for color in occupation_color_map.values()]
axes[0, 0].legend(legend_labels, occupation_color_map.keys(), title='Occupation', loc='upper left', bbox_to_anchor=(1, 1), fontsize=16)

# Plot 2: Box plot for Health Score distribution by Occupation
sns.boxplot(x='Occupation', y='Health Score', data=df_train, ax=axes[0, 1], palette=occupation_color_map)
axes[0, 1].set_title('Health Score Distribution by Occupation', fontsize=24, fontweight='bold')
axes[0, 1].set_xlabel('Occupation', fontsize=18)
axes[0, 1].set_ylabel('Health Score', fontsize=18)
axes[0, 1].tick_params(axis='x', rotation=45, labelsize=16)

# Add custom legend for the box plot
axes[0, 1].legend(legend_labels, occupation_color_map.keys(), title='Occupation', loc='upper left', bbox_to_anchor=(1, 1), fontsize=16)

# Plot 3: Count plot of Occupation (to show how many records per occupation)
countplot = sns.countplot(x='Occupation', data=df_train, ax=axes[1, 0], palette=occupation_color_map)
axes[1, 0].set_title('Count of Records by Occupation', fontsize=24, fontweight='bold')
axes[1, 0].set_xlabel('Occupation', fontsize=18)
axes[1, 0].set_ylabel('Count', fontsize=18)
axes[1, 0].tick_params(axis='x', rotation=45, labelsize=16)

# Add custom legend for the count plot
axes[1, 0].legend(legend_labels, occupation_color_map.keys(), title='Occupation', loc='upper left', bbox_to_anchor=(1, 1), fontsize=16)

# Add values above the bars in the count plot
for p in countplot.patches:
    height = p.get_height()
    countplot.text(p.get_x() + p.get_width() / 2., height + 2, str(int(height)), ha="center", fontsize=16, fontweight='bold')

# Plot 4: Grouped Histogram for Health Score distribution by Occupation
sns.histplot(data=df_train, x='Health Score', hue='Occupation', multiple='stack', kde=True, ax=axes[1, 1], palette=occupation_color_map, bins=20)
axes[1, 1].set_title('Grouped Health Score Distribution by Occupation (Histogram)', fontsize=24, fontweight='bold')
axes[1, 1].set_xlabel('Health Score', fontsize=18)
axes[1, 1].set_ylabel('Frequency', fontsize=18)

# Add custom legend for the histogram
axes[1, 1].legend(legend_labels, occupation_color_map.keys(), title='Occupation', loc='upper left', bbox_to_anchor=(1, 1), fontsize=16)

# Adjust layout for better spacing
plt.tight_layout(pad=6.0)  # Increased padding for better spacing

# Show the plots
plt.show()



# Find the most frequent Occupation for each Location
most_frequent_occupation_by_location = df_train.groupby('Location')['Occupation'].agg(lambda x: x.mode()[0])

# Output the most frequent Occupation for each Location
print("Most Frequent Occupation for each Location:\n", most_frequent_occupation_by_location)



# Calculate the distribution of Occupation for each Location
occupation_by_location = df_train.groupby('Location')['Occupation'].value_counts(normalize=True).unstack().fillna(0)

# Calculate the mean, max, and count of Health Score for each Location and Occupation
health_score_by_location_occupation = df_train.groupby(['Location', 'Occupation'])['Health Score'].agg(['count', 'mean', 'max'])

# Output the results
display("Occupation Distribution within each Location:", occupation_by_location)
display("Health Score by Location and Occupation:", health_score_by_location_occupation)



# Impute missing values in 'Occupation' with the mode (most frequent value)
occupation_imputer = SimpleImputer(strategy='most_frequent')

# Ensure the input to fit_transform is a 2D array by selecting the column as DataFrame
df_train['Occupation'] = occupation_imputer.fit_transform(df_train[['Occupation']]).flatten()
df_test['Occupation'] = occupation_imputer.transform(df_test[['Occupation']]).flatten()

# Define the custom color palette for each plot
occupation_palette = ['#B77A00', '#001F2D', '#29002d']

# Set up the figure and axes for subplots with much larger figsize
fig, axes = plt.subplots(2, 2, figsize=(40, 32))  # Much larger figsize

# Plot 1: Stacked bar plot of Occupation distribution within each Location
occupation_by_location.plot(kind='bar', stacked=True, ax=axes[0, 0], color=occupation_palette)
axes[0, 0].set_title('Occupation Distribution within Each Location', fontsize=30, fontweight='bold')
axes[0, 0].set_xlabel('Location', fontsize=24)
axes[0, 0].set_ylabel('Proportion', fontsize=24)
axes[0, 0].tick_params(axis='x', rotation=45, labelsize=24, width=2, length=10, colors='black', grid_color='gray', grid_alpha=0.5)
axes[0, 0].tick_params(axis='y', labelsize=24, width=2, length=10, colors='black', grid_color='gray', grid_alpha=0.5)
axes[0, 0].legend(title='Occupation', fontsize=20, bbox_to_anchor=(1.05, 1), loc='upper left')  # Legend outside

# Add bold values on bars for Plot 1
for p in axes[0, 0].patches:
    height = p.get_height()
    width = p.get_width()
    x, y = p.get_xy()  # Get the x and y coordinates of the rectangle
    axes[0, 0].text(x + width / 2, y + height / 2, f'{height:.2f}', ha='center', va='center', fontsize=18, color='white', fontweight='bold')

# Plot 2: Boxplot of Health Score by Location and Occupation
sns.boxplot(x='Location', y='Health Score', hue='Occupation', data=df_train, ax=axes[0, 1], palette=occupation_palette)
axes[0, 1].set_title('Health Score Distribution by Location and Occupation (Boxplot)', fontsize=30, fontweight='bold')
axes[0, 1].set_xlabel('Location', fontsize=24)
axes[0, 1].set_ylabel('Health Score', fontsize=24)
axes[0, 1].tick_params(axis='x', rotation=45, labelsize=24, width=2, length=10, colors='black', grid_color='gray', grid_alpha=0.5)
axes[0, 1].tick_params(axis='y', labelsize=24, width=2, length=10, colors='black', grid_color='gray', grid_alpha=0.5)
axes[0, 1].legend(title='Occupation', fontsize=20, bbox_to_anchor=(1.05, 1), loc='upper left')  # Legend outside

# Plot 3: Clustered bar plot of Health Score count by Location and Occupation
health_score_by_location_occupation['count'].unstack().plot(kind='bar', ax=axes[1, 0], color=occupation_palette, width=0.8)
axes[1, 0].set_title('Health Score Count by Location and Occupation', fontsize=30, fontweight='bold')
axes[1, 0].set_xlabel('Location', fontsize=24)
axes[1, 0].set_ylabel('Count', fontsize=24)
axes[1, 0].tick_params(axis='x', rotation=45, labelsize=24, width=2, length=10, colors='black', grid_color='gray', grid_alpha=0.5)
axes[1, 0].tick_params(axis='y', labelsize=24, width=2, length=10, colors='black', grid_color='gray', grid_alpha=0.5)
axes[1, 0].legend(title='Occupation', fontsize=20, bbox_to_anchor=(1.05, 1), loc='upper left')  # Legend outside

# Add bold values on bars for Plot 3
for p in axes[1, 0].patches:
    height = p.get_height()
    width = p.get_width()
    x, y = p.get_xy()  # Get the x and y coordinates of the rectangle
    axes[1, 0].text(x + width / 2, y + height / 2, f'{height:.0f}', ha='center', va='center', fontsize=18, color='white', fontweight='bold')

# Plot 4: Clustered bar plot of mean Health Score by Location and Occupation
health_score_by_location_occupation['mean'].unstack().plot(kind='bar', ax=axes[1, 1], color=occupation_palette, width=0.8)
axes[1, 1].set_title('Mean Health Score by Location and Occupation', fontsize=30, fontweight='bold')
axes[1, 1].set_xlabel('Location', fontsize=24)
axes[1, 1].set_ylabel('Mean Health Score', fontsize=24)
axes[1, 1].tick_params(axis='x', rotation=45, labelsize=24, width=2, length=10, colors='black', grid_color='gray', grid_alpha=0.5)
axes[1, 1].tick_params(axis='y', labelsize=24, width=2, length=10, colors='black', grid_color='gray', grid_alpha=0.5)
axes[1, 1].legend(title='Occupation', fontsize=20, bbox_to_anchor=(1.05, 1), loc='upper left')  # Legend outside

# Add bold values on bars for Plot 4
for p in axes[1, 1].patches:
    height = p.get_height()
    width = p.get_width()
    x, y = p.get_xy()  # Get the x and y coordinates of the rectangle
    axes[1, 1].text(x + width / 2, y + height / 2, f'{height:.2f}', ha='center', va='center', fontsize=18, color='white', fontweight='bold')

# Adjust layout for better spacing
plt.tight_layout(pad=8.0)  # Increased padding for better spacing

# Show the plots
plt.show()



# Define a threshold for low Health Score (e.g., Health Score < 20)
low_health_score_threshold = 20
# Define a threshold for high Health Score (e.g., Health Score >= 50)
high_health_score_threshold = 50
# Calculate the proportion of low-health individuals (Health Score < threshold) by Location and Occupation
low_health_score_proportion = df_train[df_train['Health Score'] < low_health_score_threshold] \
    .groupby(['Location', 'Occupation']).size() / df_train.groupby(['Location', 'Occupation']).size()

# Calculate the proportion of high-health individuals (Health Score >= threshold) by Location and Occupation
high_health_score_proportion = df_train[df_train['Health Score'] >= high_health_score_threshold] \
    .groupby(['Location', 'Occupation']).size() / df_train.groupby(['Location', 'Occupation']).size()
# Combine both low and high health score proportions into a single DataFrame for easy comparison
health_score_comparison = pd.DataFrame({
    'Low Health Score Proportion': low_health_score_proportion,
    'High Health Score Proportion': high_health_score_proportion
}).fillna(0)  # Fill missing values with 0 if there are any categories without low or high health individuals

# Output the comparative analysis
display("Comparative Analysis of Low and High Health Score Proportions by Location and Occupation:", health_score_comparison)



# Reshape the data for the bar plot
health_score_comparison_reset = health_score_comparison.reset_index()

# Aggregate proportions for the pie chart
total_low = health_score_comparison['Low Health Score Proportion'].sum()
total_high = health_score_comparison['High Health Score Proportion'].sum()

# Data for the pie chart
pie_data = [total_low, total_high]
pie_labels = ['Low Health Score', 'High Health Score']
occupation_palette = ['#B77A00', '#001F2D']

# Create the subplots
fig, axes = plt.subplots(1, 2, figsize=(25, 10))

# Bar plot
sns.barplot(
    data=health_score_comparison_reset.melt(
        id_vars=['Location', 'Occupation'],
        value_vars=['Low Health Score Proportion', 'High Health Score Proportion']
    ),
    x='Location',
    y='value',
    hue='variable',
    ci=None,
    palette=occupation_palette,
    ax=axes[0]
)
axes[0].set_title('Comparative Proportions of Low and High Health Scores by Location and Occupation', fontsize=18, fontweight='bold')
axes[0].set_xlabel('Location', fontsize=16, fontweight='bold')
axes[0].set_ylabel('Proportion', fontsize=16, fontweight='bold')
axes[0].tick_params(axis='x', rotation=45, labelsize=14, labelcolor='black')
axes[0].tick_params(axis='y', labelsize=14, labelcolor='black')

# Move legend outside the plot
axes[0].legend(
    title='Health Score Category', 
    fontsize=14, 
    loc='upper left', 
    bbox_to_anchor=(1, 1),
    title_fontsize=14
)

# Add values above bars
for p in axes[0].patches:
    height = p.get_height()
    if not pd.isna(height):
        axes[0].text(
            p.get_x() + p.get_width() / 2., 
            height + 0.01, 
            f'{height:.4f}', 
            ha='center', fontsize=12, color='black', fontweight='bold'
        )

# Pie chart
wedges, texts, autotexts = axes[1].pie(
    pie_data,
    labels=pie_labels,
    autopct='%1.3f%%',
    startangle=90,
    colors=occupation_palette,
    textprops={'fontsize': 14, 'fontweight': 'bold'},
    wedgeprops={'edgecolor': 'black', 'linewidth': 1.2}
)

# Format pie chart values
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

axes[1].set_title('Overall Proportions of Low and High Health Scores', fontsize=18, fontweight='bold')

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(right=0.8)  # Adjust to make space for the legend
plt.show()



# Calculate the mean values of Age, Health Score, and Annual Income for each Occupation
occupation_health_income_age_mean = df_train.groupby('Occupation')[['Age', 'Health Score', 'Annual Income']].mean()

# Output the mean values for Age, Health Score, and Annual Income for each Occupation
print("Mean values of Age, Health Score, and Annual Income for each Occupation:\n", occupation_health_income_age_mean)



# Define the custom color palette for each plot
occupation_palette = ['#B77A00', '#001F2D', '#29002d']

# Set a style for the plots
sns.set_style("whitegrid")

# Create subplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(24, 16))  # 2 rows and 3 columns

# --- First Row: Bar Plots ---
# Vertical Bar Plot for Age
sns.barplot(
    x=occupation_health_income_age_mean.index,  # 'Occupation' on the y-axis
    y=occupation_health_income_age_mean['Age'],  # 'Mean Age' on the x-axis
    color=occupation_palette[0],
    ax=axes[0, 0]
)
axes[0, 0].set_title('Mean Age by Occupation', fontsize=18, fontweight='bold', color=occupation_palette[0])
axes[0, 0].set_ylabel('Occupation', fontsize=16, fontweight='bold', color=occupation_palette[0])
axes[0, 0].set_xlabel('Mean Age', fontsize=16, fontweight='bold', color=occupation_palette[0])
axes[0, 0].tick_params(axis='x', labelsize=14, rotation=45, length=6)
axes[0, 0].tick_params(axis='y', labelsize=14, length=6)

# Annotating bars with values
for p in axes[0, 0].patches:
    axes[0, 0].annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 4., p.get_height()),
                        ha='center', va='center', fontsize=14, fontweight='bold', color='black', xytext=(0, 10), textcoords='offset points')

# Vertical Bar Plot for Health Score
sns.barplot(
    x=occupation_health_income_age_mean.index,  # 'Occupation' on the y-axis
    y=occupation_health_income_age_mean['Health Score'],  # 'Mean Health Score' on the x-axis
    color=occupation_palette[1],
    ax=axes[0, 1]
)
axes[0, 1].set_title('Mean Health Score by Occupation', fontsize=18, fontweight='bold', color=occupation_palette[1])
axes[0, 1].set_ylabel('Occupation', fontsize=16, fontweight='bold', color=occupation_palette[1])
axes[0, 1].set_xlabel('Mean Health Score', fontsize=16, fontweight='bold', color=occupation_palette[1])
axes[0, 1].tick_params(axis='x', labelsize=14, rotation=45, length=6)
axes[0, 1].tick_params(axis='y', labelsize=14, length=6)

# Annotating bars with values
for p in axes[0, 1].patches:
    axes[0, 1].annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 4., p.get_height()),
                        ha='center', va='center', fontsize=14, fontweight='bold', color='black', xytext=(0, 10), textcoords='offset points')

# Vertical Bar Plot for Annual Income
sns.barplot(
    x=occupation_health_income_age_mean.index,  # 'Occupation' on the y-axis
    y=occupation_health_income_age_mean['Annual Income'],  # 'Mean Annual Income' on the x-axis
    color=occupation_palette[2],
    ax=axes[0, 2]
)
axes[0, 2].set_title('Mean Annual Income by Occupation', fontsize=18, fontweight='bold', color=occupation_palette[2])
axes[0, 2].set_ylabel('Occupation', fontsize=16, fontweight='bold', color=occupation_palette[2])
axes[0, 2].set_xlabel('Mean Annual Income', fontsize=16, fontweight='bold', color=occupation_palette[2])
axes[0, 2].tick_params(axis='x', labelsize=14, rotation=45, length=6)
axes[0, 2].tick_params(axis='y', labelsize=14, length=6)

# Annotating bars with values
for p in axes[0, 2].patches:
    axes[0, 2].annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=14, fontweight='bold', color='black', xytext=(0, 10), textcoords='offset points')

# --- Second Row: Pie Charts ---
# Pie Chart for Mean Age
age_labels = occupation_health_income_age_mean.index
age_sizes = occupation_health_income_age_mean['Age']
pie1 = axes[1, 0].pie(age_sizes, labels=age_labels, autopct='%1.1f%%', colors=occupation_palette, startangle=90, 
                      textprops={'fontsize': 14, 'fontweight': 'bold', 'color': 'white'}, wedgeprops={'edgecolor': 'black'})
axes[1, 0].set_title('Mean Age Distribution by Occupation', fontsize=18, fontweight='bold', color=occupation_palette[0])

# Add legend for Age
axes[1, 0].legend(age_labels, loc='upper right', bbox_to_anchor=(1.3, 1), fontsize=14, title="Occupations")

# Pie Chart for Mean Health Score
health_score_labels = occupation_health_income_age_mean.index
health_score_sizes = occupation_health_income_age_mean['Health Score']
pie2 = axes[1, 1].pie(health_score_sizes, labels=health_score_labels, autopct='%1.1f%%', colors=occupation_palette, startangle=90, 
                      textprops={'fontsize': 14, 'fontweight': 'bold', 'color': 'white'}, wedgeprops={'edgecolor': 'black'})
axes[1, 1].set_title('Mean Health Score Distribution by Occupation', fontsize=18, fontweight='bold', color=occupation_palette[1])

# Add legend for Health Score
axes[1, 1].legend(health_score_labels, loc='upper right', bbox_to_anchor=(1.3, 1), fontsize=14, title="Occupations")

# Pie Chart for Mean Annual Income
income_labels = occupation_health_income_age_mean.index
income_sizes = occupation_health_income_age_mean['Annual Income']
pie3 = axes[1, 2].pie(income_sizes, labels=income_labels, autopct='%1.1f%%', colors=occupation_palette, startangle=90, 
                      textprops={'fontsize': 14, 'fontweight': 'bold', 'color': 'white'}, wedgeprops={'edgecolor': 'black'})
axes[1, 2].set_title('Mean Annual Income Distribution by Occupation', fontsize=18, fontweight='bold', color=occupation_palette[2])

# Add legend for Annual Income
axes[1, 2].legend(income_labels, loc='upper right', bbox_to_anchor=(1.3, 1), fontsize=14, title="Occupations")

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(hspace=0.4, wspace=0.6)
plt.show()



# Analyze the frequency of different Policy Types in each Location
policy_type_by_location = df_train.groupby('Policy Type')['Location'].value_counts().unstack().fillna(0)

# Output the distribution of Policy Type for each Location
print("Distribution of Policy Type by Location:\n", policy_type_by_location)



# Set a style for the plot
sns.set(style="whitegrid")

# Grouping the data by Policy Type and Location
policy_type_by_location = df_train.groupby('Policy Type')['Location'].value_counts().unstack().fillna(0)

# Define the custom color palette for each Policy Type (use colors from gender_palette)
gender_palette = ['#B77A00', '#001F2D', '#29002d', '#b85712']

# Create a bar plot for the distribution of Policy Type by Location
ax = policy_type_by_location.plot(kind='bar', stacked=True, figsize=(18, 8), color=gender_palette)

# Adding titles and labels
ax.set_title('Distribution of Policy Type by Location', fontsize=16, fontweight='bold')
ax.set_xlabel('Policy Type', fontsize=14)
ax.set_ylabel('Frequency', fontsize=14)
ax.tick_params(axis='x', rotation=45, labelsize=12)
ax.tick_params(axis='y', labelsize=12)

# Display the value counts inside the bars with bold text
for p in ax.patches:
    height = p.get_height()
    width = p.get_width()
    x = p.get_x() + width / 2
    y = p.get_y() + height / 2  # Positioning the text inside the bar
    ax.annotate(f'{height:.0f}', (x, y), ha='center', va='center', fontsize=10, color='white', fontweight='bold')

# Show the plot
plt.tight_layout()
plt.show()



# Group by 'Policy Type' and calculate the mean values of 'Previous Claims', 'Credit Score', and 'Vehicle Age',
# as well as the count of these columns
claims_by_policy_type = df_train.groupby('Policy Type').agg(
    Previous_Claims_Mean=('Previous Claims', 'mean'),
    Credit_Score_Mean=('Credit Score', 'mean'),
    Vehicle_Age_Mean=('Vehicle Age', 'mean'),
    Policy_Type_Count=('Policy Type', 'size'),
    Previous_Claims_Count=('Previous Claims', 'count'),
    Credit_Score_Count=('Credit Score', 'count'),
    Vehicle_Age_Count=('Vehicle Age', 'count')
)

# Reset the index to make 'Policy Type' a column, and put all column names in one row
claims_by_policy_type_reset = claims_by_policy_type.reset_index()

# Output the detailed summary by Policy Type
display(claims_by_policy_type_reset)



# Set a style for the plot
sns.set(style="whitegrid")

# Define the custom color palette for each metric (mean and count)
mean_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Default mean colors
count_colors = ['#d62728', '#9467bd', '#8c564b']  # Default count colors

# Custom color palette for 'Credit Score' (used for both mean and count)
gender_palette = ['#B77A00', '#001F2D', '#29002d', '#b85712']

# Create a figure with two subplots (one for mean and one for count)
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Bar width and position adjustment
bar_width = 0.25
index = np.arange(len(claims_by_policy_type_reset))

# Plotting Mean Values in the first subplot (not stacked)
mean_bars_1 = axes[0].bar(index, claims_by_policy_type_reset['Previous_Claims_Mean'], bar_width, color=mean_colors[0], label='Mean Previous Claims')
mean_bars_2 = axes[0].bar(index + bar_width, claims_by_policy_type_reset['Credit_Score_Mean'], bar_width, color=gender_palette[0], label='Mean Credit Score')  # Apply custom color for Credit Score
mean_bars_3 = axes[0].bar(index + 2 * bar_width, claims_by_policy_type_reset['Vehicle_Age_Mean'], bar_width, color=mean_colors[2], label='Mean Vehicle Age')

# Adding values above the bars for Mean Values
for bar in mean_bars_1:
    yval = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width() / 2, yval + 0.05, f'{yval:.2f}', ha='center', va='bottom', fontsize=10)
for bar in mean_bars_2:
    yval = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width() / 2, yval + 0.05, f'{yval:.2f}', ha='center', va='bottom', fontsize=10)
for bar in mean_bars_3:
    yval = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width() / 2, yval + 0.05, f'{yval:.2f}', ha='center', va='bottom', fontsize=10)

# Adding titles, labels, and legend for the first subplot
axes[0].set_title('Mean Values by Policy Type', fontsize=16, fontweight='bold')
axes[0].set_xlabel('Policy Type', fontsize=14)
axes[0].set_ylabel('Mean Values', fontsize=14)
axes[0].tick_params(axis='x', rotation=45, labelsize=12)
axes[0].tick_params(axis='y', labelsize=12)
axes[0].set_xticks(index + bar_width)
axes[0].set_xticklabels(claims_by_policy_type_reset['Policy Type'], fontsize=12)
axes[0].legend(title='Mean Metrics', loc='upper left', bbox_to_anchor=(1, 1))

# Plotting Count Values in the second subplot (not stacked)
count_bars_1 = axes[1].bar(index, claims_by_policy_type_reset['Previous_Claims_Count'], bar_width, color=count_colors[0], label='Count Previous Claims')
count_bars_2 = axes[1].bar(index + bar_width, claims_by_policy_type_reset['Credit_Score_Count'], bar_width, color=gender_palette[1], label='Count Credit Score')  # Apply custom color for Credit Score
count_bars_3 = axes[1].bar(index + 2 * bar_width, claims_by_policy_type_reset['Vehicle_Age_Count'], bar_width, color=count_colors[2], label='Count Vehicle Age')

# Adding values above the bars for Count Values
for bar in count_bars_1:
    yval = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width() / 2, yval + 0.05, f'{yval:.0f}', ha='center', va='bottom', fontsize=10)
for bar in count_bars_2:
    yval = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width() / 2, yval + 0.05, f'{yval:.0f}', ha='center', va='bottom', fontsize=10)
for bar in count_bars_3:
    yval = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width() / 2, yval + 0.05, f'{yval:.0f}', ha='center', va='bottom', fontsize=10)

# Adding titles, labels, and legend for the second subplot
axes[1].set_title('Count Values by Policy Type', fontsize=16, fontweight='bold')
axes[1].set_xlabel('Policy Type', fontsize=14)
axes[1].set_ylabel('Count Values', fontsize=14)
axes[1].tick_params(axis='x', rotation=45, labelsize=12)
axes[1].tick_params(axis='y', labelsize=12)
axes[1].set_xticks(index + bar_width)
axes[1].set_xticklabels(claims_by_policy_type_reset['Policy Type'], fontsize=12)
axes[1].legend(title='Count Metrics', loc='upper left', bbox_to_anchor=(1, 1))

# Adjust the layout to avoid overlapping
plt.tight_layout()

# Show the plot
plt.show()



# Create bins for Vehicle Age
vehicle_age_bins = [0, 2, 5, 10, 20]
vehicle_age_labels = ['0-2', '2-5', '5-10', '10-20']

# Categorize Vehicle Age into bins
df_train['Vehicle Age Range'] = pd.cut(df_train['Vehicle Age'], bins=vehicle_age_bins, labels=vehicle_age_labels)

# Analyze Previous Claims by Vehicle Age and Credit Score Range
claims_by_vehicle_age_credit = df_train.groupby(['Credit Score', 'Vehicle Age Range'])['Previous Claims'].mean().unstack()

# Output the relationship between Vehicle Age, Credit Score, and Previous Claims
display("Previous Claims by Vehicle Age and Credit Score:", claims_by_vehicle_age_credit.head(20))




# Create bins for Vehicle Age
vehicle_age_bins = [0, 2, 5, 10, 20]
vehicle_age_labels = ['0-2', '2-5', '5-10', '10-20']

# Categorize Vehicle Age into bins
df_train['Vehicle Age Range'] = pd.cut(df_train['Vehicle Age'], bins=vehicle_age_bins, labels=vehicle_age_labels)

# Analyze Previous Claims by Vehicle Age and Credit Score Range
claims_by_vehicle_age_credit = df_train.groupby(['Credit Score', 'Vehicle Age Range'])['Previous Claims'].mean().unstack()

# Output the relationship between Vehicle Age, Credit Score, and Previous Claims
claims_top_20 = claims_by_vehicle_age_credit.head(20)

# Reset the index so 'Credit Score' and 'Vehicle Age Range' become columns
claims_top_20 = claims_top_20.reset_index()

# Create a custom color palette for 'Credit Score'
gender_palette = ['#B77A00', '#001F2D', '#29002d', '#b85712']

# Create a scatter plot for 'Previous Claims' vs 'Vehicle Age Range', colored by 'Credit Score'
plt.figure(figsize=(12, 6))
sns.scatterplot(data=claims_top_20.melt(id_vars=['Credit Score'], value_vars=claims_top_20.columns[1:]),
                x='Vehicle Age Range', y='value', hue='Credit Score', palette=gender_palette, s=100, marker='o')

# Customize the plot
plt.title('Scatter Plot of Previous Claims by Vehicle Age Range and Credit Score', fontsize=16, fontweight='bold')
plt.xlabel('Vehicle Age Range', fontsize=14)
plt.ylabel('Average Previous Claims', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)

# Adjust legend position to be outside the plot
plt.legend(title='Credit Score', fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.tight_layout()
plt.show()



# Convert 'Customer Feedback' to numeric, coercing errors to NaN
df_train['Customer Feedback'] = pd.to_numeric(df_train['Customer Feedback'], errors='coerce')

# Calculate the average Insurance Duration by Policy Type
insurance_duration_by_policy = df_train.groupby('Policy Type')['Insurance Duration'].mean()

# Calculate the total number of policies by Policy Type and Insurance Duration
policy_count_by_type_duration = df_train.groupby(['Policy Type', 'Insurance Duration']).size().unstack(fill_value=0)

# Calculate the average Customer Feedback by Smoking Status
feedback_by_smoking_status = df_train.groupby('Smoking Status')['Customer Feedback'].mean()

# Calculate the frequency of Smoking Status by Policy Type
smoking_status_by_policy = df_train.groupby('Policy Type')['Smoking Status'].value_counts().unstack(fill_value=0)

# Merge all the results into a single DataFrame
pivot_table = pd.DataFrame({
    'Average Insurance Duration': insurance_duration_by_policy
})

# Merge the total number of policies by Policy Type and Insurance Duration
pivot_table = pivot_table.join(policy_count_by_type_duration)

# Merge the frequency of Smoking Status by Policy Type
pivot_table = pivot_table.join(smoking_status_by_policy)

# Reset the index to make 'Policy Type' a column and put all column names in one row
pivot_table_reset = pivot_table.reset_index()

# Output the combined pivot table
display("Combined Pivot Table:", pivot_table_reset)



# Group by 'Policy Type' and calculate the mean values of 'Previous Claims', 'Credit Score', and 'Vehicle Age',
# as well as the count of these columns
claims_by_policy_type = df_train.groupby('Policy Type').agg(
    Previous_Claims_Mean=('Previous Claims', 'mean'),
    Credit_Score_Mean=('Credit Score', 'mean'),
    Vehicle_Age_Mean=('Vehicle Age', 'mean'),
    Policy_Type_Count=('Policy Type', 'size'),
    Previous_Claims_Count=('Previous Claims', 'count'),
    Credit_Score_Count=('Credit Score', 'count'),
    Vehicle_Age_Count=('Vehicle Age', 'count')
)

# Reset the index to make 'Policy Type' a column, and put all column names in one row
claims_by_policy_type_reset = claims_by_policy_type.reset_index()

# Output the detailed summary by Policy Type
display(claims_by_policy_type_reset)


# Set the style for the plot
sns.set(style="whitegrid")

# Define a custom color palette based on the colors you provided
gender_palette = ['#B77A00', '#001F2D', '#29002d', '#b85712']

# Create the plot with multiple subplots (2 rows, 2 columns)
fig, axes = plt.subplots(2, 2, figsize=(20, 14), sharex=False)

# Define the policy types to use for the legend
policy_types = claims_by_policy_type_reset['Policy Type'].unique()

# Bar plot for the mean values of 'Previous Claims'
sns.barplot(x='Policy Type', y='Previous_Claims_Mean', data=claims_by_policy_type_reset, 
            hue='Policy Type', palette=gender_palette, ax=axes[0, 0])
axes[0, 0].set_title('Mean of Previous Claims by Policy Type', fontsize=18, fontweight='bold')
axes[0, 0].set_ylabel('Mean Previous Claims', fontsize=14)
axes[0, 0].tick_params(axis='x', rotation=45, labelsize=12)
axes[0, 0].tick_params(axis='y', labelsize=12)
axes[0, 0].tick_params(axis='x', direction='in', length=6)  # Add x-axis ticks

# Add the values above the bars
for p in axes[0, 0].patches:
    axes[0, 0].annotate(f'{p.get_height():.2f}', 
                        (p.get_x() + p.get_width() / 4., p.get_height()), 
                        ha='center', va='center', fontsize=12, 
                        xytext=(0, 8), textcoords='offset points')

# Bar plot for the mean values of 'Credit Score'
sns.barplot(x='Policy Type', y='Credit_Score_Mean', data=claims_by_policy_type_reset, 
            hue='Policy Type', palette=gender_palette, ax=axes[0, 1])
axes[0, 1].set_title('Mean of Credit Score by Policy Type', fontsize=18, fontweight='bold')
axes[0, 1].set_ylabel('Mean Credit Score', fontsize=14)
axes[0, 1].tick_params(axis='x', rotation=45, labelsize=12)
axes[0, 1].tick_params(axis='y', labelsize=12)
axes[0, 1].tick_params(axis='x', direction='in', length=6)  # Add x-axis ticks

# Add the values above the bars
for p in axes[0, 1].patches:
    axes[0, 1].annotate(f'{p.get_height():.2f}', 
                        (p.get_x() + p.get_width() / 4., p.get_height()), 
                        ha='center', va='center', fontsize=12, 
                        xytext=(0, 8), textcoords='offset points')

# Bar plot for the mean values of 'Vehicle Age'
sns.barplot(x='Policy Type', y='Vehicle_Age_Mean', data=claims_by_policy_type_reset, 
            hue='Policy Type', palette=gender_palette, ax=axes[1, 0])
axes[1, 0].set_title('Mean of Vehicle Age by Policy Type', fontsize=18, fontweight='bold')
axes[1, 0].set_ylabel('Mean Vehicle Age', fontsize=14)
axes[1, 0].tick_params(axis='x', rotation=45, labelsize=12)
axes[1, 0].tick_params(axis='y', labelsize=12)
axes[1, 0].tick_params(axis='x', direction='in', length=6)  # Add x-axis ticks

# Add the values above the bars
for p in axes[1, 0].patches:
    axes[1, 0].annotate(f'{p.get_height():.2f}', 
                        (p.get_x() + p.get_width() / 4., p.get_height()), 
                        ha='center', va='center', fontsize=12, 
                        xytext=(0, 8), textcoords='offset points')

# Bar plot for the counts of 'Previous Claims'
sns.barplot(x='Policy Type', y='Previous_Claims_Count', data=claims_by_policy_type_reset, 
            hue='Policy Type', palette=gender_palette, ax=axes[1, 1])
axes[1, 1].set_title('Count of Previous Claims by Policy Type', fontsize=18, fontweight='bold')
axes[1, 1].set_ylabel('Count of Previous Claims', fontsize=14)
axes[1, 1].tick_params(axis='x', rotation=45, labelsize=12)
axes[1, 1].tick_params(axis='y', labelsize=12)
axes[1, 1].tick_params(axis='x', direction='in', length=6)  # Add x-axis ticks

# Add the values above the bars
for p in axes[1, 1].patches:
    axes[1, 1].annotate(f'{p.get_height():.0f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', fontsize=12, 
                        xytext=(0, 8), textcoords='offset points')

# Set common x-axis label for all subplots
fig.text(0.5, 0.04, 'Policy Type', ha='center', fontsize=16, fontweight='bold')

# Move the legend outside the plot to the right
for ax in axes.flat:
    ax.legend(title='Policy Type', loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=12)

# Adjust layout for better spacing
plt.tight_layout(pad=4.0)

# Display all the plots
plt.show()



# Group by 'Policy Type' and calculate the average 'Insurance Duration'
insurance_duration_by_policy = df_train.groupby('Policy Type')['Insurance Duration'].mean()

# Output the average 'Insurance Duration' by Policy Type
display("Average Insurance Duration by Policy Type:", insurance_duration_by_policy)

# Group by 'Smoking Status' to calculate the average 'Customer Feedback'
feedback_by_smoking_status = df_train.groupby('Smoking Status')['Customer Feedback'].mean()
print("=============================================================================")
# Calculate the count of 'Smoking Status' for each 'Policy Type'
smoking_status_by_policy = df_train.groupby('Policy Type')['Smoking Status'].value_counts().unstack(fill_value=0)

# Output the frequency of Smoking Status by 'Policy Type'
display("Frequency of Smoking Status by Policy Type:", smoking_status_by_policy)



# Define custom color palette
gender_palette = ['#B77A00', '#001F2D', '#29002d', '#b85712']

# Group by 'Policy Type' and calculate the average 'Insurance Duration'
insurance_duration_by_policy = df_train.groupby('Policy Type')['Insurance Duration'].mean()

# Group by 'Smoking Status' to calculate the average 'Customer Feedback'
feedback_by_smoking_status = df_train.groupby('Smoking Status')['Customer Feedback'].mean()

# Group by 'Policy Type' to calculate the count of 'Smoking Status'
smoking_status_by_policy = df_train.groupby('Policy Type')['Smoking Status'].value_counts().unstack(fill_value=0)

# Create a subplot with 1 row and 2 columns
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

# Pie Chart for the distribution of 'Insurance Duration' by 'Policy Type'
wedges, texts, autotexts = axes[0].pie(
    insurance_duration_by_policy, 
    labels=insurance_duration_by_policy.index, 
    autopct='%1.1f%%', 
    startangle=90, 
    colors=gender_palette[:len(insurance_duration_by_policy)],
    textprops={'color': 'white', 'fontweight': 'bold'},  # Text color and bold inside pie chart
    wedgeprops={'edgecolor': 'black'},
    labeldistance=1.15  # Move labels outside
)

# Make the pie chart labels bold
for autotext in autotexts:
    autotext.set_fontweight('bold')

# Change label color to black
for text in texts:
    text.set_color('black')

# Set pie chart title and add labels
axes[0].set_title('Average Insurance Duration by Policy Type', fontsize=14, color='darkblue')

# Add legend to pie chart
axes[0].legend(wedges, insurance_duration_by_policy.index, title="Policy Type", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=12)

# Bar Chart for 'Smoking Status' frequency by 'Policy Type'
smoking_status_by_policy.plot(kind='bar', stacked=True, ax=axes[1], color=gender_palette, width=0.8)

# Add the values inside the stacked bars
for p in axes[1].patches:
    height = p.get_height()
    width = p.get_width()
    x, y = p.get_xy()  # Get the x and y position of the rectangle
    axes[1].text(x + width/2, y + height/2, f'{int(height)}', ha='center', va='center', color='white', fontsize=12, fontweight='bold')

# Add title, labels, and other enhancements to the bar chart
axes[1].set_title('Frequency of Smoking Status by Policy Type', fontsize=14, color='darkgreen')
axes[1].set_ylabel('Count of Smoking Status')
axes[1].set_xlabel('Policy Type')

# Improve layout and display
plt.tight_layout()
plt.show()



# Group by 'Policy Type' and 'Insurance Duration' to calculate the total number of policies
policy_count_by_type_duration = df_train.groupby(['Policy Type', 'Insurance Duration']).size().unstack(fill_value=0)

# Output the number of policies by 'Policy Type' and 'Insurance Duration'
display("Number of Policies by Policy Type and Insurance Duration:", policy_count_by_type_duration)


# Define a custom color palette
gender_palette = ['#B77A00', '#001F2D', '#29002d', '#b85712', '#34b1eb', '#eb3434', '#61eb34', '#776f7a', '#ff7d03']

# Group by 'Policy Type' and 'Insurance Duration' to calculate the total number of policies
policy_count_by_type_duration = df_train.groupby(['Policy Type', 'Insurance Duration']).size().unstack(fill_value=0)

# Create a subplot with 1 row and 1 column for the bar chart
fig, ax = plt.subplots(figsize=(10, 6))

# Stacked Bar Chart for Policy Count by Policy Type and Insurance Duration
# Apply the custom color palette to the stacked bars
bars = policy_count_by_type_duration.plot(kind='bar', stacked=True, ax=ax, width=0.8, color=gender_palette[:len(policy_count_by_type_duration.columns)])

# Set title and labels for the chart
ax.set_title('Stacked Bar Chart: Policies by Type and Duration', fontsize=14, color='darkgreen')
ax.set_ylabel('Number of Policies', fontsize=12)
ax.set_xlabel('Policy Type', fontsize=12)

# Add custom labels for Policy Type (ensure the index of the grouped data has correct Policy Types)
ax.set_xticklabels(policy_count_by_type_duration.index, rotation=45, ha='right', fontsize=10)

# Annotate values inside each bar
for container in ax.containers:
    for bar in container:
        # Get bar position and height
        height = bar.get_height()
        if height > 0:  # Only annotate non-zero bars
            x = bar.get_x() + bar.get_width() / 2  # Center of the bar
            y = bar.get_y() + height / 2  # Middle of the bar segment
            value = int(height)  # Get the value (as an integer)
            ax.text(x, y, str(value), ha='center', va='center', fontsize=10, color='white', fontweight='bold')

# Add legend outside of the plot
ax.legend(title='Insurance Duration', loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

# Improve layout and display
plt.tight_layout()
plt.show()



# Convert 'Policy Start Date' to datetime format (if not already in datetime)
df_train['Policy Start Date'] = pd.to_datetime(df_train['Policy Start Date'], errors='coerce')

# Extract the year and month from the Policy Start Date for analysis
df_train['Policy Start Year'] = df_train['Policy Start Date'].dt.year
df_train['Policy Start Month'] = df_train['Policy Start Date'].dt.month

# Group by Policy Start Year and Month to calculate the average Insurance Duration
insurance_duration_by_start_date = df_train.groupby(['Policy Start Year', 'Policy Start Month'])['Insurance Duration'].mean()

# Get the top 20 highest and lowest values
top_20_insurance_duration = insurance_duration_by_start_date.nlargest(20)
bottom_20_insurance_duration = insurance_duration_by_start_date.nsmallest(20)

# Output the top 20 highest and bottom 20 lowest average Insurance Duration by Policy Start Date
print("Highest Average Insurance Duration by Policy Start Date:\n", top_20_insurance_duration)
print("Lowest Average Insurance Duration by Policy Start Date:\n", bottom_20_insurance_duration)




# Set the style for the plot
sns.set(style="whitegrid")

# Combine the top 20 and bottom 20 values into a single DataFrame for easier plotting
combined_insurance_duration = pd.concat([top_20_insurance_duration, bottom_20_insurance_duration])

# Reset index for easier plotting and manipulation
combined_insurance_duration = combined_insurance_duration.reset_index()

# Define colors for the top and bottom groups
color_palette = gender_palette[:2]  # Use the first two colors from the palette for the top and bottom
top_color = color_palette[0]  # Color for top 20 highest
bottom_color = color_palette[1]  # Color for bottom 20 lowest

# Create a new column to categorize the data as 'Top 20' or 'Bottom 20'
combined_insurance_duration['Category'] = ['Top 20'] * len(top_20_insurance_duration) + ['Bottom 20'] * len(bottom_20_insurance_duration)

# Create the barplot
plt.figure(figsize=(14, 7))

# Plot the bars for the top 20 and bottom 20
ax = sns.barplot(x='Policy Start Year', y='Insurance Duration', hue='Category', data=combined_insurance_duration, dodge=True, palette=[top_color, bottom_color])

# Adding titles, labels, and customizing the plot
plt.title('Comparative Analysis of Insurance Duration by Policy Start Date', fontsize=16, fontweight='bold')
plt.xlabel('Policy Start Year and Month', fontsize=14)
plt.ylabel('Average Insurance Duration', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)

# Customizing the legend labels
handles, labels = plt.gca().get_legend_handles_labels()
labels = ['Highest Average Insurance Duration by Policy Start Date', 'Lowest Average Insurance Duration by Policy Start Date']
plt.legend(handles=handles, labels=labels, title='Category', loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)

# Add values above the bars (including zero values)
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height:.4f}', 
                (p.get_x() + p.get_width() / 2., height), 
                ha='center', va='center', fontsize=10, color='black', 
                xytext=(0, 5), textcoords='offset points')

# Display the plot
plt.tight_layout()
plt.show()



# Define Insurance Duration bins
insurance_duration_bins = [0, 5, 10]
insurance_duration_labels = ['0-5', '5-10']

# Categorize Insurance Duration into bins
df_train['Insurance Duration Category'] = pd.cut(df_train['Insurance Duration'], bins=insurance_duration_bins, labels=insurance_duration_labels)

# Calculate the frequency of Smoking Status within each Insurance Duration category
smoking_status_by_duration_category = df_train.groupby(['Insurance Duration Category', 'Smoking Status']).size().unstack(fill_value=0)

# Output the frequency of Smoking Status by Insurance Duration category
print("Frequency of Smoking Status by Insurance Duration Category:\n", smoking_status_by_duration_category)



# Set the style for the plots
sns.set(style="whitegrid")

# Define the color palette for Smoking Status (using similar shades as before)
smoking_palette = ['#B77A00', '#001F2D', '#29002d', '#b85712']

# Create the plot figure and axes
fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)

# Plot for Smoking Status in each Insurance Duration category
smoking_status_by_duration_category.plot(kind='bar', stacked=True, ax=axes[0], color=smoking_palette)

# Adding titles, labels, and legends for the first subplot
axes[0].set_title('Smoking Status by Insurance Duration (0-5 years)', fontsize=16, fontweight='bold')
axes[0].set_xlabel('Insurance Duration Category', fontsize=14)
axes[0].set_ylabel('Frequency', fontsize=14)
axes[0].tick_params(axis='x', rotation=45, labelsize=12)
axes[0].tick_params(axis='y', labelsize=12)

# Adding values inside the bars (bold text)
for p in axes[0].patches:
    height = p.get_height()
    width = p.get_width()
    x = p.get_x() + width / 2
    y = p.get_y() + height / 2  # Positioning the text inside the bar
    axes[0].annotate(f'{height:.0f}', (x, y), ha='center', va='center', fontsize=10, color='white', fontweight='bold')

# Create the second subplot with Smoking Status by Insurance Duration (5-10 years)
smoking_status_by_duration_category.plot(kind='bar', stacked=True, ax=axes[1], color=smoking_palette)

# Adding titles, labels, and legends for the second subplot
axes[1].set_title('Smoking Status by Insurance Duration (5-10 years)', fontsize=16, fontweight='bold')
axes[1].set_xlabel('Insurance Duration Category', fontsize=14)
axes[1].set_ylabel('Frequency', fontsize=14)
axes[1].tick_params(axis='x', rotation=45, labelsize=12)
axes[1].tick_params(axis='y', labelsize=12)

# Adding values inside the bars (bold text)
for p in axes[1].patches:
    height = p.get_height()
    width = p.get_width()
    x = p.get_x() + width / 2
    y = p.get_y() + height / 2  # Positioning the text inside the bar
    axes[1].annotate(f'{height:.0f}', (x, y), ha='center', va='center', fontsize=10, color='white', fontweight='bold')

# Adjust the layout for better presentation
plt.tight_layout()

# Show the plot
plt.show()



# Analysis of 'Exercise Frequency' column
exercise_frequency = df_train['Exercise Frequency']

# 1. Frequency Distribution of Exercise Frequency
exercise_frequency_counts = exercise_frequency.value_counts(normalize=True).sort_index()

# 2. Mean Premium Amount by Exercise Frequency
premium_by_exercise = df_train.groupby('Exercise Frequency')['Premium Amount'].mean()

# 3. Mean Age by Exercise Frequency
age_by_exercise = df_train.groupby('Exercise Frequency')['Age'].mean()

# 4. Total Premium Amount by Exercise Frequency
total_premium_by_exercise = df_train.groupby('Exercise Frequency')['Premium Amount'].sum()

# 5. Maximum Age by Exercise Frequency
max_age_by_exercise = df_train.groupby('Exercise Frequency')['Age'].max()

# Combine the results into one DataFrame
combined_exercise_analysis = pd.DataFrame({
    'Exercise Frequency Count': exercise_frequency_counts,
    'Mean Premium Amount': premium_by_exercise,
    'Mean Age': age_by_exercise,
    'Total Premium Amount': total_premium_by_exercise,
    'Max Age': max_age_by_exercise
})

# Reset the index to have column names as rows
combined_exercise_analysis = combined_exercise_analysis.reset_index()

# Display the combined results
display("Exercise Frequency Analysis:", combined_exercise_analysis)



import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# Define a custom color palette
gender_palette = ['#B77A00', '#001F2D', '#29002d', '#b85712', '#34b1eb', '#eb3434', '#61eb34', '#776f7a', '#ff7d03']

# Assuming df_train is your DataFrame and combined_exercise_analysis is properly defined
exercise_frequencies = df_train['Exercise Frequency'].unique()
colors = gender_palette[:len(exercise_frequencies)]  # Adjust the number of colors if needed

# Create a dictionary to map Exercise Frequency to a specific color
color_dict = dict(zip(exercise_frequencies, colors))

# Create subplots with 3 rows and 2 columns for different plots
fig, axes = plt.subplots(3, 2, figsize=(18, 16))  # Increased figsize for larger plots

# 1. Bar Chart for Exercise Frequency Count
for i, freq in enumerate(exercise_frequencies):
    count = combined_exercise_analysis[combined_exercise_analysis['Exercise Frequency'] == freq]['Exercise Frequency Count'].values[0]
    bar = axes[0, 0].bar(freq, count, color=color_dict[freq], label=freq)
    
    # Annotate with value above bars, formatted to three digits after the decimal point
    # Dynamically calculate the position to avoid the text going out of range
    height = bar[0].get_height()
    axes[0, 0].text(freq, height + 0.01, f'{height:.3f}', ha='center', fontsize=12, color='black', fontweight='bold')

axes[0, 0].set_title('Exercise Frequency Distribution', fontsize=16, color='darkblue')
axes[0, 0].set_ylabel('Frequency', fontsize=12)
axes[0, 0].set_xlabel('Exercise Frequency', fontsize=12)
axes[0, 0].legend(title='Exercise Frequency', loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=12)

# Increase the y-axis range to 0.260 for the first subplot
axes[0, 0].set_ylim(0, 0.280)  # This line will ensure the y-axis range starts from 0 to 0.260

# 2. Box Plot for Premium Amount by Exercise Frequency
sns.boxplot(x='Exercise Frequency', y='Premium Amount', data=df_train, ax=axes[0, 1], palette=color_dict)
axes[0, 1].set_title('Premium Amount by Exercise Frequency', fontsize=16, color='darkgreen')
axes[0, 1].set_ylabel('Premium Amount', fontsize=12)
axes[0, 1].set_xlabel('Exercise Frequency', fontsize=12)
axes[0, 1].legend(title='Exercise Frequency', loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=12)

# 3. Bar Chart for Mean Age by Exercise Frequency
for i, freq in enumerate(exercise_frequencies):
    mean_age = combined_exercise_analysis[combined_exercise_analysis['Exercise Frequency'] == freq]['Mean Age'].values[0]
    axes[1, 0].bar(freq, mean_age, color=color_dict[freq], label=freq)
    axes[1, 0].text(freq, mean_age + 0.01, f'{mean_age:.3f}', ha='center', fontsize=12, color='black', fontweight='bold')  # Add value above bars with 3 decimal places
axes[1, 0].set_title('Mean Age by Exercise Frequency', fontsize=16, color='purple')
axes[1, 0].set_ylabel('Mean Age', fontsize=12)
axes[1, 0].set_xlabel('Exercise Frequency', fontsize=12)
axes[1, 0].legend(title='Exercise Frequency', loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=12)

# 4. Stacked Bar Chart for Total Premium Amount by Exercise Frequency
for i, freq in enumerate(exercise_frequencies):
    total_premium = combined_exercise_analysis[combined_exercise_analysis['Exercise Frequency'] == freq]['Total Premium Amount'].values[0]
    axes[1, 1].bar(freq, total_premium, color=color_dict[freq], label=freq)
    axes[1, 1].text(freq, total_premium + 0.1, f'{total_premium:.1f}', ha='center', fontsize=10, color='black')  # Add value above bars
axes[1, 1].set_title('Total Premium Amount by Exercise Frequency', fontsize=16, color='red')
axes[1, 1].set_ylabel('Total Premium Amount', fontsize=12)
axes[1, 1].set_xlabel('Exercise Frequency', fontsize=12)
axes[1, 1].legend(title='Exercise Frequency', loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=12)

# Format y-axis of the 'Total Premium Amount' plot for short form
def format_currency(x, pos):
    """Function to format y-axis labels in short form (K, M, etc.)"""
    if x >= 1e6:
        return f'{x*1e-6:.1f}M'  # Millions
    elif x >= 1e3:
        return f'{x*1e-3:.1f}K'  # Thousands
    else:
        return f'{x:.0f}'

axes[1, 1].yaxis.set_major_formatter(FuncFormatter(format_currency))

# 5. Histogram for Age Distribution by Exercise Frequency (Grouped Mode)
for i, freq in enumerate(exercise_frequencies):
    filtered_data = df_train[df_train['Exercise Frequency'] == freq]['Age']
    n, bins, patches = axes[2, 0].hist(filtered_data, bins=10, alpha=0.7, color=color_dict[freq], label=freq, histtype='barstacked')
    # Removed the value above histogram bars
axes[2, 0].set_title('Age Distribution by Exercise Frequency', fontsize=16, color='orange')
axes[2, 0].set_ylabel('Count', fontsize=12)
axes[2, 0].set_xlabel('Age', fontsize=12)
axes[2, 0].legend(title='Exercise Frequency', loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=12)

# Hide the empty subplot at axes[2, 1]
axes[2, 1].axis('off')

# Improve layout and display
plt.tight_layout()
plt.show()



# 1. Bin Age into different categories
age_bins = [18, 22, 25, 30, 35, 40, 45, 50, 55, 60, 64]
age_labels = ['18-22', '23-25', '26-30', '31-35', '36-40', '41-45', '46-50', '51-55', '56-60', '61-64']
df_train['Age Group'] = pd.cut(df_train['Age'], bins=age_bins, labels=age_labels)

# 2. Analyze mean Premium Amount by Age Group
premium_by_age_group = df_train.groupby('Age Group')['Premium Amount'].mean().reset_index(name="Premium Amount")
print("\nMean Premium Amount by Age Group:")
print(premium_by_age_group)

# 3. Analyze the distribution of Premium Amount within each Age Group (only mean, count, max, and min)
premium_age_group_desc = df_train.groupby('Age Group')['Premium Amount'].agg(['mean', 'count', 'max', 'min'])
print("\nPremium Amount Distribution within Age Groups:")
print(premium_age_group_desc)



# Group by Gender and Age Group and calculate the mean Premium Amount
premium_by_age_gender = df_train.groupby(['Gender', 'Age Group'])['Premium Amount'].mean().reset_index(name="Premium Amount")

# Sort the results by 'Premium Amount' for easy comparison
premium_by_age_gender_desc = premium_by_age_gender.sort_values(by='Premium Amount', ascending=False)

display("Premium Amount by Age Group and Gender", premium_by_age_gender_desc)



# Group by Gender and Age Group and calculate the mean Premium Amount
premium_by_age_gender = df_train.groupby(['Gender', 'Age Group'])['Premium Amount'].mean().reset_index(name="Premium Amount")

# Sort the results by 'Premium Amount' for easy comparison
premium_by_age_gender_desc = premium_by_age_gender.sort_values(by='Premium Amount', ascending=False)

# Define a custom color palette for gender
gender_palette = ['#B77A00', '#001F2D', '#29002d', '#b85712', '#34b1eb', '#eb3434', '#61eb34', '#776f7a', '#ff7d03']

# Create the figure and axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

# --- Bar Plot Enhancements ---
sns.barplot(x='Premium Amount', y='Age Group', hue='Gender', data=premium_by_age_gender_desc, palette=gender_palette, ax=ax1)

# Add values above the bars in ax1 with styling adjustments
for p in ax1.patches:
    ax1.annotate(f'{p.get_width():.2f}', (p.get_x() + p.get_width() + 0.02, p.get_y() + p.get_height() / 2.),
                 ha='left', va='center', fontsize=9, color='black', fontweight='bold')

# Add title and labels for the barplot with custom font size and weight
ax1.set_title('Premium Amount by Age Group and Gender', fontsize=20, fontweight='bold', color='#2F4F4F')
ax1.set_xlabel('Mean Premium Amount', fontsize=14)
ax1.set_ylabel('Age Group', fontsize=14)
ax1.legend(title='Gender', loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=12, title_fontsize=14)

# Customize the grid and background for the bar plot
ax1.set_facecolor('#f5f5f5')
ax1.grid(True, axis='x', linestyle='--', alpha=0.7)

# --- Pie Chart Enhancements ---
gender_counts = df_train['Gender'].value_counts()

# Add shadow and highlight the largest slice in the pie chart
explode = (0.1, 0) if len(gender_counts) > 1 else (0, 0)  # Exploding the first slice if more than one
ax2.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90, colors=gender_palette, 
        wedgeprops={'edgecolor': 'black', 'linewidth': 2, 'linestyle': 'solid'}, explode=explode)

# Customize the pie chart text to be white, bold, and larger
for text in ax2.texts:
    text.set_fontsize(16)
    text.set_color('white')
    text.set_fontweight('bold')

# Add a glowing effect to the pie chart title
ax2.set_title('Gender Distribution', fontsize=20, color='#FF4500', fontweight='bold')

# --- Final Adjustments ---
plt.tight_layout()

# Display the plot
plt.show()


