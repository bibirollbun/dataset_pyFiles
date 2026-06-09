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


# ================== Load Data ==================
train1 = pd.read_csv("/kaggle/input/playground-series-s3e24/train.csv")
test1 = pd.read_csv("/kaggle/input/playground-series-s3e24/test.csv")
train2 = pd.read_csv("/kaggle/input/smoker-status-prediction-using-biosignals/train_dataset.csv")

# ================== Prepare IDs ==================
test1_ids = test1["id"]

# ================== Basic Cleaning ==================
# Standardize column names (remove spaces)
train1.columns = train1.columns.str.replace(' ', '_')
test1.columns = test1.columns.str.replace(' ', '_')
train2.columns = train2.columns.str.replace(' ', '_')


print(train1.shape)
print(train2.shape)
print(test1.shape)


print(train1.dtypes)


def classify_column(column):
    # Get the number of unique values in the column
    unique_vals = column.nunique()

    # If the number of unique values is less than a threshold, consider it discrete
    if unique_vals < 10:  # You can adjust the threshold based on your dataset
        return 'Discrete'
    else:
        return 'Continuous'

# Apply the classification function to each column
for col in train1.columns:
    print(f"Column '{col}' is {classify_column(train1[col])}")


def classify_column(column):
    # Get the number of unique values in the column
    unique_vals = column.nunique()

    # If the number of unique values is less than a threshold, consider it discrete
    if unique_vals < 10:  # You can adjust the threshold based on your dataset
        return 'Discrete'
    else:
        return 'Continuous'

# Print only the discrete columns
for col in train1.columns:
    if classify_column(train1[col]) == 'Discrete':
        print(f"Column '{col}' is Discrete")


# Add missing columns to match
missing_cols = set(train1.columns) - set(train2.columns) - {'id', 'smoking', 'is_train'}
for col in missing_cols:
    train2[col] = np.nan

# Align columns
train2 = train2[train1.drop(['id', 'smoking'], axis=1).columns.tolist() + ['smoking']]

# Label datasets
train1["is_train"] = 1
test1["is_train"] = 0
train2["is_train"] = 1

# Add dummy target to test
test1["smoking"] = -1


from scipy.stats import ks_2samp
import pandas as pd

# Assume you already have train1 and train2 loaded

# Store results
ks_results = []

# Loop over features
for col in train1.columns:
    if col in ['id', 'smoking', 'is_train']:
        continue  # skip non-features
    
    # Drop NaNs
    data1 = train1[col].dropna()
    data2 = train2[col].dropna()
    
    # KS test
    stat, p_value = ks_2samp(data1, data2)
    
    ks_results.append((col, p_value))

# Create a DataFrame
ks_df = pd.DataFrame(ks_results, columns=['feature', 'p_value'])

# Sort features by p-value (low p-value = very different distributions)
ks_df = ks_df.sort_values('p_value')

print(ks_df)



import matplotlib.pyplot as plt
import seaborn as sns

# Define how many plots per page
plots_per_page = 4
features = ks_df[ks_df['p_value'] < 0.05]['feature'].tolist()  # features you want to plot

# Loop through features
for i in range(0, len(features), plots_per_page):
    subset = features[i:i+plots_per_page]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))  # 2 rows x 2 cols
    axes = axes.flatten()
    
    for j, feature in enumerate(subset):
        sns.kdeplot(train1[feature], label='Main', fill=True, color='red', alpha=0.5, linewidth=2, ax=axes[j])
        sns.kdeplot(train2[feature], label='Original', fill=True, color='black', alpha=0.5, linewidth=2, ax=axes[j])
        
        axes[j].set_title(f'Distribution of {feature}', fontsize=14)
        axes[j].legend()
    
    # If fewer than 4 plots in last page, hide empty subplots
    for j in range(len(subset), plots_per_page):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()



# For main dataset
train1_proportions = train1['dental_caries'].value_counts(normalize=True)

# For original dataset
train2_proportions = train2['dental_caries'].value_counts(normalize=True)

print("Main dataset proportions:\n", train1_proportions)
print("Original dataset proportions:\n", train2_proportions)



import matplotlib.pyplot as plt
import pandas as pd

# Create a DataFrame to plot
compare_df = pd.DataFrame({
    'Main': train1_proportions,
    'Original': train2_proportions
}).T  # transpose so rows are datasets

# Define custom colors: black for 0, red for 1
colors = ['black', 'red']

# Plot
ax = compare_df.plot(kind='bar', stacked=True, color=colors, figsize=(8, 6))
plt.title('Dental Caries Distribution: Main vs Original', fontsize=16)
plt.xlabel('Dataset', fontsize=14)
plt.ylabel('Proportion', fontsize=14)

# Move the legend outside the plot
plt.legend(title='Dental Caries\n(0 = No, 1 = Yes)', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()



print(train1.shape)
print(train2.shape)
print(test1.shape)


print(train2.head())


full_df = pd.concat([train1, train2, test1], axis=0, ignore_index=True)


train_df = full_df[full_df["is_train"] == 1].drop(["id", "is_train"], axis=1)


print(train_df.isna().sum())


# Find unique values in 'age' and how many times each appears
age_counts = train_df['age'].value_counts()

print(age_counts)



# Define function to create age groups
def age_to_group(age):
    if 20 <= age < 30:
        return '20s'
    elif 30 <= age < 40:
        return '30s'
    elif 40 <= age < 50:
        return '40s'
    elif 50 <= age < 60:
        return '50s'
    elif 60 <= age < 70:
        return '60s'
    else:
        return '70+'

# Apply function to create a new feature
#train_df['age_group'] = train_df['age'].apply(age_to_group)

# Check distribution
#print(train_df['age_group'].value_counts())



# Assuming your column is called 'smoker'
smoker_counts = train_df['smoking'].value_counts(normalize=True) * 100

print(smoker_counts)


import matplotlib.pyplot as plt

# Calculate percentages
smoker_counts = train_df['smoking'].value_counts(normalize=True) * 100

# Labels for the pie chart
labels = ['Non-Smoker', 'Smoker']

# Colors: Non-Smoker = black, Smoker = red
colors = ['grey', 'red']

# Create the pie chart
plt.figure(figsize=(6,6))
plt.pie(smoker_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
plt.title('Smoker vs Non-Smoker Distribution', color='white')
plt.axis('equal')  # Equal aspect ratio makes the pie round
plt.show()



import seaborn as sns
import matplotlib.pyplot as plt

# Features to plot (excluding 'smoking')
features = train_df.drop('smoking', axis=1).columns
n_features = len(features)

# Color mapping: 0 = non-smoker (black), 1 = smoker (red)
palette = {0: 'black', 1: 'red'}

# Features that need log-scaling
log_scale_features = ['serum_creatinine', 'AST', 'ALT', 'Gtp']

# Loop through features, 4 per page
for i in range(0, n_features, 4):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Boxplots of Features vs Smoking", fontsize=18)
    
    for j, ax in enumerate(axes.flat):
        if i + j < n_features:
            feature = features[i + j]
            sns.boxplot(x='smoking', y=feature, data=train_df, ax=ax, palette=palette)
            ax.set_title(f'{feature} vs Smoking', fontsize=14)
            
            # Set log-scale if needed
            if feature in log_scale_features:
                ax.set_yscale('log')
                ax.set_ylabel(f'log({feature})')
        else:
            # Hide empty subplots
            ax.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make space for suptitle
    plt.show()




import seaborn as sns
import matplotlib.pyplot as plt

# Melt the data if needed, or plot each side separately

for feature in ['hearing(left)', 'hearing(right)']:
    plt.figure(figsize=(6, 4))
    prop_df = train_df.groupby(['smoking', feature]).size().reset_index(name='count')
    prop_df['percent'] = prop_df.groupby('smoking')['count'].transform(lambda x: x / x.sum() * 100)
    sns.barplot(x='smoking', y='percent', hue=feature, data=prop_df, palette={1: 'black', 2: 'red'})
    plt.title(f'Proportion of {feature} by Smoking Status', fontsize=14)
    plt.ylabel('Percentage')
    plt.xlabel('Smoking Status')
    plt.legend(title=feature)
    plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

# Set the plot style
sns.set_style("whitegrid")

# Calculate % hearing loss for left
hearing_left_loss = (train_df[train_df['hearing(left)'] == 2].groupby('smoking').size() / 
                     train_df.groupby('smoking').size()) * 100

# Calculate % hearing loss for right
hearing_right_loss = (train_df[train_df['hearing(right)'] == 2].groupby('smoking').size() / 
                      train_df.groupby('smoking').size()) * 100

# Set up side-by-side plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left ear
sns.barplot(x=hearing_left_loss.index, y=hearing_left_loss.values, palette=['black', 'red'], ax=axes[0])
axes[0].set_title('Hearing Loss (Left Ear)', fontsize=16, fontweight='bold')
axes[0].set_xlabel('Smoking Status', fontsize=13)
axes[0].set_ylabel('Percentage (%)', fontsize=13)
axes[0].set_xticklabels(['Non-Smoker', 'Smoker'], fontsize=12)

# Right ear
sns.barplot(x=hearing_right_loss.index, y=hearing_right_loss.values, palette=['black', 'red'], ax=axes[1])
axes[1].set_title('Hearing Loss (Right Ear)', fontsize=16, fontweight='bold')
axes[1].set_xlabel('Smoking Status', fontsize=13)
axes[1].set_ylabel('Percentage (%)', fontsize=13)
axes[1].set_xticklabels(['Non-Smoker', 'Smoker'], fontsize=12)

plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")

# Create % distribution
urine_protein_dist = train_df.groupby('smoking')['Urine_protein'].value_counts(normalize=True).unstack()

# Plot
urine_protein_dist.T.plot(kind='bar', figsize=(10,6), color=['black', 'red'])

plt.title('Urine Protein Levels by Smoking Status', fontsize=16, fontweight='bold')
plt.xlabel('Urine Protein Level', fontsize=14)
plt.ylabel('Proportion', fontsize=14)
plt.legend(title='Smoking', labels=['Non-smoker', 'Smoker'])
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")

# Calculate % with dental caries
dental_caries_rate = (train_df[train_df['dental_caries'] == 1].groupby('smoking').size() /
                      train_df.groupby('smoking').size()) * 100

# Plot
plt.figure(figsize=(6, 5))
sns.barplot(x=dental_caries_rate.index, y=dental_caries_rate.values, palette=['black', 'red'])
plt.title('Dental Caries Rate by Smoking Status', fontsize=16)
plt.xlabel('Smoking Status', fontsize=14)
plt.ylabel('Dental Caries (%)', fontsize=14)
plt.xticks([0,1], ['Non-Smoker', 'Smoker'], fontsize=12)
plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

# Define categorical features
categorical_features = ['Urine_protein', 'dental_caries', 'hearing(left)', 'hearing(right)']

# Loop through the features in batches of 4 plots
feature_list = [col for col in train_df.columns if col != 'smoking']  # Exclude 'smoking'

# Number of plots per page
plots_per_page = 4
num_plots = len(feature_list)

# Custom color palette
smoking_colors = {0: 'black', 1: 'red'}  # Red for smoking, black for nonsmoking

for start_idx in range(0, num_plots, plots_per_page):
    end_idx = min(start_idx + plots_per_page, num_plots)

    # Create a grid for the current batch of plots
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))  # Adjust size based on your preference
    axes = axes.flatten()

    # Loop through the features for this batch and plot
    for plot_idx, col in enumerate(feature_list[start_idx:end_idx]):
        ax = axes[plot_idx]

        if col in categorical_features:
            # Countplot for categorical features with custom color palette
            sns.countplot(data=train_df, x=col, hue='smoking', palette=smoking_colors, ax=ax)
            ax.set_title(f'Countplot of {col} by Smoking')
        else:
            # Bar plot of average values for continuous features
            sns.barplot(data=train_df, x='smoking', y=col, errorbar=None, ax=ax, palette=smoking_colors)
            ax.set_title(f'Average {col} by Smoking')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plots for the current batch
    plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'train_df' contains 'weight_kg', 'height_cm', and 'smoking' columns
train_df['height_m'] = train_df['height(cm)'] / 100  # Convert height to meters
train_df['bmi'] = train_df['weight(kg)'] / (train_df['height_m'] ** 2)  # Calculate BMI

# Plot average BMI for smokers vs nonsmokers
plt.figure(figsize=(8, 6))
sns.barplot(data=train_df, x='smoking', y='bmi', palette={0: 'black', 1: 'red'})

# Set the title and labels
plt.title('Average BMI for Smokers vs Nonsmokers')
plt.xlabel('Smoking Status')
plt.ylabel('Average BMI')

# Show the plot
plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'train_df' contains 'systolic', 'diastolic', and 'smoking' columns
train_df['bp_ratio'] = train_df['systolic'] / train_df['relaxation']  # Calculate systolic/diastolic ratio

# Plot average ratio for smokers vs nonsmokers
plt.figure(figsize=(8, 6))
sns.barplot(data=train_df, x='smoking', y='bp_ratio', palette={0: 'black', 1: 'red'})

# Set the title and labels
plt.title('Average Systolic/Diastolic Ratio for Smokers vs Nonsmokers')
plt.xlabel('Smoking Status')
plt.ylabel('Average Systolic/Diastolic Ratio')

# Show the plot
plt.tight_layout()
plt.show()


