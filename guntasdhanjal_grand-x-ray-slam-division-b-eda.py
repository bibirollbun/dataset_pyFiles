# Import required libraries for analysis and visualization
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import os

# Ensure plots display in Colab
%matplotlib inline

# Set seaborn style for clean visualizations
sns.set(style='whitegrid')

# Set seaborn style for clean visualizations
sns.set(style='whitegrid')



# Define path to training data
data_path = '/kaggle/input/grand-xray-slam-division-b/train2.csv'

# Load training dataset with error handling
try:
    train_df = pd.read_csv(data_path)
    print(f"Successfully loaded train.csv with shape: {train_df.shape}")
except FileNotFoundError:
    print(f"Error: {data_path} not found. Please check the file path.")
    raise


train_df.head(1)


# Display basic dataset info
print("Dataset Info:")
print(train_df.info())


# Summarize key metrics
total_images = len(train_df)
total_patients = train_df['Patient_ID'].nunique()
total_studies = train_df['Study'].nunique()
print(f"Total Images: {total_images}")
print(f"Total Patients: {total_patients}")
print(f"Total Studies: {total_studies}")


# Check for missing values
print("\nMissing Values:")
print(train_df.isnull().sum())


# Define the 14 condition columns
label_columns = ['No Finding', 'Lung Opacity', 'Support Devices', 'Atelectasis',
                 'Cardiomegaly', 'Pleural Effusion', 'Enlarged Cardiomediastinum',
                 'Edema', 'Consolidation', 'Pneumonia', 'Fracture', 'Lung Lesion',
                 'Pneumothorax', 'Pleural Other']

# Calculate counts and percentages for each condition
label_counts = train_df[label_columns].sum()
label_percentages = (label_counts / total_images * 100).round(2)
prevalence_df = pd.DataFrame({
    'Condition': label_counts.index,
    'Count': label_counts.values,
    'Percent (%)': label_percentages.values
})

# Display prevalence table
print("Label Prevalence:")
print(prevalence_df)


# Barplot for label prevalence
plt.figure(figsize=(12, 6))
sns.barplot(x='Count', y='Condition', data=prevalence_df, palette='viridis', hue=None)
plt.title('Label Counts (Number of Positive Cases)')
plt.xlabel('Count')
plt.ylabel('Condition')
plt.legend([],[], frameon=False)
plt.tight_layout()
plt.savefig('/content/label_counts_barplot.jpg')
plt.show()


# Donut chart for label percentages
plt.figure(figsize=(8, 8))
colors = sns.color_palette('viridis', len(prevalence_df))
plt.pie(prevalence_df['Count'], labels=prevalence_df['Condition'],
        autopct=lambda pct: f'{pct:.1f}%', startangle=140, colors=colors,
        wedgeprops={'width': 0.4})
plt.title('Label Prevalence (Positive Cases %)')
plt.tight_layout()
plt.savefig('/content/label_percent_donut.jpg')
plt.show()


# Calculate number of labels per image
train_df['Label_Count'] = train_df[label_columns].sum(axis=1)
multi_label_counts = train_df['Label_Count'].value_counts().sort_index()
multi_label_percent = (multi_label_counts / total_images * 100).round(2)

# Display multi-label distribution
print("Multi-Label Distribution:")
print(pd.DataFrame({'Number of Labels': multi_label_counts.index,
                    'Count': multi_label_counts.values,
                    'Percent (%)': multi_label_percent.values}))


# Compute co-occurrence matrix
co_occurrence = train_df[label_columns].T.dot(train_df[label_columns])
print("\nCo-Occurrence Matrix (Top 5x5 for brevity):")
print(co_occurrence.iloc[:5, :5])


# Heatmap for co-occurrence
plt.figure(figsize=(10, 8))
sns.heatmap(co_occurrence, annot=True, fmt='d', cmap='viridis')
plt.title('Label Co-Occurrence Matrix')
plt.tight_layout()
plt.savefig('/content/co_occurrence_heatmap.jpg')
plt.show()


# Visualizing Single-Label vs Multi-Label Image Distribution in Training Data
single_label_count = (train_df['Label_Count'] == 1).sum()
multi_label_count = (train_df['Label_Count'] > 1).sum()

labels = ['Single Label', 'Multi-Label']
sizes = [single_label_count, multi_label_count]
colors = ['#66c2a5', '#fc8d62']
explode = (0.05, 0.05)

plt.figure(figsize=(6,6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, explode=explode)
plt.title('Single vs Multi-Label Image Distribution')
plt.axis('equal')
plt.show()



# Label Distribution Excluding "No Finding"
filtered_df = train_df[(train_df['Label_Count'] >= 1) & (train_df['No Finding'] == 0)]

single_label_count = (filtered_df['Label_Count'] == 1).sum()
multi_label_count = (filtered_df['Label_Count'] > 1).sum()

labels = ['Single Label (Excl. No Finding)', 'Multi-Label (Excl. No Finding)']
sizes = [single_label_count, multi_label_count]
colors = ['#66c2a5', '#fc8d62']
explode = (0.05, 0.05)

plt.figure(figsize=(6,6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, explode=explode)
plt.title('Label Distribution Excluding "No Finding"')
plt.axis('equal')
plt.show()



# Replace NaNs with string 'Unknown' for plotting
sex_for_plot = train_df['Sex'].fillna('Unknown')

# Get counts including NaNs replaced
sex_counts = sex_for_plot.value_counts()

print("Sex distribution (including 'Unknown'):")
print(sex_counts)


plt.figure(figsize=(6,4))
sns.countplot(x=sex_for_plot, order=sex_counts.index, palette=['#95a5a6', '#3498db', '#e74c3c'])
plt.title('Sex Distribution (including Unknown)')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.show()



# Age distribution
valid_age_min = 0
valid_age_max = 120
ages = train_df['Age']
valid_ages = ages.dropna()
valid_ages_count = valid_ages[(valid_ages >= valid_age_min) & (valid_ages <= valid_age_max)].count()
nan_ages_count = ages.isna().sum()
invalid_ages_count = valid_ages[(valid_ages < valid_age_min) | (valid_ages > valid_age_max)].count()

print(f"Images with valid numeric Age: {valid_ages_count}")
print(f"Images with missing Age (NaN): {nan_ages_count}")
print(f"Images with invalid Age (<{valid_age_min} or >{valid_age_max}): {invalid_ages_count}")


# Histogram for Age distribution
plt.figure(figsize=(8, 6))
sns.histplot(train_df['Age'].dropna(), bins=30, kde=True, color='teal')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('/content/age_distribution.jpg')
plt.show()


# Condition prevalence by age group
age_bins = [0, 20, 40, 60, 80, 100]
train_df['AgeGroup'] = pd.cut(train_df['Age'], bins=age_bins, right=False)

age_label_counts = train_df.groupby('AgeGroup')[label_columns].apply(lambda x: (x == 1).sum())

plt.figure(figsize=(18, 10))
ax = age_label_counts.T.plot(kind='bar', stacked=False, figsize=(18,10))

plt.title('Condition Counts by Age Group', fontsize=20)
plt.xlabel('Condition (Disease)', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.xticks(rotation=90, fontsize=13)
plt.yticks(fontsize=13)
plt.legend(title='Age Group', fontsize=12, title_fontsize=14, loc='upper right')
plt.tight_layout()
plt.show()



# Image View Breakdown

# Count view categories and positions
view_cat_counts = train_df['ViewCategory'].value_counts(dropna=False)
view_pos_counts = train_df['ViewPosition'].value_counts(dropna=False)

print("ViewCategory counts:")
print(view_cat_counts)

print("\nViewPosition counts:")
print(view_pos_counts)



# Image View Distribution
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
sns.countplot(x='ViewCategory', data=train_df, order=view_cat_counts.index)
plt.title('ViewCategory Distribution')
plt.xticks(rotation=45)

plt.subplot(1,2,2)
sns.countplot(x='ViewPosition', data=train_df, order=view_pos_counts.index)
plt.title('ViewPosition Distribution')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# Path to train folder
IMAGE_ROOT_DIR = '/kaggle/input/grand-xray-slam-division-b/train2'

# List .jpg images
image_files = [f for f in os.listdir(IMAGE_ROOT_DIR) if f.endswith('.jpg')]

# Take first 5 images
image_files = image_files[:5]

# Plot images
fig, axs = plt.subplots(1, len(image_files), figsize=(20, 5))
for i, image_file in enumerate(image_files):
    image_path = os.path.join(IMAGE_ROOT_DIR, image_file)
    img = Image.open(image_path).convert('L')
    axs[i].imshow(img, cmap='gray')
    axs[i].set_title(image_file, fontsize=8)
    axs[i].axis('off')

plt.suptitle('Sample X-ray Images from Train Folder', fontsize=14)
plt.tight_layout()
plt.show()


# Check for duplicate Image_Names
duplicate_images = train_df['Image_name'].duplicated().sum()
print(f"Duplicated Image_Name entries: {duplicate_images}")

# Check for duplicate Patient_IDs (expected due to multiple images per patient)
duplicate_patients = total_images - total_patients
print(f"Duplicated Patient_ID entries: {duplicate_patients}")

# Check for invalid Age values
invalid_ages = train_df['Age'].dropna()
invalid_ages = invalid_ages[invalid_ages < 0].count()
print(f"Invalid Age values (<0): {invalid_ages}")

