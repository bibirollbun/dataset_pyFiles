import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pydicom
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, confusion_matrix
from sklearn.svm import SVC


# Suppress warnings
warnings.filterwarnings("ignore")

# Setting up Plotly and Seaborn aesthetics
import plotly.offline as py
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=True, theme='ggplot')
sns.set_style("whitegrid")
plt.style.use('fivethirtyeight')


# Load the dataset
input_dir = "/kaggle/input/siim-isic-melanoma-classification"
train_path = f"{input_dir}/train.csv"
test_path = f"{input_dir}/test.csv"



# Loading necessary CSV files
train_csv_path = os.path.join(input_dir, "/kaggle/input/siim-isic-melanoma-classification/train.csv")
test_csv_path = os.path.join(input_dir, "/kaggle/input/siim-isic-melanoma-classification/test.csv")

# Load data into DataFrames
train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)



# Load data into DataFrames
train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

# Display basic information about the datasets
print("Training DataFrame shape:", train_df.shape)
print("Testing DataFrame shape:", test_df.shape)

print("Training DataFrame sample:")
print(train_df.head())

print("Testing DataFrame sample:")
print(test_df.head())


# Checking distribution of classes in the training data
class_counts = train_df['benign_malignant'].value_counts()
print("\nClass Distribution in Training Data:")
print(class_counts)

# Visualization of class distribution
plt.figure(figsize=(8, 6))
sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis')
plt.title("Class Distribution in Training Data")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()


# Display training and testing data details
print("\nTraining DataFrame Info:")
train_df.info()

print("\nTesting DataFrame Info:")
test_df.info()

# Display first few rows of training data
print("\nFirst 5 rows of Training Data:")
print(train_df.head())

# Display first few rows of testing data
print("\nFirst 5 rows of Testing Data:")
print(test_df.head())

# Check for missing values
print("\nMissing Values in Training Data:")
missing_train = train_df.isnull().sum()
print(missing_train)

print("\nMissing Values in Testing Data:")
missing_test = test_df.isnull().sum()
print(missing_test)

# Visualize missing values
plt.figure(figsize=(10, 6))
sns.heatmap(train_df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values in Training Data")
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(test_df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values in Testing Data")
plt.show()

# Check unique patients in training data
unique_patients_train = train_df['patient_id'].nunique()
print(f"\nNumber of unique patients in Training Data: {unique_patients_train}")

# Check unique patients in testing data
unique_patients_test = test_df['patient_id'].nunique()
print(f"Number of unique patients in Testing Data: {unique_patients_test}")



# 4.3.2 Total Images and Unique IDs
print("\nTotal Images and Unique IDs")
print(f"Total training images: {len(train_df)}")
print(f"Total testing images: {len(test_df)}")
print(f"Unique patient IDs in training data: {train_df['patient_id'].nunique()}")
print(f"Unique patient IDs in testing data: {test_df['patient_id'].nunique()}")




### 4.3.3 Exploring the Target Column
print("\nExploring the Target Column")
plt.figure(figsize=(6, 6))
sns.countplot(x='benign_malignant', data=train_df, palette='Set2')
plt.title("Target Column Distribution")
plt.xlabel("Benign/Malignant")
plt.ylabel("Count")
plt.show()

# Additional Bar Plot with Percentages
percentages = (class_counts / class_counts.sum()) * 100
plt.figure(figsize=(6, 6))
ax = sns.barplot(x=percentages.index, y=percentages.values, palette='pastel')
plt.xlabel("Benign/Malignant")
plt.ylabel("Percentage")

# Add percentage annotations on top of bars
for p, percentage in zip(ax.patches, percentages):
    ax.annotate(f'{percentage:.2f}%', (p.get_x() + p.get_width() / 2, p.get_height()), 
                ha='center', va='bottom', fontsize=13)

plt.show()



# 4.3.4 Gender-wise Distribution
print("\nGender-wise Distribution")
# Plot gender count
plt.figure(figsize=(6, 6))
sns.countplot(x='sex', data=train_df, palette='coolwarm')
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()

# Plot gender percentages
gender_counts = train_df['sex'].value_counts()
gender_percentages = (gender_counts / gender_counts.sum()) * 100
plt.figure(figsize=(8, 6))
ax = sns.barplot(x=gender_counts.index, y=gender_percentages.values, palette='Greens')
plt.title("Gender Distribution (Percentage)")
plt.xlabel("Gender")
plt.ylabel("Percentage")

# Annotate percentages on the bars
for p, percentage in zip(ax.patches, gender_percentages):
    ax.annotate(f'{percentage:.2f}%', (p.get_x() + p.get_width() / 2, p.get_height()), 
                ha='center', va='bottom', fontsize=12)

plt.show()


# 4.3.5 Gender with Target
print("Gender with Target Distribution")
gender_target = train_df.groupby(['sex', 'benign_malignant']).size().unstack()

# Prepare data for plotting
categories = ['Benign Cases (0)', 'Malignant Cases (1)']
gender_labels = ['Female', 'Male']
benign_counts = [gender_target.loc['female', 'benign'], gender_target.loc['male', 'benign']]
malignant_counts = [gender_target.loc['female', 'malignant'], gender_target.loc['male', 'malignant']]

# Create a bar plot
x = np.arange(len(gender_labels))  # Label locations
width = 0.35  # Width of the bars

fig, ax = plt.subplots(figsize=(10, 6))

# Add bars for benign and malignant cases
bars1 = ax.bar(x - width/2, benign_counts, width, label='Benign Cases (0)', color='skyblue')
bars2 = ax.bar(x + width/2, malignant_counts, width, label='Malignant Cases (1)', color='salmon')

# Add text annotations on the top of bars
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{int(height)}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # Offset text by 3 units above the bar
                textcoords="offset points",
                ha='center', va='bottom')

for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{int(height)}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom')


# 4.3.6 Location of Imaged Site
print("Location of Imaged Site")
plt.figure(figsize=(10, 6))
sns.countplot(y='anatom_site_general_challenge', data=train_df, order=train_df['anatom_site_general_challenge'].value_counts().index, palette='cubehelix')
plt.title("Distribution of Imaged Site")
plt.xlabel("Count")
plt.ylabel("Imaged Site")
plt.show()

# Percentage-based bar plot
site_counts = train_df['anatom_site_general_challenge'].value_counts()
site_percentages = (site_counts / site_counts.sum()) * 100

# Create a bar plot for percentages
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=site_percentages.values, y=site_percentages.index, palette='cubehelix')
plt.title("Distribution of Imaged Site (Percentage)")
plt.xlabel("Percentage")
plt.ylabel("Imaged Site")

# Add percentage annotations on the bars
for p, percentage in zip(ax.patches, site_percentages):
    ax.annotate(f'{percentage:.2f}%', (p.get_width(), p.get_y() + p.get_height() / 2), 
                xytext=(5, 0), textcoords="offset points",
                ha='left', va='center', fontsize=10)

plt.show()


# 4.3.7 Location of Imaged Site with Respect to Gender
print("Location of Imaged Site")

# Prepare data for plotting
site_gender_counts = train_df.groupby(['anatom_site_general_challenge', 'sex']).size().unstack(fill_value=0)
site_labels = site_gender_counts.index
female_counts = site_gender_counts['female']
male_counts = site_gender_counts['male']

# Create a grouped bar plot
x = np.arange(len(site_labels))  # Label locations
width = 0.4  # Width of the bars

fig, ax = plt.subplots(figsize=(12, 8))

# Add bars for female and male counts
bars1 = ax.bar(x - width/2, female_counts, width, label='Female', color='skyblue')
bars2 = ax.bar(x + width/2, male_counts, width, label='Male', color='salmon')

# Add text annotations on the top of bars
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{int(height)}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # Offset text by 3 units above the bar
                textcoords="offset points",
                ha='center', va='bottom')

for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{int(height)}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom')

# Customize the plot
ax.set_xlabel('Location of Imaged Site')
ax.set_ylabel('Count of Melanoma Cases')
ax.set_title('Location of Imaged Site')
ax.set_xticks(x)
ax.set_xticklabels(site_labels, rotation=45, ha='right')
ax.legend()

# Show the plot
plt.tight_layout()
plt.show()



# 4.3.8 Age Distribution of Patients
print("\nAge Distribution of Patients")
plt.figure(figsize=(10, 6))
sns.histplot(train_df['age_approx'], kde=True, bins=30, color='blue')
plt.title("Age Distribution")
plt.xlabel("Approximate Age")
plt.ylabel("Frequency")
plt.show()


# 4.3.9 Age Distribution with Respect to Target
print("\nAge Distribution w.r.t Target")
plt.figure(figsize=(10, 6))
sns.kdeplot(data=train_df, x='age_approx', hue='benign_malignant', fill=True, common_norm=False, palette='Set2')
plt.title("Age Distribution by Target")
plt.xlabel("Approximate Age")
plt.ylabel("Density")
plt.show()



# 4.3.10 Diagnosis Distribution
print("\nDiagnosis Distribution")
plt.figure(figsize=(10, 6))
sns.countplot(y='diagnosis', data=train_df, order=train_df['diagnosis'].value_counts().index, palette='plasma')
plt.title("Distribution of Diagnosis")
plt.xlabel("Count")
plt.ylabel("Diagnosis")
plt.show()




