!pip install --no-deps /kaggle/input/imbalanced-learn-0-11-0/imbalanced_learn-0.11.0-py3-none-any.whl
!pip install --no-deps /kaggle/input/scikit-learn-1-3-0/scikit_learn-1.3.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl



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


# -----------------------------
# 1. Core Python Modules
# -----------------------------
import os
import sys
import json
import warnings
import logging

# -----------------------------
# 2. Numerical & Data Handling
# -----------------------------
import numpy as np
import pandas as pd

# -----------------------------
# 3. Visualization
# -----------------------------
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 4. Machine Learning & Data Processing
# -----------------------------
# Scikit-learn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_curve, roc_auc_score, precision_recall_curve,
    accuracy_score, f1_score, confusion_matrix,
    classification_report, average_precision_score
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.utils import parallel_backend

# Imbalanced Learning
from imblearn.over_sampling import SMOTE

# XGBoost
import xgboost as xgb

# SHAP for model interpretability
import shap

# Feature Engineering with automated feature engineering
import featuretools as ft

# -----------------------------
# 5. Parallel Processing
# -----------------------------
import joblib

# -----------------------------
# 6. GPU Acceleration
# -----------------------------
import torch            # PyTorch for deep learning
import cudf             # GPU-accelerated DataFrame library
import cuml             # RAPIDS machine learning library on GPU
from cuml.decomposition import PCA as cuPCA
from cuml.neighbors import KNeighborsRegressor

# -------------------------------------------------
# Example usage print for version information:
# -------------------------------------------------
import sklearn
import imblearn
print("scikit-learn:", sklearn.__version__)
print("imbalanced-learn:", imblearn.__version__)
print("numpy:", np.__version__)



# ================================
#       INITIAL EXPLORATION
# ================================


# Set base and output paths
base_path = "/kaggle/input/playground-series-s4e11/"
output_path = "/kaggle/working/"

# Load Datasets safely with encoding and separator
train = pd.read_csv(base_path + "train.csv", encoding='utf-8', sep=',')
test = pd.read_csv(base_path + "test.csv", encoding='utf-8', sep=',')
submission = pd.read_csv(base_path + "sample_submission.csv", encoding='utf-8', sep=',')

# ================================
#           DATA SUMMARY
# ================================

print("\n" + "="*50)
print("DATA SUMMARY".center(50))
print("="*50)

# Dataset shapes
print(f"\nTrain shape: {train.shape}")
print(f"Test shape: {test.shape}")

# Data types and missing values
print("\nTrain Info:")
display(train.info(verbose=True))

print("\nTest Info:")
display(test.info(verbose=True))

# First rows
print("\nFirst 10 rows of Train:")
display(train.head(10).style.background_gradient(cmap='Blues', low=0.5, high=0))

print("\nFirst 10 rows of Test:")
display(test.head(10).style.background_gradient(cmap='Greens', low=0.5, high=0))

# Descriptive statistics
print("\nDescriptive Statistics - Train:")
display(train.describe().style.background_gradient(cmap='Blues', axis=0))

print("\nDescriptive Statistics - Test:")
display(test.describe().style.background_gradient(cmap='Greens', axis=0))

# Sample submission format
print("\nSample Submission Format:")
display(submission.head(5))

# ================================
#      TARGET VARIABLE OVERVIEW
# ================================

print("\n" + "="*50)
print("TARGET DISTRIBUTION".center(50))
print("="*50)

# Depression target distribution
if 'Depression' in train.columns:
    depression_counts = train['Depression'].value_counts().sort_index()
    depression_percentage = train['Depression'].value_counts(normalize=True).sort_index() * 100

    # Styled table
    target_df = pd.DataFrame({
        'Value': ['Not Depressed (0)', 'Depressed (1)'],
        'Count': depression_counts.values,
        'Percentage (%)': depression_percentage.round(2).values
    })

    print("\nDepression Distribution:")
    display(target_df.style.background_gradient(cmap='Blues', subset=['Count', 'Percentage (%)']))

    # Bar plot
    plt.figure(figsize=(8, 5))
    sns.barplot(x=depression_counts.index, y=depression_counts.values, palette=['#AED6F1', '#F5B7B1'])
    plt.title('Depression Distribution', fontsize=14)
    plt.xlabel('Depression (0 = No, 1 = Yes)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for i, v in enumerate(depression_counts.values):
        plt.text(i, v + 500, f"{v:,}", ha='center', fontsize=11)
        plt.text(i, v / 2, f"{depression_percentage.values[i]:.2f}%", ha='center', fontsize=11, color='black')

    plt.tight_layout()
    plt.show()
else:
    print("Warning: 'Depression' column not found in the training dataset.")
#######################################################################################################################################

# Missing values analysis
def plot_missing_values(df, title):

    missing = df.isnull().sum().sort_values(ascending=False)
    missing_percent = (missing / len(df) * 100).round(4)
    missing_data = pd.DataFrame({'Missing Count': missing, 'Percentage': missing_percent})
    missing_data = missing_data[missing_data['Missing Count'] > 0]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=missing_data['Percentage'], y=missing_data.index, 
                     palette="Reds_r", orient='h')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.title(f'Missing Values in {title}', fontsize=14)
    plt.xlabel('Percentage Missing', fontsize=12)
    plt.ylabel('')
    
    # Ensure x-axis extends beyond the maximum value to fit labels
    max_pct = missing_data['Percentage'].max()
    plt.xlim(0, max_pct * 1.15)  # Add more padding for the text

    # Add percentage labels at the end of each bar
    for i, (pct, y_val) in enumerate(zip(missing_data['Percentage'], missing_data.index)):
        ax.text(pct + max_pct * 0.02, i, f'{pct:.2f}%', va='center', fontsize=10)

    plt.tight_layout()
    plt.show()
    
    # Print numerical values
    print(f"Missing Values (%) in {title}")
    for col, pct in missing_percent[missing_percent > 0].items():
        print(f"{col:<20} {pct:.6f}")
    print()
    
    # Also print raw counts
    print(f"Missing values in {title}:")
    for col, count in missing[missing > 0].items():
        print(f"{col:<35} {count}")
    if len(missing[missing > 0]) > 0:
        if 'float' in str(df[missing[missing > 0].index[0]].dtype):
            print(f"dtype: {df[missing[missing > 0].index[0]].dtype}")
    print()

# Generate missing values visualizations
plot_missing_values(train, 'Train')
plot_missing_values(test, 'Test')

# Display a comparison of dataset sizes with properly positioned labels
plt.figure(figsize=(10, 6))
bars = plt.bar(['Train', 'Test'], 
        [len(train), len(test)], 
        color=['blue', 'green'])

# Add the count values inside the top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height * 0.95,
            f'{int(height):,}',
            ha='center', va='top', fontsize=10, 
            color='white', fontweight='bold')

plt.title('ID Count by Dataset')
plt.ylabel('Count')
plt.xlabel('Dataset')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylim(0, max(len(train), len(test)) * 1.05)
plt.tight_layout()
plt.show()


######################################################################################################################################################
# 3. Categorical Features Analysis

# Define a more refined profession grouping system
profession_groups = {
    # Healthcare
    'Doctor': 'Healthcare',
    'Medical Doctor': 'Healthcare',
    'MBBS': 'Healthcare',
    'Pharmacist': 'Healthcare',
    'Surgeon': 'Healthcare',
    'MD': 'Healthcare',
    'B.Pharm': 'Healthcare',
    'M.Pharm': 'Healthcare',
    'PhD': 'Healthcare', 
    
    # Education
    'Teacher': 'Education',
    'Educational Consultant': 'Education',
    'Academic': 'Education',
    'B.Ed': 'Education',
    'M.Ed': 'Education',
    
    # Business & Finance
    'Accountant': 'Business & Finance',
    'Business Analyst': 'Business & Finance',
    'Financial Analyst': 'Business & Finance',
    'Finanancial Analyst': 'Business & Finance',
    'Investment Banker': 'Business & Finance',
    'Entrepreneur': 'Business & Finance',
    'MBA': 'Business & Finance',
    'B.Com': 'Business & Finance',
    'BBA': 'Business & Finance',
    
    # Technology & Engineering
    'Software Engineer': 'Technology & Engineering',
    'Data Scientist': 'Technology & Engineering',
    'UX/UI Designer': 'Technology & Engineering',
    'Dev': 'Technology & Engineering',
    'Civil Engineer': 'Technology & Engineering',
    'Mechanical Engineer': 'Technology & Engineering',
    'M.Tech': 'Technology & Engineering',
    'MCA': 'Technology & Engineering',
    'BE': 'Technology & Engineering',
    'ME': 'Technology & Engineering',
    
    # Legal
    'Lawyer': 'Legal',
    'Judge': 'Legal',
    'LLM': 'Legal',
    
    # Creative & Media
    'Content Writer': 'Creative & Media',
    'Graphic Designer': 'Creative & Media',
    'Digital Marketer': 'Creative & Media',
    
    # Service & Hospitality
    'Chef': 'Service & Hospitality',
    'Customer Support': 'Service & Hospitality',
    'Travel Consultant': 'Service & Hospitality',
    'Pilot': 'Service & Hospitality',
    
    # Architecture & Design
    'Architect': 'Architecture & Design',
    
    # Management & Consulting
    'HR Manager': 'Management & Consulting',
    'Manager': 'Management & Consulting',
    'Marketing Manager': 'Management & Consulting',
    'Consultant': 'Management & Consulting',
    'Family Consultant': 'Management & Consulting',
    'City Consultant': 'Management & Consulting',
    'City Manager': 'Management & Consulting',
    'Research Analyst': 'Management & Consulting',
    'Researcher': 'Management & Consulting',
    'Analyst': 'Management & Consulting',
    
    # Skilled Trades
    'Electrician': 'Skilled Trades',
    'Plumber': 'Skilled Trades',
    'Chemist': 'Skilled Trades',
    
    # Sales & Marketing
    'Sales Executive': 'Sales & Marketing',
    
    # Students & Unemployed
    'Student': 'Students & Unemployed',
    'Unemployed': 'Students & Unemployed',
    
    # General Professional
    'Working Professional': 'General Professional',
    'Unveil': 'General Professional',
    'Profession': 'General Professional',
    
    # Data Errors (locations, names, etc.)
    'FamilyVirar': 'Data Error',
    'Nagpur': 'Data Error',
    'Patna': 'Data Error',
    'Visakhapatnam': 'Data Error',
    'Surat': 'Data Error',
    'Yogesh': 'Data Error',
    'Yuvraj': 'Data Error',
    'Pranav': 'Data Error',
    'Manvi': 'Data Error',
    'Samar': 'Data Error',
    'Simran': 'Data Error',
    '24th': 'Data Error',
    '3M': 'Data Error',
    'Unhealthy': 'Data Error',
    'No': 'Data Error',
    'Name': 'Data Error',
    'Moderate': 'Data Error'
}

# Define a more refined degree classification system
degree_groups = {
    # High School
    'Class 12': 'High School',
    'Class 11': 'High School',
    
    # Bachelor's - Business/Commerce
    'B.Com': 'Bachelor of Business/Commerce',
    'B B.Com': 'Bachelor of Business/Commerce',
    'BBA': 'Bachelor of Business/Commerce',
    'B.BA': 'Bachelor of Business/Commerce',
    'BBM': 'Bachelor of Business/Commerce',
    'BMS': 'Bachelor of Business/Commerce',
    'P.Com': 'Bachelor of Business/Commerce',
    'LL.Com': 'Bachelor of Business/Commerce',
    'LLCom': 'Bachelor of Business/Commerce',
    'ACA': 'Bachelor of Business/Commerce',
    'B_Com': 'Bachelor of Business/Commerce',
    'B Financial Analyst': 'Bachelor of Business/Commerce',
    
    # Bachelor's - Science/Engineering/Technology
    'B.Sc': 'Bachelor of Science/Engineering',
    'BSc': 'Bachelor of Science/Engineering',
    'B.Pharm': 'Bachelor of Science/Engineering',
    'BPharm': 'Bachelor of Science/Engineering',
    'B._Pharm': 'Bachelor of Science/Engineering',
    'H_Pharm': 'Bachelor of Science/Engineering',
    'S.Pharm': 'Bachelor of Science/Engineering',
    'P.Pharm': 'Bachelor of Science/Engineering',
    'N.Pharm': 'Bachelor of Science/Engineering',
    'B.Tech': 'Bachelor of Science/Engineering',
    'BTech': 'Bachelor of Science/Engineering',
    'B B.Tech': 'Bachelor of Science/Engineering',
    'BE': 'Bachelor of Science/Engineering',
    'BCA': 'Bachelor of Science/Engineering',
    'B.CA': 'Bachelor of Science/Engineering',
    'B BCA': 'Bachelor of Science/Engineering',
    'RCA': 'Bachelor of Science/Engineering',
    'PCA': 'Bachelor of Science/Engineering',
    'LCA': 'Bachelor of Science/Engineering',
    'GCA': 'Bachelor of Science/Engineering',
    'HCA': 'Bachelor of Science/Engineering',
    'BHCA': 'Bachelor of Science/Engineering',
    'E.Tech': 'Bachelor of Science/Engineering',
    'S.Tech': 'Bachelor of Science/Engineering',
    
    # Bachelor's - Arts/Humanities
    'BA': 'Bachelor of Arts/Humanities',
    'B.A': 'Bachelor of Arts/Humanities',
    'B BA': 'Bachelor of Arts/Humanities',
    'BPA': 'Bachelor of Arts/Humanities',
    
    # Bachelor's - Education
    'B.Ed': 'Bachelor of Education',
    'BEd': 'Bachelor of Education',
    'LL B.Ed': 'Bachelor of Education',
    'K.Ed': 'Bachelor of Education',
    'L.Ed': 'Bachelor of Education',
    'LLEd': 'Bachelor of Education',
    'A.Ed': 'Bachelor of Education',
    'E.Ed': 'Bachelor of Education',
    'G.Ed': 'Bachelor of Education',
    'I.Ed': 'Bachelor of Education',
    'J.Ed': 'Bachelor of Education',
    
    # Bachelor's - Other Specialized
    'LLB': 'Bachelor of Law',
    'LLBA': 'Bachelor of Law',
    'LLS': 'Bachelor of Law',
    'B.Arch': 'Bachelor of Architecture',
    'BArch': 'Bachelor of Architecture',
    'B.B.Arch': 'Bachelor of Architecture',
    'S.Arch': 'Bachelor of Architecture',
    'BHM': 'Bachelor of Hotel Management',
    'LHM': 'Bachelor of Hotel Management',
    'B': 'General Bachelor Degree',
    'BB': 'General Bachelor Degree',
    'BH': 'General Bachelor Degree',
    'B.H': 'General Bachelor Degree',
    
    # Master's - Business/Commerce
    'M.Com': 'Master of Business/Commerce',
    'B.M.Com': 'Master of Business/Commerce',
    'MBA': 'Master of Business/Commerce',
    'PGDM': 'Master of Business/Commerce',
    
    # Master's - Science/Engineering/Technology
    'M.Sc': 'Master of Science/Engineering',
    'MSc': 'Master of Science/Engineering',
    'M.Tech': 'Master of Science/Engineering',
    'MTech': 'Master of Science/Engineering',
    'M_Tech': 'Master of Science/Engineering',
    'ME': 'Master of Science/Engineering',
    'MCA': 'Master of Science/Engineering',
    'M.Pharm': 'Master of Science/Engineering',
    'MPharm': 'Master of Science/Engineering',
    'M': 'Master of Science/Engineering',
    'M.': 'Master of Science/Engineering',
    'LLTech': 'Master of Science/Engineering',
    'M.UI': 'Master of Science/Engineering',
    
    # Master's - Arts/Humanities
    'MA': 'Master of Arts/Humanities',
    'M.A': 'Master of Arts/Humanities',
    'MPA': 'Master of Arts/Humanities',
    
    # Master's - Education
    'M.Ed': 'Master of Education',
    'MEd': 'Master of Education',
    'M.B.Ed': 'Master of Education',
    'M.M.Ed': 'Master of Education',
    
    # Master's - Other Specialized
    'LLM': 'Master of Law',
    'M.Arch': 'Master of Architecture',
    'MHM': 'Master of Hotel Management',
    
    # Doctoral & Professional Medical
    'PhD': 'Doctoral Degree',
    'Ph.D': 'Doctoral Degree',
    'M.Phil': 'Doctoral Degree',
    'MBBS': 'Medical Degree',
    'MD': 'Medical Degree',
    'BDS': 'Medical Degree',
    'M.S': 'Medical Degree',
    'MS': 'Medical Degree',
    
    # Data errors
    '0': 'Data Error',
    '20': 'Data Error',
    '24': 'Data Error',
    '29': 'Data Error',
    '3.0': 'Data Error',
    '5.56': 'Data Error',
    '5.61': 'Data Error',
    '5.65': 'Data Error',
    '5.88': 'Data Error',
    '7.06': 'Data Error',
    '8.56': 'Data Error',
    '8.95': 'Data Error',
    'B.03': 'Data Error',
    'B.3.79': 'Data Error',
    'CGPA': 'Data Error',
    'B.Study_Hours': 'Data Error',
    'B.Press': 'Data Error',
    'B.Student': 'Data Error',
    'B. Gender': 'Data Error',
    'Degree': 'Data Error',
    'Unite': 'Data Error',
    'M. Business Analyst': 'Data Error',
    
    # Occupation entries (misplaced)
    'Business Analyst': 'Occupation Entry',
    'Data Scientist': 'Occupation Entry',
    'Doctor': 'Occupation Entry',
    'Entrepreneur': 'Occupation Entry',
    'HR Manager': 'Occupation Entry',
    'Plumber': 'Occupation Entry',
    'UX/UI Designer': 'Occupation Entry',
    'Working Professional': 'Occupation Entry',
    'Mechanical Engineer': 'Occupation Entry',
    'Travel Consultant': 'Occupation Entry',
    
    # Names (likely errors)
    'Aarav': 'Name Entry',
    'Aadhya': 'Name Entry',
    'Advait': 'Name Entry',
    'Badhya': 'Name Entry',
    'Banchal': 'Name Entry',
    'Bhavesh': 'Name Entry',
    'Bian': 'Name Entry',
    'Brit': 'Name Entry',
    'Brithika': 'Name Entry',
    'Esha': 'Name Entry',
    'Eshita': 'Name Entry',
    'Gagan': 'Name Entry',
    'Jhanvi': 'Name Entry',
    'Kavya': 'Name Entry',
    'Lata': 'Name Entry',
    'Magan': 'Name Entry',
    'Mahika': 'Name Entry',
    'Marsh': 'Name Entry',
    'Mihir': 'Name Entry',
    'Moham': 'Name Entry',
    'Mthanya': 'Name Entry',
    'Nalini': 'Name Entry',
    'Navya': 'Name Entry',
    'Pihu': 'Name Entry',
    'Ritik': 'Name Entry',
    'Rupak': 'Name Entry',
    'Veda': 'Name Entry',
    'Vibha': 'Name Entry',
    'Vivaan': 'Name Entry',
    'Vrinda': 'Name Entry',
    
    # Locations (likely errors)
    'Bhopal': 'Location Entry',
    'Kalyan': 'Location Entry',
    'Pune': 'Location Entry'
}

# Function to map profession to group
def map_profession_to_group(profession):
    if pd.isna(profession):
        return 'Unknown'
    elif profession in profession_groups:
        return profession_groups[profession]
    else:
        return 'Other'

# Function to map degree to group
def map_degree_to_group(degree):
    if pd.isna(degree):
        return 'Unknown'
    elif degree in degree_groups:
        return degree_groups[degree]
    else:
        return 'Other'

# Apply mappings
train['Profession_Group'] = train['Profession'].apply(map_profession_to_group)
test['Profession_Group'] = test['Profession'].apply(map_profession_to_group)
train['Degree_Group'] = train['Degree'].apply(map_degree_to_group)
test['Degree_Group'] = test['Degree'].apply(map_degree_to_group)

# Display profession group distribution
print("="*80)
print("REFINED PROFESSION GROUP DISTRIBUTION".center(80))
print("="*80)

# Create a more visually appealing display for profession groups
prof_group_train = train['Profession_Group'].value_counts().reset_index()
prof_group_train.columns = ['Profession_Group', 'Count']
prof_group_train['Percentage'] = prof_group_train['Count'] / len(train) * 100

prof_group_test = test['Profession_Group'].value_counts().reset_index()
prof_group_test.columns = ['Profession_Group', 'Count']
prof_group_test['Percentage'] = prof_group_test['Count'] / len(test) * 100

# Visualize profession groups with count labels COMPLETELY visible
plt.figure(figsize=(16, 8))  # Make figure wider
ax = sns.barplot(x='Percentage', y='Profession_Group', 
            data=prof_group_train.sort_values('Percentage', ascending=False).head(10),
            color='royalblue')
plt.title('Top 10 Profession Groups in Training Set', fontsize=14)
plt.xlabel('Percentage (%)', fontsize=12)
plt.ylabel('Profession Group', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Get the maximum percentage value to set the x-axis limit properly
max_pct = prof_group_train['Percentage'].max()
# Add 40% padding to ensure labels fit within the figure
plt.xlim(0, max_pct * 1.4)

# Add percentage and count annotations with clear positioning
for i, row in enumerate(prof_group_train.sort_values('Percentage', ascending=False).head(10).itertuples()):
    # For large percentage values, place text inside the bar
    if row.Percentage > 10:
        # Place text at 80% of the way across the bar
        text_x = row.Percentage * 0.8
        plt.text(text_x, i, f'{row.Percentage:.2f}%', 
                 va='center', color='white', fontweight='bold')
        
        # Place count closer to the end of the bar
        plt.text(row.Percentage + 0.5, i, f'(n={row.Count:,})', 
                 va='center', ha='left')
    else:
        # For smaller bars, place percentage at the end of bar
        plt.text(row.Percentage + 0.3, i, f'{row.Percentage:.2f}%', 
                 va='center', ha='left')
        
        # Place count right after the percentage
        plt.text(row.Percentage + 3, i, f'(n={row.Count:,})', 
                 va='center', ha='left')

plt.tight_layout()
plt.show()

# Display degree group distribution
print("\n" + "="*80)
print("REFINED DEGREE GROUP DISTRIBUTION".center(80))
print("="*80)

# Create a more visually appealing display for degree groups
degree_group_train = train['Degree_Group'].value_counts().reset_index()
degree_group_train.columns = ['Degree_Group', 'Count']
degree_group_train['Percentage'] = degree_group_train['Count'] / len(train) * 100

degree_group_test = test['Degree_Group'].value_counts().reset_index()
degree_group_test.columns = ['Degree_Group', 'Count']
degree_group_test['Percentage'] = degree_group_test['Count'] / len(test) * 100

# Visualize degree groups with count labels COMPLETELY visible
plt.figure(figsize=(16, 10))  # Make figure wider
ax = sns.barplot(x='Percentage', y='Degree_Group', 
            data=degree_group_train.sort_values('Percentage', ascending=False).head(12),
            color='seagreen')
plt.title('Top 12 Degree Groups in Training Set', fontsize=14)
plt.xlabel('Percentage (%)', fontsize=12)
plt.ylabel('Degree Group', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Get the maximum percentage value to set the x-axis limit properly
max_pct = degree_group_train['Percentage'].max()
# Add 40% padding to ensure labels fit within the figure
plt.xlim(0, max_pct * 1.4)

# Add percentage and count annotations with clear positioning
for i, row in enumerate(degree_group_train.sort_values('Percentage', ascending=False).head(12).itertuples()):
    # For large percentage values, place text inside the bar
    if row.Percentage > 10:
        # Place text at 80% of the way across the bar
        text_x = row.Percentage * 0.8
        plt.text(text_x, i, f'{row.Percentage:.2f}%', 
                 va='center', color='white', fontweight='bold')
        
        # Place count closer to the end of the bar
        plt.text(row.Percentage + 0.5, i, f'(n={row.Count:,})', 
                 va='center', ha='left')
    else:
        # For smaller bars, place percentage at the end of bar
        plt.text(row.Percentage + 0.3, i, f'{row.Percentage:.2f}%', 
                 va='center', ha='left')
        
        # Place count right after the percentage
        plt.text(row.Percentage + 3, i, f'(n={row.Count:,})', 
                 va='center', ha='left')

plt.tight_layout()
plt.show()

# Check impact of profession group on depression
print("\n" + "="*80)
print("PROFESSION GROUP vs DEPRESSION".center(80))
print("="*80)

# Calculate depression rate by profession group
prof_depression = train.groupby('Profession_Group')['Depression'].agg(['mean', 'count']).reset_index()
prof_depression['Depression_Rate'] = prof_depression['mean'] * 100
prof_depression = prof_depression.sort_values('Depression_Rate', ascending=False)

# Visualize profession group vs depression - FIXED to show all labels
plt.figure(figsize=(16, 10))  # Wider figure
bars = plt.barh(prof_depression['Profession_Group'], prof_depression['Depression_Rate'], 
        color=plt.cm.RdYlBu_r(prof_depression['Depression_Rate']/30))
plt.title('Depression Rate by Profession Group', fontsize=14)
plt.xlabel('Depression Rate (%)', fontsize=12)
plt.ylabel('Profession Group', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Add much more padding (60%) to ensure all annotations fit
plt.xlim(0, max(prof_depression['Depression_Rate']) * 1.6)

# Add annotations - with clearer separation
for i, bar in enumerate(bars):
    width = bar.get_width()
    count = prof_depression.iloc[i]['count']
    
    # If the bar is wide enough, place rate inside the bar
    if width > 15:
        plt.text(width * 0.5, i, f'{width:.2f}%', 
                va='center', ha='center', color='white', fontweight='bold')
        
        # Place count after the bar
        plt.text(width + 2, i, f'(n={count:,})', va='center')
    else:
        # For smaller bars, place rate just after end of bar
        plt.text(width + 1, i, f'{width:.2f}%', va='center')
        
        # Place count after the rate with clear space
        plt.text(width + 10, i, f'(n={count:,})', va='center')

plt.tight_layout()
plt.show()

# Check impact of degree group on depression
print("\n" + "="*80)
print("DEGREE GROUP vs DEPRESSION".center(80))
print("="*80)

# Calculate depression rate by degree group
deg_depression = train.groupby('Degree_Group')['Depression'].agg(['mean', 'count']).reset_index()
deg_depression['Depression_Rate'] = deg_depression['mean'] * 100
deg_depression = deg_depression.sort_values('Depression_Rate', ascending=False)

# Filter to show only groups with significant sample size (n > 1000)
deg_depression_filtered = deg_depression[deg_depression['count'] > 1000].copy()

# Visualize degree group vs depression - FIXED to show all labels
plt.figure(figsize=(16, 10))  # Wider figure
bars = plt.barh(deg_depression_filtered['Degree_Group'], deg_depression_filtered['Depression_Rate'], 
        color=plt.cm.RdYlBu_r(deg_depression_filtered['Depression_Rate']/30))
plt.title('Depression Rate by Degree Group (Groups with n > 1000)', fontsize=14)
plt.xlabel('Depression Rate (%)', fontsize=12)
plt.ylabel('Degree Group', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Add much more padding (60%) to ensure all annotations fit
plt.xlim(0, max(deg_depression_filtered['Depression_Rate']) * 1.6)

# Add annotations - with clearer separation
for i, bar in enumerate(bars):
    width = bar.get_width()
    count = deg_depression_filtered.iloc[i]['count']
    
    # If the bar is wide enough, place rate inside the bar
    if width > 20:
        plt.text(width * 0.5, i, f'{width:.2f}%', 
                va='center', ha='center', color='white', fontweight='bold')
        
        # Place count after the bar
        plt.text(width + 2, i, f'(n={count:,})', va='center')
    else:
        # For smaller bars, place rate just after end of bar
        plt.text(width + 1, i, f'{width:.2f}%', va='center')
        
        # Place count after the rate with clear space
        plt.text(width + 10, i, f'(n={count:,})', va='center')

plt.tight_layout()
plt.show()

# Check for unmapped values
unmapped_prof_train = [p for p in train['Profession'].dropna().unique() if p not in profession_groups]
unmapped_prof_test = [p for p in test['Profession'].dropna().unique() if p not in profession_groups]
unmapped_deg_train = [d for d in train['Degree'].dropna().unique() if d not in degree_groups]
unmapped_deg_test = [d for d in test['Degree'].dropna().unique() if d not in degree_groups]

print("\nUnmapped values summary:")
print(f"- Profession (train): {len(unmapped_prof_train)} unmapped values")
print(f"- Profession (test): {len(unmapped_prof_test)} unmapped values")
print(f"- Degree (train): {len(unmapped_deg_train)} unmapped values")
print(f"- Degree (test): {len(unmapped_deg_test)} unmapped values")

if unmapped_prof_train or unmapped_prof_test or unmapped_deg_train or unmapped_deg_test:
    print("\nList of unmapped values:")
    if unmapped_prof_train:
        print("Unmapped professions in train:", unmapped_prof_train)
    if unmapped_prof_test:
        print("Unmapped professions in test:", unmapped_prof_test)
    if unmapped_deg_train:
        print("Unmapped degrees in train:", unmapped_deg_train)
    if unmapped_deg_test:
        print("Unmapped degrees in test:", unmapped_deg_test)
############################################################################################################################################################

# 4. Numeric Features Analysis

# Identify meaningful numeric columns (excluding id and target)
numeric_cols = train.select_dtypes(include=['int64', 'float64']).columns
numeric_cols = [col for col in numeric_cols if col not in ['id', 'Depression']]

print("\n" + "="*80)
print("NUMERIC FEATURES ANALYSIS".center(80))
print("="*80)

# Function to create boxplots with properly positioned count labels
def create_boxplot_by_depression(feature):
    plt.figure(figsize=(14, 7))
    
    # Create the boxplot
    ax = sns.boxplot(x='Depression', y=feature, data=train.dropna(subset=[feature]))
    
    # Set labels and title
    plt.title(f'{feature} by Depression Status', fontsize=14)
    plt.ylabel(feature, fontsize=12)
    plt.xlabel('Depression (0=No, 1=Yes)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Get counts for each group
    counts = train.groupby('Depression')[feature].count()
    
    # Add count annotations with proper positioning - FIXED
    # Determine the lowest y-value shown on the plot
    y_min, y_max = plt.ylim()
    
    # Increase the bottom margin to make room for the labels
    plt.ylim(y_min - (y_max - y_min) * 0.1, y_max)
    
    # Place the count labels in the increased margin space
    for i, count in enumerate(counts):
        ax.annotate(f'n = {count:,}', 
                    xy=(i, y_min - (y_max - y_min) * 0.05),
                    ha='center', va='center', fontsize=10)
    
    # Ensure tight layout with proper margins for annotations
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Add extra bottom margin
    plt.show()
    
    # Print basic statistics
    print(f"\nStatistics for {feature} by Depression Status:")
    stats = train.groupby('Depression')[feature].agg(['mean', 'median', 'std', 'min', 'max']).round(2)
    stats.index = ['Not Depressed (0)', 'Depressed (1)']
    display(stats)
    
    # Calculate and print correlation with Depression
    corr = train[[feature, 'Depression']].corr().iloc[0, 1]
    print(f"Correlation with Depression: {corr:.4f}")
    print("-"*60)

# Create boxplot visualizations for each numeric feature
for col in numeric_cols:
    create_boxplot_by_depression(col)

# Create visualization of correlation with depression - FIXED
print("\nFeature Correlation with Depression:")
corr_data = train[numeric_cols + ['Depression']].corr()['Depression'].drop('Depression').sort_values()

# Create a horizontal bar chart with proper margins for labels
plt.figure(figsize=(14, 8))
bars = plt.barh(corr_data.index, corr_data.values, 
        color=['royalblue' if x < 0 else 'salmon' for x in corr_data.values])

plt.title('Feature Correlation with Depression', fontsize=14)
plt.xlabel('Correlation Coefficient', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)

# Ensure x-axis has enough padding for labels
x_min, x_max = plt.xlim()
range_x = x_max - x_min
plt.xlim(x_min - range_x * 0.05, x_max + range_x * 0.2)  # Add 20% padding to the right

# Add correlation value annotations with better positioning
for i, bar in enumerate(bars):
    width = bar.get_width()
    # Determine position based on correlation value
    if width > 0:
        # Positive correlations - place after bar
        x_pos = width + 0.01
        ha = 'left'
    else:
        # Negative correlations - place before bar
        x_pos = width - 0.01
        ha = 'right'
    
    plt.text(x_pos, i, f' {width:.3f}', va='center', ha=ha)

plt.tight_layout()
plt.show()

# Create a correlation heatmap
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(train[numeric_cols + ['Depression']].corr(), dtype=bool))
sns.heatmap(train[numeric_cols + ['Depression']].corr(), mask=mask, annot=True, 
            fmt='.2f', cmap='coolwarm', center=0, square=True, linewidths=0.5)
plt.title('Feature Correlation Matrix', fontsize=14)
plt.tight_layout()
plt.show()

# Feature distribution histograms
for col in numeric_cols:
    plt.figure(figsize=(14, 6))
    
    # Ensure we only plot non-null values
    train_data = train[col].dropna()
    
    # Create histogram with both general distribution and distribution by depression status
    plt.subplot(1, 2, 1)
    sns.histplot(train_data, bins=20, kde=True, color='steelblue')
    plt.title(f'{col} Overall Distribution')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.subplot(1, 2, 2)
    sns.histplot(data=train.dropna(subset=[col]), x=col, hue='Depression', 
                 bins=20, kde=True, palette=['steelblue', 'darkorange'], 
                 multiple='dodge', shrink=0.8)
    plt.title(f'{col} by Depression Status')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

# Compare the distribution of features between depressed and non-depressed individuals
# using mean values and standard error bars
def plot_feature_means_by_depression():
    means = []
    errors = []
    features = numeric_cols
    
    for feature in features:
        # Calculate means for each depression group
        feature_means = train.groupby('Depression')[feature].mean()
        # Calculate standard errors
        feature_errors = train.groupby('Depression')[feature].sem()
        
        means.append(feature_means.values)
        errors.append(feature_errors.values)
    
    # Prepare data for plotting
    means_df = pd.DataFrame(means, index=features, columns=['Not Depressed', 'Depressed'])
    errors_df = pd.DataFrame(errors, index=features, columns=['Not Depressed', 'Depressed'])
    
    # Z-score standardize the means for better comparison
    from scipy import stats
    means_standardized = pd.DataFrame(index=features, columns=['Not Depressed', 'Depressed'])
    for feature in features:
        values = train[feature].dropna().values
        mean = np.mean(values)
        std = np.std(values)
        means_standardized.loc[feature, 'Not Depressed'] = (means_df.loc[feature, 'Not Depressed'] - mean) / std
        means_standardized.loc[feature, 'Depressed'] = (means_df.loc[feature, 'Depressed'] - mean) / std
    
    # Plot standardized means
    plt.figure(figsize=(12, 8))
    bar_width = 0.35
    index = np.arange(len(features))
    
    plt.bar(index - bar_width/2, means_standardized['Not Depressed'], bar_width, 
            label='Not Depressed', color='steelblue', alpha=0.7)
    plt.bar(index + bar_width/2, means_standardized['Depressed'], bar_width, 
            label='Depressed', color='darkorange', alpha=0.7)
    
    plt.title('Standardized Feature Means by Depression Status', fontsize=14)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Standardized Mean (Z-Score)', fontsize=12)
    plt.xticks(index, features, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Plot mean feature values by depression status
plot_feature_means_by_depression()


# --- 1. Setup Environment and Data Paths ---
print("Starting preprocessing pipeline...")
n_jobs = min(20, os.cpu_count())
print(f"Using {n_jobs} CPU cores for parallel processing")

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    print(f"Active GPU: {torch.cuda.get_device_name(0)}")
    use_gpu = True
else:
    use_gpu = False
    print("No GPU available, using CPU")

# Update paths (Kaggle environment)
base_path = "/kaggle/input/playground-series-s4e11/"
output_path = "/kaggle/working/"
preprocessed_path = os.path.join(output_path, "preprocessed")
os.makedirs(preprocessed_path, exist_ok=True)

# --- 2. Enhanced Feature Engineering Class Definition ---
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Enhanced transformer for mental health data preprocessing with optional GPU acceleration."""
    
    def __init__(self, profession_groups=None, degree_groups=None, 
                 high_risk_professions=None, high_risk_degrees=None,
                 use_knn_imputer=True, n_neighbors=5, knn_weights='distance',
                 use_pca=True, pca_components=15, apply_scaling=True,
                 n_jobs=n_jobs, use_gpu=use_gpu):
        
        self.use_knn_imputer = use_knn_imputer
        self.n_neighbors = n_neighbors
        self.knn_weights = knn_weights
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.apply_scaling = apply_scaling
        self.n_jobs = n_jobs
        self.use_gpu = use_gpu
        
        self.categorical_mappings = {}
        self.numeric_medians = {}
        self.categorical_modes = {}
        self.numeric_iqrs = {}
        self.high_risk_professions = high_risk_professions or []
        self.high_risk_degrees = high_risk_degrees or []
        
        # Choose the scaler implementation; try GPU-based from cuML if possible.
        try:
            if self.use_gpu:
                import cuml
                self.scaler = cuml.preprocessing.StandardScaler()
            else:
                raise ImportError
        except:
            self.scaler = StandardScaler()
        
        # Standardize profession and degree group keys.
        if profession_groups is not None:
            self.profession_groups = {str(k).lower().strip(): v for k, v in profession_groups.items()}
        else:
            self.profession_groups = {}
            
        if degree_groups is not None:
            self.degree_groups = {str(k).lower().strip(): v for k, v in degree_groups.items()}
        else:
            self.degree_groups = {}
    
    def clean_text(self, x):
        """Clean and normalize text values."""
        return x.lower().strip() if isinstance(x, str) else x
    
    def is_valid_name(self, text):
        """Check if a string appears to be a valid name."""
        if not isinstance(text, str):
            return False
        text = text.lower().strip()
        invalid_patterns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', 
                            'unhealthy', 'moderate', 'healthy', 'yes', 'no']
        return not any(pattern in text for pattern in invalid_patterns)

    def fit(self, X, y=None):
        """Learn medians, modes, and scaling parameters from the training data."""
        # Set up standard categorical mappings
        self.categorical_mappings = {
            'sleep duration': {'less than 5 hours': 4, '5-6 hours': 5.5, '7-8 hours': 7.5, 'more than 8 hours': 9},
            'dietary habits': {'unhealthy': 0, 'moderate': 1, 'healthy': 2},
            'working professional or student': {'student': 1, 'working professional': 0},
            'have you ever had suicidal thoughts ?': {'yes': 1, 'no': 0},
            'family history of mental illness': {'yes': 1, 'no': 0},
            'gender': {'male': 0, 'female': 1, 'other': 2}
        }
        
        # Create student and professional masks
        student_mask = X['Working Professional or Student'].str.lower().str.strip() == 'student'
        professional_mask = ~student_mask
        
        numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
        self.student_medians = {}
        self.professional_medians = {}
        
        for col in numeric_cols:
            self.numeric_medians[col] = X[col].median()
            if col in ['Academic Pressure', 'CGPA', 'Study Satisfaction'] and student_mask.sum() > 100:
                self.student_medians[col] = X.loc[student_mask, col].median()
            if col in ['Work Pressure', 'Job Satisfaction'] and professional_mask.sum() > 100:
                self.professional_medians[col] = X.loc[professional_mask, col].median()
            
            # Calculate IQR for potential scaling
            q75, q25 = np.percentile(X[col].dropna(), [75, 25])
            iqr = q75 - q25
            self.numeric_iqrs[col] = iqr if iqr > 0 else 1
        
        # Store modes for additional categorical columns
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col.lower().strip() not in self.categorical_mappings and X[col].notna().any():
                self.categorical_modes[col] = X[col].mode()[0]
        
        # Initialize the KNN imputer (using scikit-learn without the n_jobs parameter)
        if self.use_knn_imputer:
            try:
                if self.use_gpu:
                    import cuml
                    self.knn_imputer = cuml.neighbors.KNeighborsRegressor(
                        n_neighbors=self.n_neighbors,
                        weights=self.knn_weights
                    )
                else:
                    raise ImportError
            except:
                self.knn_imputer = KNNImputer(
                    n_neighbors=self.n_neighbors,
                    weights=self.knn_weights
                )
        
        # Initialize PCA transformer
        if self.use_pca:
            try:
                if self.use_gpu:
                    from cuml.decomposition import PCA as cuPCA
                    self.pca = cuPCA(n_components=self.pca_components)
                else:
                    raise ImportError
            except:
                self.pca = PCA(n_components=self.pca_components, random_state=42)
            
        return self

    def map_profession_to_group(self, profession):
        """Map profession values to standardized groups."""
        if pd.isna(profession):
            return 'Unknown'
        profession = self.clean_text(profession)
        return self.profession_groups.get(profession, 'Other')

    def map_degree_to_group(self, degree):
        """Map degree values to standardized groups."""
        if pd.isna(degree):
            return 'Unknown'
        degree = self.clean_text(degree)
        return self.degree_groups.get(degree, 'Other')
    
    def transform(self, X):
        """Perform extensive feature engineering and ensure no missing values remain."""
        df = X.copy()
        using_gpu_df = False
        if self.use_gpu:
            try:
                import cudf
                df = cudf.DataFrame.from_pandas(df)
                using_gpu_df = True
                print("Using GPU-accelerated DataFrame processing")
            except Exception as e:
                print("Failed to convert to GPU DataFrame, using pandas:", e)
        
        # --- 1. Basic text cleaning and normalization ---
        text_cols = ['Name', 'Gender', 'City', 'Profession', 'Degree']
        if using_gpu_df:
            for col in text_cols:
                if col in df.columns:
                    df[col] = df[col].str.lower().str.strip()
        else:
            with parallel_backend('threading', n_jobs=self.n_jobs):
                for col in text_cols:
                    if col in df.columns:
                        df[col] = df[col].apply(self.clean_text)
        
        # --- 2. Data cleaning: fix incorrect entries for 'Profession' ---
        if 'Profession' in df.columns:
            if using_gpu_df:
                name_list = ['yogesh', 'yuvraj', 'pranav', 'manvi', 'samar', 'simran']
                for name in name_list:
                    df.loc[df['Profession'].str.lower() == name, 'Profession'] = None
            else:
                name_mask = df['Profession'].apply(lambda x: isinstance(x, str) and x.lower() in 
                                                     ['yogesh', 'yuvraj', 'pranav', 'manvi', 'samar', 'simran'])
                df.loc[name_mask, 'Profession'] = np.nan
        
        # --- 3. Create missing indicators for key numeric variables ---
        for col in ['Academic Pressure', 'Work Pressure', 'CGPA', 'Study Satisfaction', 'Job Satisfaction']:
            if col in df.columns:
                df[f'missing_{col.lower().replace(" ", "_")}'] = df[col].isna().astype(int)
        
        # --- 4. Basic feature transformations for student indicator ---
        if 'Working Professional or Student' in df.columns:
            if using_gpu_df:
                mapping = self.categorical_mappings['working professional or student']
                df['Working Professional or Student'] = df['Working Professional or Student'].str.lower().str.strip()
                for key, value in mapping.items():
                    df.loc[df['Working Professional or Student'] == key, 'is_student'] = value
            else:
                df['is_student'] = df['Working Professional or Student'].apply(self.clean_text)\
                                  .map(self.categorical_mappings['working professional or student']).fillna(0)
        else:
            df['is_student'] = 0
        
        # --- 5. Suicidal thoughts and family history indicators ---
        if 'Have you ever had suicidal thoughts ?' in df.columns:
            if using_gpu_df:
                mapping = self.categorical_mappings['have you ever had suicidal thoughts ?']
                df['Have you ever had suicidal thoughts ?'] = df['Have you ever had suicidal thoughts ?'].str.lower().str.strip()
                for key, value in mapping.items():
                    df.loc[df['Have you ever had suicidal thoughts ?'] == key, 'has_suicidal_thoughts'] = value
            else:
                df['has_suicidal_thoughts'] = df['Have you ever had suicidal thoughts ?'].apply(self.clean_text)\
                                             .map(self.categorical_mappings['have you ever had suicidal thoughts ?']).fillna(0)
            
            if 'Sleep Duration' in df.columns:
                if using_gpu_df:
                    df['suicidal_thoughts_x_poor_sleep'] = df['has_suicidal_thoughts'] * \
                        (df['Sleep Duration'].str.lower().str.strip() == 'less than 5 hours').astype(int)
                else:
                    df['suicidal_thoughts_x_poor_sleep'] = df['has_suicidal_thoughts'] * \
                        (df['Sleep Duration'].apply(self.clean_text) == 'less than 5 hours').astype(int)
        
        if 'Family History of Mental Illness' in df.columns:
            if using_gpu_df:
                mapping = self.categorical_mappings['family history of mental illness']
                df['Family History of Mental Illness'] = df['Family History of Mental Illness'].str.lower().str.strip()
                for key, value in mapping.items():
                    df.loc[df['Family History of Mental Illness'] == key, 'has_family_history'] = value
            else:
                df['has_family_history'] = df['Family History of Mental Illness'].apply(self.clean_text)\
                                          .map(self.categorical_mappings['family history of mental illness']).fillna(0)
            if 'has_suicidal_thoughts' in df.columns:
                df['family_history_x_suicidal_thoughts'] = df['has_family_history'] * df['has_suicidal_thoughts']
        
        # --- 6. Sleep duration features ---
        if 'Sleep Duration' in df.columns:
            if using_gpu_df:
                mapping = self.categorical_mappings['sleep duration']
                df['Sleep Duration'] = df['Sleep Duration'].str.lower().str.strip()
                for key, value in mapping.items():
                    df.loc[df['Sleep Duration'] == key, 'Sleep_Duration_Hours'] = value
            else:
                df['Sleep_Duration_Hours'] = df['Sleep Duration'].apply(self.clean_text)\
                                            .map(self.categorical_mappings['sleep duration'])\
                                            .fillna(self.numeric_medians.get('Sleep Duration', 7.5))
            
            df['Poor_Sleep'] = (df['Sleep_Duration_Hours'] < 5.5).astype(int)
            df['Optimal_Sleep'] = ((df['Sleep_Duration_Hours'] >= 7) & (df['Sleep_Duration_Hours'] <= 9)).astype(int)
        
        # --- 7. Dietary habits features ---
        if 'Dietary Habits' in df.columns:
            if using_gpu_df:
                mapping = self.categorical_mappings['dietary habits']
                df['Dietary Habits'] = df['Dietary Habits'].str.lower().str.strip()
                for key, value in mapping.items():
                    df.loc[df['Dietary Habits'] == key, 'Dietary_Health'] = value
            else:
                df['Dietary_Health'] = df['Dietary Habits'].apply(self.clean_text)\
                                      .map(self.categorical_mappings['dietary habits']).fillna(1)
            df['Unhealthy_Diet'] = (df['Dietary_Health'] == 0).astype(int)
        
        # --- 8. Age group features ---
        if 'Age' in df.columns:
            if using_gpu_df:
                df['Young_Adult'] = (df['Age'] <= 25).astype(int)
                df['Middle_Aged'] = ((df['Age'] >= 35) & (df['Age'] <= 55)).astype(int)
                df['Age_Group'] = 'Unknown'
                df.loc[(df['Age'] > 17) & (df['Age'] <= 25), 'Age_Group'] = '18-25'
                df.loc[(df['Age'] > 25) & (df['Age'] <= 35), 'Age_Group'] = '26-35'
                df.loc[(df['Age'] > 35) & (df['Age'] <= 45), 'Age_Group'] = '36-45'
                df.loc[(df['Age'] > 45) & (df['Age'] <= 55), 'Age_Group'] = '46-55'
                df.loc[(df['Age'] > 55) & (df['Age'] <= 65), 'Age_Group'] = '56-60'
            else:
                bins = [17, 25, 35, 45, 55, 65]
                labels = ['18-25', '26-35', '36-45', '46-55', '56-60']
                df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels)
                df['Young_Adult'] = (df['Age'] <= 25).astype(int)
                df['Middle_Aged'] = ((df['Age'] >= 35) & (df['Age'] <= 55)).astype(int)
        
        # --- 9. Profession and degree grouping ---
        if 'Profession' in df.columns:
            if using_gpu_df:
                temp_prof = df['Profession'].to_pandas() if hasattr(df['Profession'], 'to_pandas') else df['Profession']
                df['Profession_Group'] = temp_prof.apply(self.map_profession_to_group).values
                df['High_Risk_Profession'] = df['Profession_Group'].isin(self.high_risk_professions).astype(int)
            else:
                df['Profession_Group'] = df['Profession'].apply(self.map_profession_to_group)
                df['High_Risk_Profession'] = df['Profession_Group'].isin(self.high_risk_professions).astype(int)
        
        if 'Degree' in df.columns:
            if using_gpu_df:
                temp_degree = df['Degree'].to_pandas() if hasattr(df['Degree'], 'to_pandas') else df['Degree']
                df['Degree_Group'] = temp_degree.apply(self.map_degree_to_group).values
                df['High_Risk_Degree'] = df['Degree_Group'].isin(self.high_risk_degrees).astype(int)
            else:
                df['Degree_Group'] = df['Degree'].apply(self.map_degree_to_group)
                df['High_Risk_Degree'] = df['Degree_Group'].isin(self.high_risk_degrees).astype(int)
        
        # --- 10. Context-based imputation for academic/professional variables ---
        student_mask = df['is_student'] == 1
        professional_mask = df['is_student'] == 0
        
        if 'Academic Pressure' in df.columns:
            student_val = self.student_medians.get('Academic Pressure', self.numeric_medians.get('Academic Pressure', 3))
            df.loc[student_mask & df['Academic Pressure'].isna(), 'Academic Pressure'] = student_val
            df.loc[professional_mask, 'Academic Pressure'] = df.loc[professional_mask, 'Academic Pressure'].fillna(0)
        
        if 'CGPA' in df.columns:
            student_val = self.student_medians.get('CGPA', self.numeric_medians.get('CGPA', 7.5))
            df.loc[student_mask & df['CGPA'].isna(), 'CGPA'] = student_val
            df.loc[professional_mask, 'CGPA'] = df.loc[professional_mask, 'CGPA'].fillna(0)
        
        if 'Study Satisfaction' in df.columns:
            student_val = self.student_medians.get('Study Satisfaction', self.numeric_medians.get('Study Satisfaction', 3))
            df.loc[student_mask & df['Study Satisfaction'].isna(), 'Study Satisfaction'] = student_val
            df.loc[professional_mask, 'Study Satisfaction'] = df.loc[professional_mask, 'Study Satisfaction'].fillna(0)
        
        if 'Work Pressure' in df.columns:
            prof_val = self.professional_medians.get('Work Pressure', self.numeric_medians.get('Work Pressure', 3))
            df.loc[professional_mask & df['Work Pressure'].isna(), 'Work Pressure'] = prof_val
            df.loc[student_mask, 'Work Pressure'] = df.loc[student_mask, 'Work Pressure'].fillna(0)
        
        if 'Job Satisfaction' in df.columns:
            prof_val = self.professional_medians.get('Job Satisfaction', self.numeric_medians.get('Job Satisfaction', 3))
            df.loc[professional_mask & df['Job Satisfaction'].isna(), 'Job Satisfaction'] = prof_val
            df.loc[student_mask, 'Job Satisfaction'] = df.loc[student_mask, 'Job Satisfaction'].fillna(0)
        
        # --- 11. Create combined features ---
        df['Overall_Satisfaction'] = 0
        if 'Job Satisfaction' in df.columns and 'Study Satisfaction' in df.columns:
            df.loc[professional_mask, 'Overall_Satisfaction'] = df.loc[professional_mask, 'Job Satisfaction']
            df.loc[student_mask, 'Overall_Satisfaction'] = df.loc[student_mask, 'Study Satisfaction']
        
        df['Overall_Pressure'] = 0
        if 'Work Pressure' in df.columns and 'Academic Pressure' in df.columns:
            df.loc[professional_mask, 'Overall_Pressure'] = df.loc[professional_mask, 'Work Pressure']
            df.loc[student_mask, 'Overall_Pressure'] = df.loc[student_mask, 'Academic Pressure']
        
        # --- 12. Advanced imputation using KNN if enabled ---
        if self.use_knn_imputer:
            print("Applying KNN imputation for missing values...")
            numeric_cols = [col for col in df.select_dtypes(include=['float64', 'int64']).columns 
                            if not col.startswith('missing_')]
            if using_gpu_df:
                temp_df = df[numeric_cols].to_pandas()
            else:
                temp_df = df[numeric_cols]
            
            if self.use_gpu and hasattr(self, 'knn_imputer') and 'predict' in dir(self.knn_imputer):
                for col in numeric_cols:
                    mask = temp_df[col].isna()
                    if mask.sum() > 0:
                        X_train = temp_df.loc[~mask, numeric_cols].drop(columns=[col])
                        y_train = temp_df.loc[~mask, col]
                        X_test = temp_df.loc[mask, numeric_cols].drop(columns=[col])
                        if len(X_train) > 0 and len(X_test) > 0:
                            self.knn_imputer.fit(X_train, y_train)
                            temp_df.loc[mask, col] = self.knn_imputer.predict(X_test)
            else:
                imputed = self.knn_imputer.fit_transform(temp_df)
                temp_df = pd.DataFrame(imputed, columns=numeric_cols, index=temp_df.index)
            
            if using_gpu_df:
                for col in numeric_cols:
                    df[col] = temp_df[col].values
            else:
                df[numeric_cols] = temp_df
        
        # --- 13. Apply scaling if enabled ---
        if self.apply_scaling:
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            if using_gpu_df:
                temp_df = df[numeric_cols].to_pandas()
            else:
                temp_df = df[numeric_cols]
            scaled_data = self.scaler.fit_transform(temp_df)
            scaled_df = pd.DataFrame(scaled_data, columns=numeric_cols, index=temp_df.index)
            if using_gpu_df:
                for col in numeric_cols:
                    df[col] = scaled_df[col].values
            else:
                df[numeric_cols] = scaled_df
        
        # --- 14. Apply PCA if enabled ---
        if self.use_pca:
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            if using_gpu_df:
                temp_df = df[numeric_cols].to_pandas()
            else:
                temp_df = df[numeric_cols]
            pca_data = self.pca.fit_transform(temp_df)
            for i in range(min(self.pca_components, pca_data.shape[1])):
                df[f'PCA_{i+1}'] = pca_data[:, i]
        
        # --- 15. Final cleanup for any remaining missing values ---
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            if df[col].isna().sum() > 0:
                df[col] = df[col].fillna(self.numeric_medians.get(col, 0))
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isna().sum() > 0:
                df[col] = df[col].fillna(self.categorical_modes.get(col, 'Unknown'))
        # Use forward-fill and backward-fill to ensure no missing values remain.
        df = df.ffill().bfill()
        
        if using_gpu_df:
            df = df.to_pandas()
        
        assert df.isnull().sum().sum() == 0, "There are still missing values in the processed data."
        print("No missing values remain in the processed data.")
        
        return df

# --- 3. Define High Risk Groups and Mappings ---
high_risk_professions = ['Creative & Media', 'Education', 'Technology & Engineering']
high_risk_degrees = ['Bachelor of Science/Engineering', 'Master of Science/Engineering', 'Bachelor of Arts/Humanities']

profession_groups = {
    'doctor': 'Healthcare', 'medical doctor': 'Healthcare', 'mbbs': 'Healthcare', 'pharmacist': 'Healthcare',
    'surgeon': 'Healthcare', 'md': 'Healthcare', 'b.pharm': 'Healthcare', 'm.pharm': 'Healthcare', 'phd': 'Healthcare',
    'teacher': 'Education', 'educational consultant': 'Education', 'academic': 'Education', 'b.ed': 'Education',
    'm.ed': 'Education', 'accountant': 'Business & Finance', 'business analyst': 'Business & Finance',
    'financial analyst': 'Business & Finance', 'finanancial analyst': 'Business & Finance', 'investment banker': 'Business & Finance',
    'entrepreneur': 'Business & Finance', 'mba': 'Business & Finance', 'b.com': 'Business & Finance', 'bba': 'Business & Finance',
    'software engineer': 'Technology & Engineering', 'data scientist': 'Technology & Engineering', 'ux/ui designer': 'Technology & Engineering',
    'dev': 'Technology & Engineering', 'civil engineer': 'Technology & Engineering', 'mechanical engineer': 'Technology & Engineering',
    'm.tech': 'Technology & Engineering', 'mca': 'Technology & Engineering', 'be': 'Technology & Engineering', 'me': 'Technology & Engineering',
    'lawyer': 'Legal', 'judge': 'Legal', 'llm': 'Legal', 'content writer': 'Creative & Media', 'graphic designer': 'Creative & Media',
    'digital marketer': 'Creative & Media', 'chef': 'Service & Hospitality', 'customer support': 'Service & Hospitality',
    'travel consultant': 'Service & Hospitality', 'pilot': 'Service & Hospitality', 'architect': 'Architecture & Design',
    'hr manager': 'Management & Consulting', 'manager': 'Management & Consulting', 'marketing manager': 'Management & Consulting',
    'consultant': 'Management & Consulting', 'family consultant': 'Management & Consulting', 'city consultant': 'Management & Consulting',
    'city manager': 'Management & Consulting', 'research analyst': 'Management & Consulting', 'researcher': 'Management & Consulting',
    'analyst': 'Management & Consulting', 'electrician': 'Skilled Trades', 'plumber': 'Skilled Trades', 'chemist': 'Skilled Trades',
    'sales executive': 'Sales & Marketing', 'student': 'Students & Unemployed', 'unemployed': 'Students & Unemployed',
    'working professional': 'General Professional', 'unveil': 'General Professional', 'profession': 'General Professional',
    'familyvirar': 'Data Error', 'nagpur': 'Data Error', 'patna': 'Data Error', 'visakhapatnam': 'Data Error',
    'surat': 'Data Error', 'yogesh': 'Data Error', 'yuvraj': 'Data Error', 'pranav': 'Data Error', 'manvi': 'Data Error',
    'samar': 'Data Error', 'simran': 'Data Error', '24th': 'Data Error', '3m': 'Data Error', 'unhealthy': 'Data Error',
    'no': 'Data Error', 'name': 'Data Error', 'moderate': 'Data Error'
}

degree_groups = {
    'class 12': 'High School', 'class 11': 'High School', 'b.com': 'Bachelor of Business/Commerce', 'b b.com': 'Bachelor of Business/Commerce',
    'bba': 'Bachelor of Business/Commerce', 'b.ba': 'Bachelor of Business/Commerce', 'bbm': 'Bachelor of Business/Commerce',
    'bms': 'Bachelor of Business/Commerce', 'p.com': 'Bachelor of Business/Commerce', 'll.com': 'Bachelor of Business/Commerce',
    'llcom': 'Bachelor of Business/Commerce', 'aca': 'Bachelor of Business/Commerce', 'b_com': 'Bachelor of Business/Commerce',
    'b financial analyst': 'Bachelor of Business/Commerce', 'b.sc': 'Bachelor of Science/Engineering', 'bsc': 'Bachelor of Science/Engineering',
    'b.pharm': 'Bachelor of Science/Engineering', 'bpharm': 'Bachelor of Science/Engineering', 'b._pharm': 'Bachelor of Science/Engineering',
    'h_pharm': 'Bachelor of Science/Engineering', 's.pharm': 'Bachelor of Science/Engineering', 'p.pharm': 'Bachelor of Science/Engineering',
    'n.pharm': 'Bachelor of Science/Engineering', 'b.tech': 'Bachelor of Science/Engineering', 'btech': 'Bachelor of Science/Engineering',
    'b b.tech': 'Bachelor of Science/Engineering', 'be': 'Bachelor of Science/Engineering', 'bca': 'Bachelor of Science/Engineering',
    'b.ca': 'Bachelor of Science/Engineering', 'b bca': 'Bachelor of Science/Engineering', 'rca': 'Bachelor of Science/Engineering',
    'pca': 'Bachelor of Science/Engineering', 'lca': 'Bachelor of Science/Engineering', 'gca': 'Bachelor of Science/Engineering',
    'hca': 'Bachelor of Science/Engineering', 'bhca': 'Bachelor of Science/Engineering', 'e.tech': 'Bachelor of Science/Engineering',
    's.tech': 'Bachelor of Science/Engineering', 'ba': 'Bachelor of Arts/Humanities', 'b.a': 'Bachelor of Arts/Humanities',
    'b ba': 'Bachelor of Arts/Humanities', 'bpa': 'Bachelor of Arts/Humanities', 'b.ed': 'Bachelor of Education',
    'bed': 'Bachelor of Education', 'll b.ed': 'Bachelor of Education', 'k.ed': 'Bachelor of Education', 'l.ed': 'Bachelor of Education',
    'lled': 'Bachelor of Education', 'a.ed': 'Bachelor of Education', 'e.ed': 'Bachelor of Education', 'g.ed': 'Bachelor of Education',
    'i.ed': 'Bachelor of Education', 'j.ed': 'Bachelor of Education', 'llb': 'Bachelor of Law', 'llba': 'Bachelor of Law',
    'lls': 'Bachelor of Law', 'b.arch': 'Bachelor of Architecture', 'barch': 'Bachelor of Architecture', 'b.b.arch': 'Bachelor of Architecture',
    's.arch': 'Bachelor of Architecture', 'bhm': 'Bachelor of Hotel Management', 'lhm': 'Bachelor of Hotel Management',
    'b': 'General Bachelor Degree', 'bb': 'General Bachelor Degree', 'bh': 'General Bachelor Degree', 'b.h': 'General Bachelor Degree',
    'm.com': 'Master of Business/Commerce', 'b.m.com': 'Master of Business/Commerce', 'mba': 'Master of Business/Commerce',
    'pgdm': 'Master of Business/Commerce', 'm.sc': 'Master of Science/Engineering', 'msc': 'Master of Science/Engineering',
    'm.tech': 'Master of Science/Engineering', 'mtech': 'Master of Science/Engineering', 'm_tech': 'Master of Science/Engineering',
    'me': 'Master of Science/Engineering', 'mca': 'Master of Science/Engineering', 'm.pharm': 'Master of Science/Engineering',
    'mpharm': 'Master of Science/Engineering', 'm': 'Master of Science/Engineering', 'm.': 'Master of Science/Engineering',
    'lltech': 'Master of Science/Engineering', 'm.ui': 'Master of Science/Engineering', 'ma': 'Master of Arts/Humanities',
    'm.a': 'Master of Arts/Humanities', 'mpa': 'Master of Arts/Humanities', 'm.ed': 'Master of Education', 'med': 'Master of Education',
    'm.b.ed': 'Master of Education', 'm.m.ed': 'Master of Education', 'llm': 'Master of Law', 'm.arch': 'Master of Architecture',
    'mhm': 'Master of Hotel Management', 'phd': 'Doctoral Degree', 'ph.d': 'Doctoral Degree', 'm.phil': 'Doctoral Degree',
    'mbbs': 'Medical Degree', 'md': 'Medical Degree', 'bds': 'Medical Degree', 'm.s': 'Medical Degree', 'ms': 'Medical Degree',
    '0': 'Data Error', '20': 'Data Error', '24': 'Data Error', '29': 'Data Error', '3.0': 'Data Error', '5.56': 'Data Error',
    '5.61': 'Data Error', '5.65': 'Data Error', '5.88': 'Data Error', '7.06': 'Data Error', '8.56': 'Data Error', '8.95': 'Data Error',
    'b.03': 'Data Error', 'b.3.79': 'Data Error', 'cgpa': 'Data Error', 'b.study_hours': 'Data Error', 'b.press': 'Data Error',
    'b.student': 'Data Error', 'b. gender': 'Data Error', 'degree': 'Data Error', 'unite': 'Data Error', 'm. business analyst': 'Data Error',
    'business analyst': 'Occupation Entry', 'data scientist': 'Occupation Entry', 'doctor': 'Occupation Entry',
    'entrepreneur': 'Occupation Entry', 'hr manager': 'Occupation Entry', 'plumber': 'Occupation Entry', 'ux/ui designer': 'Occupation Entry',
    'working professional': 'Occupation Entry', 'mechanical engineer': 'Occupation Entry', 'travel consultant': 'Occupation Entry',
    'aarav': 'Name Entry', 'aadhya': 'Name Entry', 'advait': 'Name Entry', 'badhya': 'Name Entry', 'banchal': 'Name Entry',
    'bhavesh': 'Name Entry', 'bian': 'Name Entry', 'brit': 'Name Entry', 'brithika': 'Name Entry', 'esha': 'Name Entry',
    'eshita': 'Name Entry', 'gagan': 'Name Entry', 'jhanvi': 'Name Entry', 'kavya': 'Name Entry', 'lata': 'Name Entry',
    'magan': 'Name Entry', 'mahika': 'Name Entry', 'marsh': 'Name Entry', 'mihir': 'Name Entry', 'moham': 'Name Entry',
    'mthanya': 'Name Entry', 'nalini': 'Name Entry', 'navya': 'Name Entry', 'pihu': 'Name Entry', 'ritik': 'Name Entry',
    'rupak': 'Name Entry', 'veda': 'Name Entry', 'vibha': 'Name Entry', 'vivaan': 'Name Entry', 'vrinda': 'Name Entry',
    'bhopal': 'Location Entry', 'kalyan': 'Location Entry', 'pune': 'Location Entry'
}

# --- 4. Load and Process Data ---
print("Loading data...")
train = pd.read_csv(os.path.join(base_path, "train.csv"), encoding='utf-8', sep=',')
test = pd.read_csv(os.path.join(base_path, "test.csv"), encoding='utf-8', sep=',')
submission = pd.read_csv(os.path.join(base_path, "sample_submission.csv"), encoding='utf-8', sep=',')

print("Original data shapes:")
print(f"Train: {train.shape}")
print(f"Test: {test.shape}")

target_column = [col for col in submission.columns if col != 'id'][0]
print(f"Target column identified as: {target_column}")

if target_column in train.columns:
    y = train[target_column]
    X = train.drop(columns=[target_column])
else:
    raise ValueError(f"Target column '{target_column}' not found in training data")

# --- 5. Process Data with Enhanced Feature Engineering ---
print("\n=== Applying Enhanced Feature Engineering ===")
fe = FeatureEngineer(
    profession_groups=profession_groups, 
    degree_groups=degree_groups, 
    high_risk_professions=high_risk_professions, 
    high_risk_degrees=high_risk_degrees,
    use_knn_imputer=True,
    n_neighbors=5,
    knn_weights='distance',
    use_pca=True,
    pca_components=15,
    apply_scaling=True,
    n_jobs=n_jobs
)

print("Preprocessing training data...")
X_processed = fe.fit_transform(X)
print(f"Processed training data shape: {X_processed.shape}")
print("\nSample of processed training data:")
print(X_processed.head())

print("\nPreprocessing test data...")
test_processed = fe.transform(test)
print(f"Processed test data shape: {test_processed.shape}")
print("\nSample of processed test data:")
print(test_processed.head())

new_cols = [col for col in X_processed.columns if col not in X.columns]
print(f"\nAdded {len(new_cols)} new features including:")
for col in new_cols[:10]:
    print(f"- {col}")
if len(new_cols) > 10:
    print(f"... and {len(new_cols) - 10} more")

# --- 6. Save Processed Data and Preprocessing Objects ---
print("\n=== Saving Processed Data ===")
X_processed.to_csv(os.path.join(preprocessed_path, 'X_processed.csv'), index=False)
pd.DataFrame(y).to_csv(os.path.join(preprocessed_path, 'y_train.csv'), index=False)
test_processed.to_csv(os.path.join(preprocessed_path, 'test_processed.csv'), index=False)
joblib.dump(fe, os.path.join(preprocessed_path, 'feature_engineer.joblib'))

feature_info = {
    'original_columns': list(X.columns),
    'processed_columns': list(X_processed.columns),
    'new_features': new_cols,
    'high_risk_professions': high_risk_professions,
    'high_risk_degrees': high_risk_degrees,
    'pca_components': fe.pca_components if fe.use_pca else 0,
    'knn_imputation_used': fe.use_knn_imputer
}

with open(os.path.join(preprocessed_path, 'feature_info.json'), 'w') as f:
    json.dump(feature_info, f, indent=2)

print(f"\nPreprocessing complete. Files saved to {preprocessed_path}")
print("You can now run the modeling script with the preprocessed data.")



# --- 1. Setup Environment and Data Paths ---
n_jobs = min(20, os.cpu_count())
print(f"Using {n_jobs} CPU cores for parallel processing")

# Check for GPU availability for XGBoost
try:
    gpu_available = xgb.config.get_config().get('use_gpu', False)
    tree_method = 'gpu_hist' if gpu_available else 'hist'
    print(f"GPU acceleration for XGBoost: {'Available' if gpu_available else 'Not available'}")
except Exception as e:
    tree_method = 'hist'
    print("GPU not detected, using CPU for XGBoost")

# Set base and output paths for Kaggle
base_path = "/kaggle/input/playground-series-s4e11/"
output_path = "/kaggle/working/"

# Load Datasets safely with encoding and separator
train = pd.read_csv(base_path + "train.csv", encoding='utf-8', sep=',')
test = pd.read_csv(base_path + "test.csv", encoding='utf-8', sep=',')
submission = pd.read_csv(base_path + "sample_submission.csv", encoding='utf-8', sep=',')

# Preprocessed files are saved in "output/preprocessed"
preprocessed_path = os.path.join(output_path, "preprocessed")
os.makedirs(preprocessed_path, exist_ok=True)
# Directory to save model results
model_results_path = os.path.join(output_path, "model_results")
os.makedirs(model_results_path, exist_ok=True)

print("\n=== Loading Preprocessed Data ===")
X_processed = pd.read_csv(os.path.join(preprocessed_path, 'X_processed.csv'))
y_train = pd.read_csv(os.path.join(preprocessed_path, 'y_train.csv')).iloc[:, 0]
test_processed = pd.read_csv(os.path.join(preprocessed_path, 'test_processed.csv'))

print(f"Training features shape: {X_processed.shape}")
print(f"Target vector shape: {y_train.shape}")
print(f"Test features shape: {test_processed.shape}")

with open(os.path.join(preprocessed_path, 'feature_info.json'), 'r') as f:
    feature_info = json.load(f)

print(f"Loaded {len(feature_info['processed_columns'])} processed features")
print(f"Added {len(feature_info['new_features'])} engineered features during preprocessing")

# --- 3. Train-Validation Split ---
X_train, X_val, y_train_split, y_val = train_test_split(
    X_processed, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print("Class distribution in target variable:")
print(pd.Series(y_train).value_counts(normalize=True))
print(pd.Series(y_train).value_counts())

# --- 4. Feature Selection ---
print("\n=== Applying Feature Selection ===")
# Exclude 'id' column if present
feature_cols = [col for col in X_train.columns if col != 'id']
X_train_fs = X_train[feature_cols].copy()
X_val_fs = X_val[feature_cols].copy()

# Convert all columns to numeric (non-convertible values become NaN)
X_train_fs = X_train_fs.apply(lambda col: pd.to_numeric(col, errors='coerce'))
X_val_fs = X_val_fs.apply(lambda col: pd.to_numeric(col, errors='coerce'))

# Fill missing values with 0
X_train_fs = X_train_fs.fillna(0)
X_val_fs = X_val_fs.fillna(0)

k_best_features = min(30, X_train_fs.shape[1])
with parallel_backend('threading', n_jobs=n_jobs):
    selector = SelectKBest(mutual_info_classif, k=k_best_features)
    X_train_selected = selector.fit_transform(X_train_fs, y_train_split)
    selected_indices = selector.get_support(indices=True)
    selected_features = X_train_fs.columns[selected_indices].tolist()

print(f"Selected {k_best_features} best features:")
feature_scores = list(zip(X_train_fs.columns, selector.scores_))
feature_scores.sort(key=lambda x: x[1], reverse=True)
for feature, score in feature_scores[:10]:
    print(f"- {feature}: {score:.4f}")
if len(feature_scores) > 10:
    print(f"... and {len(feature_scores) - 10} more")

X_val_selected = X_val_fs[selected_features]
test_selected = test_processed[selected_features]
print(f"Shape after feature selection - Train: {X_train_selected.shape}, Val: {X_val_selected.shape}")

# --- 5. Class Balancing with SMOTE ---
print("\n=== Applying SMOTE for Class Balancing ===")
print("Original class distribution:")
print(pd.Series(y_train_split).value_counts(normalize=True))

with parallel_backend('threading', n_jobs=n_jobs):
    smote = SMOTE(random_state=42)  # Removed n_jobs parameter
    X_train_resampled, y_train_resampled = smote.fit_resample(
        pd.DataFrame(X_train_selected, columns=selected_features), 
        y_train_split
    )

print("Class distribution after SMOTE:")
print(pd.Series(y_train_resampled).value_counts(normalize=True))
print(f"Resampled training data shape: {X_train_resampled.shape}")

# --- 6. Model Training ---
print("\n=== Training XGBoost Model ===")
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'eta': 0.05,
    'max_depth': 5,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': tree_method,
    'random_state': 42,
    'n_jobs': n_jobs
}

model = xgb.XGBClassifier(**params)
model.fit(
    X_train_resampled, 
    y_train_resampled, 
    eval_set=[(X_val_selected, y_val)],
    early_stopping_rounds=20,
    verbose=True
)

# --- 7. Cross-Validation ---
print("\n=== Performing Cross-Validation ===")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Implementação manual de cross-validation para evitar o erro com __sklearn_tags__
cv_scores = []

for train_idx, val_idx in cv.split(X_train_selected, y_train_split):
    X_cv_train = pd.DataFrame(X_train_selected, columns=selected_features).iloc[train_idx]
    y_cv_train = y_train_split.iloc[train_idx]
    X_cv_val = pd.DataFrame(X_train_selected, columns=selected_features).iloc[val_idx]
    y_cv_val = y_train_split.iloc[val_idx]
    
    cv_model = xgb.XGBClassifier(**params)
    cv_model.fit(X_cv_train, y_cv_train)
    y_cv_pred_proba = cv_model.predict_proba(X_cv_val)[:, 1]
    cv_scores.append(roc_auc_score(y_cv_val, y_cv_pred_proba))

cv_scores = np.array(cv_scores)
print(f"Cross-validation ROC AUC scores: {cv_scores}")
print(f"Mean CV ROC AUC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# --- 8. Model Evaluation ---
print("\n=== Evaluating Model Performance ===")
y_pred_proba = model.predict_proba(X_val_selected)[:, 1]
y_pred = model.predict(X_val_selected)

auc = roc_auc_score(y_val, y_pred_proba)
ap = average_precision_score(y_val, y_pred_proba)
accuracy = accuracy_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
conf_matrix = confusion_matrix(y_val, y_pred)

print(f"ROC AUC: {auc:.4f}")
print(f"Average Precision: {ap:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_report(y_val, y_pred))

# --- 9. Generate Predictions for Test Data ---
print("\n=== Generating Predictions for Test Data ===")
test_pred_proba = model.predict_proba(test_selected)[:, 1]
test_predictions = model.predict(test_selected)

print("\nTest Predictions (first 10 samples):")
print(pd.DataFrame({
    'ID': test_processed['id'][:10] if 'id' in test_processed.columns else range(10),
    'Predicted': test_predictions[:10],
    'Probability': test_pred_proba[:10]
}))

submission = pd.read_csv(base_path + "sample_submission.csv", encoding='utf-8', sep=',')
target_column = [col for col in submission.columns if col != 'id'][0]
submission[target_column] = test_predictions
submission.to_csv(os.path.join(output_path, 'submission.csv'), index=False)
print(f"Submission file saved to {os.path.join(output_path, 'submission.csv')}")

# --- 10. Feature Importance Analysis ---
print("\n=== Feature Importance Analysis ===")
feature_importance = model.feature_importances_
sorted_idx = np.argsort(feature_importance)[::-1]
top_features = [selected_features[i] for i in sorted_idx[:15]]

print("Top 15 important features:")
for i, feature in enumerate(top_features):
    print(f"{i+1}. {feature}: {feature_importance[sorted_idx[i]]:.4f}")

# --- 11. SHAP Analysis ---
print("\n=== SHAP Values Analysis ===")
try:
    explainer = shap.TreeExplainer(model)
    shap_sample_size = min(1000, X_val_selected.shape[0])
    shap_sample_indices = np.random.choice(X_val_selected.shape[0], shap_sample_size, replace=False)
    X_val_sample = X_val_selected.iloc[shap_sample_indices]
    
    with parallel_backend('threading', n_jobs=n_jobs):
        shap_values = explainer.shap_values(X_val_sample)

    shap.summary_plot(shap_values, X_val_sample, plot_type="bar")
    shap.summary_plot(shap_values, X_val_sample)
    print("SHAP analysis completed successfully")
except Exception as e:
    print(f"SHAP analysis failed with error: {str(e)}")

# --- 12. ROC and PR Curves ---
print("\n=== Generating ROC and PR Curves ===")
fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
precision, recall, _ = precision_recall_curve(y_val, y_pred_proba)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(recall, precision, label=f'AP = {ap:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.tight_layout()
plt.show()

# --- 13. Threshold Optimization ---
print("\n=== Optimizing Classification Threshold ===")
thresholds = np.linspace(0.1, 0.9, 50)
f1_scores = [f1_score(y_val, (y_pred_proba >= t).astype(int)) for t in thresholds]

best_threshold_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_threshold_idx]
best_f1 = f1_scores[best_threshold_idx]

print(f"Best threshold: {best_threshold:.4f} (F1: {best_f1:.4f})")

plt.figure(figsize=(10, 6))
plt.plot(thresholds, f1_scores, marker='o')
plt.axvline(x=best_threshold, color='r', linestyle='--', label=f'Best Threshold = {best_threshold:.4f}')
plt.xlabel('Threshold')
plt.ylabel('F1 Score')
plt.title('F1 Score vs. Threshold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

optimized_predictions = (test_pred_proba >= best_threshold).astype(int)
print("\nOptimized Test Predictions (first 10 samples):")
print(pd.DataFrame({
    'ID': test_processed['id'][:10] if 'id' in test_processed.columns else range(10),
    'Optimized Predicted': optimized_predictions[:10],
    'Probability': test_pred_proba[:10]
}))

# --- 14. Save Model and Artifacts ---
print("\n=== Saving Model and Analysis Objects ===")
joblib.dump(model, os.path.join(output_path, 'xgboost_model.joblib'))
with open(os.path.join(output_path, 'selected_features.json'), 'w') as f:
    json.dump(selected_features, f)
with open(os.path.join(output_path, 'threshold.json'), 'w') as f:
    json.dump({'best_threshold': float(best_threshold), 'best_f1': float(best_f1)}, f)

model_metrics = {
    'roc_auc': float(auc),
    'average_precision': float(ap),
    'accuracy': float(accuracy),
    'f1_score': float(f1),
    'cv_scores': cv_scores.tolist(),
    'cv_mean': float(cv_scores.mean()),
    'cv_std': float(cv_scores.std()),
    'best_threshold': float(best_threshold),
    'best_f1': float(best_f1),
    'confusion_matrix': conf_matrix.tolist(),
    'feature_importance': {selected_features[i]: float(feature_importance[i]) for i in range(len(selected_features))}
}

with open(os.path.join(output_path, 'model_metrics.json'), 'w') as f:
    json.dump(model_metrics, f, indent=2)

print("\n=== Processing Complete ===")
print(f"Model and predictions saved to {output_path}")
print(f"Final model performance on validation set: ROC AUC = {auc:.4f}, F1 = {f1:.4f}")


# visualization

# --- 1. Setup Environment and Data Paths ---
base_path = "/kaggle/input/playground-series-s4e11/"
output_path = "/kaggle/working/"
preprocessed_path = os.path.join(output_path, "preprocessed")
os.makedirs(preprocessed_path, exist_ok=True)

print("="*80)
print("MENTAL HEALTH PREDICTION MODEL - RESULTS VISUALIZATION".center(80))
print("="*80)

# --- 2. Load the Required Files ---
print("\nLoading saved model data...")
try:
    # Load model and metrics
    model = joblib.load(os.path.join(output_path, 'xgboost_model.joblib'))
    
    with open(os.path.join(output_path, 'model_metrics.json'), 'r') as f:
        metrics = json.load(f)
    
    with open(os.path.join(output_path, 'selected_features.json'), 'r') as f:
        selected_features = json.load(f)
    
    with open(os.path.join(output_path, 'threshold.json'), 'r') as f:
        threshold_info = json.load(f)
    
    # Load submissions
    submission = pd.read_csv(os.path.join(output_path, 'submission.csv'))
    
    # Check if submission_optimized exists
    optimized_path = os.path.join(output_path, 'submission_optimized.csv')
    if os.path.exists(optimized_path):
        submission_optimized = pd.read_csv(optimized_path)
    else:
        print("WARNING: submission_optimized.csv not found. Using default submission.")
        submission_optimized = submission.copy()
    
    # Load original test data from the correct path
    original_test = pd.read_csv(os.path.join(base_path, "test.csv"))
    
    # Try to load the processed data
    try:
        test_processed = pd.read_csv(os.path.join(preprocessed_path, 'test_processed.csv'))
        X_processed = pd.read_csv(os.path.join(preprocessed_path, 'X_processed.csv'))
        y_train = pd.read_csv(os.path.join(preprocessed_path, 'y_train.csv')).iloc[:, 0]
        print("Processed data loaded successfully.")
    except Exception as e:
        print(f"Note: Processed data not found, but this is not critical: {str(e)}")
        test_processed = None
        X_processed = None
        y_train = None
    
    print("Main data loaded successfully!")
except Exception as e:
    print(f"Error loading data: {str(e)}")
    print("Please run the preprocessing and modeling scripts first.")
    # Don't use sys.exit() to avoid terminating the notebook
    raise

# --- 3. Display Model Performance Summary ---
print("\n" + "="*80)
print("MODEL PERFORMANCE SUMMARY".center(80))
print("="*80)
print(f"ROC AUC Score: {metrics['roc_auc']:.4f}")
print(f"Average Precision: {metrics['average_precision']:.4f}")
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 Score: {metrics['f1_score']:.4f}")
print(f"\nCross-Validation ROC AUC: {metrics['cv_mean']:.4f} (±{metrics['cv_std']:.4f})")
print(f"Best Threshold: {metrics['best_threshold']:.4f} (F1: {metrics['best_f1']:.4f})")

# --- 4. Display the Confusion Matrix ---
print("\n" + "="*80)
print("CONFUSION MATRIX".center(80))
print("="*80)
cm = np.array(metrics['confusion_matrix'])
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
print(f"True Negatives: {cm[0, 0]} | False Positives: {cm[0, 1]}")
print(f"False Negatives: {cm[1, 0]} | True Positives: {cm[1, 1]}")

# --- 5. Display Top Features by Importance ---
print("\n" + "="*80)
print("TOP FEATURES BY IMPORTANCE".center(80))
print("="*80)
feature_importance = {k: v for k, v in sorted(metrics['feature_importance'].items(),
                                             key=lambda item: item[1], reverse=True)}
top_n = 15
top_features = list(feature_importance.keys())[:top_n]
top_values = list(feature_importance.values())[:top_n]

plt.figure(figsize=(12, 8))
plt.barh(range(len(top_features)), top_values, align='center')
plt.yticks(range(len(top_features)), top_features)
plt.xlabel('Feature Importance')
plt.title('Top Features by Importance')
plt.tight_layout()
plt.show()

for i, (feature, importance) in enumerate(zip(top_features, top_values)):
    print(f"{i+1}. {feature}: {importance:.4f}")

# --- 6. Display SHAP plots (if available) ---
print("\n" + "="*80)
print("SHAP FEATURE IMPORTANCE".center(80))
print("="*80)
shap_files = ['shap_summary.png', 'shap_detailed.png']
any_shap_found = False

for shap_file in shap_files:
    try:
        plt.figure(figsize=(14, 10))
        img = plt.imread(os.path.join(output_path, shap_file))
        plt.imshow(img)
        plt.axis('off')
        plt.title(shap_file.replace('.png', '').replace('_', ' ').title())
        plt.show()
        any_shap_found = True
    except Exception:
        pass

if not any_shap_found:
    print("SHAP analysis not found as image files. This is expected if SHAP analysis was displayed but not saved.")

# --- 9. Display Sample Predictions ---
print("\n" + "="*80)
print("SAMPLE PREDICTIONS".center(80))
print("="*80)
target_column = [col for col in submission.columns if col != 'id'][0]
predictions_df = pd.DataFrame({
    'ID': submission['id'],
    'Default_Prediction': submission[target_column],
    'Optimized_Prediction': submission_optimized[target_column]
})

# Try to add original test features
additional_cols = []
if original_test is not None:
    for col in ['Age', 'Gender', 'CGPA', 'Career_Service']:
        if col in original_test.columns:
            additional_cols.append(col)
    
    if additional_cols:
        predictions_df = predictions_df.merge(
            original_test[['id'] + additional_cols], 
            left_on='ID', 
            right_on='id', 
            how='left'
        ).drop('id', axis=1)

sample_size = min(20, len(predictions_df))
samples = predictions_df.sample(sample_size, random_state=42)

print("\nSample of predictions (default vs. optimized):")
different_predictions = predictions_df[predictions_df['Default_Prediction'] != predictions_df['Optimized_Prediction']]
if len(different_predictions) > 0:
    different_samples = different_predictions.head(10)
    print(different_samples)
    print(f"\nTotal records with different predictions: {len(different_predictions)} ({(len(different_predictions)/len(predictions_df))*100:.2f}%)")
else:
    print("No differences between default and optimized predictions")
    print(samples)

# --- 10. Display Predicted Class Distribution ---
print("\n" + "="*80)
print("PREDICTED CLASS DISTRIBUTION".center(80))
print("="*80)
default_distribution = pd.Series(submission[target_column]).value_counts(normalize=True)
optimized_distribution = pd.Series(submission_optimized[target_column]).value_counts(normalize=True)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
default_distribution.plot(kind='bar', color=['lightblue', 'salmon'])
plt.title('Default Threshold Predictions')
plt.ylabel('Proportion')
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)

plt.subplot(1, 2, 2)
optimized_distribution.plot(kind='bar', color=['lightblue', 'salmon'])
plt.title(f'Optimized Threshold ({threshold_info["best_threshold"]:.2f}) Predictions')
plt.ylabel('Proportion')
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

print("Default threshold predictions:")
print(pd.Series(submission[target_column]).value_counts())
print(f"Class 0: {default_distribution.get(0, 0):.2%}, Class 1: {default_distribution.get(1, 0):.2%}")

print("\nOptimized threshold predictions:")
print(pd.Series(submission_optimized[target_column]).value_counts())
print(f"Class 0: {optimized_distribution.get(0, 0):.2%}, Class 1: {optimized_distribution.get(1, 0):.2%}")

# --- 11. Try to display preprocessing information ---
print("\n" + "="*80)
print("PREPROCESSING INFORMATION".center(80))
print("="*80)
try:
    with open(os.path.join(preprocessed_path, 'feature_info.json'), 'r') as f:
        preproc_info = json.load(f)

    print(f"Original features: {len(preproc_info['original_columns'])}")
    print(f"Processed features: {len(preproc_info['processed_columns'])}")
    print(f"Added features: {len(preproc_info['new_features'])}")

    if 'pca_components' in preproc_info:
        print(f"\nPCA Components added: {preproc_info['pca_components']}")
    
    if 'knn_imputation_used' in preproc_info:
        print(f"KNN Imputation used: {'Yes' if preproc_info['knn_imputation_used'] else 'No'}")

    if len(preproc_info['new_features']) > 0:
        print("\nTop 10 new engineered features:")
        for i, feature in enumerate(preproc_info['new_features'][:10]):
            print(f"{i+1}. {feature}")
except Exception as e:
    print(f"Preprocessing information not available: {str(e)}")

# --- 12. Conclusion ---
print("\n" + "="*80)
print("CONCLUSION".center(80))
print("="*80)
print(f"Model Performance: ROC AUC = {metrics['roc_auc']:.4f}, F1 = {metrics['f1_score']:.4f}")
print(f"Optimized with threshold = {threshold_info['best_threshold']:.4f}")
print(f"Files saved to {output_path}")
print("="*80)

#See first rows submission file

# Define the path to the submission file
submission_path = "/kaggle/working/submission.csv"
# Load the submission file
submission = pd.read_csv(submission_path)
# Display the first 30 rows
print(submission.head(30))

