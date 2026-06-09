import numpy as np 
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
import lightgbm as lgb, xgboost as xgb, catboost as cb
from gc import collect
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np



# Function to load all data
def get_feats(mode='train'):
    
    # Load quantitative metadata
    feats = pd.read_excel(f"/kaggle/input/widsdatathon2025/{mode}/{mode}_QUANTITATIVE_METADATA.xlsx")
    
    # Load categorical metadata with the correct filename depending on mode
    if mode == 'TRAIN':
        cate = pd.read_excel(f"/kaggle/input/widsdatathon2025/{mode}/{mode}_CATEGORICAL_METADATA.xlsx")
    else:
        cate = pd.read_excel(f"/kaggle/input/widsdatathon2025/{mode}/{mode}_CATEGORICAL.xlsx")
    
    # Merge categorical data
    feats = feats.merge(cate, on='participant_id', how='left')
    
    # Load functional connectome matrices
    func = pd.read_csv(f"/kaggle/input/widsdatathon2025/{mode}/{mode}_FUNCTIONAL_CONNECTOME_MATRICES.csv")
    feats = feats.merge(func, on='participant_id', how='left')
    
    # If training data, merge with solution file
    if mode == 'TRAIN':
        solution = pd.read_excel("/kaggle/input/widsdatathon2025/TRAIN/TRAINING_SOLUTIONS.xlsx")
        feats = feats.merge(solution, on='participant_id', how='left')
    
    return feats




# Load data
train = get_feats(mode='TRAIN')
test = get_feats(mode='TEST')

sub = pd.read_excel('/kaggle/input/widsdatathon2025/SAMPLE_SUBMISSION.xlsx')
y = pd.read_excel(f"/kaggle/input/widsdatathon2025/TRAIN/TRAINING_SOLUTIONS.xlsx")


# Set index
train.set_index('participant_id', inplace=True)
test.set_index('participant_id', inplace=True)

# Define targets and features
targets = ['ADHD_Outcome', 'Sex_F']
non_connectome_features = [col for col in train.columns if not (col.startswith('0throw') or col.startswith('1throw') or 
                                                             col.startswith('2throw') or col.startswith('3throw') or 
                                                             col.startswith('4throw') or col.startswith('5throw') or
                                                             col.startswith('6throw') or col.startswith('7throw') or
                                                             col.startswith('8throw') or col.startswith('9throw') or
                                                             col in targets)]

connectome_features = [col for col in train.columns if (col.startswith('0throw') or col.startswith('1throw') or 
                                                      col.startswith('2throw') or col.startswith('3throw') or 
                                                      col.startswith('4throw') or col.startswith('5throw') or
                                                      col.startswith('6throw') or col.startswith('7throw') or
                                                      col.startswith('8throw') or col.startswith('9throw'))]


train.shape


test.shape


print("Number of rows: ",train.shape[0])
print("Number of columns: ",train.shape[1])


print("Number of rows: ",test.shape[0])
print("Number of columns: ",test.shape[1])


print("\nDataset Overview:")
print(f"Training data: {train.shape[0]} participants, {train.shape[1]} features")
print(f"Test data: {test.shape[0]} participants, {test.shape[1]} features")
print(f"Number of non-connectome features: {len(non_connectome_features)}")
print(f"Number of brain connectivity features: {len(connectome_features)}")



train.describe()



train.head(10)


# Check for missing values in the training data
missing = train.isnull().sum()
missing_percent = 100 * missing / len(train)
missing_df = pd.DataFrame({
    'Missing Values': missing,
    'Percentage': missing_percent
})


# Display features with missing values
missing_features = missing_df[missing_df['Missing Values'] > 0].sort_values('Percentage', ascending=False)
print("\nFeatures with missing values in training data:")
missing_features



# Import the imputer
from sklearn.impute import SimpleImputer

# Get the list of columns that have missing values
columns_with_missing = missing_features.index.tolist()
print(f"Number of columns with missing values: {len(columns_with_missing)}")

# Create an imputer that will replace missing values with the mean of each column
imputer = SimpleImputer(strategy='mean')

# Apply the imputer only to the columns that have missing values
train[columns_with_missing] = imputer.fit_transform(train[columns_with_missing])

# Do the same for the test set if needed
test[columns_with_missing] = imputer.transform(test[columns_with_missing])

# Verify that missing values have been filled
missing_after = train[columns_with_missing].isnull().sum().sum()
print(f"Missing values after imputation: {missing_after}")
print("All missing values have been replaced with column means")


# Check for missing values in the testing data
missing = test.isnull().sum()
missing_percent = 100 * missing / len(test)
missing_df = pd.DataFrame({
    'Missing Values': missing,
    'Percentage': missing_percent
})


# Display features with missing values
missing_features = missing_df[missing_df['Missing Values'] > 0].sort_values('Percentage', ascending=False)
print("\nFeatures with missing values in training data:")
missing_features




# Categorical columns - we'll use the most common value to fill in missing data
categorical_columns = [
    'Barratt_Barratt_P1_Edu', 
    'Barratt_Barratt_P2_Edu',
    'Barratt_Barratt_P1_Occ',
    'Barratt_Barratt_P2_Occ',
    'PreInt_Demos_Fam_Child_Race'
]

# Numerical columns - we'll use the average value to fill in missing data
numerical_columns = [
    'SDQ_SDQ_Difficulties_Total',
    'SDQ_SDQ_Prosocial',
    'SDQ_SDQ_Peer_Problems',
    'SDQ_SDQ_Internalizing',
    'SDQ_SDQ_Hyperactivity',
    'SDQ_SDQ_Generating_Impact',
    'SDQ_SDQ_Emotional_Problems',
    'SDQ_SDQ_Externalizing',
    'SDQ_SDQ_Conduct_Problems',
    'APQ_P_APQ_P_PP',
    'APQ_P_APQ_P_PM',
    'APQ_P_APQ_P_OPD',
    'APQ_P_APQ_P_INV',
    'APQ_P_APQ_P_ID',
    'APQ_P_APQ_P_CP',
    'ColorVision_CV_Score',
    'EHQ_EHQ_Total'
]


mode_imputer = SimpleImputer(strategy='most_frequent')  # For categorical data
mean_imputer = SimpleImputer(strategy='mean')           # For numerical data


print("Filling in missing categorical data with most common values...")
mode_imputer.fit(train[categorical_columns])  
test[categorical_columns] = mode_imputer.transform(test[categorical_columns])


print("Filling in missing numerical data with average values...")
mean_imputer.fit(train[numerical_columns])  
test[numerical_columns] = mean_imputer.transform(test[numerical_columns])

print("Done! All missing values have been filled in.")


test.isnull().sum()



sdq_vars = [col for col in train.columns if col.startswith('SDQ')]

# 1. Distribution overview
fig, ax = plt.subplots(figsize=(12, 6))
train[sdq_vars].mean().sort_values().plot(kind='bar', ax=ax)
plt.title('Average SDQ Scores')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 2. Correlation with ADHD
plt.figure(figsize=(10, 8))
sdq_adhd_corr = train[sdq_vars + ['ADHD_Outcome']].corr()['ADHD_Outcome'].drop('ADHD_Outcome').sort_values()
sdq_adhd_corr.plot(kind='bar')
plt.title('Correlation between SDQ Scores and ADHD')
plt.tight_layout()
plt.show()

# 3. Box plots by ADHD diagnosis
plt.figure(figsize=(14, 10))
for i, var in enumerate(sdq_vars):
    plt.subplot(3, 3, i+1)
    sns.boxplot(x='ADHD_Outcome', y=var, data=train)
    plt.title(var.split('SDQ_SDQ_')[1])
plt.tight_layout()
plt.show()


# APQ Features EDA
apq_vars = [col for col in train.columns if 'APQ_P_APQ_P' in col]

# 1. Distribution of APQ scores
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for i, var in enumerate(apq_vars):
    sns.histplot(train[var], kde=True, ax=axes[i])
    var_name = var.replace('APQ_P_APQ_P_', '')
    axes[i].set_title(f'{var_name} Distribution')
    
# Add descriptive labels for distribution charts
labels = [
    "Most parents report minimal use of physical discipline",
    "Normal distribution of discipline inconsistency",
    "High levels of parental engagement reported",
    "Balanced distribution of alternative discipline strategies",
    "Bimodal distribution suggesting two distinct supervision groups",
    "Strong right-skew: frequent positive reinforcement"
]

for i, label in enumerate(labels):
    axes[i].set_xlabel(label, wrap=True)
    
plt.tight_layout()
plt.show()

# 2. Correlation between parenting measures and ADHD
corr_apq_adhd = train[apq_vars + ['ADHD_Outcome']].corr()['ADHD_Outcome'].drop('ADHD_Outcome')
plt.figure(figsize=(10, 6))
corr_plot = corr_apq_adhd.sort_values().plot(kind='bar')
plt.title('Correlation between Parenting Measures and ADHD')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Correlation with ADHD Diagnosis')
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.xlabel('Parenting Measures\n(Higher positive values suggest stronger relationship with ADHD)', wrap=True)
plt.tight_layout()
plt.show()

# 3. Compare distributions by ADHD diagnosis
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for i, var in enumerate(apq_vars):
    sns.boxplot(x='ADHD_Outcome', y=var, data=train, ax=axes[i])
    var_name = var.replace('APQ_P_APQ_P_', '')
    axes[i].set_title(f'{var_name} by ADHD Status')
    axes[i].set_xlabel('ADHD Diagnosis (1=Yes, 0=No)')
    
# Add descriptive labels for ADHD status charts
labels = [
    "CP: Physical discipline differences",
    "ID: Discipline consistency variations",
    "INV: Parental involvement by ADHD status",
    "OPD: Alternative discipline strategies",
    "PM: Monitoring practices across groups",
    "PP: Positive reinforcement patterns"
]

for i, label in enumerate(labels):
    axes[i].set_ylabel(label, rotation=90, ha='right')
    
plt.tight_layout()
plt.show()

# 4. Relationship between parenting styles and sex
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for i, var in enumerate(apq_vars):
    sns.boxplot(x='Sex_F', y=var, data=train, ax=axes[i])
    var_name = var.replace('APQ_P_APQ_P_', '')
    axes[i].set_title(f'{var_name} by Sex')
    axes[i].set_xlabel('Sex (1=Female, 0=Male)')
    
# Add descriptive labels for sex comparison charts
labels = [
    "CP: Physical discipline by gender",
    "ID: Discipline consistency across sexes",
    "INV: Parental involvement differences",
    "OPD: Alternative discipline strategies",
    "PM: Monitoring practices by sex",
    "PP: Positive reinforcement patterns"
]

for i, label in enumerate(labels):
    axes[i].set_ylabel(label, rotation=90, ha='right')
    
plt.tight_layout()
plt.show()

"""
Detailed Observations:
- Corporal Punishment (CP): Most parents report minimal use of physical discipline, with scores heavily concentrated at the low end.
- Inconsistent Discipline (ID): Parents show a normal distribution of inconsistency in rule enforcement, with most showing moderate levels.
- Involvement (INV): Parents typically report high levels of engagement with their children, with most scores in the upper range.
- Other Discipline Practices (OPD): Alternative discipline strategies show a balanced distribution with most parents using a moderate amount.
- Poor Monitoring (PM): The bimodal distribution suggests two distinct groups of parents regarding supervision practices.
- Positive Parenting (PP): The strong right-skew indicates that most parents report frequent use of praise and positive reinforcement.

Correlation Insights:
- Higher levels of parental engagement (spending time with children, participating in their activities, talking about their day, etc.) is associated with somewhat lower rates of ADHD diagnosis.
"""


# Clinical Measures EDA (Handedness and Color Vision)
clinical_vars = ['EHQ_EHQ_Total', 'ColorVision_CV_Score']

# 1. Distribution of clinical measures
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Handedness distribution
sns.histplot(train['EHQ_EHQ_Total'], kde=True, ax=axes[0])
axes[0].set_title('Distribution of Handedness Scores')
axes[0].set_xlabel('EHQ Score (-100 = Left, 100 = Right)')
axes[0].axvline(x=0, color='r', linestyle='--')

# Add vertical lines for handedness categories
axes[0].axvline(x=-28, color='g', linestyle='--', alpha=0.6)
axes[0].axvline(x=48, color='g', linestyle='--', alpha=0.6)
# Add text for handedness regions
axes[0].text(-75, axes[0].get_ylim()[1]*0.9, 'Left-handed', ha='center')
axes[0].text(0, axes[0].get_ylim()[1]*0.9, 'Ambidextrous', ha='center')
axes[0].text(75, axes[0].get_ylim()[1]*0.9, 'Right-handed', ha='center')

# Color vision distribution
sns.countplot(x=train['ColorVision_CV_Score'], ax=axes[1])
axes[1].set_title('Distribution of Color Vision Test Scores')
axes[1].set_xlabel('Color Vision Score')

plt.tight_layout()
plt.show()

# 2. Relationship with ADHD and Sex
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Handedness by ADHD
sns.boxplot(x='ADHD_Outcome', y='EHQ_EHQ_Total', data=train, ax=axes[0, 0])
axes[0, 0].set_title('Handedness by ADHD Status')
axes[0, 0].set_xlabel('ADHD Diagnosis (1=Yes, 0=No)')
axes[0, 0].set_ylabel('EHQ Score (-100 = Left, 100 = Right)')

# Handedness by Sex
sns.boxplot(x='Sex_F', y='EHQ_EHQ_Total', data=train, ax=axes[0, 1])
axes[0, 1].set_title('Handedness by Sex')
axes[0, 1].set_xlabel('Sex (1=Female, 0=Male)')
axes[0, 1].set_ylabel('EHQ Score (-100 = Left, 100 = Right)')

# Color vision by ADHD
sns.countplot(x='ColorVision_CV_Score', hue='ADHD_Outcome', data=train, ax=axes[1, 0])
axes[1, 0].set_title('Color Vision Scores by ADHD Status')
axes[1, 0].set_xlabel('Color Vision Score')
axes[1, 0].legend(title='ADHD', labels=['No', 'Yes'])

# Color vision by Sex
sns.countplot(x='ColorVision_CV_Score', hue='Sex_F', data=train, ax=axes[1, 1])
axes[1, 1].set_title('Color Vision Scores by Sex')
axes[1, 1].set_xlabel('Color Vision Score')
axes[1, 1].legend(title='Sex', labels=['Male', 'Female'])

plt.tight_layout()
plt.show()

# 3. Statistical tests for differences
print("Statistical Tests for Clinical Measures:")

# Handedness by ADHD
from scipy.stats import ttest_ind
adhd_positive = train.loc[train['ADHD_Outcome'] == 1, 'EHQ_EHQ_Total']
adhd_negative = train.loc[train['ADHD_Outcome'] == 0, 'EHQ_EHQ_Total']
t_stat, p_val = ttest_ind(adhd_positive, adhd_negative)
print(f"Handedness difference by ADHD: t-stat={t_stat:.3f}, p-value={p_val:.3f}")

# Handedness by Sex
male = train.loc[train['Sex_F'] == 0, 'EHQ_EHQ_Total']
female = train.loc[train['Sex_F'] == 1, 'EHQ_EHQ_Total']
t_stat, p_val = ttest_ind(male, female)
print(f"Handedness difference by Sex: t-stat={t_stat:.3f}, p-value={p_val:.3f}")

# Color vision by ADHD
from scipy.stats import chi2_contingency
color_adhd_contingency = pd.crosstab(train['ColorVision_CV_Score'], train['ADHD_Outcome'])
chi2, p_val, dof, expected = chi2_contingency(color_adhd_contingency)
print(f"Color vision association with ADHD: chi2={chi2:.3f}, p-value={p_val:.3f}")

# Color vision by Sex
color_sex_contingency = pd.crosstab(train['ColorVision_CV_Score'], train['Sex_F'])
chi2, p_val, dof, expected = chi2_contingency(color_sex_contingency)
print(f"Color vision association with Sex: chi2={chi2:.3f}, p-value={p_val:.3f}")


import matplotlib.pyplot as plt
import seaborn as sns

# Study Site Distribution
plt.figure(figsize=(10, 6))
site_labels = ['Staten Island', 'MRV', 'Midtown', 'Harlem']
site_counts = train['Basic_Demos_Study_Site'].value_counts().sort_index()
plt.bar(site_labels, site_counts)
plt.title('Distribution of Study Sites')
plt.xlabel('Study Site')
plt.ylabel('Number of Participants')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Ethnicity Distribution
plt.figure(figsize=(10, 6))
ethnicity_counts = train['PreInt_Demos_Fam_Child_Ethnicity'].value_counts().sort_index()
ethnicity_labels = {
    0: 'Not Hispanic/Latino', 
    1: 'Hispanic/Latino', 
    2: 'Decline to Specify', 
    3: 'Unknown'
}

# Use the actual index values and corresponding labels
plt.bar(
    [ethnicity_labels.get(x, str(x)) for x in ethnicity_counts.index], 
    ethnicity_counts
)
plt.title('Distribution of Child Ethnicity')
plt.xlabel('Ethnicity')
plt.ylabel('Number of Participants')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Race Distribution
# Race Distribution
plt.figure(figsize=(12, 6))
race_labels_dict = {
    0: 'White/Caucasian', 
    1: 'Black/African American', 
    2: 'Hispanic', 
    3: 'Asian', 
    4: 'Indian', 
    5: 'Native American Indian', 
    6: 'American Indian/Alaskan Native', 
    7: 'Native Hawaiian/Pacific Islander', 
    8: 'Two or More Races', 
    9: 'Other Race', 
    10: 'Unknown', 
    11: 'Choose Not to Specify'
}

race_counts = train['PreInt_Demos_Fam_Child_Race'].value_counts().sort_index()

# Use actual index values and corresponding labels
plt.bar(
    [race_labels_dict.get(x, str(x)) for x in race_counts.index], 
    race_counts
)
plt.title('Distribution of Child Race')
plt.xlabel('Race')
plt.ylabel('Number of Participants')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Parent 1 Education Distribution
# Parent 1 Education Distribution
plt.figure(figsize=(10, 6))
edu_labels_dict = {
    3: 'Less than 7th grade',
    6: 'Junior High',
    9: 'Partial High School',
    12: 'High School Graduate',
    15: 'Partial College',
    18: 'College Education',
    21: 'Graduate Degree'
}

edu_counts = train['Barratt_Barratt_P1_Edu'].value_counts().sort_index()

# Use actual index values and corresponding labels
plt.bar(
    [edu_labels_dict.get(x, str(x)) for x in edu_counts.index], 
    edu_counts
)
plt.title('Distribution of Parent 1 Education')
plt.xlabel('Education Level')
plt.ylabel('Number of Participants')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Optional: Print out the exact counts for reference
print("Study Site Counts:")
print(site_counts)
print("\nEthnicity Counts:")
print(ethnicity_counts)
print("\nRace Counts:")
print(race_counts)
print("\nParent 1 Education Counts:")
print(edu_counts)


print("=== BRAIN CONNECTIVITY ANALYSIS ===")

# 1. Identify brain connectivity features
connectome_cols = [col for col in train.columns if (col.startswith('0throw') or 
                                                    col.startswith('1throw') or 
                                                    col.startswith('2throw'))]
print(f"Number of brain connectivity features: {len(connectome_cols)}")
print(f"First few connectivity features: {connectome_cols[:5]}")

# 2. Distribution of connectivity values (sample)
plt.figure(figsize=(10, 6))
# Take a random sample to avoid memory issues
sample_size = min(1000, len(connectome_cols))
np.random.seed(42)
sample_cols = np.random.choice(connectome_cols, sample_size)
all_values = train[sample_cols].values.flatten()
all_values = all_values[~np.isnan(all_values)]  # Remove NaN values if any

plt.hist(all_values, bins=50)
plt.title('Distribution of Brain Connectivity Values (Sample)')
plt.xlabel('Correlation Value')
plt.ylabel('Frequency')
plt.grid(alpha=0.3)
plt.show()

# 3. Simple visualization of connectivity difference between ADHD and non-ADHD
# Take just 20 random features for a simple comparison
sample_size = 20
np.random.seed(42)
tiny_sample = np.random.choice(connectome_cols, sample_size)

# Calculate means by group
adhd_means = train[train['ADHD_Outcome']==1][tiny_sample].mean()
non_adhd_means = train[train['ADHD_Outcome']==0][tiny_sample].mean()
diff = adhd_means - non_adhd_means

# Plot differences
plt.figure(figsize=(12, 6))
diff.sort_values().plot(kind='bar')
plt.title('Difference in Brain Connectivity: ADHD vs Non-ADHD (20 Random Connections)')
plt.xlabel('Brain Connection')
plt.ylabel('Difference (ADHD - Non-ADHD)')
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.xticks([])  # Hide feature names to avoid clutter
plt.tight_layout()
plt.show()




print("=== UNDERSTANDING BRAIN CONNECTIVITY DATA ===")

# 1. First, let's understand what our connectivity data looks like
connectome_cols = [col for col in train.columns if (col.startswith('0throw') or 
                                                  col.startswith('1throw') or 
                                                  col.startswith('2throw'))]
print(f"Number of brain connectivity features: {len(connectome_cols)}")
print(f"First few connectivity features: {connectome_cols[:5]}")

print("\nWhat do these features mean?")
print("- Each feature (like '0throw_1stcolumn') represents the correlation between two brain regions")
print("- Values range from -1 to +1")
print("- Positive values: regions activate together")
print("- Negative values: when one region activates, the other tends to deactivate")
print("- Values close to zero: little relationship between regions")


# 3. Let's look at some sample connectivity values first
print("\n=== SAMPLE BRAIN CONNECTIVITY VALUES ===")
# Show a few sample values from the first subject
sample_subject = train.iloc[0]
print("For the first subject in our dataset:")
for i, col in enumerate(connectome_cols[:5]):
    print(f"{col}: {sample_subject[col]:.4f}")

# 4. Distribution of connectivity values
plt.figure(figsize=(10, 6))
# Take a random sample to avoid memory issues
sample_size = min(1000, len(connectome_cols))
np.random.seed(42)
sample_cols = np.random.choice(connectome_cols, sample_size)
all_values = train[sample_cols].values.flatten()
all_values = all_values[~np.isnan(all_values)]  # Remove NaN values if any

plt.hist(all_values, bins=50)
plt.title('Distribution of Brain Connectivity Values (Sample)')
plt.xlabel('Correlation Value')
plt.ylabel('Frequency')
plt.grid(alpha=0.3)
plt.show()

print("\nThe histogram shows us how brain connectivity values are distributed.")
print("Most values are centered around zero, with fewer strong positive or negative correlations.")

# 5. Simple comparison between ADHD and non-ADHD
print("\n=== COMPARING BRAIN CONNECTIVITY: ADHD VS NON-ADHD ===")
# Take just 10 random features for a simple comparison
sample_size = 10
np.random.seed(42)
tiny_sample = np.random.choice(connectome_cols, sample_size)

# Calculate means by group
adhd_means = train[train['ADHD_Outcome']==1][tiny_sample].mean()
non_adhd_means = train[train['ADHD_Outcome']==0][tiny_sample].mean()
diff = adhd_means - non_adhd_means

# Create a comparative table
comparison = pd.DataFrame({
    'ADHD': adhd_means,
    'Non-ADHD': non_adhd_means,
    'Difference': diff
})
print(comparison.round(4))

# Plot differences
plt.figure(figsize=(12, 6))
diff.plot(kind='bar')
plt.title('Difference in Brain Connectivity: ADHD vs Non-ADHD (10 Random Connections)')
plt.xlabel('Brain Connection')
plt.ylabel('Difference (ADHD - Non-ADHD)')
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nThis bar chart shows differences in brain connectivity between ADHD and non-ADHD groups.")
print("Bars above zero: connections stronger in ADHD subjects")
print("Bars below zero: connections stronger in non-ADHD subjects")
print("Even with just 10 random connections, we can see differences between the groups!")



# 4. Simple PCA on a subset of connectivity features
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Take 500 random features for PCA
sample_size = min(500, len(connectome_cols)) 
np.random.seed(42)
pca_sample = np.random.choice(connectome_cols, sample_size)

# Standardize the data
X = train[pca_sample].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create a DataFrame with PCA results
pca_df = pd.DataFrame({
    'PC1': X_pca[:, 0],
    'PC2': X_pca[:, 1],
    'ADHD': train['ADHD_Outcome'].values,
    'Sex': train['Sex_F'].values
})

# Plot PCA results
plt.figure(figsize=(12, 5))

# By ADHD
plt.subplot(1, 2, 1)
sns.scatterplot(x='PC1', y='PC2', hue='ADHD', data=pca_df, palette=['blue', 'red'])
plt.title('PCA of Brain Connectivity by ADHD Status')

# By Sex
plt.subplot(1, 2, 2)
sns.scatterplot(x='PC1', y='PC2', hue='Sex', data=pca_df, palette=['green', 'purple'])
plt.title('PCA of Brain Connectivity by Sex')

plt.tight_layout()
plt.show()

# Print explained variance
print(f"Variance explained by PC1: {pca.explained_variance_ratio_[0]*100:.2f}%")
print(f"Variance explained by PC2: {pca.explained_variance_ratio_[1]*100:.2f}%")


print(train.shape, test.shape, y.shape)
print(train.isnull().sum().sum())
print(test.isnull().sum().sum())
print(train.dtypes.value_counts())
print(y.head())
print(y.value_counts())


train_cols = set(train.columns)
test_cols = set(test.columns)

missing_in_test = train_cols - test_cols
missing_in_train = test_cols - train_cols

print("Missing in test:", missing_in_test)
print("Missing in train:", missing_in_train)



##removing columns from train that is not in test
y_ADHD = train['ADHD_Outcome']
y_Sex = train['Sex_F']

train.drop(columns=['ADHD_Outcome', 'Sex_F'], inplace=True)



print(train.shape, test.shape, y_ADHD.shape, y_Sex.shape)



print(train.shape, test.shape, y_ADHD.shape, y_Sex.shape)


print("Each train datasets in each fold",1213/10)
print("Total train dataset in each fold",(1213/10)*9)


from sklearn.model_selection import KFold

kf = KFold(n_splits=10, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
    X_train, X_val = train.iloc[train_idx], train.iloc[val_idx]
    y_train, y_val = y_ADHD.iloc[train_idx], y_ADHD.iloc[val_idx]
    print(f'Fold {fold + 1}')
    print('X_train:', X_train.shape, 'X_val:', X_val.shape)
    print('y_train:', y_train.value_counts(normalize=True))
    print('y_val:', y_val.value_counts(normalize=True))
    print('-------------------------------------------\n')



from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np

kf = KFold(n_splits=10, shuffle=True, random_state=42)

plt.figure(figsize=(10, 6))

for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
    fold_assignments = np.full(len(train), np.nan)
    fold_assignments[train_idx] = 0
    fold_assignments[val_idx] = 1
    
    plt.scatter(range(len(fold_assignments)), 
                [fold]*len(fold_assignments),
                c=fold_assignments, cmap='coolwarm', marker='|', s=200)

plt.yticks(range(10), [f'Fold {i+1}' for i in range(10)])
plt.xlabel('Sample Index')
plt.title('10-Fold Cross-Validation Splits (Non-stratified)')
plt.show()



from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(train, y_ADHD)):
    y_train_fold = y_ADHD.iloc[train_idx]
    y_val_fold = y_ADHD.iloc[val_idx]

    print(f"Stratified Fold {fold+1} - Training Class Distribution:")
    print(y_train_fold.value_counts(normalize=True))
    print(f"Stratified Fold {fold+1} - Validation Class Distribution:")
    print(y_val_fold.value_counts(normalize=True))
    print("-" * 40)



from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

plt.figure(figsize=(10, 6))

for fold, (train_idx, val_idx) in enumerate(skf.split(train, y_ADHD)):
    fold_labels = np.full(len(y_ADHD), np.nan)
    fold_labels[train_idx] = 0  # Training data
    fold_labels[val_idx] = 1  # Validation data

    plt.scatter(range(len(fold_labels)), [fold]*len(fold_labels),
                c=fold_labels, cmap='coolwarm', marker='|', s=200)

plt.xlabel("Sample Index")
plt.ylabel("Folds")
plt.title("Stratified K-Fold Cross-Validation Splits (Balanced Classes)")
plt.show()




# Define scoring metrics
scoring = {
    "Accuracy": "accuracy",
    "Precision": make_scorer(precision_score),
    "Recall": make_scorer(recall_score),
    "F1-Score": make_scorer(f1_score),
    "ROC-AUC": make_scorer(roc_auc_score)
}

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Gradient Boosting": GradientBoostingClassifier(),
    # "SVM": SVC(probability=True),
    # "KNN": KNeighborsClassifier(),
    # "Naive Bayes": GaussianNB()
}

# Apply cross-validation for all models
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\nğŸ“Œ Model: {name}")
    for metric_name, metric in scoring.items():
        scores = cross_val_score(model, train, y_ADHD, cv=skf, scoring=metric, n_jobs=-1)
        print(f"{metric_name}: {np.mean(scores):.4f}")



best_model = LogisticRegression(max_iter=1000)
best_model.fit(train, y_ADHD)  
y_test_pred_adhd = best_model.predict(test)



y_test_pred_sex = best_model.predict(test)



test.index[:5]



submission = pd.DataFrame({
    "participant_id": test.index,  
    "ADHD_Outcome": y_test_pred_adhd,
    "Sex_F": y_test_pred_sex
})


submission.to_csv("submission.csv", index=False)



import os
print(os.listdir("/kaggle/working/"))  # Lists files in the working directory





