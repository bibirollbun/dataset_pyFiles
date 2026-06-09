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


# Get an idea of what it looks like 
test_categorical = "/kaggle/input/widsdatathon2025/TEST/TEST_CATEGORICAL.xlsx"
test_categorical_df = pd.read_excel(test_categorical)

test_categorical_df.head()


# Get an idea of what it looks like 
test_quantitative = "/kaggle/input/widsdatathon2025/TEST/TEST_QUANTITATIVE_METADATA.xlsx"
test_quantitative_df = pd.read_excel(test_quantitative)

test_quantitative_df.head()


# Get an idea of what it looks like 
test_functional = "/kaggle/input/widsdatathon2025/TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv"
test_functional_df = pd.read_csv(test_functional)

test_functional_df.head()


# Get an idea of what submission file should look like 
submission_sample = "/kaggle/input/widsdatathon2025/SAMPLE_SUBMISSION.xlsx"
submission_sample_df = pd.read_excel(submission_sample)

print("This is a sample submission file, providing an example of the expected format for your final submission: \n")
print(submission_sample_df.head())


# Learn the dictionary
data_dict_path = "/kaggle/input/widsdatathon2025/Data Dictionary.xlsx"
data_dict = pd.ExcelFile(data_dict_path)

# Display Tabs / Sheet Names
print("Tabs:", data_dict.sheet_names) 

for tab in data_dict.sheet_names: 
    print(f"\nTab: {tab}")
    df = data_dict.parse(tab)

    # Display the entire sheet
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False) # Prevent line wrapping 
    display(df)


import matplotlib.pyplot as plt

# File paths
train_categorical_path = "/kaggle/input/widsdatathon2025/TRAIN/TRAIN_CATEGORICAL_METADATA.xlsx"
test_categorical_path = "/kaggle/input/widsdatathon2025/TEST/TEST_CATEGORICAL.xlsx"
training_solution_path = "/kaggle/input/widsdatathon2025/TRAIN/TRAINING_SOLUTIONS.xlsx"  

# Load the categorical data
train_categorical = pd.read_excel(train_categorical_path)
test_categorical = pd.read_excel(test_categorical_path)
train_df = pd.read_excel(training_solution_path)

# Combine both datasets
categorical_data = pd.concat([train_categorical, test_categorical], ignore_index=True)

# Define mappings for ethnicity and race
ethnicity_mapping = {0: "Not Hispanic or Latino", 1: "Hispanic or Latino"}
race_mapping = {
    0: "White/Caucasian",
    1: "Black/African American",
    2: "Asian",
    3: "American Indian/Alaska Native",
    4: "Native Hawaiian/Other Pacific Islander",
    5: "More than one race"
}

# Apply mappings
categorical_data["Ethnicity_Label"] = categorical_data["PreInt_Demos_Fam_Child_Ethnicity"].map(ethnicity_mapping)
categorical_data["Race_Label"] = categorical_data["PreInt_Demos_Fam_Child_Race"].map(race_mapping)

# Plot ethnicity distribution
plt.figure(figsize=(8, 5))
categorical_data["Ethnicity_Label"].value_counts().plot(kind="bar", color=["blue", "orange"])
plt.title("Ethnicity Distribution")
plt.xlabel("Ethnicity")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# Plot race distribution
plt.figure(figsize=(10, 5))
categorical_data["Race_Label"].value_counts().plot(kind="bar", color=["red", "green", "blue", "purple", "brown", "cyan"])
plt.title("Race Distribution")
plt.xlabel("Race")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# Plot gender distribution
plt.figure(figsize=(10, 5))
train_df["Sex_F"].value_counts().plot(kind="bar", color=["blue", "purple"])
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()


# File paths
train_quantitative_path = "/kaggle/input/widsdatathon2025/TRAIN/TRAIN_QUANTITATIVE_METADATA.xlsx"
test_quantitative_path = "/kaggle/input/widsdatathon2025/TEST/TEST_QUANTITATIVE_METADATA.xlsx"

# Load the categorical data
train_quantitative = pd.read_excel(train_quantitative_path)
test_quantitative = pd.read_excel(test_quantitative_path)

# Combine both datasets
quantitative_data = pd.concat([train_quantitative, test_quantitative], ignore_index=True)
quantitative_data.head()




# Trim spaces and clean column names
quantitative_data.columns = quantitative_data.columns.str.strip()

# Verify available column names
print(quantitative_data.columns.tolist())  # Print column names to check for discrepancies



import seaborn as sns

# Select relevant columns
quantitative_data = quantitative_data[["SDQ_SDQ_Hyperactivity", "MRI_Track_Age_at_Scan"]]

# Rename for readability
quantitative_data.rename(columns={"SDQ_SDQ_Hyperactivity": "Hyperactivity_Scale", 
                         "MRI_Track_Age_at_Scan": "Age_at_Scan"}, inplace=True)

# Plot 1: Histogram of Hyperactivity Scale
plt.figure(figsize=(8, 5))
sns.histplot(quantitative_data["Hyperactivity_Scale"], bins=10, kde=True, color="blue")
plt.title("Distribution of Hyperactivity Scores")
plt.xlabel("Hyperactivity Score")
plt.ylabel("Frequency")
plt.show()

# Plot 2: Histogram of Age at MRI Scan
plt.figure(figsize=(8, 5))
sns.histplot(quantitative_data["Age_at_Scan"], bins=10, kde=True, color="green")
plt.title("Distribution of Age at MRI Scan")
plt.xlabel("Age at Scan (Years)")
plt.ylabel("Frequency")
plt.show()



# File paths
train_quantitative_path = "/kaggle/input/widsdatathon2025/TRAIN/TRAIN_QUANTITATIVE_METADATA.xlsx"
test_quantitative_path = "/kaggle/input/widsdatathon2025/TEST/TEST_QUANTITATIVE_METADATA.xlsx"

# Load the categorical data
train_quantitative = pd.read_excel(train_quantitative_path)
test_quantitative = pd.read_excel(test_quantitative_path)

# Combine both datasets
quantitative_data = pd.concat([train_quantitative, test_quantitative], ignore_index=True)

# Load datasets
training_solution_path = "/kaggle/input/widsdatathon2025/TRAIN/TRAINING_SOLUTIONS.xlsx"  

train_df = pd.read_excel(training_solution_path)

# Clean column names (remove spaces)
train_df.columns = train_df.columns.str.strip()
quantitative_data.columns = quantitative_data.columns.str.strip()

# Identify relevant columns
participant_id_col = "participant_id" 
adhd_col = "ADHD_Outcome"  # Update with actual column name in training solutions
hyperactivity_col = "SDQ_SDQ_Hyperactivity"  # Ensure this matches the actual column name

# Merge dataframes on Participant ID
merged_df = pd.merge(train_df[[participant_id_col, adhd_col]], 
                      quantitative_data[[participant_id_col, hyperactivity_col]], 
                      on=participant_id_col, how="inner")

# Drop missing values (if any)
merged_df.dropna(inplace=True)

# Compute correlation
correlation_value = merged_df[[adhd_col, hyperactivity_col]].corr().iloc[0, 1]
print(f"Correlation between ADHD Outcome and Hyperactivity: {correlation_value:.3f}")

# Box plot: ADHD Outcome categories vs. Hyperactivity Scores
plt.figure(figsize=(8, 5))
sns.boxplot(x=merged_df[adhd_col], y=merged_df[hyperactivity_col], palette="coolwarm")
plt.title("Hyperactivity Scores by ADHD Outcome")
plt.xlabel("ADHD Outcome")
plt.ylabel("Hyperactivity Score")
plt.grid(True)
plt.show()



# File paths
train_quantitative_path = "/kaggle/input/widsdatathon2025/TRAIN/TRAIN_QUANTITATIVE_METADATA.xlsx"
test_quantitative_path = "/kaggle/input/widsdatathon2025/TEST/TEST_QUANTITATIVE_METADATA.xlsx"

# Load the categorical data
train_quantitative = pd.read_excel(train_quantitative_path)
test_quantitative = pd.read_excel(test_quantitative_path)

# Combine both datasets
quantitative_data = pd.concat([train_quantitative, test_quantitative], ignore_index=True)

# Load datasets
training_solution_path = "/kaggle/input/widsdatathon2025/TRAIN/TRAINING_SOLUTIONS.xlsx"  # Update with actual file path

train_df = pd.read_excel(training_solution_path)

# Clean column names (remove spaces)
train_df.columns = train_df.columns.str.strip()
quantitative_data.columns = quantitative_data.columns.str.strip()

# Columns of interest
participant_id_col = "participant_id"
adhd_col = "ADHD_Outcome"


# Merge datasets on Participant ID
merged_df = pd.merge(train_df[[participant_id_col, adhd_col]], quantitative_data, on=participant_id_col, how="inner")

# Drop non-numeric columns (except ADHD Outcome)
numeric_cols = merged_df.select_dtypes(include=["number"]).columns
filtered_df = merged_df[numeric_cols]

# Compute correlation with ADHD Outcome
correlation_matrix = filtered_df.corr()[adhd_col].drop(adhd_col)  # Exclude self-correlation

# Get top 10 quantitative features most correlated with ADHD Outcome
top_10_features = correlation_matrix.abs().nlargest(10)

# Print top features
print("Top 10 Quantitative Features Correlated with ADHD Outcome:")
print(top_10_features)

# Visualization: Heatmap for Top 10 Features
plt.figure(figsize=(8, 6))
sns.heatmap(filtered_df[top_10_features.index].corr(), annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap - Top 10 Features Related to ADHD Outcome")
plt.show()


# File paths for Connectome data
train_connectome_path = "/kaggle/input/widsdatathon2025/TRAIN/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES.csv"
test_connectome_path = "/kaggle/input/widsdatathon2025/TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv"

# Load the connectome datasets
train_connectome = pd.read_csv(train_connectome_path)
test_connectome = pd.read_csv(test_connectome_path)

# Combine Train & Test connectome data
connectome_data = pd.concat([train_connectome, test_connectome], ignore_index=True)

# Load the Training Solutions dataset
training_solution_path = "/kaggle/input/widsdatathon2025/TRAIN/TRAINING_SOLUTIONS.xlsx"
train_df = pd.read_excel(training_solution_path)

# Clean column names (remove spaces)
train_df.columns = train_df.columns.str.strip()
connectome_data.columns = connectome_data.columns.str.strip()

# Columns of interest
participant_id_col = "participant_id"
adhd_col = "ADHD_Outcome"

# Merge ADHD Outcome with Connectome data
merged_df = pd.merge(train_df[[participant_id_col, adhd_col]], connectome_data, on=participant_id_col, how="inner")

# Drop non-numeric columns (except ADHD Outcome)
numeric_cols = merged_df.select_dtypes(include=["number"]).columns
filtered_df = merged_df[numeric_cols]

# Compute correlation with ADHD Outcome
correlation_matrix = filtered_df.corr()[adhd_col].drop(adhd_col)  # Exclude self-correlation

# Get top 50 features most correlated with ADHD Outcome
top_50_features = correlation_matrix.abs().nlargest(50)

# Print top features
print("Top 50 Connectome Features Correlated with ADHD Outcome:")
print(top_50_features)

# Visualization: Heatmap for Top 50 Features
plt.figure(figsize=(12, 8))
sns.heatmap(filtered_df[top_50_features.index].corr(), annot=False, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap - Top 50 Connectome Features Related to ADHD Outcome")
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# File paths for Connectome data
train_connectome_path = "/kaggle/input/widsdatathon2025/TRAIN/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES.csv"
test_connectome_path = "/kaggle/input/widsdatathon2025/TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv"

# Load the connectome datasets
train_connectome = pd.read_csv(train_connectome_path)
test_connectome = pd.read_csv(test_connectome_path)

# Combine Train & Test connectome data
connectome_data = pd.concat([train_connectome, test_connectome], ignore_index=True)

# Load the Training Solutions dataset
training_solution_path = "/kaggle/input/widsdatathon2025/TRAIN/TRAINING_SOLUTIONS.xlsx"
train_df = pd.read_excel(training_solution_path)

# Clean column names (remove spaces)
train_df.columns = train_df.columns.str.strip()
connectome_data.columns = connectome_data.columns.str.strip()

# Columns of interest
participant_id_col = "participant_id"
sex_f_col = "Sex_F"  # Ensure this column name matches exactly

# Merge Sex_F with Connectome data
merged_df = pd.merge(train_df[[participant_id_col, sex_f_col]], connectome_data, on=participant_id_col, how="inner")

# Drop non-numeric columns (except Sex_F)
numeric_cols = merged_df.select_dtypes(include=["number"]).columns
filtered_df = merged_df[numeric_cols]

# Compute correlation with Sex_F
correlation_matrix = filtered_df.corr()[sex_f_col].drop(sex_f_col)  # Exclude self-correlation

# Get top 50 features most correlated with Sex_F
top_50_features = correlation_matrix.abs().nlargest(50)

# Print top features
print("Top 50 Connectome Features Correlated with Sex_F:")
print(top_50_features)

# Visualization: Heatmap for Top 50 Features
plt.figure(figsize=(12, 8))
sns.heatmap(filtered_df[top_50_features.index].corr(), annot=False, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap - Top 50 Connectome Features Related to Sex_F")
plt.show()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# File paths for data
train_connectome_path = "/kaggle/input/widsdatathon2025/TRAIN/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES.csv"
test_connectome_path = "/kaggle/input/widsdatathon2025/TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv"
training_solution_path = "/kaggle/input/widsdatathon2025/TRAIN/TRAINING_SOLUTIONS.xlsx"

# Load datasets
train_connectome = pd.read_csv(train_connectome_path)
test_connectome = pd.read_csv(test_connectome_path)
train_df = pd.read_excel(training_solution_path)

# Combine Train & Test connectome data
connectome_data = pd.concat([train_connectome, test_connectome], ignore_index=True)

# Clean column names
train_df.columns = train_df.columns.str.strip()
connectome_data.columns = connectome_data.columns.str.strip()

# Define Participant ID (for merging)
participant_id_col = "participant_id"
sex_f_col = "Sex_F"  # Target variable

# Merge Connectome Data with Training Solutions
merged_df = pd.merge(train_df[[participant_id_col, sex_f_col]], connectome_data, on=participant_id_col, how="inner")

# Select the top most correlated features (update based on previous correlation results)
top_features = [
    "164throw_189thcolumn", "164throw_173thcolumn", "158throw_191thcolumn", "28throw_188thcolumn",
    "131throw_198thcolumn", "133throw_182thcolumn", "183throw_189thcolumn", "114throw_121thcolumn",
    "172throw_188thcolumn", "180throw_182thcolumn", "71throw_73thcolumn", "160throw_190thcolumn",
    "183throw_197thcolumn", "73throw_133thcolumn", "89throw_91thcolumn"
]  

# Prepare features (X) - Only using Connectome data, no Sex_F
X = merged_df[top_features]  # Features
y = merged_df[sex_f_col]  # Target (0 = Male, 1 = Female)

# Drop Sex_F from dataset to avoid leakage
X = X.drop(columns=[sex_f_col], errors="ignore")

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the data (important for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])

print(f"Model Accuracy: {accuracy:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature Importance (Weights)
feature_importance = pd.DataFrame({"Feature": top_features, "Weight": model.coef_[0]})
feature_importance = feature_importance.sort_values(by="Weight", ascending=False)

# Visualization: Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance["Weight"], y=feature_importance["Feature"], palette="coolwarm")
plt.title("Feature Importance (Logistic Regression) - Predicting Sex_F")
plt.xlabel("Weight")
plt.ylabel("Connectome Feature")
plt.show()



# File paths for data
train_connectome_path = "/kaggle/input/widsdatathon2025/TRAIN/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES.csv"
test_connectome_path = "/kaggle/input/widsdatathon2025/TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv"
training_solution_path = "/kaggle/input/widsdatathon2025/TRAIN/TRAINING_SOLUTIONS.xlsx"

# Load datasets
train_connectome = pd.read_csv(train_connectome_path)
test_connectome = pd.read_csv(test_connectome_path)
train_df = pd.read_excel(training_solution_path)

# Combine Train & Test connectome data
connectome_data = pd.concat([train_connectome, test_connectome], ignore_index=True)

# Clean column names
train_df.columns = train_df.columns.str.strip()
connectome_data.columns = connectome_data.columns.str.strip()

# Define columns of interest
participant_id_col = "participant_id"
sex_f_col = "Sex_F"  # Target variable

# Merge Sex_F with Connectome data
merged_df = pd.merge(train_df[[participant_id_col, sex_f_col]], connectome_data, on=participant_id_col, how="inner")

# Separate Male and Female data
male_df = merged_df[merged_df[sex_f_col] == 0].drop(columns=[sex_f_col, participant_id_col])
female_df = merged_df[merged_df[sex_f_col] == 1].drop(columns=[sex_f_col, participant_id_col])

# Compute absolute mean differences between males and females for each feature
mean_diff = (male_df.mean() - female_df.mean()).abs().sort_values(ascending=False)

# Get the top 10 features with the biggest mean differences
top_10_features = mean_diff.head(10).to_frame(name="Mean Difference")

# Print the results
print("Top 10 Connectome Features with Biggest Mean Differences Between Males and Females:")
print(top_10_features)

