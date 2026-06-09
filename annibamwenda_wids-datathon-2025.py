# Description


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# !pip install geomstats --target=/kaggle/working/


# !pip install openpyxl --target=/kaggle/working/


import numpy as np # linear algebra and statistics
import pandas as pd # data processing
import seaborn as sns # data visualization
import matplotlib.pyplot as plt # data visualization
# import geomstats.backend as gs
# import openpyxl
import math
from sklearn.preprocessing import LabelEncoder # for feature engineering
from sklearn.preprocessing import OneHotEncoder # for feature engineering
from sklearn.preprocessing import StandardScaler # for data normalization
from sklearn.preprocessing import MinMaxScaler # for data normalization
from sklearn.preprocessing import RobustScaler # for data normalization
from sklearn.metrics import f1_score # for model evaluation
from sklearn.model_selection import train_test_split # for splitting the dataset
from sklearn.model_selection import GridSearchCV # hyperparamater tuning
from sklearn.impute import SimpleImputer # for feature engineering
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score  # For evaluation on validation, if you have a holdout set
# from tqdm import tqdm  # For progress bars

import os



# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# loading the train datasets

# train_mri = pd.read_csv('/kaggle/input/widsdatathon2025/TRAIN/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES.csv')
train_mri = pd.read_csv('/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv')
train_labels = pd.read_excel('/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAINING_SOLUTIONS.xlsx')
train_categorical = pd.read_excel('/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_CATEGORICAL_METADATA_new.xlsx')
train_numerical = pd.read_excel('/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_QUANTITATIVE_METADATA_new.xlsx')


# loading the test datasets
test_mri = pd.read_csv('/kaggle/input/widsdatathon2025/TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv')
test_categorical = pd.read_excel('/kaggle/input/widsdatathon2025/TEST/TEST_CATEGORICAL.xlsx')
test_numerical = pd.read_excel('/kaggle/input/widsdatathon2025/TEST/TEST_QUANTITATIVE_METADATA.xlsx')


# print("Train targets: \n", train_labels.head(5))


# print("Train MRI data: \n", train_mri.head(5))


# print("Train categorical data: \n", train_categorical.head(5))


# print("Train numerical features: \n", train_numerical.head(5))


print("Shapes of the train datasets:")
print("Shape of train_labels: \n", train_labels.shape)
print("Shape of train_mri: \n", train_mri.shape)
print("Shape of train_categorical: \n", train_categorical.shape)
print("Shape of train_numerical: \n", train_numerical.shape)


# Concise summary of the train datasets
print(train_mri.info())


print(train_categorical.info())


print(train_numerical.info())


# Statistical summary of our dataset
# print(train_mri.describe())


# print(train_categorical.describe())


# print(train_numerical.describe())


# train_numerical.columns


# EHQ_EHQ_Total
plt.figure(figsize=(12, 8))
sns.histplot(x='EHQ_EHQ_Total', data=train_numerical, color='green')
plt.title("Edinburgh Handedness Questionnaire", fontsize=16)
plt.xlabel("Laterality Index Score", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.show()
# -100 = 10th left 
# −28 ≤ LI < 48 = middle 
# 100 = 10th right


# SDQ_SDQ_Conduct_Problems
plt.figure(figsize=(12, 8))
sns.countplot(x='SDQ_SDQ_Conduct_Problems', data=train_numerical, palette = 'coolwarm')
plt.title("Strength and Difficult Questionaire for Conduct Problems", fontsize=16)
plt.xlabel("Conduct Problems Scale", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.show()


# SDQ_SDQ_Emotional_Problems
plt.figure(figsize=(12, 8))
sns.countplot(x='SDQ_SDQ_Emotional_Problems', data=train_numerical, palette = 'pastel')
plt.title("Strength and Difficult Questionaire for Emotional Problems", fontsize=16)
plt.xlabel("Emotional Problems Scale", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.show()


# SDQ_SDQ_Externalizing and Internalizing
plt.figure(figsize=(12, 6))
sns.countplot(x='SDQ_SDQ_Externalizing', data=train_numerical, palette = 'Set2')
plt.title("Externalizing Scores Distribution", fontsize=16)
plt.xlabel("Externalizing Score", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.show()
plt.figure(figsize=(12, 6))
sns.countplot(x='SDQ_SDQ_Internalizing', data=train_numerical, palette = 'Set2')
plt.title("Internalizing Scores Distribution", fontsize=16)
plt.xlabel("Internalizing score", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.show()


# MRI_Track_Age_at_Scan
plt.figure(figsize=(12,6))
sns.histplot(x='MRI_Track_Age_at_Scan', kde=True, data=train_numerical, color='Maroon')
plt.title("Distribution of Age during MRI Scan", fontsize=16)
plt.xlabel("Age", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.show()


# ADHD Distribution
print(train_labels['ADHD_Outcome'].value_counts())
plt.figure(figsize=(12,6))
sns.countplot(x='ADHD_Outcome', data=train_labels, color='Skyblue')
plt.title("ADHD Distribution", fontsize=16)
plt.xlabel("Outcome (1=Yes, 0=No)", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.show()


# Gender Distribution
print(train_labels['Sex_F'].value_counts())
plt.figure(figsize=(12,6))
sns.countplot(x='Sex_F', data=train_labels, color='Green')
plt.title("Gender Distribution", fontsize=16)
plt.xlabel("Gender (0 = Male, 1 = Female)", fontsize=12)
plt.ylabel("Count",fontsize=12)
plt.show()


# Correlation of Emotional Problems with ADHD outcome
train_numerical_copy = train_numerical.copy()
train_numerical_copy['ADHD_Outcome'] = train_labels['ADHD_Outcome']

plt.figure(figsize=(8, 6))
sns.boxplot(x='ADHD_Outcome', y='SDQ_SDQ_Emotional_Problems', data=train_numerical_copy)
plt.title('SDQ_SDQ_Emotional_Problems vs ADHD Outcome')
plt.xlabel('ADHD Outcome')
plt.ylabel('SDQ_SDQ_Emotional_Problems')
plt.show()


# Barratt_Barratt_P2_Occ - Barratt Simplified Measure of Social Status - Parent 1 Occupation
train_categorical['Barratt_Barratt_P2_Occ'].value_counts()

# 0=Homemaker, stay at home parent.
# 5=Day laborer, janitor, house cleaner, farm worker, food counter,preparation worker, busboy.
# 10=Garbage collector, short-order cook, cab driver, shoe sales, assembly line workers, masons, baggage porter.
# 15=Pa


plt.figure(figsize=(12,6))
sns.countplot(x='Barratt_Barratt_P2_Occ', data=train_categorical[['Barratt_Barratt_P2_Occ']])
plt.title(f"Distribution of Barratt Social Status Measure - Parent 2 Occupation", fontsize=14)
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(12,6))
sns.countplot(x='Barratt_Barratt_P1_Occ', data=train_categorical[['Barratt_Barratt_P1_Occ']])
plt.title(f"Distribution of Barratt Social Status Measure - Parent 1 Occupation", fontsize=14)
plt.xticks(rotation=45)
plt.show()


# Let's compare Parent level of education with ADHD Outcome

sns.countplot(data=train_categorical, x='Barratt_Barratt_P1_Edu', hue=train_labels['ADHD_Outcome'])
plt.title('ADHD Prevalence by Parent 1 Education')
plt.show()



sns.countplot(data=train_categorical, x='Barratt_Barratt_P2_Edu', hue=train_labels['ADHD_Outcome'])
plt.title("ADHD Outcome Distribution by Parent 2 Education")
plt.show()


# Comparing color vision test and gender

sns.countplot(data=train_numerical, x='ColorVision_CV_Score', hue=train_labels['Sex_F'])
plt.title('Color Vision Score Distribution by Gender')
plt.show()


# Demographics - Race of Child vs ADHD Outcomes
print(train_categorical['PreInt_Demos_Fam_Child_Race'].value_counts())

# 0= White/Caucasian 
# 1= Black/African American 
# 2= Hispanic 
# 3= Asian 
# 4= Indian
# 5= Native American India...


sns.countplot(data=train_categorical, x='PreInt_Demos_Fam_Child_Race', hue=train_labels['ADHD_Outcome'])
plt.title('ADHD Outcomes by Race of Child')
plt.xlabel("Child Race")
plt.show()


# Correlation Matrix

# Correlation of numerical features with train_labels

# let's merge train_numerical with train_labels to create our dataset for correlation
cat_corr_data = pd.merge(train_numerical, train_labels, on='participant_id')
cat_corr_data.drop('participant_id', axis=1, inplace=True) # we won't need to check correlation with ids
cat_corr_matrix = cat_corr_data.corr()


# Detailed heat map
# sns.heatmap(cat_corr_data, 
#             cmap='YlGnBu', # choosing a yellow-green-blue colormap
#             annot=True, # Turning on annotations
#             fmt="d", # displaying annotations as integer
#             linewidths=.5, # Add gridlines with width 0.5
#             cbar=True, # Include color bar
# )
# plt.show()


# Checking for duplicates in our train data
print("Len of train_numerical before: ", len(train_numerical))
train_numerical.drop_duplicates() # removing duplicates if any
print("Len of train_numerical after: ",len(train_numerical))


print("Len of train_categorical before: ", len(train_categorical))
train_categorical.drop_duplicates()#
print("Len of train_categorical before: ", len(train_categorical))


# Checking for duplicates in test data
print("Len of test_numerical before: ", len(test_numerical))
test_numerical.drop_duplicates() # removing duplicates if any
print("Len of test_numerical after: ", len(test_numerical))

print("Len of test_categorical before: ", len(test_categorical))
test_categorical.drop_duplicates()
print("Len of test_categorical after: ", len(test_categorical))


# Checking for missing values
print("Missing values in train_numerical: ")
train_numerical.isnull().sum()


train_numerical['MRI_Track_Age_at_Scan'].describe()


train_numerical[train_numerical['MRI_Track_Age_at_Scan'] == 0]['MRI_Track_Age_at_Scan'].value_counts()


train_numerical[train_numerical['MRI_Track_Age_at_Scan'] == 0].index


# Drop the two rows with 'MRI_Track_Age_at_Scan' as 0
train_numerical = train_numerical[train_numerical['MRI_Track_Age_at_Scan'] != 0]
print(train_numerical[train_numerical['MRI_Track_Age_at_Scan'] == 0]['MRI_Track_Age_at_Scan'].value_counts())
# From the output, the two rows are now dropped


# Let's check the descriptive statistics of MRI column again.
train_numerical['MRI_Track_Age_at_Scan'].describe()


# We'll now replace the missing values in 'MRI' with the mean
train_numerical['MRI_Track_Age_at_Scan'].fillna(train_numerical['MRI_Track_Age_at_Scan'].mean(), inplace=True)


# Let's check again for missing values in train numerical
train_numerical.isnull().sum()


# Missing values in train_categorical
train_categorical.isnull().sum()


# Let's further investigate the 'PreInt_Demos_Fam_Child_Ethnicity' feature
# 'PreInt_Demos_Fam_Child_Ethnicity' feature indicates the ethnicity of the Child
# 0= Not Hispanic or Latino 
# 1= Hispanic or Latino 
# 2= Decline to specify 
# 3= Unknown
print("Unique values for PreInt_Demos_Fam_Child_Ethnicity Feature: ")
print(train_categorical['PreInt_Demos_Fam_Child_Ethnicity'].unique(), '\n')
print("Value counts for each unique value in PreInt_Demos_Fam_Child_Ethnicity:")
print(train_categorical['PreInt_Demos_Fam_Child_Ethnicity'].value_counts())


# Since category 0 has the highest frequency, We'll replace the missing values with mode(0.0)
train_categorical['PreInt_Demos_Fam_Child_Ethnicity'].fillna(train_categorical['PreInt_Demos_Fam_Child_Ethnicity'].mode().iloc[0], inplace=True)
# We'll replace missing values in other categorical eatures with their mode as well
train_categorical['PreInt_Demos_Fam_Child_Race'].fillna(train_categorical['PreInt_Demos_Fam_Child_Race'].mode().iloc[0], inplace = True)
train_categorical['Barratt_Barratt_P1_Edu'].fillna(train_categorical['Barratt_Barratt_P1_Edu'].mode().iloc[0], inplace = True)
train_categorical['Barratt_Barratt_P1_Occ'].fillna(train_categorical['Barratt_Barratt_P1_Occ'].mode().iloc[0], inplace = True)
train_categorical['MRI_Track_Scan_Location'].fillna(train_categorical['MRI_Track_Scan_Location'].mode().iloc[0], inplace = True)


# We'll drop Barratt_Barratt_P2_Edu and Barratt_Barratt_P2_Occ because they both have too many missing values
drop_cols = ['Barratt_Barratt_P2_Edu', 'Barratt_Barratt_P2_Occ']
train_categorical.drop(drop_cols, axis = 1, inplace = True)
test_categorical.drop(drop_cols, axis = 1, inplace = True)


# Final check to see if we removed all missing values from train_categorical
train_categorical.isnull().sum()


# Starting with 'APQ_P_APQ_P_CP'
Q1 = train_numerical['APQ_P_APQ_P_CP'].quantile(0.25)
Q3 = train_numerical['APQ_P_APQ_P_CP'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5*IQR
upper_bound = Q3 + 1.5 * IQR
print(f'Lower bound for "APQ_P_APQ_P_CP" is: {lower_bound}')
print(f'Upper bound for "APQ_P_APQ_P_CP" is: {upper_bound}')
# Let's check how many values lie above the upper bound
print(len(train_numerical[train_numerical['APQ_P_APQ_P_CP'] > upper_bound]))
upper_bound_df = train_numerical[train_numerical['APQ_P_APQ_P_CP'] > upper_bound]
print(upper_bound_df['APQ_P_APQ_P_CP'].value_counts())


# We'll use the standard scaler to standardize our numerical columns

scaler = StandardScaler() #initializing the scaler

# dropping the participant_id column before standardizing the numerical columns
train_numerical_scaled = scaler.fit_transform(train_numerical.drop(columns ='participant_id'))
test_numerical_scaled = scaler.fit_transform(test_numerical.drop(columns = 'participant_id'))

# reconstructing the dataframes
train_num_scaled_df = pd.DataFrame(train_numerical_scaled, columns=train_numerical.columns.drop('participant_id'))
train_num_scaled_df['participant_id'] = train_numerical['participant_id'].values

test_num_scaled_df = pd.DataFrame(test_numerical_scaled, columns=test_numerical.columns.drop('participant_id'))
test_num_scaled_df['participant_id'] = test_numerical['participant_id'].values


# Let's combine numerical and categorical datasets into one dataframe
train_combined = pd.merge(train_num_scaled_df, train_categorical,on ="participant_id", how ="inner")
test_combined = pd.merge(test_num_scaled_df, test_categorical, on = "participant_id", how = "inner")
# assert all(train_combined.index == train_labels.index), "Label IDs don't match train IDs"


train_combined.shape


test_combined.shape


train_combined["participant_id"].head(5)


# --- Function to load and flatten the mri data ---

def preprocess_mri_data(mri_df):
    mri_dict = {}
    for _, row in mri_df.iterrows():
        pid = row['participant_id']
        matrix_values = row.iloc[1:].values.flatten()
        side_len = int(math.sqrt(matrix_values.shape[0]))
        mri_scan = matrix_values[:side_len*side_len].reshape(side_len, side_len)
        mri_scan = mri_scan / mri_scan.max() if mri_scan.max() > 0 else mri_scan
        mri_dict[pid] = mri_scan.flatten()  # Store flattened MRI scan
    return mri_dict

# Preprocess MRI data
mri_train_dict = preprocess_mri_data(train_mri)  # Your train MRI DataFrame
mri_test_dict = preprocess_mri_data(test_mri)  # Your test MRI DataFrame


# Merging mri and tabular data into final features
def merge_mri_with_tabular(tabular_df, mri_dict):
    combined_features = []
    valid_ids = []

    # Loop through each row of tabular data and add corresponding MRI features
    for _, row in tabular_df.iterrows():
        pid = row['participant_id']
        if pid in mri_dict:
            # Merge MRI features and tabular features
            tabular_features = row.drop('participant_id').values  # Exclude participant_id
            mri_features = mri_dict[pid]
            combined = np.concatenate([tabular_features, mri_features])  # Merge both
            combined_features.append(combined)
            valid_ids.append(pid)

    return np.array(combined_features), valid_ids

# Combine the data (MRI + tabular features)
X_train_combined, valid_train_ids = merge_mri_with_tabular(train_combined, mri_train_dict)
X_test_combined, valid_test_ids = merge_mri_with_tabular(test_combined, mri_test_dict)


# Match labels to valid training IDs (those with MRI + tabular data)
y_train = train_labels[train_labels['participant_id'].isin(valid_train_ids)]
print(y_train.shape)
print(y_train.head(5))
y_train = y_train.set_index('participant_id').loc[valid_train_ids][['Sex_F', 'ADHD_Outcome']].values
print(y_train.shape)


from sklearn.impute import SimpleImputer

# Initialize imputer
imputer = SimpleImputer(strategy='median')

# Fit on training data and transform both train and test
X_train_combined = imputer.fit_transform(X_train_combined)
X_test_combined = imputer.transform(X_test_combined)


# Initialize base classifier
base_clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Wrap it for multi-output
multi_clf = MultiOutputClassifier(base_clf)

# Train
multi_clf.fit(X_train_combined, y_train)


from sklearn.metrics import accuracy_score, classification_report

# Predict on training data
y_train_pred = multi_clf.predict(X_train_combined)

# Convert predictions and labels to DataFrames for easier analysis
train_pred_df = pd.DataFrame(y_train_pred, columns=['Sex_F_pred', 'ADHD_pred'])
train_pred_df['participant_id'] = valid_train_ids

# Merge with true labels for comparison
y_train_true_df = train_labels[train_labels['participant_id'].isin(valid_train_ids)]
y_train_true_df = y_train_true_df.set_index('participant_id').loc[valid_train_ids].reset_index()
train_analysis_df = pd.merge(train_pred_df, y_train_true_df, on='participant_id')

# Accuracy scores
sex_acc = accuracy_score(train_analysis_df['Sex_F'], train_analysis_df['Sex_F_pred'])*100
adhd_acc = accuracy_score(train_analysis_df['ADHD_Outcome'], train_analysis_df['ADHD_pred'])*100

print(f"Training Accuracy - Sex_F: {sex_acc:.2f}%")
print(f"Training Accuracy - ADHD_Outcome: {adhd_acc:.2f}%")

# Classification reports
print("\nClassification Report - Sex_F:")
print(classification_report(train_analysis_df['Sex_F'], train_analysis_df['Sex_F_pred']))

print("\nClassification Report - ADHD_Outcome:")
print(classification_report(train_analysis_df['ADHD_Outcome'], train_analysis_df['ADHD_pred']))

# Prediction counts
adhd_count = train_analysis_df['ADHD_pred'].sum()
female_count = train_analysis_df['Sex_F_pred'].sum()
female_adhd_count = train_analysis_df[
    (train_analysis_df['Sex_F_pred'] == 1) & (train_analysis_df['ADHD_pred'] == 1)
].shape[0]

# Original counts
og_adhd_count = train_labels['ADHD_Outcome'].sum()
og_female_count = train_labels['Sex_F'].sum()
og_female_adhd_count = train_labels[
    (train_labels['Sex_F'] == 1) & (train_labels['ADHD_Outcome'] == 1)
].shape[0]

print("\nTraining Set Prediction Counts:")
print(f"Predicted ADHD cases: {adhd_count}")
print(f"Predicted Female cases: {female_count}")
print(f"Predicted Female with ADHD: {female_adhd_count}")

print("\nTraining Set Original Label Counts:")
print(f"Predicted ADHD cases: {og_adhd_count}")
print(f"Predicted Female cases: {og_female_count}")
print(f"Predicted Female with ADHD: {og_female_adhd_count}")


test_preds = multi_clf.predict(X_test_combined)


# Convert predictions to a DataFrame for easier analysis
test_pred_df = pd.DataFrame(test_preds, columns=['Sex_F', 'ADHD_Outcome'])
test_pred_df['participant_id'] = valid_test_ids

# Count how many test predictions had ADHD
adhd_count = test_pred_df['ADHD_Outcome'].sum()

# Count how many were predicted to be women
women_count = test_pred_df['Sex_F'].sum()

# Count how many were predicted as both women and ADHD
women_adhd_count = test_pred_df[(test_pred_df['Sex_F'] == 1) & (test_pred_df['ADHD_Outcome'] == 1)].shape[0]

# Print the results
print(f"\n--- Test Prediction Summary ---")
print(f"Total test samples: {len(test_pred_df)}")
print(f"Predicted ADHD cases: {adhd_count}")
print(f"Predicted Female participants: {women_count}")
print(f"Predicted Female participants with ADHD: {women_adhd_count}")



#-- Commented this out to reduce runtime
# let's tune our existing random forest model
# Tune only ADHD label (index 1)
# y_adhd = y_train[:, 1]

# # Define the base model
# base_rf = RandomForestClassifier(random_state=42)

# # Define hyperparameter grid
# param_grid = {
#     'n_estimators': [100, 200],
#     'max_depth': [None, 10, 20],
#     'min_samples_split': [2, 5],
# }

# # Run grid search
# grid_search = GridSearchCV(
#     estimator=base_rf,
#     param_grid=param_grid,
#     cv=3,
#     scoring='accuracy',
#     n_jobs=-1,
#     verbose=1
# )

# # Train using just one label for tuning speed — ADHD_Outcome
# grid_search.fit(X_train_combined, y_adhd)

# # Output best params
# print("Best Parameters for ADHD:", grid_search.best_params_)


# -- Create feature names
# get tabular data feature names (train numerical and categorical)
num_mri_features = len(next(iter(mri_train_dict.values())))
# get the number of MRI features and name them generically (e.g., MRI_0, MRI_1, …)
tabular_feature_names = train_combined.drop(columns=['participant_id']).columns.tolist() 
mri_feature_names = [f'MRI_{i}' for i in range(num_mri_features)]
# combine both lists
combined_feature_names = tabular_feature_names + mri_feature_names


# -- Check feature importances for ADHD
adhd_model = multi_clf.estimators_[1]
adhd_importances = adhd_model.feature_importances_

# Get top 20 features
top_adhd_idx = np.argsort(adhd_importances)[-20:]
top_features = [combined_feature_names[i] for i in top_adhd_idx]
top_importances = adhd_importances[top_adhd_idx]
top_adhd_set = set(top_adhd_idx)


# Plot
plt.figure(figsize=(10, 6))
plt.barh(range(20), top_importances)
plt.yticks(ticks=range(20), labels=top_features)
plt.xlabel("Importance")
plt.title("Top 20 Feature Importances (ADHD Prediction)")
plt.tight_layout()
plt.show()


# Check feature importances for Sex_F
sex_model = multi_clf.estimators_[0]
importances_sex = sex_model.feature_importances_

# Get top 20 features
top_sex_idx = np.argsort(importances_sex)[-20:]
top_features_sex = [combined_feature_names[i] for i in top_sex_idx]
top_importances_sex = importances_sex[top_sex_idx]
top_sex_set = set(top_sex_idx)

# Plot
plt.figure(figsize=(10, 6))
plt.barh(range(20), top_importances_sex)
plt.yticks(ticks=range(20), labels=top_features_sex)
plt.xlabel("Importance")
plt.title("Top 20 Feature Importances (Sex_F Prediction)")
plt.tight_layout()
plt.show()


# We'll only use the top 20 features from each of the plots above to train our model

# Union of top features (no duplicates)
combined_top_indices = sorted(top_adhd_set.union(top_sex_set))

# Reduce X to important features
X_train_reduced = X_train_combined[:, combined_top_indices]
X_test_reduced = X_test_combined[:, combined_top_indices]


# Since running the random forest classifier with tuned paramaters resulted in a very low score, let's train with HistGradientboost

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier

base_clf_best = HistGradientBoostingClassifier(random_state=42)
multi_clf_best = MultiOutputClassifier(base_clf)
multi_clf_best.fit(X_train_reduced, y_train)

# # Let's replace with tuned parameters
# best_params = {'n_estimators': 100, 'min_samples_split': 5, 'max_depth': None}
# # best_params = grid_search.best_params_
# base_clf_best = RandomForestClassifier(**best_params, random_state=42)

# # Wrap in MultiOutput
# multi_clf_best = MultiOutputClassifier(base_clf_best)

# # Train on full y_train (Sex_F and ADHD_Outcome)
# multi_clf_best.fit(X_train_reduced, y_train)

# train prediction accuracy
train_preds_best = multi_clf_best.predict(X_train_reduced)
print("Training Accuracy (Sex_F):", accuracy_score(y_train[:, 0], train_preds_best[:, 0]) * 100, "%")
print("Training Accuracy (ADHD):", accuracy_score(y_train[:, 1], train_preds_best[:, 1]) * 100, "%")



# Test predictions
test_preds_best = multi_clf_best.predict(X_test_reduced)


# Convert predictions to a DataFrame for easier analysis
test_pred_df = pd.DataFrame(test_preds_best, columns=['Sex_F', 'ADHD_Outcome'])
test_pred_df['participant_id'] = valid_test_ids

# Count how many test predictions had ADHD
adhd_count = test_pred_df['ADHD_Outcome'].sum()

# Count how many were predicted to be women
women_count = test_pred_df['Sex_F'].sum()

# Count how many were predicted as both women and ADHD
women_adhd_count = test_pred_df[(test_pred_df['Sex_F'] == 1) & (test_pred_df['ADHD_Outcome'] == 1)].shape[0]

# Print the results
print(test_pred_df.head(5))
print(f"\n--- Test Prediction Summary ---")
print(f"Total test samples: {len(test_pred_df)}")
print(f"Predicted ADHD cases: {adhd_count}")
print(f"Predicted Female participants: {women_count}")
print(f"Predicted Female participants with ADHD: {women_adhd_count}")


# Wrap predictions with participant IDs
submission_df = pd.DataFrame(test_preds_best, columns=['Sex_F', 'ADHD_Outcome'])
submission_df['participant_id'] = valid_test_ids
submission_df = submission_df[['participant_id', 'Sex_F', 'ADHD_Outcome']]
print(submission_df.head(5))

# Save to CSV
submission_df.to_csv("submission.csv", index=False)
print("Submission saved as submission.csv")

