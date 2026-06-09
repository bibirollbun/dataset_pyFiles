%matplotlib inline
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import inspect

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif

train_category_path = "/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_CATEGORICAL_METADATA_new.xlsx"
train_quant_path = "/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_QUANTITATIVE_METADATA_new.xlsx"
train_fmri_path = "/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv"
train_solutions_path="/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAINING_SOLUTIONS.xlsx"

df_train_cat = pd.read_excel(train_category_path)
df_train_function = pd.read_csv(train_fmri_path)
df_train_quantit = pd.read_excel(train_quant_path)
df_training_sol = pd.read_excel(train_solutions_path)
print('done')


df_train_cat.head()


df_train_function.head()


df_train_quantit.head()


df_training_sol.head()


def get_dataframe_name(df):
    caller_frame = inspect.currentframe().f_back
    caller_locals = caller_frame.f_locals
    return [name for name, value in caller_locals.items() if value is df][0]

def summary_dataframe(df):
    summary = {}
    df_name = get_dataframe_name(df)
    summary['DataFrame'] = df_name

    # Check for missing values in each column
    missing_values = df.isnull().sum()

    # Filter columns that have missing values
    missing_columns = missing_values[missing_values > 0]

    # If there are missing values, print and add to the summary
    if not missing_columns.empty:
        print(f"Missing values per column in {df_name}:")
        for column, missing_count in missing_columns.items():
            print(f"- {column}: {missing_count} missing values")
        summary['Missing values columns'] = missing_columns.to_dict()  # Add to summary
    else:
        print(f"No missing values in any columns in {df_name}")
        summary['Missing values columns'] = "No missing values"

    return summary


miss_cols_df_train_cat=summary_dataframe(df_train_cat)


miss_cols_df_train_function=summary_dataframe(df_train_function)


miss_cols_df_train_quant=summary_dataframe(df_train_quantit)


sdq_cols = [
    'SDQ_SDQ_Difficulties_Total', 'SDQ_SDQ_Emotional_Problems',
    'SDQ_SDQ_Externalizing', 'SDQ_SDQ_Generating_Impact',
    'SDQ_SDQ_Hyperactivity', 'SDQ_SDQ_Internalizing',
    'SDQ_SDQ_Peer_Problems', 'SDQ_SDQ_Prosocial'
]

missing_in_same_row = df_train_quantit[sdq_cols].isnull().all(axis=1)
missing_rows = df_train_quantit[missing_in_same_row]
print(missing_rows[['participant_id'] + sdq_cols])



apq_cols=['APQ_P_APQ_P_CP',                
'APQ_P_APQ_P_ID','APQ_P_APQ_P_INV',
'APQ_P_APQ_P_OPD','APQ_P_APQ_P_PM','APQ_P_APQ_P_PP']              

missing_in_same_row_2 = df_train_quantit[apq_cols].isnull().all(axis=1)
missing_rows_2 = df_train_quantit[missing_in_same_row_2]
print(f"Number of rows with all SDQ columns missing: {missing_rows_2.shape[0]}")
print(missing_rows_2[['participant_id'] + apq_cols])



df_train_quantit['ColorVision_CV_Score'].unique()



df_train_quantit.columns = df_train_quantit.columns.str.strip()
df_train_quantit['MRI_Track_Age_at_Scan'].describe()


summary_dataframe(df_training_sol)


df_train_function.shape


df_train_function.columns


def plot_label_distribution(labels, label_column,labels_map):
    label_counts = labels[label_column].value_counts().reindex([0, 1], fill_value=0)
    colors = ['#4682B4','#FF6347']
    plt.figure(figsize=(8, 6))
    bars = plt.bar(label_counts.index, label_counts.values, color=colors, width=0.4)
    plt.title(f'{label_column} Label Distribution', fontsize=16)
    plt.xlabel(label_column, fontsize=14)
    plt.xticks([0, 1], labels=['0', '1'], fontsize=12)
    plt.ylabel('Count', fontsize=14)
    plt.legend(bars, [labels_map[key] for key in label_counts.index], title=label_column, fontsize=12, title_fontsize=13)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

labels_map_adhd={0: "No", 1: "Yes"}
plot_label_distribution(df_training_sol, 'ADHD_Outcome',labels_map_adhd)



plot_label_distribution(df_training_sol, 'Sex_F',labels_map_adhd)


df_merged = df_train_quantit.merge(df_training_sol, on='participant_id', how='inner')


df_merged_copy=df_merged.copy()


miss_cols_names_quant = list(miss_cols_df_train_quant['Missing values columns'].keys())
df_merged_copy.dropna(subset=miss_cols_names_quant, inplace=True)


missing_values_after_drop = df_merged_copy.isnull().sum()
print("Missing values per column after dropping rows:")
print(missing_values_after_drop[missing_values_after_drop > 0])


df_females = df_merged_copy[df_merged_copy['Sex_F'] == 1]
df_males = df_merged_copy[df_merged_copy['Sex_F'] == 0]


df_females['Sex_F'].unique()


df_males['Sex_F'].unique()


positive_adhd_females = df_females[df_females['ADHD_Outcome'] == 1].shape[0]
negative_adhd_females = df_females[df_females['ADHD_Outcome'] == 0].shape[0]
diagnosis_counts = [positive_adhd_females, negative_adhd_females]
diagnosis_labels = ['Positive ADHD', 'Negative ADHD']

plt.figure(figsize=(8, 6))
sns.barplot(x=diagnosis_labels, y=diagnosis_counts, width=0.4,  palette=["purple", "darkorange"])
plt.xlabel('ADHD Diagnosis')
plt.ylabel('Count')
plt.title('ADHD Diagnosis for Females')
plt.grid(axis='y', linestyle='--', alpha=0.7)


positive_adhd_males = df_males[df_males['ADHD_Outcome'] == 1].shape[0]
negative_adhd_males = df_males[df_males['ADHD_Outcome'] == 0].shape[0]
diagnosis_counts_m = [positive_adhd_males, negative_adhd_males]
diagnosis_labels_m = ['Positive ADHD', 'Negative ADHD']

plt.figure(figsize=(8, 6))
sns.barplot(x=diagnosis_labels_m, y=diagnosis_counts_m, width=0.4,  palette=["mediumseagreen", "royalblue"])
plt.xlabel('ADHD Diagnosis')
plt.ylabel('Count')
plt.title('ADHD Diagnosis for Males')
plt.grid(axis='y', linestyle='--', alpha=0.7)


prop_females = positive_adhd_females / (positive_adhd_females + negative_adhd_females)
prop_males = positive_adhd_males / (positive_adhd_males + negative_adhd_males)

print(f"ADHD Diagnosis Rate - Females: {prop_females:.2%}")
print(f"ADHD Diagnosis Rate - Males: {prop_males:.2%}")


df_merged_copy.dropna(subset=['MRI_Track_Age_at_Scan'], inplace=True)


df_merged_copy['Sex_Label'] = df_merged_copy['Sex_F'].map({0: 'Male', 1: 'Female'})


bins = [6, 12, 18, 30]  
labels = ['6-12', '13-18', '19-30']
df_merged_copy['Age_Group'] = pd.cut(df_merged_copy['MRI_Track_Age_at_Scan'], bins=bins, labels=labels, right=False)

df_positive_adhd_m_f = df_merged_copy[df_merged_copy['ADHD_Outcome'] == 1]
df_grouped_positive = df_positive_adhd_m_f.groupby(['Age_Group', 'Sex_F'])['participant_id'].nunique().reset_index()
df_grouped_positive = df_grouped_positive.rename(columns={'participant_id': 'Positive_Diagnosis_Count'})

sns.barplot(data=df_grouped_positive, x='Age_Group', y='Positive_Diagnosis_Count', hue='Sex_F', palette='Set2',errorbar=None)
plt.xlabel("Age Group")
plt.ylabel("Count")
plt.title("Number of Positive ADHD Diagnosis")
plt.grid(True, axis='y', linestyle='--', alpha=0.7)


df_positive_adhd_females = df_females[df_females['ADHD_Outcome'] == 1]
df_negative_adhd_females = df_females[df_females['ADHD_Outcome'] == 0]

plt.figure(figsize=(8, 6))
sns.histplot(df_negative_adhd_females['ColorVision_CV_Score'], kde=False, color='red', label='Negative ADHD', bins=8)
plt.xlabel('Color Vision Score')
plt.ylabel('Frequency')
plt.title('Negative ADHD Diagnosis: Females')
plt.xlim(0,15)
plt.ylim(0,210)


plt.figure(figsize=(8, 6))
sns.set(style="whitegrid", palette="muted")
sns.histplot(df_positive_adhd_females['ColorVision_CV_Score'], kde=False, color='blue', label='Positive ADHD', bins=15, alpha=0.6)
plt.xlabel('Color Vision Score')
plt.ylabel('Frequency')
plt.title('Positive ADHD Diagnosis: Females')
plt.ylim(0,210)
plt.xlim(0,15)


select_cols=['SDQ_SDQ_Difficulties_Total','SDQ_SDQ_Emotional_Problems',
'SDQ_SDQ_Hyperactivity','SDQ_SDQ_Externalizing']

fig, axes = plt.subplots(len(select_cols), 1, figsize=(8, 12))

# Loop through SDQ columns and plot for both positive and negative ADHD females
for i, col in enumerate(select_cols):
    ax = axes[i]
    
    sns.histplot(df_positive_adhd_females[col], kde=False, color='seagreen', label='Positive ADHD', ax=ax, bins=20)
    sns.histplot(df_negative_adhd_females[col], kde=False, color='blue', label='Negative ADHD', ax=ax, bins=20)
    
    ax.set_title(f'{col} - Females')
    ax.set_ylabel('Frequency')
    ax.set_xlabel('')
    ax.legend()

plt.subplots_adjust(hspace=0.5)
plt.tight_layout()



df_positive_adhd_males = df_males[df_males['ADHD_Outcome'] == 1]
df_negative_adhd_males = df_males[df_males['ADHD_Outcome'] == 0]


fig, axes = plt.subplots(len(select_cols), 1, figsize=(8, 12))

for i, col in enumerate(select_cols):
    ax = axes[i]
    
    sns.histplot(df_positive_adhd_males[col], kde=False, color='purple', label='Positive ADHD', ax=ax, bins=20)
    sns.histplot(df_negative_adhd_males[col], kde=False, color='darkorange', label='Negative ADHD', ax=ax, bins=20)
    
    ax.set_title(f'{col} - Males')
    ax.set_ylabel('Frequency')
    ax.set_xlabel('')
    ax.legend()

plt.subplots_adjust(hspace=0.5)
plt.tight_layout()


df_merged_copy = df_merged_copy.merge(df_train_cat, on='participant_id', how='inner')
miss_cols_names_cat = list(miss_cols_df_train_cat['Missing values columns'].keys())
df_merged_copy.dropna(subset=miss_cols_names_cat, inplace=True)


plt.figure(figsize=(10, 6))
sns.countplot(data=df_merged_copy, x='Basic_Demos_Enroll_Year', hue='ADHD_Outcome', palette='viridis')
plt.title('ADHD Outcome by Enrollment Year')
plt.xlabel('Enrollment Year')
plt.ylabel('Number of Participants')
plt.legend(title='ADHD Outcome')
plt.grid(axis='y', linestyle='--', alpha=0.7)


df_merged_copy['MRI_Track_Scan_Location'].unique()


location_labels = {
    1: "Staten Island",
    2: "RUBIC",
    3: "CBIC",
    4: "CUNY"
}

site_labels = {
    1: "Staten Island",
    2: "MRV",
    3: "Midtown",
    4: "Harlem",
    5: "SI RUMC",
}

df_merged_copy['MRI_Track_Scan_Location'] = df_merged_copy['MRI_Track_Scan_Location'].map(location_labels)
df_merged_copy['Basic_Demos_Study_Site'] = df_merged_copy['Basic_Demos_Study_Site'].map(site_labels)

df_merged_copy_2018 = df_merged_copy[df_merged_copy['Basic_Demos_Enroll_Year'] == 2018]
df_females_2018 = df_merged_copy_2018[df_merged_copy_2018['Sex_F'] == 1]
df_males_2018 = df_merged_copy_2018[df_merged_copy_2018['Sex_F'] == 0]


plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1) 
sns.countplot(data=df_females_2018, x='MRI_Track_Scan_Location', hue='ADHD_Outcome', palette='Set2', width=0.4)
plt.title('ADHD Outcome by MRI Scan Location for Females')
plt.xlabel('Scan Location')
plt.ylabel('Number of Female Participants')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.legend(title='ADHD Outcome', loc='upper right')
plt.ylim(0,75)

plt.subplot(1, 2, 2)
sns.countplot(data=df_males_2018, x='MRI_Track_Scan_Location', hue='ADHD_Outcome', palette='deep', width=0.4)
plt.title('ADHD Outcome by MRI Scan Location for Males')
plt.xlabel('Scan Location')
plt.ylabel('Number of Male Participants')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.legend(title='ADHD Outcome', loc='upper right')
plt.ylim(0,75)


plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.countplot(data=df_females_2018, x='Basic_Demos_Study_Site', hue='ADHD_Outcome', palette='viridis', width=0.4)
plt.title('ADHD Outcome by Site of Phenotypic Testing for Females (2018)')
plt.xlabel('Site of Phenotypic Testing')
plt.ylabel('Number of Female Participants')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.legend(title='ADHD Outcome', loc='upper right')
plt.ylim(0,75)


plt.subplot(1, 2, 2) 
sns.countplot(data=df_males_2018, x='Basic_Demos_Study_Site', hue='ADHD_Outcome', palette='dark', width=0.4)
plt.title('ADHD Outcome by Site of Phenotypic Testing for Males (2018)')
plt.xlabel('Site of Phenotypic Testing')
plt.ylabel('Number of Male Participants')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.legend(title='ADHD Outcome', loc='upper right')
plt.ylim(0,75)


df_merged = df_merged.merge(df_train_function, on='participant_id', how='inner')


print(df_train_cat.shape)
print(df_train_function.shape)
print(df_train_quantit.shape)
print(df_training_sol.shape)
print(df_merged.shape)


duplicate_columns = df_merged.columns[df_merged.columns.duplicated()]
print("Duplicate column names:", duplicate_columns)


y_ADHD_outcome = df_merged['ADHD_Outcome']
y_Sex = df_merged['Sex_F']
X = df_merged.drop(['participant_id', 'ADHD_Outcome', 'Sex_F'], axis=1)


X_train, X_val, y_train_adhd, y_val_adhd, y_train_sex, y_val_sex = train_test_split(
    X, 
    y_ADHD_outcome, 
    y_Sex, 
    test_size=0.2, 
    random_state=42,
)


miss_cols_quant_keys = list(miss_cols_df_train_quant['Missing values columns'].keys())
mode_cols_train = X_train[miss_cols_quant_keys].mode().iloc[0]
medians_cols_train = X_train[miss_cols_quant_keys].median()

comparison_df = pd.DataFrame({
    'Mode': mode_cols_train,
    'Median': medians_cols_train
})

print(comparison_df)


X_train[miss_cols_quant_keys] = X_train[miss_cols_quant_keys].apply(lambda col: col.fillna(medians_cols_train[col.name]))
X_val[miss_cols_quant_keys] = X_val[miss_cols_quant_keys].apply(lambda col: col.fillna(medians_cols_train[col.name]))


missing_values_final1 = X_train.isnull().sum().sum()
print(f'Total missing values in X_train: {missing_values_final1}')

missing_values_final2 = X_val.isnull().sum().sum()
print(f'Total missing values in X_val: {missing_values_final2}')


X_train_scaled = StandardScaler().fit_transform(X_train)
X_val_scaled = StandardScaler().fit_transform(X_val)


selector_adhd = SelectKBest(mutual_info_classif,k=20)
X_train_selected_adhd = selector_adhd.fit_transform(X_train_scaled, y_train_adhd)
X_val_selected_adhd = selector_adhd.transform(X_val_scaled)


selected_features_adhd = X_train.columns[selector_adhd.get_support()]
print("Selected features for ADHD prediction:")
print(selected_features_adhd)


selector_sex = SelectKBest(mutual_info_classif, k=50)
X_train_selected_sex = selector_sex.fit_transform(X_train_scaled, y_train_sex)
X_val_selected_sex = selector_sex.transform(X_val_scaled)


selected_features_sex = X_train.columns[selector_sex.get_support()]
print("Selected features for sex prediction:")
print(selected_features_sex)


rf_model_adhd = RandomForestClassifier(n_estimators=400,random_state=1,class_weight='balanced').fit(X_train_selected_adhd, y_train_adhd)
adhd_pred_rf = rf_model_adhd.predict(X_val_selected_adhd)
adhd_f1_rf = f1_score(y_val_adhd, adhd_pred_rf)
print('ADHD F1 score using Random Forest: ', adhd_f1_rf)


lr_model_adhd = LogisticRegression(penalty='l2',class_weight='balanced',max_iter=5000,solver='liblinear').fit(X_train_selected_adhd, y_train_adhd)
adhd_pred_lr = lr_model_adhd.predict(X_val_selected_adhd)
adhd_f1_lr = f1_score(y_val_adhd, adhd_pred_lr)
print('ADHD F1 Score using Logistic Regression:', adhd_f1_lr)


rf_model_sex = RandomForestClassifier(n_estimators=400,random_state=1,class_weight='balanced').fit(X_train_selected_sex, y_train_sex)
sex_pred_rf = rf_model_sex.predict(X_val_selected_sex)
sex_f1_rf = f1_score(y_val_sex, sex_pred_rf)
print('Sex F1 Score using Random Forest: ', sex_f1_rf)


lr_model_sex = LogisticRegression(penalty='l2',class_weight='balanced',max_iter=5000,solver='liblinear').fit(X_train_selected_sex, y_train_sex)
sex_pred_lr = lr_model_sex.predict(X_val_selected_sex)
sex_f1_lr = f1_score(y_val_sex, sex_pred_lr)
print('Sex Classification F1 Score using Logistic Regression: ', sex_f1_lr)


test_fmri_path="/kaggle/input/widsdatathon2025/TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv"
test_quant_path="/kaggle/input/widsdatathon2025/TEST/TEST_QUANTITATIVE_METADATA.xlsx"

df_test_function = pd.read_csv(test_fmri_path)
df_test_quantit = pd.read_excel(test_quant_path)


df_test_merged = df_test_function.merge(df_test_quantit, on='participant_id',how='inner')


miss_cols_df_test_merged=summary_dataframe(df_test_merged)


miss_cols_df_test_keys = list(miss_cols_df_test_merged['Missing values columns'].keys())
medians_test = df_test_merged[miss_cols_df_test_keys].median()

df_test_merged[miss_cols_df_test_keys] = df_test_merged[miss_cols_df_test_keys].apply(lambda col: col.fillna(medians_test[col.name]))


X_test = df_test_merged.drop(['participant_id'], axis=1)


X_test_scaled = StandardScaler().fit_transform(X_test)


selected_indices_adhd = selector_adhd.get_support(indices=True)
X_test_selected_adhd = X_test_scaled[:, selected_indices_adhd]
y_test_pred_adhd_final = rf_model_adhd.predict(X_test_selected_adhd)


selected_indices_sex= selector_sex.get_support(indices=True)
X_test_selected_sex = X_test_scaled[:,selected_indices_sex]
y_test_pred_sex_final = lr_model_sex.predict(X_test_selected_sex)


submission = pd.DataFrame({
    'participant_id': df_test_merged['participant_id'],
    'ADHD_Outcome': y_test_pred_adhd_final,
    'Sex_F': y_test_pred_sex_final
})


submission


submission.to_csv('submmission.csv', index=False)

