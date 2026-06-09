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


!pip install --upgrade --force-reinstall numpy scipy scikit-learn


#Importing Necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE


#Reading data files

df_train = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')

df_test = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')

df_sample = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv')


df_train.head()


df_train.columns


df_train.info()


#Calculating null value percentage for the columns
null_percentage = (df_train.isnull().sum() / len(df_train)) * 100
null_percentage = null_percentage.sort_values(ascending=False).reset_index()
null_percentage.columns = ['Column', 'Null_Percentage']

print(null_percentage)


df_train.describe()


#Missing Value Distribution

missing_values = df_train.isnull().sum().sort_values(ascending=False)

plt.figure(figsize=(15, 6))
sns.barplot(x=missing_values[missing_values > 0].index, y=missing_values[missing_values > 0].values)
plt.xticks(rotation=45, ha="right")
plt.title("Missing Values per Feature")
plt.ylabel("Count")
plt.show()


#Distribution of columns with high null values
if 'mrd_hct' in df_train.columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(x='mrd_hct', data=df_train)
    plt.title("mrd_hct Distribution (efs)")
    plt.show()


#Distribution of columns with high null values
if 'cyto_score' in df_train.columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(x='cyto_score', data=df_train)
    plt.title("mrd_hct Distribution (efs)")
    plt.show()


#Distribution of columns with high null values
if 'tce_match' in df_train.columns:
    plt.figure(figsize=(10, 4))
    sns.countplot(x='tce_match', data=df_train)
    plt.title("tce_match Distribution (efs)")
    plt.show()


#Target variable
if 'efs' in df_train.columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(x='efs', data=df_train)
    plt.title("Target Variable Distribution (efs)")
    plt.show()


#Checking through Kde's
if "ID" in df_train.columns:
    df_num = df_train.select_dtypes(include=np.number).drop(columns=["ID"])
else:
    df_num = df_train.select_dtypes(include=np.number)

# Plot the KDE using histograms
plt.subplots(figsize=(20, 20))
for i, c in enumerate(df_num, 1):
    plt.subplot(10, 3, i)
    sns.histplot(df_num[c], kde=True)  # Add kde=True to include KDE in the histogram
    plt.title(c)
plt.tight_layout()
plt.show()


if 'efs_time' in df_train.columns:
    plt.figure(figsize=(6, 4))
    sns.histplot(df_train['efs_time'], kde=True, bins=30)
    plt.title("Distribution of Event-Free Survival Time (efs_time)")
    plt.xlabel("Time")
    plt.show()


# Ensure ID column is excluded
if "ID" in df_train.columns:
    df_num = df_train.select_dtypes(include=np.number).drop(columns=["ID"])
else:
    df_num = df_train.select_dtypes(include=np.number)

# df_num = df.select_dtypes(include=np.number)
plt.figure(figsize=(12,8))
sns.heatmap(df_num.corr(),annot=True)
plt.tight_layout()


# Dividing categorical and numerical columns

categorical_col = [col for col in df_train.columns if df_train[col].dtype == 'object']
numerical_col = [col for col in df_train.columns if df_train[col].dtype != 'object']

print("Categorical Columns:", categorical_col)
print("Numerical Columns:", numerical_col)


from scipy.stats import chi2_contingency

chi_square_results = {}

for col in categorical_col:
    contingency_table = pd.crosstab(df_train[col], df_train['efs'])
    chi2, p, _, _ = chi2_contingency(contingency_table)
    chi_square_results[col] = {'Chi2': chi2, 'p-value': p}

# Display results sorted by significance
chi_square_results_df = pd.DataFrame(chi_square_results).T.sort_values(by='p-value')
print(chi_square_results_df)


#Identifying Significant Categorical variable having association with target variable

significant_vars = chi_square_results_df[chi_square_results_df['p-value'] < 0.05].index.tolist()
print(f"\nSignificant Variables: {significant_vars}")

Insignificant_vars = chi_square_results_df[chi_square_results_df['p-value'] >= 0.05].index.tolist()
print(f"\nInsignificant Variables: {Insignificant_vars}")


drop_col = ['mrd_hct','tce_match', 'cyto_score']

df_train = df_train.drop(drop_col, axis=1)

df_train.columns


df_train['tce_div_match'] = df_train.groupby('efs')['tce_div_match'].transform(lambda x: x.fillna(x.mode()[0]))
df_train['tce_imm_match'] = df_train.groupby('efs')['tce_imm_match'].transform(lambda x: x.fillna(x.mode()[0]))
df_train['hla_high_res_10'] = df_train.groupby('efs')['hla_high_res_10'].transform(lambda x: x.fillna(x.mode()[0]))
df_train['hla_high_res_8'] = df_train.groupby('efs')['hla_high_res_8'].transform(lambda x: x.fillna(x.mode()[0]))
df_train['hla_high_res_6'] = df_train.groupby('efs')['hla_high_res_6'].transform(lambda x: x.fillna(x.mode()[0]))
df_train['hla_match_dqb1_high'] = df_train.groupby('efs')['hla_match_dqb1_high'].transform(lambda x: x.fillna(x.mode()[0]))
df_train['hla_low_res_10'] = df_train.groupby('efs')['hla_low_res_10'].transform(lambda x: x.fillna(x.mode()[0]))
df_train['diabetes'] = df_train.groupby('efs')['diabetes'].transform(lambda x: x.fillna(x.mode()[0]))


df_train['psych_disturb'] = df_train.groupby('efs')['psych_disturb'].transform(lambda x: x.fillna(x.mode()[0]))
df_train['pulm_moderate'] = df_train.groupby('efs')['pulm_moderate'].transform(lambda x: x.fillna(x.mode()[0]))
df_train['hepatic_mild'] = df_train.groupby('efs')['hepatic_mild'].transform(lambda x: x.fillna(x.mode()[0]))
df_train['hepatic_severe'] = df_train.groupby('efs')['hepatic_severe'].transform(lambda x: x.fillna(x.mode()[0]))
df_train['renal_issue'] = df_train.groupby('efs')['renal_issue'].transform(lambda x: x.fillna(x.mode()[0]))
df_train['donor_age'] = df_train['donor_age'].fillna(df_train['donor_age'].mean())
df_train['obesity'] = df_train.groupby('efs')['obesity'].transform(lambda x: x.fillna(x.mode()[0]))


df_train['prior_tumor'] = df_train.groupby('efs')['prior_tumor'].transform(lambda x: x.fillna(x.mode()[0]))
df_train['melphalan_dose'] = df_train.groupby('efs')['melphalan_dose'].transform(lambda x: x.fillna(x.mode()[0]))
df_train['karnofsky_score'] = df_train.groupby('efs')['karnofsky_score'].transform(lambda x: x.fillna(x.mode()[0]))
df_train['cmv_status'] = df_train.groupby('efs')['cmv_status'].transform(lambda x: x.fillna(x.mode()[0]))
df_train['ethnicity'] = df_train.groupby('efs') ['ethnicity'].transform(lambda x : x.fillna(x.mode()[0]))
df_train['comorbidity_Score'] = df_train.groupby('efs')['comorbidity_score'].transform(lambda x : x.fillna(x.mode()[0]))
df_train['sex_match'] = df_train.groupby('efs')['sex_match'].transform(lambda x : x.fillna(x.mode()[0]))


df_train['vent_hist'] = df_train.groupby('efs') ['vent_hist'].transform(lambda x : x.fillna(x.mode()[0]))
df_train['in_vivo_tcd'] = df_train.groupby('efs') ['in_vivo_tcd'].transform(lambda x : x.fillna(x.mode()[0]))
df_train['gvhd_proph'] = df_train.groupby('efs') ['gvhd_proph'].transform(lambda x : x.fillna(x.mode()[0]))
df_train['donor_related'] = df_train.groupby('efs')['donor_related'].transform(lambda x : x.fillna(x.mode()[0]))
df_train['dri_score'] = df_train.groupby('efs')['dri_score'].transform(lambda x: x.fillna(x.mode()[0]))
df_train['conditioning_intensity'] = df_train.groupby('efs')['conditioning_intensity'].transform(lambda x: x.fillna(x.mode()[0]))


df_train['hla_match_c_high'] = df_train.groupby('efs')['hla_match_c_high'].transform(lambda x: x.fillna(x.mode()[0]))
df_train['hla_match_a_high'] = df_train.groupby('efs')['hla_match_a_high'].transform(lambda x: x.fillna(x.mode()[0]))
df_train['hla_nmdp_6'] = df_train.groupby('efs')['hla_nmdp_6'].transform(lambda x: x.fillna(x.mode()[0]))
df_train['hla_match_dqb1_low'] = df_train.groupby('efs')['hla_match_dqb1_low'].transform(lambda x: x.fillna(x.mode()[0]))
df_train['hla_match_b_high'] = df_train.groupby('efs')['hla_match_b_high'].transform(lambda x: x.fillna(x.mode()[0]))


df_train['hla_low_res_8'] = df_train.groupby('efs')['hla_low_res_8'].transform(lambda x: x.fillna(x.mode()[0]))
df_train['hla_match_drb1_high'] = df_train.groupby('efs')['hla_match_drb1_high'].transform(lambda x: x.fillna(x.mode()[0]))
df_train['hla_low_res_6'] = df_train.groupby('efs')['hla_low_res_6'].transform(lambda x: x.fillna(x.mode()[0]))
df_train['hla_match_c_low'] = df_train.groupby('efs')['hla_match_c_low'].transform(lambda x: x.fillna(x.mode()[0]))
df_train['hla_match_drb1_low '] = df_train.groupby('efs')['hla_match_drb1_low'].transform(lambda x: x.fillna(x.mode()[0]))
df_train['hla_match_b_low'] = df_train.groupby('efs')['hla_match_b_low'].transform(lambda x: x.fillna(x.mode()[0]))


df_train['arrhythmia'] = df_train.groupby('efs')['arrhythmia'].transform(lambda x: x.fillna(x.mode()[0]))
df_train['pulm_severe'] = df_train.groupby('efs')['pulm_severe'].transform(lambda x: x.fillna(x.mode()[0]))
df_train['rituximab'] = df_train.groupby('efs')['rituximab'].transform(lambda x: x.fillna(x.mode()[0]))
df_train['hla_match_drb1_low'] = df_train.groupby('efs')['arrhythmia'].transform(lambda x: x.fillna(x.mode()[0]))
df_train['cyto_score_detail'] = df_train.groupby('efs')['cyto_score_detail'].transform(lambda x: x.fillna(x.mode()[0]))
df_train['peptic_ulcer'] = df_train.groupby('efs')['peptic_ulcer'].transform(lambda x: x.fillna(x.mode()[0]))
df_train['hla_match_a_low'] = df_train.groupby('efs')['hla_match_a_low'].transform(lambda x: x.fillna(x.mode()[0]))
df_train['rheum_issue'] = df_train.groupby('efs')['rheum_issue'].transform(lambda x: x.fillna(x.mode()[0]))
df_train['comorbidity_score'] = df_train.groupby('efs')['comorbidity_score'].transform(lambda x: x.fillna(x.median()))
df_train['cardiac'] = df_train.groupby('efs')['cardiac'].transform(lambda x: x.fillna(x.mode()[0]))


null_columns = df_train.isnull().sum()
null_columns = null_columns[null_columns > 0]
print(null_columns)


category_col = ['dri_score', 'psych_disturb', 'diabetes', 'tbi_status', 'arrhythmia', 'graft_type', 'vent_hist', 'renal_issue', 'pulm_severe', 'prim_disease_hct', 'cmv_status', 'tce_imm_match', 'rituximab', 'prod_type', 'cyto_score_detail', 'conditioning_intensity', 'ethnicity', 'obesity', 'in_vivo_tcd', 'hepatic_severe', 'prior_tumor', 'peptic_ulcer', 'gvhd_proph', 'rheum_issue', 'sex_match', 'race_group', 'hepatic_mild', 'tce_div_match', 'donor_related', 'melphalan_dose', 'cardiac', 'pulm_moderate']


def category_onehot_multicolumns(df, multicolumns):
    df_result = df.copy() 

    for i, field in enumerate(multicolumns):
        print(f"Processing column: {field}")

        df_onehot = pd.get_dummies(df_result[field], drop_first=True).astype(int)

        df_result.drop([field], axis=1, inplace=True)

        df_result = pd.concat([df_result, df_onehot], axis=1)

    return df_result

categorical_columns = category_col  
df_train = category_onehot_multicolumns(df_train, categorical_columns)


#For hla_match_drb1_low
df_train = category_onehot_multicolumns(df_train,['hla_match_drb1_low'])


df_train.head()


def calc_vif(X):
    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return vif

pd.set_option('display.max_rows', None) 
pd.set_option('display.max_columns', None)  
pd.set_option('display.width', None)  
pd.set_option('display.max_colwidth', None)  


vif_result = calc_vif(df_train)
# Display the VIF results
# Assuming vif_result is a DataFrame with a 'VIF' column
sorted_df = vif_result.sort_values(by='VIF', ascending=False)
print(sorted_df)


columns_multicollinear = ['hla_low_res_6', 'hla_low_res_8', 'hla_high_res_6', 'year_hct', 'hla_high_res_10',
                          'hla_low_res_6', 'hla_high_res_8', 'comorbidity_score', 'hla_low_res_10','Yes', 'Not done',
                          'hla_nmdp_6', 'FK+ MMF +- others', 'karnofsky_score', 'hla_match_b_low', 'hla_match_c_high', 'hla_match_c_low', 'hla_match_drb1_low ', 'hla_match_drb1_high',
                          'P/P', 'hla_match_a_high', 'hla_match_a_low', 'Permissive mismatched', 'Related', 'hla_match_dqb1_low', 'hla_match_dqb1_high', 'hla_match_b_high',
                          'Intermediate','Peripheral blood' ]


df_train = df_train.drop(columns_multicollinear, axis=1)


df_train.shape


import numpy as np

corr_matrix = df_train.corr()
high_corr = corr_matrix[abs(corr_matrix) > 0.5]

plt.figure(figsize=(10, 8))
sns.heatmap(high_corr, cmap='coolwarm')
plt.title("Filtered Correlation Heatmap")
plt.show()


df_target = df_train['efs']
df_train = df_train.drop(columns=['efs'])
df_id = df_train['ID']
df_train = df_train.drop(columns=['ID'])


scaler = MinMaxScaler()
df_train_scaled = scaler.fit_transform(df_train)
print(df_train_scaled)


df_final = pd.concat([df_id, pd.DataFrame(df_train_scaled, columns=df_train.columns)], axis=1)
df_final = pd.concat([df_final, df_target], axis=1)
df_final.head()


df_test.head()


print(df_test.shape)


df_test.isnull().sum()


#Calculating null value percentage for the columns
null_percentage = (df_test.isnull().sum() / len(df_test)) * 100
null_percentage = null_percentage.sort_values(ascending=False).reset_index()
null_percentage.columns = ['Column', 'Null_Percentage']

print(null_percentage)


#These three columns are already dropped in train data so dropping here
drop_col = ['mrd_hct','tce_match', 'cyto_score']

df_test = df_test.drop(drop_col, axis=1)


#Calculating null value percentage for the columns
null_percentage = (df_test.isnull().sum() / len(df_test)) * 100
null_percentage = null_percentage.sort_values(ascending=False).reset_index()
null_percentage.columns = ['Column', 'Null_Percentage']

# Filter columns with null percentage > 0
columns_with_nulls = null_percentage[null_percentage['Null_Percentage'] > 0]

print(columns_with_nulls)


df_test['donor_age'] = df_test['donor_age'].transform(lambda x: x.fillna(x.mean()))
df_test['conditioning_intensity'] = df_test['conditioning_intensity'].transform(lambda x: x.fillna(x.mode()[0]))
df_test['cyto_score_detail'] = df_test['cyto_score_detail'].transform(lambda x: x.fillna(x.mode()[0]))
df_test['hla_high_res_10'] = df_test['hla_high_res_10'].transform(lambda x: x.fillna(x.mode()[0]))
df_test['tce_imm_match'] = df_test['tce_imm_match'].transform(lambda x: x.fillna(x.mode()[0]))
df_test['hla_match_c_high'] = df_test['hla_match_c_high'].transform(lambda x: x.fillna(x.mode()[0]))
df_test['hla_high_res_8'] = df_test['hla_high_res_8'].transform(lambda x: x.fillna(x.mode()[0]))
df_test['tce_div_match'] = df_test['tce_div_match'].transform(lambda x: x.fillna(x.mode()[0]))


df_test.isnull().sum()


category_col = ['dri_score', 'psych_disturb', 'diabetes', 'tbi_status', 'arrhythmia', 'graft_type', 'vent_hist', 'renal_issue', 'pulm_severe', 'prim_disease_hct', 'cmv_status', 'tce_imm_match', 'rituximab', 'prod_type', 'cyto_score_detail', 'conditioning_intensity', 'ethnicity', 'obesity', 'in_vivo_tcd', 'hepatic_severe', 'prior_tumor', 'peptic_ulcer', 'gvhd_proph', 'rheum_issue', 'sex_match', 'race_group', 'hepatic_mild', 'tce_div_match', 'donor_related', 'melphalan_dose', 'cardiac', 'pulm_moderate']


import pandas as pd

def encode_test_data_with_training_schema(train_encoded, test_data, categorical_columns):
  
    test_encoded = test_data.copy()

    for column in categorical_columns:
        test_onehot = pd.get_dummies(test_encoded[column], prefix=column, drop_first=True)

        test_encoded.drop([column], axis=1, inplace=True)

        train_onehot_columns = [col for col in train_encoded.columns if col.startswith(column + "_")]
        test_onehot = test_onehot.reindex(columns=train_onehot_columns, fill_value=0)

        test_encoded = pd.concat([test_encoded, test_onehot], axis=1)

    test_encoded = test_encoded.reindex(columns=train_encoded.columns, fill_value=0)

    return test_encoded


categorical_columns = category_col
test_encoded = encode_test_data_with_training_schema(df_final, df_test, categorical_columns)


test_encoded.shape


df_final.shape


df_id_1 = test_encoded['ID']
test_encoded = test_encoded.drop(columns=['ID'])

scaler = MinMaxScaler()
test_encoded_scaled = scaler.fit_transform(test_encoded)


df_test_final = pd.concat([df_id_1, pd.DataFrame(test_encoded_scaled, columns=test_encoded.columns)], axis=1)
df_test_final = df_test_final.drop(columns = ['efs'])
df_final = df_final.loc[:, ~df_final.columns.duplicated()]


df_train_final = df_final
df_train_final.shape


df_test_final.shape


df_train_final.head()


X = df_train_final.drop(columns=['efs', 'ID']) #Removing ID
Y = df_train_final['efs']

smote = SMOTE(random_state=42)
X_resampled, Y_resampled = smote.fit_resample(X, Y)

print("Original shape of X_train:", df_train_final.shape)
print("Resampled shape of X_train:", X_resampled.shape)


duplicate_columns = df_train_final.columns[df_train_final.columns.duplicated()]
print(f"Duplicate columns: {duplicate_columns}")

print(f"Shape of X_resampled: {X_resampled.shape}")
print(f"Number of columns in df_train_final: {len(df_train_final.columns)}")


X_resampled_df = pd.DataFrame(X_resampled, columns = df_train_final.columns)
y_resampled_df = pd.Series(Y_resampled, name="efs")


df_train_resampled = pd.concat([X_resampled_df, y_resampled_df], axis=1)
df_train_resampled.head()


df_train_resampled = df_train_resampled.drop(columns = ['ID', 'efs'])
df_train_resampled.head()


df_train_resampled = pd.concat([df_train_resampled, y_resampled_df], axis=1)
df_train_resampled.head()


df_test_final = df_test_final.drop(columns = ['ID'])
print(df_train_resampled.shape)
print(df_test_final.shape)


pip install lifelines


from sklearn.model_selection import KFold
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

# Initialize KFold cross-validation
n_splits = 3  # Number of folds
cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)


X_train = df_train_resampled.drop(columns=['efs', 'efs_time'])  # Features
y_train = df_train_resampled[['efs_time', 'efs']]  # Target (efs_time and efs)


X_train.head(3)


# # Store C-index scores for each fold
# c_index_scores = []

# # Perform k-fold cross-validation
# for train_index, val_index in cv.split(X_train):
#     # Split the data into training and validation sets
#     X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
#     y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

#     # Prepare the training data for CoxPH
#     df_train_fold = X_train_fold.copy()
#     df_train_fold['efs_time'] = y_train_fold['efs_time']
#     df_train_fold['efs'] = y_train_fold['efs']

#     # Initialize and fit the CoxPH model
#     cph = CoxPHFitter()
#     cph.fit(df_train_fold, duration_col='efs_time', event_col='efs')

#     # Predict risk scores for the validation set
#     val_predictions = cph.predict_partial_hazard(X_val_fold)

#     # Calculate the C-index for the validation set
#     c_index = concordance_index(
#         event_times=y_val_fold['efs_time'],  # Actual event times
#         predicted_scores=-val_predictions,  # Negative risk scores (higher risk = shorter survival)
#         event_observed=y_val_fold['efs']  # Event indicator
#     )

#     # Store the C-index score
#     c_index_scores.append(c_index)
#     print(f"Fold C-index: {c_index:.4f}")


# mean_c_index = np.mean(c_index_scores)
# print(f"Mean C-index across {n_splits} folds: {mean_c_index:.4f}")


#Final Training
# Prepare the entire training data for CoxPH
df_train_final = X_train.copy()
df_train_final['efs_time'] = y_train['efs_time']
df_train_final['efs'] = y_train['efs']


X_test = df_test_final[X_train.columns]

X_test = X_test.loc[:, ~X_test.columns.duplicated()]


cph_final = CoxPHFitter()
cph_final.fit(df_train_final, duration_col='efs_time', event_col='efs')


test_predictions = cph_final.predict_partial_hazard(X_test)


test_predictions


submission = pd.DataFrame({
    'ID': df_test['ID'],  # Replace 'id' with the actual ID column in the test data
    'predicted_risk': test_predictions
})
submission.to_csv('submission.csv', index=False)


submission.head()




