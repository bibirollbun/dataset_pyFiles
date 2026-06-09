# import Libraries
# Plan
import pandas as pd 
import numpy as np 
from scipy.stats import chi2_contingency
from lifelines import KaplanMeierFitter
from statsmodels.formula.api import ols
from statsmodels.formula.api import logit
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import learning_curve
import seaborn as sns
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib 


# Replace missing values with NaN
nulls_values = ['','Other','TBD','Not tested','Not done','N/A - disease not classifiable','unknown dose','No drugs reported','N/A, F(pre-TED) not submitted']
data_train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv" , na_values=nulls_values)
data_test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")


# Show Data
print(data_train.shape)


# Info 
print(data_train.info())


# Check if there are duplicate rows?
print(data_train.duplicated().sum())


#Check if there are any missing values? 
missing_values = data_train.isna().sum()
print(missing_values)


missing_columns = missing_values[missing_values > 9000].index

plt.figure(figsize=(12,8))
sns.heatmap(data_train[missing_columns].isnull(), cbar=True)
 
plt.tight_layout()
plt.show


# Using OrdinalEncoder to convert categorical text values into numeric values for each column.
def ordinalencoder_data(data):

    # Select the categorical columns (object type) from the dataframe
    categorical_cols = data.select_dtypes(include=['object']).columns

    # Create a mask to identify missing values (NaN) in the categorical columns
    mask = data[categorical_cols].isna()

    # Fill missing values with the string 'missing' to ensure no NaN values before encoding
    temp_data = data[categorical_cols].fillna('missing')

    # Initialize the OrdinalEncoder, which will convert categorical values to numerical labels
    oe = OrdinalEncoder()

    # Fit the encoder to the data and transform the categorical columns into numerical labels
    encoded_data = oe.fit_transform(temp_data)

    # Convert the encoded data to a DataFrame, keeping the original column names and specifying the dtype as 'Int64'
    encoded_series = pd.DataFrame(encoded_data, columns=categorical_cols, dtype='Int64')

    # Restore the missing values (NaN) in the original positions, using pd.NA to indicate missing values
    encoded_series[mask] = pd.NA

    # Update the original dataframe with the encoded columns while preserving the missing value positions
    data[categorical_cols] = encoded_series

    return data


# Apply ordinal encoding to the training and test datasets  
data_train = ordinalencoder_data(data_train)
data_test = ordinalencoder_data(data_test)


print(data_train.info())


# Fill missing values in the columns selected for analysis
def fillna_data(data):

    # Save the original data types of the columns in a copy to refer back to after imputation
    original_dtypes = data.dtypes.copy()

    # Initialize the IterativeImputer to fill missing values using multiple imputation iterations
    imputer = SimpleImputer(strategy='most_frequent')

    # Apply the imputer to the data and perform the imputation
    imputer_data = imputer.fit_transform(data)

    # Convert the imputed data back into a DataFrame with the original column names
    data = pd.DataFrame(imputer_data, columns=data.columns)

    # Convert columns that were originally of type 'int64' or 'Int64' back to integer type after imputation
    data = data.apply(lambda col: col.astype(int) if original_dtypes[col.name] in ['int64','Int64'] else round(col , 1))

    return data


# Fill missing values in the training and test datasets  
data_train = fillna_data(data_train)
data_test = fillna_data(data_test)


# Calculate and print the total number of missing values per column (if needed) after filling
missing_values = data_train.isna().sum()
print(f"Missing values filling: {missing_values}")


print(data_train.info())


# Statistical Summary of the Data
print(data_train.describe())


# What is the distribution of the dri_score and psych_disturb and cyto_score and diabetes ?

plt.figure(figsize=(12,8))

plt.subplot(2,2,1)

dri_score_counts = data_train['dri_score'].value_counts().sort_index()

dri_score_counts.plot(kind='bar' , color='skyblue' , edgecolor='black')
plt.title("Distribution of DRI Score")
plt.xlabel("DRI Score")
plt.ylabel("Count")

plt.subplot(2,2,2)

psych_disturb = data_train['psych_disturb'].value_counts().sort_index()

psych_disturb.plot(kind='bar' , color='skyblue' , edgecolor='black')
plt.title("Distrubution of Psych Disturb")
plt.xlabel("Psych Disturb")
plt.ylabel("Count")

plt.subplot(2,2,3)

cyto_score = data_train['cyto_score'].value_counts().sort_index()

cyto_score.plot(kind='bar' , color='skyblue' , edgecolor='black')
plt.title("Distrubution of Cyto Score")
plt.xlabel("Cyto Score")
plt.ylabel("Count")

plt.subplot(2,2,4)

diabetes = data_train['diabetes'].value_counts().sort_index()

diabetes.plot(kind='bar' , color='skyblue' , edgecolor='black')
plt.title("Distrubution of Diabetes")
plt.xlabel("Diabetes")
plt.ylabel("Count")

plt.tight_layout()
plt.show()


# What is the distribution of the hla_match_c_high and hla_high_res_8 and hla_low_res_6 and hla_high_res_6 ?

plt.figure(figsize=(12,8))

plt.subplot(2,2,1)

sns.countplot(data=data_train , x='hla_match_c_high' , color='blue')
plt.title('Distribution of HLA Match C High')
plt.xlabel('HLA Match C High')
plt.ylabel('Count')


plt.subplot(2,2,2)

sns.countplot(data=data_train , x='hla_high_res_8' , color='blue')
plt.title('Distrubution of HLA High Res 8')
plt.xlabel("HLA High Res 8")
plt.ylabel("Count")

plt.subplot(2,2,3)

sns.countplot(data=data_train , x='hla_low_res_6' , color='blue')
plt.title("Distrubution of HLA Low Res 6")
plt.xlabel("HLA Low Res 6")
plt.ylabel("Count")

plt.subplot(2,2,4)

sns.countplot(data=data_train , x='hla_high_res_6' , color='blue')
plt.title('Distrubution of HLA High Res 6')
plt.xlabel("HLA High Res 6")
plt.ylabel("Count")

plt.tight_layout()
plt.show()


# What is the distribution of the tbi_status and arrhythmia and graft_type and vent_hist ?

plt.figure(figsize=(12,8))

plt.subplot(2,2,1)

tbi_status = data_train['tbi_status'].value_counts().sort_index()

tbi_status.plot(kind='bar' , color='skyblue' , edgecolor='black')
plt.title("Distrubution of TDI Status")
plt.xlabel("TDI Status")
plt.ylabel("Count")

plt.subplot(2,2,2)

arrhythmia = data_train['arrhythmia'].value_counts().sort_index()

arrhythmia.plot(kind='bar' , color='skyblue' , edgecolor='black')
plt.title("Distrubution of Arrhythmia")
plt.xlabel("Arrhythmia")
plt.ylabel("Count")

plt.subplot(2,2,3)

graft_type = data_train['graft_type'].value_counts().sort_index()

graft_type.plot(kind='bar' , color='skyblue' , edgecolor='black')
plt.title("Distrubution of Graft Type")
plt.xlabel("Graft type")
plt.ylabel("Count")

plt.subplot(2,2,4)

vent_hist = data_train['vent_hist'].value_counts().sort_index()

vent_hist.plot(kind='bar' , color='skyblue' , edgecolor='black')
plt.title("Distrubution of Vent Hist")
plt.xlabel("Vent Hist")
plt.ylabel("Count")

plt.tight_layout()
plt.show()



# What is the distribution of the renal_issue and pulm_severe and prim_disease_hct and cmv_status ?
        
plt.figure(figsize=(12,8))

plt.subplot(2,2,1)

renal_issue = data_train['renal_issue'].value_counts().sort_index()

renal_issue.plot(kind='bar' , color='skyblue' , edgecolor='black')
plt.title("Distrubution of Renal Issue")
plt.xlabel("Renal Issue")
plt.ylabel("Count")

plt.subplot(2,2,2)

pulm_severe = data_train['pulm_severe'].value_counts().sort_index()

pulm_severe.plot(kind='bar' , color='skyblue' , edgecolor='black')
plt.title("Distrubution of Pulm Severe")
plt.xlabel("Pulm Severe")
plt.ylabel("Count")

plt.subplot(2,2,3)

prim_disease_hct = data_train['prim_disease_hct'].value_counts().sort_index()

prim_disease_hct.plot(kind='bar' , color='skyblue' , edgecolor='black')
plt.title("Distrubution of Prim Disease HCT")
plt.xlabel("Prim Disease HCT")
plt.ylabel("Count")

plt.subplot(2,2,4)

cmv_status = data_train['cmv_status'].value_counts().sort_index()

cmv_status.plot(kind='bar' , color='skyblue' , edgecolor='black')
plt.title("Distrubution of CMV Status")
plt.xlabel("CMV Status")
plt.ylabel("Count")
plt.tight_layout()
plt.show()


# What is the distribution of the hla_high_res_10 and hla_match_dqb1_high and tce_imm_match and hla_nmdp_6 ?

plt.figure(figsize=(12,8))

plt.subplot(2,2,1)

sns.countplot(data=data_train , x='hla_high_res_10' , color='blue')
plt.title('Distrubution of HLA High Res 10')
plt.xlabel("HLA High Res 10")
plt.ylabel("Count")

plt.subplot(2,2,2)

sns.countplot(data=data_train , x='hla_match_dqb1_high' , color='blue')
plt.title('Distrubution of HLA Match DQB1 High')
plt.xlabel("HLA Match DQB1 High")
plt.ylabel("Count")

plt.subplot(2,2,3)

sns.countplot(data=data_train , x='tce_imm_match' , color='blue')
plt.title('Distrubution of TCE IMM Match')
plt.xlabel("ICE IMM Match")
plt.ylabel("Count")

plt.subplot(2,2,4)

sns.countplot(data=data_train , x='hla_nmdp_6' , color='blue')
plt.title('Distrubution of HLA NMDP 6')
plt.xlabel("HLA NMDP 6")
plt.ylabel("Count")

plt.tight_layout()
plt.show()



# What is the distribution of the rituximab  and prod_type and conditioning_intensity and ethnicity
        
plt.figure(figsize=(12,8))

plt.subplot(2,2,1)

rituximab = data_train['rituximab'].value_counts().sort_index()

rituximab.plot(kind='bar' , color='skyblue' , edgecolor='black')
plt.title("Distribution of Rituximab")
plt.xlabel("Rituximab")
plt.ylabel("Count")

plt.subplot(2,2,2)

prod_type = data_train['prod_type'].value_counts().sort_index()

prod_type.plot(kind='bar' , color='skyblue' , edgecolor='black')
plt.title("Distribution of Prod Type")
plt.xlabel("Prod Type")
plt.ylabel("Count")

plt.subplot(2,2,3)

conditioning_intensity = data_train['conditioning_intensity'].value_counts().sort_index()

conditioning_intensity.plot(kind='bar' , color='skyblue' , edgecolor='black')
plt.title("Distribution of Conditioning Intensity")
plt.xlabel("Conditioning Intensity")
plt.ylabel("Count")

plt.subplot(2,2,4)

ethnicity = data_train['ethnicity'].value_counts().sort_index()

ethnicity.plot(kind='bar' , color='skyblue' , edgecolor='black')
plt.title("Distribution of Ethnicity")
plt.xlabel("Ethnicity")
plt.ylabel("Count")

plt.tight_layout()
plt.show()




# What is the distribution of the hla_match_c_low and  hla_match_drb1_low and hla_match_dqb1_low and hla_match_a_high ?
        
plt.figure(figsize=(12,8))

plt.subplot(2,2,1)

sns.countplot(data=data_train , x='hla_match_c_low' , color='blue')
plt.title('Distrubution of HLA Match C Low')
plt.xlabel("HLA Match C Low")
plt.ylabel("Count")

plt.subplot(2,2,2)

sns.countplot(data=data_train , x='hla_match_drb1_low' , color='blue')
plt.title('Distrubution of HLA Match DRB1  Low')
plt.xlabel("HLA Match DRB1 Low")
plt.ylabel("Count")

plt.subplot(2,2,3)

sns.countplot(data=data_train , x='hla_match_dqb1_low' , color='blue')
plt.title('Distrubution of HLA Match DQB1 Low')
plt.xlabel("HLA Match DQB1 Low")
plt.ylabel("Count")

plt.subplot(2,2,4)

sns.countplot(data=data_train , x='hla_match_a_high' , color='blue')
plt.title('Distrubution of HLA Match A High')
plt.xlabel("HLA Match A High")
plt.ylabel("Count")

plt.tight_layout()
plt.show()



# What is the distribution of the year_hct and obesity and mrd_hct and in_vivo_tcd ?

plt.figure(figsize=(12,8))

plt.subplot(2,2,1)

sns.histplot(data=data_train['year_hct'] , bins=10 , kde=False , color='skyblue')
plt.title('Distribution of Year HCT')
plt.xlabel('Year of HCT')
plt.ylabel('Frequency')

plt.subplot(2,2,2)

obesity = data_train['obesity'].value_counts().sort_index()

obesity.plot(kind='bar' , color='skyblue' , edgecolor='black')
plt.title("Distribution of Obesity")
plt.xlabel("Obesity")
plt.ylabel("Count")

plt.subplot(2,2,3)

mrd_hct = data_train['mrd_hct'].value_counts().sort_index()

mrd_hct.plot(kind='bar' , color='skyblue' , edgecolor='black')
plt.title("Distribution of MRD HCT")
plt.xlabel("MRD HCT")
plt.ylabel("Count")

plt.subplot(2,2,4)

in_vivo_tcd = data_train['in_vivo_tcd'].value_counts().sort_index()

in_vivo_tcd.plot(kind='bar' , color='skyblue' , edgecolor='black')
plt.title("Distribution of IN VIVO TCD")
plt.xlabel("IN VIVO TCD")
plt.ylabel("Count")

plt.tight_layout()
plt.show()



# What is the distribution of the tce_match and hepatic_severe and donor_age and prior_tumor ?

plt.figure(figsize=(12,8))

plt.subplot(2,2,1)

tce_match = data_train['tce_match'].value_counts().sort_index() 

tce_match.plot(kind='bar' , color='skyblue' , edgecolor='black')
plt.title("Distribution of TCE Match")
plt.xlabel("TCE Match")
plt.ylabel("Count")

plt.subplot(2,2,2)

hepatic_severe = data_train['hepatic_severe'].value_counts().sort_index()
hepatic_severe.plot(kind='bar' , color='skyblue' , edgecolor='black')
plt.title("Distribution of Hepatic Severe")
plt.xlabel("Hepatic Severe")
plt.ylabel("Count")

plt.subplot(2,2,3)

sns.violinplot(x=data_train['donor_age'] , color='skyblue')
plt.title("Distribution of Donor Age")
plt.xlabel("Donor Age")
plt.ylabel("Count")

plt.subplot(2,2,4)

prior_tumor = data_train['prior_tumor'].value_counts().sort_index()
prior_tumor.plot(kind='bar' , color='skyblue' , edgecolor='black')
plt.title("Distribution of Prior Tumor")
plt.xlabel("Prior Tumor")
plt.ylabel("Count")

plt.tight_layout()
plt.show()


# What is the distribution of the hla_match_b_low and age_at_hct and hla_match_a_low and hla_match_b_high ?
        
plt.figure(figsize=(12,8))

plt.subplot(2,2,1)

sns.countplot(data=data_train , x='hla_match_b_low' , color='blue')
plt.title('Distrubution of HLA Match D Low')
plt.xlabel("HLA Match D Low")
plt.ylabel("Count")

plt.subplot(2,2,2)

sns.violinplot(x=data_train['age_at_hct'] , color='blue')
plt.title("Distribution of Age At HCT")
plt.xlabel("Donor Age At HCT")
plt.ylabel("Count")

plt.subplot(2,2,3)

sns.countplot(data=data_train , x='hla_match_a_low' , color='blue')
plt.title('Distrubution of HLA Match A Low')
plt.xlabel("HLA Match A Low")
plt.ylabel("Count")

plt.subplot(2,2,4)

sns.countplot(data=data_train , x='hla_match_b_high' , color='blue')
plt.title('Distrubution of HLA Match D High')
plt.xlabel("HLA Match D High")
plt.ylabel("Count")

plt.tight_layout()
plt.show()



# What is the distribution of the peptic_ulcer and gvhd_proph and rheum_issue and sex_match ?
        
plt.figure(figsize=(12,8))

plt.subplot(2,2,1)

peptic_ulcer = data_train['peptic_ulcer'].value_counts().sort_index()

peptic_ulcer.plot(kind='bar' , color='skyblue' , edgecolor='black')
plt.title('Distrubution of Peptic Ulcer')
plt.xlabel("Peptic Ulcer")
plt.ylabel("Count")

plt.subplot(2,2,2)

gvhd_proph = data_train['gvhd_proph'].value_counts().sort_index()

gvhd_proph.plot(kind='bar' , color='skyblue' , edgecolor='black')
plt.title('Distrubution of Gvhd Proph')
plt.xlabel("Gvhd Proph")
plt.ylabel("Count")

plt.subplot(2,2,3)

rheum_issue = data_train['rheum_issue'].value_counts()

rheum_issue.plot(kind='bar' , color='skyblue' , edgecolor='black')
plt.title('Distrubution of Rheum Issue')
plt.xlabel("Rheum Issue")
plt.ylabel("Count")

plt.subplot(2,2,4)

sex_match = data_train['sex_match'].value_counts().sort_index()

sex_match.plot(kind='bar' , color='skyblue' , edgecolor='black')
plt.title('Distrubution of Sex Match')
plt.xlabel("Sex Match")
plt.ylabel("Count")

plt.tight_layout()
plt.show()
        


# What is the distribution of the tce_div_match and hla_low_res_8 and hla_match_drb1_high hla_low_res_10 ?

plt.figure(figsize=(12,8))

plt.subplot(2,2,1)

tce_div_match = data_train['tce_div_match'].value_counts().sort_index()
tce_div_match.plot(kind='bar' , color='blue' , edgecolor='black')
plt.title('Distrubution of TCE DIV Match')
plt.xlabel("TCE DIV Match")
plt.ylabel("Count")

plt.subplot(2,2,2)

sns.countplot(data=data_train , x='hla_low_res_8' , color='blue')
plt.title('Distrubution of HLA Low Res 8')
plt.xlabel("HLA Low  Res Low")
plt.ylabel("Count")

plt.subplot(2,2,3)

sns.countplot(data=data_train , x='hla_match_drb1_high' , color='blue')
plt.title('Distrubution of HLA Match DRB1 High')
plt.xlabel("HLA Match DRB1 High")
plt.ylabel("Count")

plt.subplot(2,2,4)

sns.countplot(data=data_train , x='hla_low_res_10' , color='blue')
plt.title('Distrubution of HLA LOW RES 10')
plt.xlabel("HLA LOW RES 10")
plt.ylabel("Count")

plt.tight_layout()
plt.show()


# What is the distribution of the race_group and comorbidity_score and karnofsky_score and hepatic_mild ?

plt.figure(figsize=(12,8))

plt.subplot(2,2,1)

race_group = data_train['race_group'].value_counts().sort_index()

race_group.plot(kind='bar' , color='skyblue' , edgecolor='black')
plt.title('Distrubution of Race Group')
plt.xlabel("Race Group")
plt.ylabel("Count")

plt.subplot(2,2,2)

sns.countplot(data=data_train , x='comorbidity_score' , color='skyblue')
plt.title('Distrubution of Comorbidity Score')
plt.xlabel("Comorbidity Score")
plt.ylabel("Count")

plt.subplot(2,2,3)

sns.countplot(data=data_train , x='karnofsky_score' , color='skyblue')
plt.title('Distrubution of Karnofsky_Score')
plt.xlabel("Karnofsky Score")
plt.ylabel("Count")

plt.subplot(2,2,4)

hepatic_mild = data_train['hepatic_mild'].value_counts().sort_index()

hepatic_mild.plot(kind='bar' , color='skyblue' , edgecolor='black')
plt.title('Distrubution of Hepatic Mild')
plt.xlabel("Hepatic Mild")
plt.ylabel("Count")

plt.tight_layout()
plt.show()


# What is the distribution of the donor_related and melphalan_dose and cardiac and pulm_moderate ?

plt.figure(figsize=(12,8))

plt.subplot(2,2,1)

donor_related = data_train['donor_related'].value_counts().sort_index()

donor_related.plot(kind='bar' , color='skyblue' , edgecolor='black')
plt.title('Distrubution of Donor Related')
plt.xlabel("Donor Related")
plt.ylabel("Count")

plt.subplot(2,2,2)

melphalan_dose = data_train['melphalan_dose'].value_counts().sort_index()

melphalan_dose.plot(kind='bar' , color='skyblue' , edgecolor='black')
plt.title('Distrubution of Melphalan Dose')
plt.xlabel("Melphalan Dose")
plt.ylabel("Count")

plt.subplot(2,2,3)

cardiac = data_train['cardiac'].value_counts().sort_index()

cardiac.plot(kind='bar' , color='skyblue' , edgecolor='black')
plt.title('Distrubution of Cardiac')
plt.xlabel("Cardiac")
plt.ylabel("Count")

plt.subplot(2,2,4)

pulm_moderate = data_train['pulm_moderate'].value_counts().sort_index()

pulm_moderate.plot(kind='bar' , color='skyblue' , edgecolor='black')
plt.title('Distrubution of Pulm Moderate')
plt.xlabel("Pulm Moderate")
plt.ylabel("Count")

plt.tight_layout()
plt.show()




# What is the distribution of the efs and efs_time ?

plt.figure(figsize=(12,8))

plt.subplot(2,2,1)

efs = data_train['efs'].value_counts().sort_index()

efs.plot(kind='bar' , color='skyblue' , edgecolor='black')
plt.title('Distrubution of EFS')
plt.xlabel("EFS")
plt.ylabel("Count")

plt.subplot(2,2,2)

sns.kdeplot(data_train['efs_time'], fill=True)
plt.title('Kernel Density Estimation of efs_time (Time to Event-Free Survival)')
plt.xlabel('Time (months)')
plt.ylabel('Density')

plt.tight_layout()
plt.show()



# 5 - Is there a relationship between the 'dri_score' column and the and the 'efs' or 'efs_time' columns?

# Test Chi-Squared 

# Create a table of frequencies between the two taxonomic columns
contingency_table = pd.crosstab(data_train['dri_score'] , data_train['efs'])


# Apply the chi-square test
chi2_stat , p_value , dof , expected = chi2_contingency(contingency_table)

# print the results 
print(f"Chi-squared statistic: {chi2_stat}")
print(f"P-value: {p_value}")
print(f"Degrees of freedom: {dof}")
print(f"Expected Frequencies: {expected}")


# Logistic Regression Model using the Logit function ?

# Build a logistic regression model
model_logistic = logit(formula='efs ~ C(dri_score)' , data=data_train).fit()
print(model_logistic.summary())


# Regression Model using the OLS function ?

# Model with categorical variable dri_score
model_ols = ols(formula='efs_time ~ C(dri_score)' , data=data_train).fit()
print(model_ols.summary())


# Function to plot Kaplan-Meier survival curve for any categorical group
def plot_kaplan_meier(group_column , plot_title):

    kaplanmeierfitter = KaplanMeierFitter()

    # Loop through each unique value of the group column
    for group in sorted(data_train[group_column].unique()):
        # Filter the data for the current group
        group_data = data_train[data_train[group_column] == group]
        
        
        # Fit the Kaplan-Meier estimator
        kaplanmeierfitter.fit(group_data['efs_time'], event_observed=group_data['efs'], label=f"{group_column} {group}")
        
        # Plot the survival function
        kaplanmeierfitter.plot_survival_function(ci_show=False)

    # Set the title and labels for the plot
    plt.title(plot_title)
    plt.xlabel("Time(months)")  # Label for the x-axis
    plt.ylabel("Survival Probability")  # Label for the y-axis
    
    # Adjust the layout to avoid clipping
    plt.tight_layout()
    # Save or show the plot
    plt.show()


# Plot Kaplan-Meier curve for 'dri_score' column
plot_kaplan_meier('dri_score', 'Kaplan-Meier Survival Curve for Cyto-Score')


# 6-  Is there a relationship between the 'Psych-disturd' column and the 'diabetes'  and 'arrhythmia' 
# and renal-issue' and efs' or 'efs_time' columns?

# This function performs a Chi-squared test of independence between two categorical variables.
def chi2_contingency_test(data1, data2):
    # Create a contingency table using the specified columns from the dataset ('data1' and 'data2').
    contingency_table = pd.crosstab(data_train[data1], data_train[data2])

    # Perform the Chi-squared test using the contingency table.
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

    # Calculate Cramér's V
    n = contingency_table.values.sum()  # Total number of observations
    k = min(contingency_table.shape)  # Use the largest dimension of the table to calculate Cramér's V
    v = np.sqrt(chi2_stat / (n * (k - 1)))

    # Title
    print(f"Title: {data1}")
    # Print the Chi-squared statistic to show the strength of the association.
    print(f"Chi2: {chi2_stat}")
    # Print the p-value to determine the statistical significance of the association.
    print(f"P-value: {p_value}")
    # Print the degrees of freedom used in the test.
    print(f"Degrees of Freedom: {dof}")
    # Print the expected frequencies to compare against the observed frequencies.
    print(f"Expected Frequencies: {expected}")
    # Print Cramér's V to show the strength of the association between the variables.
    print(f"Cramér's V: {v}")



# Test the function with 'Psych-disturd' as the first variable and 'efs' as the second variable.
# This checks if there's a significant association between these two categorical variables.
chi2_contingency_test('psych_disturb', 'efs')



# Test the function with 'diabetes' as the first variable and 'efs' as the second variable.
chi2_contingency_test('diabetes' , 'efs')


# Test the function with 'arrhythmia' as the first variable and 'efs' as the second variable.
chi2_contingency_test('arrhythmia' , 'efs')


# Test the function with 'renal-issue' as the first variable and 'efs' as the second variable.
chi2_contingency_test('renal_issue' , 'efs')



# Logistic Regression Model using the Logit function ?

# Build a logistic regression model
model_logistic = logit(formula='efs ~ C(psych_disturb) + C(diabetes) + C(arrhythmia) + C(renal_issue)' , data=data_train).fit()
# Print the Model Logistic results
print(model_logistic.summary())


# Regression Model using the OLS function ?
# Model with categorical variable dri_score
model_ols = ols(formula='efs_time ~ C(psych_disturb) + C(diabetes) + C(arrhythmia) + (renal_issue)' , data=data_train).fit()
# print the model OLS results
print(model_ols.summary())


# Set the figure size for the plot to make it larger and more readable
plt.figure(figsize=(12,8))

# First subplot: Logistic regression plot for 'psych_disturb' vs 'efs'
# Creates the first subplot in a 2x2 grid
plt.subplot(2,2,1)  

sns.regplot(x='psych_disturb' , y='efs' , data=data_train , logistic=True)  # Logistic regression plot
# Set title and axis labels for this plot
plt.title("Logistic Regression: Effect of Psych-disturb on EFS")  # Title of the plot
plt.xlabel("Psych-Disturb (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS (Event-Free Survival)")  # Y-axis label

# Second subplot: Logistic regression plot for 'diabetes' vs 'efs'
# Creates the second subplot in the grid
plt.subplot(2,2,2) 

sns.regplot(x='diabetes' , y='efs' , data=data_train , logistic=True)  # Logistic regression plot
# Set title and axis labels for this plot
plt.title("Logistic Regression: Effect of Diabetes on EFS")  # Title of the plot
plt.xlabel("Diabetes (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS (Event-Free Survival)")  # Y-axis label

# Third subplot: Logistic regression plot for 'arrhythmia' vs 'efs'
# Creates the third subplot in the grid
plt.subplot(2,2,3)

sns.regplot(x='arrhythmia' , y='efs' , data=data_train , logistic=True)  # Logistic regression plot
# Set title and axis labels for this plot
plt.title("Logistic Regression: Effect of Arrhythmia on EFS")  # Title of the plot
plt.xlabel("Arrhythmia (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS (Event-Free Survival)")  # Y-axis label

# Fourth subplot: Logistic regression plot for 'renal_issue' vs 'efs'
# Creates the fourth subplot in the grid
plt.subplot(2,2,4)  

sns.regplot(x='renal_issue' , y='efs' , data=data_train , logistic=True)  # Logistic regression plot
# Set title and axis labels for this plot
plt.title("Logistic Regression: Effect of Renal-Issue on EFS")  # Title of the plot
plt.xlabel("Renal-Issue (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS (Event-Free Survival)")  # Y-axis label

# Adjust the layout to prevent overlap and ensure the plots fit well
plt.tight_layout()
# Show the complete plot with all four subplots
plt.show()



# Set the figure size for the plot to make it larger and more readable
plt.figure(figsize=(12,8))

# First subplot: Liner regression plot for 'psych_disturb' vs 'efs-time'
# Creates the first subplot in a 2x2 grid
plt.subplot(2,2,1) 

sns.regplot(x='psych_disturb' , y='efs_time' , data=data_train , logistic=False)  # Liner regression plot
# Set title and axis labels for this plot
plt.title("Liner Regression: Effect of Psych-disturb on EFS-Time")  # Title of the plot
plt.xlabel("Psych-Disturb (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS-Time (Event-Free Survival)")  # Y-axis label

# Second subplot: Liner regression plot for 'diabetes' vs 'efs-time'
# Creates the second subplot in the grid
plt.subplot(2,2,2) 

sns.regplot(x='diabetes' , y='efs_time' , data=data_train , logistic=False)  # Liner regression plot
# Set title and axis labels for this plot
plt.title("Liner Regression: Effect of Diabetes on EFS-Time")  # Title of the plot
plt.xlabel("Diabetes (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS-Time (Event-Free Survival)")  # Y-axis label


# Third subplot: Liner regression plot for 'arrhythmia' vs 'efs-time'
# Creates the third subplot in the grid
plt.subplot(2,2,3)

sns.regplot(x='arrhythmia' , y='efs_time' , data=data_train , logistic=False)  # Liner regression plot
# Set title and axis labels for this plot
plt.title("Liner Regression: Effect of Arrhythmia on EFS-Time")  # Title of the plot
plt.xlabel("Arrhythmia (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS-Time (Event-Free Survival)")  # Y-axis label


# Fourth subplot: Liner regression plot for 'renal_issue' vs 'efs-time'
# Creates the fourth subplot in the grid
plt.subplot(2,2,4)  

sns.regplot(x='renal_issue' , y='efs_time' , data=data_train , logistic=False)  # Liner regression plot
# Set title and axis labels for this plot
plt.title("Liner Regression: Effect of Renal-Issue on EFS-Time")  # Title of the plot
plt.xlabel("Renal-Issue (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS-Time (Event-Free Survival)")  # Y-axis label


# Adjust the layout to prevent overlap and ensure the plots fit well
plt.tight_layout()
# Show the complete plot with all four subplots
plt.show()



# Set the overall figure sizecc
plt.figure(figsize=(12, 8))

# Plot 1: Psych_disturb vs EFS
plt.subplot(2, 2, 1)
sns.countplot(x='psych_disturb', hue='efs', data=data_train, palette='Set2')  # Create a count plot
plt.title("Distribution of Psych_disturb vs EFS")  # Add title for the plot
plt.xlabel("Psych_disturb")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Plot 2: Diabetes vs EFS
plt.subplot(2, 2, 2)
sns.countplot(x='diabetes', hue='efs', data=data_train, palette='Set2')  # Create a count plot
plt.title("Distribution of Diabetes vs EFS")  # Add title for the plot
plt.xlabel("Diabetes")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Plot 3: Arrhythmia vs EFS
plt.subplot(2, 2, 3)
sns.countplot(x='arrhythmia', hue='efs', data=data_train, palette='Set2')  # Create a count plot
plt.title("Distribution of Arrhythmia vs EFS")  # Add title for the plot
plt.xlabel("Arrhythmia")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Plot 4: Renal_issue vs EFS
plt.subplot(2, 2, 4)
sns.countplot(x='renal_issue', hue='efs', data=data_train, palette='Set2')  # Create a count plot
plt.title("Distribution of Renal_Issue vs EFS")  # Add title for the plot
plt.xlabel("Renal_Issue")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Adjust the layout to prevent overlap and ensure everything fits
plt.tight_layout()
# Display the plot
plt.show()


# 7-  Is there a relationship between the 'cyto_score' column and the 'tbi_status' 
# and 'graft_type' and 'vent_hist' and  'efs' or 'efs_time' columns?
# Logistic Regression Model using the Logit function ?

# Build a logistic regression model
model_logistic = logit(formula='efs ~ C(cyto_score) + C(tbi_status) + C(graft_type) + C(vent_hist)' , data=data_train).fit()

# Print the Model Logistic results
print(model_logistic.summary())


# Model with categorical variable dri_score
model_ols = ols(formula='efs_time ~ C(cyto_score) + C(tbi_status) + C(graft_type) + (vent_hist)' , data=data_train).fit()
# print the model OLS results
print(model_ols.summary())



# Plot Kaplan-Meier curve for 'cyto_score' column
plot_kaplan_meier('cyto_score', 'Kaplan-Meier Survival Curve for Cyto-Score')


# Plot Kaplan-Meier curve for 'tbi_status' column
plot_kaplan_meier('tbi_status' , 'Kaplan-Meier Survival Curve for TBI Status')


# plot Kaplan-Meier curve for 'graft_type' column
plot_kaplan_meier('graft_type' , 'Kaplan-Meier Survival Curve for Graft Type')


# plot Kaplan-Meier curve for 'vent_hist' column
plot_kaplan_meier('vent_hist' , 'Kaplan-Meier Survival Curve for Vent Hist')


plt.figure(figsize=(12,8))

# First subplot: Logistic regression plot for 'cyto_score' vs 'efs'
# Creates the first subplot in a 2x2 grid
plt.subplot(2,2,1)  

sns.regplot(x='cyto_score' , y='efs' , data=data_train , logistic=True)  # Logistic regression plot
# Set title and axis labels for this plot
plt.title("Logistic Regression: Effect of Cyto_Score on EFS")  # Title of the plot
plt.xlabel("Cyto_Score (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS (Event-Free Survival)")  # Y-axis label

# Second subplot: Logistic regression plot for 'tbi_status' vs 'efs'
# Creates the second subplot in the grid
plt.subplot(2,2,2) 

sns.regplot(x='tbi_status' , y='efs' , data=data_train , logistic=True)  # Logistic regression plot
# Set title and axis labels for this plot
plt.title("Logistic Regression: Effect of Tbi_Status on EFS")  # Title of the plot
plt.xlabel("Tbi_Status (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS (Event-Free Survival)")  # Y-axis label

# Third subplot: Logistic regression plot for 'graft_type' vs 'efs'
# Creates the third subplot in the grid
plt.subplot(2,2,3)

sns.regplot(x='graft_type' , y='efs' , data=data_train , logistic=True)  # Logistic regression plot
# Set title and axis labels for this plot
plt.title("Logistic Regression: Effect of Graft_Type on EFS")  # Title of the plot
plt.xlabel("Graft_Type (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS (Event-Free Survival)")  # Y-axis label

# Fourth subplot: Logistic regression plot for 'vent_hist' vs 'efs'
# Creates the fourth subplot in the grid
plt.subplot(2,2,4)  

sns.regplot(x='vent_hist' , y='efs' , data=data_train , logistic=True)  # Logistic regression plot
# Set title and axis labels for this plot
plt.title("Logistic Regression: Effect of Vent_Hist on EFS")  # Title of the plot
plt.xlabel("Vent_Hist (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS (Event-Free Survival)")  # Y-axis label

# Adjust the layout to prevent overlap and ensure the plots fit well
plt.tight_layout()
# Show the complete plot with all four subplots
plt.show()


# Set the figure size for the plot to make it larger and more readable
plt.figure(figsize=(12,8))

# First subplot: Liner regression plot for 'cyto_score' vs 'efs-time'
# Creates the first subplot in a 2x2 grid
plt.subplot(2,2,1) 

sns.regplot(x='cyto_score' , y='efs_time' , data=data_train , logistic=False)  # Liner regression plot
# Set title and axis labels for this plot
plt.title("Liner Regression: Effect of Cyto_Score on EFS-Time")  # Title of the plot
plt.xlabel("Cyto_Score (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS-Time (Event-Free Survival)")  # Y-axis label

# Second subplot: Liner regression plot for 'tbi_status' vs 'efs-time'
# Creates the second subplot in the grid
plt.subplot(2,2,2) 

sns.regplot(x='tbi_status' , y='efs_time' , data=data_train , logistic=False)  # Liner regression plot
# Set title and axis labels for this plot
plt.title("Liner Regression: Effect of Tbi_Status on EFS-Time")  # Title of the plot
plt.xlabel("Tbi_Status (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS-Time (Event-Free Survival)")  # Y-axis label


# Third subplot: Liner regression plot for 'graft_type' vs 'efs-time'
# Creates the third subplot in the grid
plt.subplot(2,2,3)

sns.regplot(x='graft_type' , y='efs_time' , data=data_train , logistic=False)  # Liner regression plot
# Set title and axis labels for this plot
plt.title("Liner Regression: Effect of Graft_Type on EFS-Time")  # Title of the plot
plt.xlabel("Graft_Type (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS-Time (Event-Free Survival)")  # Y-axis label


# Fourth subplot: Liner regression plot for 'vent_hist' vs 'efs-time'
# Creates the fourth subplot in the grid
plt.subplot(2,2,4)  

sns.regplot(x='vent_hist' , y='efs_time' , data=data_train , logistic=False)  # Liner regression plot
# Set title and axis labels for this plot
plt.title("Liner Regression: Effect of Vent_Hist on EFS-Time")  # Title of the plot
plt.xlabel("Vent_Hist (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS-Time (Event-Free Survival)")  # Y-axis label


# Adjust the layout to prevent overlap and ensure the plots fit well
plt.tight_layout()
# Show the complete plot with all four subplots
plt.show()


# Test the function with 'cyto_score' as the first variable and 'efs' as the second variable.
# This checks if there's a significant association between these two categorical variables.
chi2_contingency_test('cyto_score', 'efs')




# Test the function with 'tbi_status' as the first variable and 'efs' as the second variable.
chi2_contingency_test('tbi_status' , 'efs')



# Test the function with 'graft_type' as the first variable and 'efs' as the second variable.
chi2_contingency_test('graft_type' , 'efs')



# Test the function with 'vent_hist' as the first variable and 'efs' as the second variable.
chi2_contingency_test('vent_hist' , 'efs')


# Set the overall figure size
plt.figure(figsize=(12, 8))

# Plot 1: Cyto-Score vs EFS
plt.subplot(2, 2, 1)
sns.countplot(x='cyto_score', hue='efs', data=data_train, palette='Set2')  # Create a count plot
plt.title("Distribution of Cyto_Score vs EFS")  # Add title for the plot
plt.xlabel("Cyto_Score")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Plot 2: Tbi_Status vs EFS
plt.subplot(2, 2, 2)
sns.countplot(x='tbi_status', hue='efs', data=data_train, palette='Set2')  # Create a count plot
plt.title("Distribution of Tbi_Status vs EFS")  # Add title for the plot
plt.xlabel("Tbi_Status")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Plot 3: Graft_Type vs EFS
plt.subplot(2, 2, 3)
sns.countplot(x='graft_type', hue='efs', data=data_train, palette='Set2')  # Create a count plot
plt.title("Distribution of Graft_Type vs EFS")  # Add title for the plot
plt.xlabel("Graft_Type")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Plot 4: Vent_Hist vs EFS
plt.subplot(2, 2, 4)
sns.countplot(x='vent_hist', hue='efs', data=data_train, palette='Set2')  # Create a count plot
plt.title("Distribution of Vent_Hist vs EFS")  # Add title for the plot
plt.xlabel("Vent_Hist")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Adjust the layout to prevent overlap and ensure everything fits
plt.tight_layout()
# Display the plot
plt.show()



# 8- Is there a relationship between the 'hla_match_c_high' column and 'hla_match_dqb1_high' and 
# 'hla_match_c_low' and 'hla_match_drb1_low'  the 'efs' or 'efs_time' columns?

# Test the function with 'hla_match_c_high' as the first variable and 'efs' as the second variable.
# This checks if there's a significant association between these two categorical variables.
chi2_contingency_test('hla_match_c_high', 'efs')


# Test the function with 'hla_match_dqb1_high' as the first variable and 'efs' as the second variable.
chi2_contingency_test('hla_match_dqb1_high' , 'efs')


# Test the function with 'hla_match_c_low' as the first variable and 'efs' as the second variable.
chi2_contingency_test('hla_match_c_low' , 'efs')


# Test the function with 'hla_match_drb1_low' as the first variable and 'efs' as the second variable.
chi2_contingency_test('hla_match_drb1_low' , 'efs')


# Logistic Regression Model using the Logit function ?

# Build a logistic regression model
model_logistic = logit(formula='efs ~ C(hla_match_c_high) + C(hla_match_dqb1_high) + C(hla_match_c_low) + C(hla_match_drb1_low)' , data=data_train).fit()

# Print the Model Logistic results
print(model_logistic.summary())


# Regression Model using the OLS function ?
model_ols = ols(formula='efs_time ~ C(hla_match_c_high) + C(hla_match_dqb1_high) + C(hla_match_c_low) + (hla_match_drb1_low)' , data=data_train).fit()
# print the model OLS results
print(model_ols.summary())


plt.figure(figsize=(12,8))

# First subplot: Logistic regression plot for 'hla_match_c_high' vs 'efs'
# Creates the first subplot in a 2x2 grid
plt.subplot(2,2,1)  

sns.regplot(x='hla_match_c_high' , y='efs' , data=data_train , logistic=True)  # Logistic regression plot
# Set title and axis labels for this plot
plt.title("Logistic Regression: Effect of HLA_match_C_high on EFS")  # Title of the plot
plt.xlabel("HLA_match_C_high (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS (Event-Free Survival)")  # Y-axis label

# Second subplot: Logistic regression plot for 'hla_match_dqb1_high' vs 'efs'
# Creates the second subplot in the grid
plt.subplot(2,2,2) 

sns.regplot(x='hla_match_dqb1_high' , y='efs' , data=data_train , logistic=True)  # Logistic regression plot
# Set title and axis labels for this plot
plt.title("Logistic Regression: Effect of HLA_match_dqb1_high on EFS")  # Title of the plot
plt.xlabel("HLA_match_dqb1_high (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS (Event-Free Survival)")  # Y-axis label

# Third subplot: Logistic regression plot for 'hla_match_c_low' vs 'efs'
# Creates the third subplot in the grid
plt.subplot(2,2,3)

sns.regplot(x='hla_match_c_low' , y='efs' , data=data_train , logistic=True)  # Logistic regression plot
# Set title and axis labels for this plot
plt.title("Logistic Regression: Effect of HLA_match_c_low on EFS")  # Title of the plot
plt.xlabel("HLA_match_c_low (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS (Event-Free Survival)")  # Y-axis label

# Fourth subplot: Logistic regression plot for 'hla_match_drb1_low' vs 'efs'
# Creates the fourth subplot in the grid
plt.subplot(2,2,4)  

sns.regplot(x='hla_match_drb1_low' , y='efs' , data=data_train , logistic=True)  # Logistic regression plot
# Set title and axis labels for this plot
plt.title("Logistic Regression: Effect of HLA_match_drb1_low on EFS")  # Title of the plot
plt.xlabel("HLA_match_drb1_low (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS (Event-Free Survival)")  # Y-axis label

# Adjust the layout to prevent overlap and ensure the plots fit well
plt.tight_layout()
# Show the complete plot with all four subplots
plt.show()


plt.figure(figsize=(12,8))

# First subplot: Linear regression plot for 'hla_match_c_high' vs 'efs-time'
# Creates the first subplot in a 2x2 grid
plt.subplot(2,2,1)  

sns.regplot(x='hla_match_c_high' , y='efs_time' , data=data_train , logistic=False)  # Linear regression plot
# Set title and axis labels for this plot
plt.title("Linear Regression: Effect of HLA_match_C_high on EFS-Time")  # Title of the plot
plt.xlabel("HLA_match_C_high (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS-Time (Event-Free Survival)")  # Y-axis label

# Second subplot: Liner regression plot for 'hla_match_dqb1_high' vs 'efs-time'
# Creates the second subplot in the grid
plt.subplot(2,2,2) 

sns.regplot(x='hla_match_dqb1_high' , y='efs_time' , data=data_train , logistic=False)  # Linear  regression plot
# Set title and axis labels for this plot
plt.title("Linear Regression: Effect of HLA_match_dqb1_high on EFS-Time")  # Title of the plot
plt.xlabel("HLA_match_dqb1_high (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS-Time (Event-Free Survival)")  # Y-axis label

# Third subplot: Liner regression plot for 'hla_match_c_low' vs 'efs-time'
# Creates the third subplot in the grid
plt.subplot(2,2,3)

sns.regplot(x='hla_match_c_low' , y='efs_time' , data=data_train , logistic=False)  # Linear  regression plot
# Set title and axis labels for this plot
plt.title("Linear Regression: Effect of HLA_match_c_low on EFS-Time")  # Title of the plot
plt.xlabel("HLA_match_c_low (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS-Time (Event-Free Survival)")  # Y-axis label

# Fourth subplot: Liner regression plot for 'hla_match_drb1_low' vs 'efs_time'
# Creates the fourth subplot in the grid
plt.subplot(2,2,4)  

sns.regplot(x='hla_match_drb1_low' , y='efs_time' , data=data_train , logistic=False)  # Linear  regression plot
# Set title and axis labels for this plot
plt.title("Linear Regression: Effect of HLA_match_drb1_low on EFS-Time")  # Title of the plot
plt.xlabel("HLA_match_drb1_low (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS-Time (Event-Free Survival)")  # Y-axis label

# Adjust the layout to prevent overlap and ensure the plots fit well
plt.tight_layout()
# Show the complete plot with all four subplots
plt.show()





# Plot Kaplan-Meier curve for 'hla_match_c_high' column
plot_kaplan_meier('hla_match_c_high','Kaplan-Meier Survival Curve for HLA_match_c_high')


# Plot Kaplan-Meier curve for 'hla_match_dqb1_high' column
plot_kaplan_meier('hla_match_dqb1_high', 'Kaplan-Meier Survival Curve for HLA_match_dqb1_high')


# Plot Kaplan-Meier curve for 'hla_match_c_low' column
plot_kaplan_meier('hla_match_c_low','Kaplan-Meier Survival Curve for HLA_match_c_low')



# Plot Kaplan-Meier curve for 'hla_match_drb1_low' column
plot_kaplan_meier('hla_match_drb1_low','Kaplan-Meier Survival Curve for HLA_match_drb1_low')


# Set the overall figure size
plt.figure(figsize=(12, 8))

# Plot 1: Hla_Match_c_High vs EFS
plt.subplot(2, 2, 1)
sns.countplot(x='hla_match_c_high', hue='efs', data=data_train, palette='Set2')  # Create a count plot
plt.title("Distribution of HLA_match_c_high vs EFS")  # Add title for the plot
plt.xlabel("HLA_match_c_high")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Plot 2: Hla_Match_Dqb1_High vs EFS
plt.subplot(2, 2, 2)
sns.countplot(x='hla_match_dqb1_high', hue='efs', data=data_train, palette='Set2')  # Create a count plot
plt.title("Distribution of HLA_match_dqb1_high vs EFS")  # Add title for the plot
plt.xlabel("HLA_match_dqb1_high")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Plot 3: Hla_Match_c_Low vs EFS
plt.subplot(2, 2, 3)
sns.countplot(x='hla_match_c_low', hue='efs', data=data_train, palette='Set2')  # Create a count plot
plt.title("Distribution of HLA_match_c_low vs EFS")  # Add title for the plot
plt.xlabel("HLA_match_c_low")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Plot 4: hla_Match_Drb1_Low vs EFS
plt.subplot(2, 2, 4)
sns.countplot(x='hla_match_drb1_low', hue='efs', data=data_train, palette='Set2')  # Create a count plot
plt.title("Distribution of HLA_match_drb1_low vs EFS")  # Add title for the plot
plt.xlabel("HLA_match_drb1_low")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Adjust the layout to prevent overlap and ensure everything fits
plt.tight_layout()
# Display the plot
plt.show()



# Plot Kaplan-Meier curve for 'hla_high_res_8' column
plot_kaplan_meier('hla_high_res_8', 'Kaplan-Meier Survival Curve for Hla_High_Res_8')


# Plot Kaplan-Meier curve for 'hla_low_res_6' column
plot_kaplan_meier('hla_low_res_6', 'Kaplan-Meier Survival Curve for Hla_Low_Res_6')


# plot Kaplan-Meier curve for 'hla_high_res_6' column
plot_kaplan_meier('hla_high_res_6' , 'Kaplan-Meier Survival Curve for Hla_High_Res_6')


# plot Kaplan-Meier curve for 'hla_high_res_10' column
plot_kaplan_meier('hla_high_res_10' , 'Kaplan-Meier Survival Curve for Hla_High_Res_10')


# Logistic Regression Model using the Logit function ?

# Build a logistic regression model
model_logistic = logit(formula='efs ~ C(hla_high_res_8) + C(hla_low_res_6) + C(hla_high_res_6) + C(hla_high_res_10)' , data=data_train).fit()

# Print the Model Logistic results
print(model_logistic.summary())


# Regression Model using the OLS function ?
model_ols = ols(formula='efs_time ~ C(hla_high_res_8) + C(hla_low_res_6) + C(hla_high_res_6) + (hla_high_res_10)' , data=data_train).fit()
# print the model OLS results
print(model_ols.summary())


plt.figure(figsize=(12,8))

# First subplot: Logistic regression plot for 'hla_high_res_8' vs 'efs'
# Creates the first subplot in a 2x2 grid
plt.subplot(2,2,1)  

sns.regplot(x='hla_high_res_8' , y='efs' , data=data_train , logistic=True)  # Logistic regression plot
# Set title and axis labels for this plot
plt.title("Logistic Regression: Effect of HLA_high_res_8 on EFS")  # Title of the plot
plt.xlabel("HLA_high_res_8 (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS (Event-Free Survival)")  # Y-axis label

# Second subplot: Logistic regression plot for 'hla_low_res_6' vs 'efs'
# Creates the second subplot in the grid
plt.subplot(2,2,2) 

sns.regplot(x='hla_low_res_6' , y='efs' , data=data_train , logistic=True)  # Logistic regression plot
# Set title and axis labels for this plot
plt.title("Logistic Regression: Effect of HLA_low_res_6 on EFS")  # Title of the plot
plt.xlabel("HLA_low_res_6 (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS (Event-Free Survival)")  # Y-axis label

# Third subplot: Logistic regression plot for 'hla_high_res_6' vs 'efs'
# Creates the third subplot in the grid
plt.subplot(2,2,3)

sns.regplot(x='hla_high_res_6' , y='efs' , data=data_train , logistic=True)  # Logistic regression plot
# Set title and axis labels for this plot
plt.title("Logistic Regression: Effect of HLA_high_res_6 on EFS")  # Title of the plot
plt.xlabel("HLA_high_res_6 (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS (Event-Free Survival)")  # Y-axis label

# Fourth subplot: Logistic regression plot for 'hla_high_res_10' vs 'efs'
# Creates the fourth subplot in the grid
plt.subplot(2,2,4)  

sns.regplot(x='hla_high_res_10' , y='efs' , data=data_train , logistic=True)  # Logistic regression plot
# Set title and axis labels for this plot
plt.title("Logistic Regression: Effect of HLA_high_res_10 on EFS")  # Title of the plot
plt.xlabel("HLA_high_res_10 (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS (Event-Free Survival)")  # Y-axis label

# Adjust the layout to prevent overlap and ensure the plots fit well
plt.tight_layout()
# Show the complete plot with all four subplots
plt.show()



plt.figure(figsize=(12,8))

# First subplot: Linear regression plot for 'hla_high_res_8' vs 'efs_time'
# Creates the first subplot in a 2x2 grid
plt.subplot(2,2,1)  

sns.regplot(x='hla_high_res_8' , y='efs_time' , data=data_train , logistic=False)  # Linear regression plot
# Set title and axis labels for this plot
plt.title("Linear Regression: Effect of HLA_high_res_8 on EFS-Time")  # Title of the plot
plt.xlabel("HLA_high_res_8 (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS-Time (Event-Free Survival)")  # Y-axis label

# Second subplot: Linear regression plot for 'hla_low_res_6' vs 'efs_time'
# Creates the second subplot in the grid
plt.subplot(2,2,2) 

sns.regplot(x='hla_low_res_6' , y='efs_time' , data=data_train , logistic=False)  # Linear regression plot
# Set title and axis labels for this plot
plt.title("Linear Regression: Effect of HLA_low_res_6 on EFS-Time")  # Title of the plot
plt.xlabel("HLA_low_res_6 (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS-Time (Event-Free Survival)")  # Y-axis label

# Third subplot: Linear regression plot for 'hla_high_res_6' vs 'efs_time'
# Creates the third subplot in the grid
plt.subplot(2,2,3)

sns.regplot(x='hla_high_res_6' , y='efs_time' , data=data_train , logistic=False)  # Linear regression plot
# Set title and axis labels for this plot
plt.title("Linear Regression: Effect of HLA_high_res_6 on EFS-Time")  # Title of the plot
plt.xlabel("HLA_high_res_6 (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS-Time (Event-Free Survival)")  # Y-axis label

# Fourth subplot: Linear regression plot for 'hla_high_res_10' vs 'efs_time'
# Creates the fourth subplot in the grid
plt.subplot(2,2,4)  

sns.regplot(x='hla_high_res_10' , y='efs_time' , data=data_train , logistic=False)  # Linear regression plot
# Set title and axis labels for this plot
plt.title("Linear Regression: Effect of HLA_high_res_10 on EFS-Time")  # Title of the plot
plt.xlabel("HLA_high_res_10 (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS-Time (Event-Free Survival)")  # Y-axis label

# Adjust the layout to prevent overlap and ensure the plots fit well
plt.tight_layout()
# Show the complete plot with all four subplots
plt.show()



# Set the overall figure size
plt.figure(figsize=(12, 8))

# Plot 1: hla_high_res_8 vs EFS
plt.subplot(2, 2, 1)

sns.countplot(x='hla_high_res_8', hue='efs', data=data_train, palette='Set2')  # Create a count plot
plt.title("Distribution of HLA_high_res_8 vs EFS")  # Add title for the plot
plt.xlabel("HLA_high_res_8")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Plot 2: hla_low_res_6 vs EFS
plt.subplot(2, 2, 2)

sns.countplot(x='hla_low_res_6', hue='efs', data=data_train, palette='Set2')  # Create a count plot
plt.title("Distribution of HLA_low_res_6 vs EFS")  # Add title for the plot
plt.xlabel("HLA_low_res_6")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Plot 3: hla_high_res_6 vs EFS
plt.subplot(2, 2, 3)

sns.countplot(x='hla_high_res_6', hue='efs', data=data_train, palette='Set2')  # Create a count plot
plt.title("Distribution of HLA_high_res_6 vs EFS")  # Add title for the plot
plt.xlabel("HLA_high_res_6")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Plot 4: hla_high_res_10 vs EFS
plt.subplot(2, 2, 4)

sns.countplot(x='hla_high_res_10', hue='efs', data=data_train, palette='Set2')  # Create a count plot
plt.title("Distribution of HLA_high_res_10 vs EFS")  # Add title for the plot
plt.xlabel("HLA_high_res_10")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Adjust the layout to prevent overlap and ensure everything fits
plt.tight_layout()
# Display the plot
plt.show()


#10- Is there a relationship between the 'pulm_severe' column and 'prim_disease_hct' and 'cmv_status' 
# and 'tce_imm_match'  the 'efs' or 'efs_time' columns?

# Test the function with 'pulm_severe' as the first variable and 'efs' as the second variable.
# This checks if there's a significant association between these two categorical variables.
chi2_contingency_test('pulm_severe', 'efs')


# Test the function with 'prim_disease_hct' as the first variable and 'efs' as the second variable.
chi2_contingency_test('prim_disease_hct' , 'efs')


# Test the function with 'cmv_status' as the first variable and 'efs' as the second variable.
chi2_contingency_test('cmv_status' , 'efs')


# Test the function with 'tce_imm_match' as the first variable and 'efs' as the second variable.
chi2_contingency_test('tce_imm_match' , 'efs')


# Plot Kaplan-Meier curve for 'pulm_severe' column
plot_kaplan_meier('pulm_severe', 'Kaplan-Meier Survival Curve for Pulm_Severe')


# Plot Kaplan-Meier curve for 'prim_disease_hct' column
plot_kaplan_meier('prim_disease_hct', 'Kaplan-Meier Survival Curve for Prim_Disease_Hct')


# plot Kaplan-Meier curve for 'cmv_status' column
plot_kaplan_meier('cmv_status' , 'Kaplan-Meier Survival Curve for Cmv_Status')


# plot Kaplan-Meier curve for 'tce_imm_match' column
plot_kaplan_meier('tce_imm_match' , 'Kaplan-Meier Survival Curve for Tce_Imm_Match')


# Logistic Regression Model using the Logit function ?
# Build a logistic regression model
model_logistic = logit(formula='efs ~ C(pulm_severe) + C(prim_disease_hct) + C(cmv_status) + C(tce_imm_match)' , data=data_train).fit()

# Print the Model Logistic results
print(model_logistic.summary())


# Regression Model using the OLS function ?
model_ols = ols(formula='efs_time ~ C(pulm_severe) + C(prim_disease_hct) + C(cmv_status) + (tce_imm_match)' , data=data_train).fit()

# print the model OLS results
print(model_ols.summary())


plt.figure(figsize=(12,8))

# First subplot: Logistic regression plot for 'pulm_severe' vs 'efs'
# Creates the first subplot in a 2x2 grid
plt.subplot(2,2,1)  

sns.regplot(x='pulm_severe' , y='efs' , data=data_train , logistic=True)  # Logistic regression plot
# Set title and axis labels for this plot
plt.title("Logistic Regression: Effect of Pulm_Severe on EFS")  # Title of the plot
plt.xlabel("Pulm_Severe (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS (Event-Free Survival)")  # Y-axis label

# Second subplot: Logistic regression plot for 'prim_disease_hct' vs 'efs'
# Creates the second subplot in the grid
plt.subplot(2,2,2) 

sns.regplot(x='prim_disease_hct' , y='efs' , data=data_train , logistic=True)  # Logistic regression plot
# Set title and axis labels for this plot
plt.title("Logistic Regression: Effect of Prim_Disease_Hct on EFS")  # Title of the plot
plt.xlabel("Prim_Disease_Hct (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS (Event-Free Survival)")  # Y-axis label

# Third subplot: Logistic regression plot for 'cmv_status' vs 'efs'
# Creates the third subplot in the grid
plt.subplot(2,2,3)

sns.regplot(x='cmv_status' , y='efs' , data=data_train , logistic=True)  # Logistic regression plot
# Set title and axis labels for this plot
plt.title("Logistic Regression: Effect of Cmv_Status on EFS")  # Title of the plot
plt.xlabel("Cmv_Status (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS (Event-Free Survival)")  # Y-axis label

# Fourth subplot: Logistic regression plot for 'tce_imm_match' vs 'efs'
# Creates the fourth subplot in the grid
plt.subplot(2,2,4)  

sns.regplot(x='tce_imm_match' , y='efs' , data=data_train , logistic=True)  # Logistic regression plot
# Set title and axis labels for this plot
plt.title("Logistic Regression: Effect of Tce_Imm_Match on EFS")  # Title of the plot
plt.xlabel("Tce_Imm_Match (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS (Event-Free Survival)")  # Y-axis label

# Adjust the layout to prevent overlap and ensure the plots fit well
plt.tight_layout()
# Show the complete plot with all four subplots
plt.show()


plt.figure(figsize=(12,8))

# First subplot: Linear regression plot for 'pulm_severe' vs 'efs_time'
# Creates the first subplot in a 2x2 grid
plt.subplot(2,2,1)  

sns.regplot(x='pulm_severe' , y='efs_time' , data=data_train , logistic=False)  # Linear regression plot
# Set title and axis labels for this plot
plt.title("Linear Regression: Effect of Pulm_Severe on EFS-Time")  # Title of the plot
plt.xlabel("Pulm_Severe (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS-Time (Event-Free Survival)")  # Y-axis label

# Second subplot: Linear regression plot for 'prim_disease_hct' vs 'efs_time'
# Creates the second subplot in the grid
plt.subplot(2,2,2) 

sns.regplot(x='prim_disease_hct' , y='efs_time' , data=data_train , logistic=False)  # Linear regression plot
# Set title and axis labels for this plot
plt.title("Linear Regression: Effect of Prim_Disease_Hct on EFS-Time")  # Title of the plot
plt.xlabel("Prim_Disease_Hct (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS-Time (Event-Free Survival)")  # Y-axis label

# Third subplot: Logistic regression plot for 'cmv_status' vs 'efs_time'
# Creates the third subplot in the grid
plt.subplot(2,2,3)

sns.regplot(x='cmv_status' , y='efs_time' , data=data_train , logistic=False)  # Linear regression plot
# Set title and axis labels for this plot
plt.title("Linear Regression: Effect of Cmv_Status on EFS-Time")  # Title of the plot
plt.xlabel("Cmv_Status (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS-Time (Event-Free Survival)")  # Y-axis label

# Fourth subplot: Logistic regression plot for 'tce_imm_match' vs 'efs_time'
# Creates the fourth subplot in the grid
plt.subplot(2,2,4)  

sns.regplot(x='tce_imm_match' , y='efs_time' , data=data_train , logistic=False)  # Linear regression plot
# Set title and axis labels for this plot
plt.title("Linear Regression: Effect of Tce_Imm_Match on EFS-Time")  # Title of the plot
plt.xlabel("Tce_Imm_Match (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS-Time (Event-Free Survival)")  # Y-axis label

# Adjust the layout to prevent overlap and ensure the plots fit well
plt.tight_layout()
# Show the complete plot with all four subplots
plt.show()


# Set the overall figure size
plt.figure(figsize=(12, 8))

# Plot 1: pulm_severe vs EFS
plt.subplot(2, 2, 1)

sns.countplot(x='pulm_severe', hue='efs', data=data_train, palette='Set2')  # Create a count plot
plt.title("Distribution of Pulm_Severe vs EFS")  # Add title for the plot
plt.xlabel("Pulm_Severe")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Plot 2: prim_disease_hct vs EFS
plt.subplot(2, 2, 2)

sns.countplot(x='prim_disease_hct', hue='efs', data=data_train, palette='Set2')  # Create a count plot
plt.title("Distribution of Prim_Disease_Hct vs EFS")  # Add title for the plot
plt.xlabel("Prim_Disease_Hct")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Plot 3: cmv_status vs EFS
plt.subplot(2, 2, 3)

sns.countplot(x='cmv_status', hue='efs', data=data_train, palette='Set2')  # Create a count plot
plt.title("Distribution of Cmv_Status vs EFS")  # Add title for the plot
plt.xlabel("Cmv_Status")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Plot 4: tce_imm_match vs EFS
plt.subplot(2, 2, 4)

sns.countplot(x='tce_imm_match', hue='efs', data=data_train, palette='Set2')  # Create a count plot
plt.title("Distribution of Tce_Imm_Match vs EFS")  # Add title for the plot
plt.xlabel("Tce_Imm_Match")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Adjust the layout to prevent overlap and ensure everything fits
plt.tight_layout()
# Display the plot
plt.show()


# 11-  Is there a relationship between the 'hla_nmdp_6' column and 'hla_match_b_low' 
#and 'hla_match_a_low' and 'hla_match_dqb1_low' and  the 'efs' or 'efs_time' columns?

# Plot Kaplan-Meier curve for 'hla_nmdp_6' column
plot_kaplan_meier('hla_nmdp_6', 'Kaplan-Meier Survival Curve for HLA_Nmdp_6')


# Plot Kaplan-Meier curve for 'hla_match_b_low' column
plot_kaplan_meier('hla_match_b_low', 'Kaplan-Meier Survival Curve for HLA_match_B_low')


# plot Kaplan-Meier curve for 'hla_match_a_low' column
plot_kaplan_meier('hla_match_a_low' , 'Kaplan-Meier Survival Curve for HLA_match_A_low')


# plot Kaplan-Meier curve for 'hla_match_dqb1_low' column
plot_kaplan_meier('hla_match_dqb1_low' , 'Kaplan-Meier Survival Curve for HLA_match_dqb1_low')


# Logistic Regression Model using the Logit function ?
# Build a logistic regression model
model_logistic = logit(formula='efs ~ C(hla_nmdp_6) + C(hla_match_b_low) + C(hla_match_a_low) + C(hla_match_dqb1_low)' , data=data_train).fit()

# Print the Model Logistic results
print(model_logistic.summary()) 


# Regression Model using the OLS function ?
model_ols = ols(formula='efs_time ~ C(hla_nmdp_6) + C(hla_match_b_low) + C(hla_match_a_low) + (hla_match_dqb1_low)' , data=data_train).fit()

# print the model OLS results
print(model_ols.summary())



plt.figure(figsize=(12,8))

# First subplot: Logistic regression plot for 'hla_nmdp_6' vs 'efs'
# Creates the first subplot in a 2x2 grid
plt.subplot(2,2,1)  

sns.regplot(x='hla_nmdp_6' , y='efs' , data=data_train , logistic=True)  # Logistic regression plot
# Set title and axis labels for this plot
plt.title("Logistic Regression: Effect of HLA_nmdp_6 on EFS")  # Title of the plot
plt.xlabel("HLA_nmdp_6 (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS (Event-Free Survival)")  # Y-axis label

# Second subplot: Logistic regression plot for 'hla_match_b_low' vs 'efs'
# Creates the second subplot in the grid
plt.subplot(2,2,2) 

sns.regplot(x='hla_match_b_low' , y='efs' , data=data_train , logistic=True)  # Logistic regression plot
# Set title and axis labels for this plot
plt.title("Logistic Regression: Effect of HLA_match_b_low on EFS")  # Title of the plot
plt.xlabel("HLA_match_b_low (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS (Event-Free Survival)")  # Y-axis label

# Third subplot: Logistic regression plot for 'hla_match_a_low' vs 'efs'
# Creates the third subplot in the grid
plt.subplot(2,2,3)

sns.regplot(x='hla_match_a_low' , y='efs' , data=data_train , logistic=True)  # Logistic regression plot
# Set title and axis labels for this plot
plt.title("Logistic Regression: Effect of HLA_match_a_low on EFS")  # Title of the plot
plt.xlabel("HLA_match_a_low (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS (Event-Free Survival)")  # Y-axis label

# Fourth subplot: Logistic regression plot for 'hla_match_dqb1_low' vs 'efs'
# Creates the fourth subplot in the grid
plt.subplot(2,2,4)  

sns.regplot(x='hla_match_dqb1_low' , y='efs' , data=data_train , logistic=True)  # Logistic regression plot
# Set title and axis labels for this plot
plt.title("Logistic Regression: Effect of HLA_match_dqb1_low on EFS")  # Title of the plot
plt.xlabel("HLA_match_dqb1_low (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS (Event-Free Survival)")  # Y-axis label

# Adjust the layout to prevent overlap and ensure the plots fit well
plt.tight_layout()
# Show the complete plot with all four subplots
plt.show()


plt.figure(figsize=(12,8))

# First subplot: Linear regression plot for 'hla_nmdp_6' vs 'efs_time'
# Creates the first subplot in a 2x2 grid
plt.subplot(2,2,1)  

sns.regplot(x='hla_nmdp_6' , y='efs_time' , data=data_train , logistic=False)  # Linear regression plot
# Set title and axis labels for this plot
plt.title("Linear Regression: Effect of HLA_nmdp_6 on EFS-Time")  # Title of the plot
plt.xlabel("Pulm_Severe (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS-Time (Event-Free Survival)")  # Y-axis label

# Second subplot: Linear regression plot for 'hla_match_b_low' vs 'efs_time'
# Creates the second subplot in the grid
plt.subplot(2,2,2) 

sns.regplot(x='hla_match_b_low' , y='efs_time' , data=data_train , logistic=False)  # Linear regression plot
# Set title and axis labels for this plot
plt.title("Linear Regression: Effect of HLA_match_b_low on EFS-Time")  # Title of the plot
plt.xlabel("HLA_match_b_low (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS-Time (Event-Free Survival)")  # Y-axis label

# Third subplot: Logistic regression plot for 'hla_match_a_low' vs 'efs_time'
# Creates the third subplot in the grid
plt.subplot(2,2,3)

sns.regplot(x='hla_match_a_low' , y='efs_time' , data=data_train , logistic=False)  # Linear regression plot
# Set title and axis labels for this plot
plt.title("Linear Regression: Effect of HLA_match_a_low on EFS-Time")  # Title of the plot
plt.xlabel("HLA_match_a_low (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS-Time (Event-Free Survival)")  # Y-axis label

# Fourth subplot: Logistic regression plot for 'hla_match_dqb1_low' vs 'efs_time'
# Creates the fourth subplot in the grid
plt.subplot(2,2,4)  

sns.regplot(x='hla_match_dqb1_low' , y='efs_time' , data=data_train , logistic=False)  # Linear regression plot
# Set title and axis labels for this plot
plt.title("Linear Regression: Effect of HLA_match_dqb1_low on EFS-Time")  # Title of the plot
plt.xlabel("HLA_match_dqb1_low (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS-Time (Event-Free Survival)")  # Y-axis label

# Adjust the layout to prevent overlap and ensure the plots fit well
plt.tight_layout()
# Show the complete plot with all four subplots
plt.show()



# Set the overall figure size
plt.figure(figsize=(12, 8))

# Plot 1: hla_nmdp_6 vs EFS
plt.subplot(2, 2, 1)

sns.countplot(x='hla_nmdp_6', hue='efs', data=data_train, palette='Set2')  # Create a count plot
plt.title("Distribution of HLA_nmdp_6 vs EFS")  # Add title for the plot
plt.xlabel("HLA_nmdp_6")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Plot 2: hla_match_b_low vs EFS
plt.subplot(2, 2, 2)

sns.countplot(x='hla_match_b_low', hue='efs', data=data_train, palette='Set2')  # Create a count plot
plt.title("Distribution of HLA_match_b_low vs EFS")  # Add title for the plot
plt.xlabel("HLA_match_b_low")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Plot 3: hla_match_a_low vs EFS
plt.subplot(2, 2, 3)

sns.countplot(x='hla_match_a_low', hue='efs', data=data_train, palette='Set2')  # Create a count plot
plt.title("Distribution of HLA_match_a_low vs EFS")  # Add title for the plot
plt.xlabel("HLA_match_a_low")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Plot 4: hla_match_dqb1_low vs EFS
plt.subplot(2, 2, 4)

sns.countplot(x='hla_match_dqb1_low', hue='efs', data=data_train, palette='Set2')  # Create a count plot
plt.title("Distribution of HLA_match_dqb1_low vs EFS")  # Add title for the plot
plt.xlabel("HLA_match_dqb1_low")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Adjust the layout to prevent overlap and ensure everything fits
plt.tight_layout()
# Display the plot
plt.show()


# 12- Is there a relationship between the 'rituximab' column and 'prod_type' 
# and 'conditioning_intensity' and 'ethnicity ' the 'efs' or 'efs_time' columns?

# Test the function with 'rituximab' as the first variable and 'efs' as the second variable.
# This checks if there's a significant association between these two categorical variables.
chi2_contingency_test('rituximab', 'efs')


# Test the function with 'prod_type' as the first variable and 'efs' as the second variable.
chi2_contingency_test('prod_type' , 'efs')


# Test the function with 'conditioning_intensity' as the first variable and 'efs' as the second variable.
chi2_contingency_test('conditioning_intensity' , 'efs')


# Test the function with 'ethnicity' as the first variable and 'efs' as the second variable.
chi2_contingency_test('ethnicity' , 'efs')


# Plot Kaplan-Meier curve for 'rituximab' column
plot_kaplan_meier('rituximab', 'Kaplan-Meier Survival Curve for Rituximab')


# Plot Kaplan-Meier curve for 'prod_type' column
plot_kaplan_meier('prod_type', 'Kaplan-Meier Survival Curve for Prod_Type')


# plot Kaplan-Meier curve for 'conditioning_intensity' column
plot_kaplan_meier('conditioning_intensity' , 'Kaplan-Meier Survival Curve for Conditioning_Intensity')



# plot Kaplan-Meier curve for 'hla_match_dqb1_low' column
plot_kaplan_meier('ethnicity' , 'Kaplan-Meier Survival Curve for Ethnicity')


# Logistic Regression Model using the Logit function ?
# Build a logistic regression model
model_logistic = logit(formula='efs ~ C(rituximab) + C(prod_type) + C(conditioning_intensity) + C(ethnicity)' , data=data_train).fit()

# Print the Model Logistic results
print(model_logistic.summary()) 


# Regression Model using the OLS function ?
model_ols = ols(formula='efs_time ~ C(rituximab) + C(prod_type) + C(conditioning_intensity) + (ethnicity)' , data=data_train).fit()

# print the model OLS results
print(model_ols.summary())



plt.figure(figsize=(12,8))

# First subplot: Logistic regression plot for 'rituximab' vs 'efs'
# Creates the first subplot in a 2x2 grid
plt.subplot(2,2,1)  

sns.regplot(x='rituximab' , y='efs' , data=data_train , logistic=True)  # Logistic regression plot
# Set title and axis labels for this plot
plt.title("Logistic Regression: Effect of Rituximab on EFS")  # Title of the plot
plt.xlabel("Rituximab (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS (Event-Free Survival)")  # Y-axis label

# Second subplot: Logistic regression plot for 'prod_type' vs 'efs'
# Creates the second subplot in the grid
plt.subplot(2,2,2) 

sns.regplot(x='prod_type' , y='efs' , data=data_train , logistic=True)  # Logistic regression plot
# Set title and axis labels for this plot
plt.title("Logistic Regression: Effect of Prod_Type on EFS")  # Title of the plot
plt.xlabel("Prod_Type (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS (Event-Free Survival)")  # Y-axis label

# Third subplot: Logistic regression plot for 'conditioning_intensity' vs 'efs'
# Creates the third subplot in the grid
plt.subplot(2,2,3)

sns.regplot(x='conditioning_intensity' , y='efs' , data=data_train , logistic=True)  # Logistic regression plot
# Set title and axis labels for this plot
plt.title("Logistic Regression: Effect of Conditioning_Intensity on EFS")  # Title of the plot
plt.xlabel("Conditioning_Intensity (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS (Event-Free Survival)")  # Y-axis label

# Fourth subplot: Logistic regression plot for 'ethnicity' vs 'efs'
# Creates the fourth subplot in the grid
plt.subplot(2,2,4)  

sns.regplot(x='ethnicity' , y='efs' , data=data_train , logistic=True)  # Logistic regression plot
# Set title and axis labels for this plot
plt.title("Logistic Regression: Effect of Ethnicity on EFS")  # Title of the plot
plt.xlabel("Ethnicity (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS (Event-Free Survival)")  # Y-axis label

# Adjust the layout to prevent overlap and ensure the plots fit well
plt.tight_layout()
# Show the complete plot with all four subplots
plt.show()


plt.figure(figsize=(12,8))

# First subplot: Linear regression plot for 'rituximab' vs 'efs_time'
# Creates the first subplot in a 2x2 grid
plt.subplot(2,2,1)  

sns.regplot(x='rituximab' , y='efs_time' , data=data_train , logistic=False)  # Linear regression plot
# Set title and axis labels for this plot
plt.title("Linear Regression: Effect of Rituximab on EFS-Time")  # Title of the plot
plt.xlabel("Rituximab (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS-Time (Event-Free Survival)")  # Y-axis label

# Second subplot: Linear regression plot for 'prod_type' vs 'efs_time'
# Creates the second subplot in the grid
plt.subplot(2,2,2) 

sns.regplot(x='prod_type' , y='efs_time' , data=data_train , logistic=False)  # Linear regression plot
# Set title and axis labels for this plot
plt.title("Linear Regression: Effect of Prod_Type on EFS-Time")  # Title of the plot
plt.xlabel("Prod_Type (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS-Time (Event-Free Survival)")  # Y-axis label

# Third subplot: Logistic regression plot for 'conditioning_intensity' vs 'efs_time'
# Creates the third subplot in the grid
plt.subplot(2,2,3)

sns.regplot(x='conditioning_intensity' , y='efs_time' , data=data_train , logistic=False)  # Linear regression plot
# Set title and axis labels for this plot
plt.title("Linear Regression: Effect of Conditioning_Intensity on EFS-Time")  # Title of the plot
plt.xlabel("Conditioning_Intensity (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS-Time (Event-Free Survival)")  # Y-axis label

# Fourth subplot: Logistic regression plot for 'ethnicity' vs 'efs_time'
# Creates the fourth subplot in the grid
plt.subplot(2,2,4)  

sns.regplot(x='ethnicity' , y='efs_time' , data=data_train , logistic=False)  # Linear regression plot
# Set title and axis labels for this plot
plt.title("Linear Regression: Effect of Ethnicity on EFS-Time")  # Title of the plot
plt.xlabel("Ethnicity (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS-Time (Event-Free Survival)")  # Y-axis label

# Adjust the layout to prevent overlap and ensure the plots fit well
plt.tight_layout()
# Show the complete plot with all four subplots
plt.show()


# Set the overall figure size
plt.figure(figsize=(12, 8))

# Plot 1: rituximab vs EFS
plt.subplot(2, 2, 1)

sns.countplot(x='rituximab', hue='efs', data=data_train, palette='Set2')  # Create a count plot
plt.title("Distribution of Rituximab vs EFS")  # Add title for the plot
plt.xlabel("Rituximab")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Plot 2: prod_type vs EFS
plt.subplot(2, 2, 2)

sns.countplot(x='prod_type', hue='efs', data=data_train, palette='Set2')  # Create a count plot
plt.title("Distribution of Prod_Type vs EFS")  # Add title for the plot
plt.xlabel("Prod_Type")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Plot 3: conditioning_intensity vs EFS
plt.subplot(2, 2, 3)

sns.countplot(x='conditioning_intensity', hue='efs', data=data_train, palette='Set2')  # Create a count plot
plt.title("Distribution of Conditioning_Intensity vs EFS")  # Add title for the plot
plt.xlabel("Conditioning_Intensity")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Plot 4: ethnicity vs EFS
plt.subplot(2, 2, 4)

sns.countplot(x='ethnicity', hue='efs', data=data_train, palette='Set2')  # Create a count plot
plt.title("Distribution of Ethnicity vs EFS")  # Add title for the plot
plt.xlabel("Ethnicity")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Adjust the layout to prevent overlap and ensure everything fits
plt.tight_layout()
# Display the plot
plt.show()


# 14- Is there a relationship between the 'year_hct' column and 'obesity' and 
# 'mrd_hct' and 'in_vivo_tcd'  the 'efs' or 'efs_time' columns?

# Test the function with 'year_hct' as the first variable and 'efs' as the second variable.
# This checks if there's a significant association between these two categorical variables.
chi2_contingency_test('year_hct', 'efs')


# Test the function with 'obesity' as the first variable and 'efs' as the second variable.
chi2_contingency_test('obesity' , 'efs')


# Test the function with 'mrd_hct' as the first variable and 'efs' as the second variable.
chi2_contingency_test('mrd_hct' , 'efs')


# Test the function with 'in_vivo_tcd' as the first variable and 'efs' as the second variable.
chi2_contingency_test('in_vivo_tcd' , 'efs')


# Plot Kaplan-Meier curve for 'year_hct' column
plot_kaplan_meier('year_hct', 'Kaplan-Meier Survival Curve for Year_Hct')


# Plot Kaplan-Meier curve for 'obesity' column
plot_kaplan_meier('obesity', 'Kaplan-Meier Survival Curve for Obesity')


# plot Kaplan-Meier curve for 'mrd_hct' column
plot_kaplan_meier('mrd_hct' , 'Kaplan-Meier Survival Curve for Mrd_Hct')


# plot Kaplan-Meier curve for 'in_vivo_tcd' column
plot_kaplan_meier('in_vivo_tcd' , 'Kaplan-Meier Survival Curve for In_Vivo_Tcd')


# Logistic Regression Model using the Logit function ?
# Build a logistic regression model
model_logistic = logit(formula='efs ~ C(obesity) + C(mrd_hct) + C(year_hct) + C(in_vivo_tcd)' , data=data_train).fit()

# Print the Model Logistic results
print(model_logistic.summary()) 


# Regression Model using the OLS function ?
model_ols = ols(formula='efs_time ~ C(obesity) + C(mrd_hct) + C(year_hct) + (in_vivo_tcd)' , data=data_train).fit()

# print the model OLS results
print(model_ols.summary())


# 15- Why do the years 2015 to 2019 see a decrease in efs_time compared to other years?

# Function to compare the distribution of data between two columns
def compare_distribution(column1, column2):
    # Returns the distribution of data between two columns as a pivot table.
    disease_distribution = data_train.groupby([column1, column2]).size().unstack(fill_value=0)
    return disease_distribution


# Comparing the distribution of data between 'year_hct' and 'prim_disease_hct' columns
column_prim_disease_hct = compare_distribution('year_hct', 'prim_disease_hct')

# Comparing the distribution of data between 'year_hct' and 'conditioning_intensity' columns
column_conditioning_intensity = compare_distribution('year_hct', 'conditioning_intensity')

# Comparing the distribution of data between 'year_hct' and 'tbi_status' columns
column_tbi_status = compare_distribution('year_hct', 'tbi_status')

# Comparing the distribution of data between 'year_hct' and 'rituximab' columns
column_rituximab = compare_distribution('year_hct', 'rituximab')



# Create a figure and axes for 2 subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))  # 1 row, 2 columns

# Plot the first graph on the first axis
column_prim_disease_hct.plot(kind='bar', stacked=True, ax=axes[0,0])

# Set the title and labels for the first subplot
axes[0,0].set_title('Distribution of Primary Disease by Year of HCT')
axes[0,0].set_xlabel('Year of HCT')
axes[0,0].set_ylabel('Number of Cases')
axes[0,0].legend(title='Primary Disease', bbox_to_anchor=(1.05, 1), loc='upper left')

# Plot the second graph on the second axis
column_conditioning_intensity.plot(kind='bar', stacked=True, ax=axes[0,1])

# Set the title and labels for the second subplot
axes[0,1].set_title('Distribution of Primary Disease by Conditioning Intensity')
axes[0,1].set_xlabel('Year of CI')
axes[0,1].set_ylabel('Number of Cases')
axes[0,1].legend(title='Primary Disease', bbox_to_anchor=(1.05, 1), loc='upper left')

# Plot the second graph on the second axis
column_tbi_status.plot(kind='bar' , stacked=True , ax=axes[1,0])

# Set the title and labels for the second subplot
axes[1,0].set_title('Distribution of Primary Disease by TS')
axes[1,0].set_xlabel('Year of TS')
axes[1,0].set_ylabel('Number of Cases')
axes[1,0].legend(title='Primary Disease', bbox_to_anchor=(1.05, 0.70), loc='upper left')

# Plot the second graph on the second axis
column_rituximab.plot(kind='bar' , stacked=True , ax=axes[1,1])

# Set the title and labels for the second subplot
axes[1,1].set_title('Distribution of Primary Disease by Rituximab')
axes[1,1].set_xlabel('Year of Rituximab')
axes[1,1].set_ylabel('Number of Cases')
axes[1,1].legend(title='Primary Disease', bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust the layout to prevent overlap
plt.tight_layout()
# Show the plot
plt.show()


plt.figure(figsize=(12,8))

# First subplot: Logistic regression plot for 'obesity' vs 'efs'
# Creates the first subplot in a 2x2 grid
plt.subplot(2,2,1)  

sns.regplot(x='obesity' , y='efs' , data=data_train , logistic=True)  # Logistic regression plot
# Set title and axis labels for this plot
plt.title("Logistic Regression: Effect of Obesity on EFS")  # Title of the plot
plt.xlabel("Obesity (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS (Event-Free Survival)")  # Y-axis label

# Second subplot: Logistic regression plot for 'mrd_hct' vs 'efs'
# Creates the second subplot in the grid
plt.subplot(2,2,2) 

sns.regplot(x='mrd_hct' , y='efs' , data=data_train , logistic=True)  # Logistic regression plot
# Set title and axis labels for this plot
plt.title("Logistic Regression: Effect of Mrd_Hct on EFS")  # Title of the plot
plt.xlabel("Mrd_Hct (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS (Event-Free Survival)")  # Y-axis label

# Third subplot: Logistic regression plot for 'year_hct' vs 'efs'
# Creates the third subplot in the grid
plt.subplot(2,2,3)

sns.regplot(x='year_hct' , y='efs' , data=data_train , logistic=True)  # Logistic regression plot
# Set title and axis labels for this plot
plt.title("Logistic Regression: Effect of Year_Hct on EFS")  # Title of the plot
plt.xlabel("Year_Hct (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS (Event-Free Survival)")  # Y-axis label

# Fourth subplot: Logistic regression plot for 'in_vivo_tcd' vs 'efs'
# Creates the fourth subplot in the grid
plt.subplot(2,2,4)  

sns.regplot(x='in_vivo_tcd' , y='efs' , data=data_train , logistic=True)  # Logistic regression plot
# Set title and axis labels for this plot
plt.title("Logistic Regression: Effect of In_Vivo_Tcd on EFS")  # Title of the plot
plt.xlabel("In_Vivo_Tcd (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS (Event-Free Survival)")  # Y-axis label

# Adjust the layout to prevent overlap and ensure the plots fit well
plt.tight_layout()
# Show the complete plot with all four subplots
plt.show()



plt.figure(figsize=(12,8))

# First subplot: Linear regression plot for 'obesity' vs 'efs_time'
# Creates the first subplot in a 2x2 grid
plt.subplot(2,2,1)  

sns.regplot(x='obesity' , y='efs_time' , data=data_train , logistic=False)  # Linear regression plot
# Set title and axis labels for this plot
plt.title("Linear Regression: Effect of Obesity on EFS-Time")  # Title of the plot
plt.xlabel("Obesity (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS-Time (Event-Free Survival)")  # Y-axis label

# Second subplot: Linear regression plot for 'mrd_hct' vs 'efs_time'
# Creates the second subplot in the grid
plt.subplot(2,2,2) 

sns.regplot(x='mrd_hct' , y='efs_time' , data=data_train , logistic=False)  # Linear regression plot
# Set title and axis labels for this plot
plt.title("Linear Regression: Effect of Mrd_Hct on EFS-Time")  # Title of the plot
plt.xlabel("Mrd_Hct (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS-Time (Event-Free Survival)")  # Y-axis label

# Third subplot: Logistic regression plot for 'year_hct' vs 'efs_time'
# Creates the third subplot in the grid
plt.subplot(2,2,3)

sns.regplot(x='year_hct' , y='efs_time' , data=data_train , logistic=False)  # Linear regression plot
# Set title and axis labels for this plot
plt.title("Linear Regression: Effect of Year_Hct on EFS-Time")  # Title of the plot
plt.xlabel("Year_Hct (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS-Time (Event-Free Survival)")  # Y-axis label

# Fourth subplot: Logistic regression plot for 'in_vivo_tcd' vs 'efs_time'
# Creates the fourth subplot in the grid
plt.subplot(2,2,4)  

sns.regplot(x='in_vivo_tcd' , y='efs_time' , data=data_train , logistic=False)  # Linear regression plot
# Set title and axis labels for this plot
plt.title("Linear Regression: Effect of In_Vivo_Tcd on EFS-Time")  # Title of the plot
plt.xlabel("In_Vivo_Tcd (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS-Time (Event-Free Survival)")  # Y-axis label

# Adjust the layout to prevent overlap and ensure the plots fit well
plt.tight_layout()
# Show the complete plot with all four subplots
plt.show()


# Set the overall figure size
plt.figure(figsize=(12, 8))

# Plot 1: obesity vs EFS
plt.subplot(2, 2, 1)

sns.countplot(x='obesity', hue='efs', data=data_train, palette='Set2')  # Create a count plot
plt.title("Distribution of Rituximab vs EFS")  # Add title for the plot
plt.xlabel("Obesity")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Plot 2: mrd_hct vs EFS
plt.subplot(2, 2, 2)

sns.countplot(x='mrd_hct', hue='efs', data=data_train, palette='Set2')  # Create a count plot
plt.title("Distribution of Mrd_Hct vs EFS")  # Add title for the plot
plt.xlabel("Mrd_Hct")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Plot 3: year_hct vs EFS
plt.subplot(2, 2, 3)

sns.countplot(x='year_hct', hue='efs', data=data_train, palette='Set2')  # Create a count plot
plt.title("Distribution of Year_Hct vs EFS")  # Add title for the plot
plt.xlabel("Year_Hct")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Plot 4: in_vivo_tcd vs EFS
plt.subplot(2, 2, 4)

sns.countplot(x='in_vivo_tcd', hue='efs', data=data_train, palette='Set2')  # Create a count plot
plt.title("Distribution of In_Vivo_Tcd vs EFS")  # Add title for the plot
plt.xlabel("In_Vivo_Tcd")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Adjust the layout to prevent overlap and ensure everything fits
plt.tight_layout()
# Display the plot
plt.show()


# 16- Is there a relationship between the 'cyto_score_detail' column and 'tce_match'
# and 'hepatic_severe' and 'prior_tumor'  the 'efs' or 'efs_time' columns?


# Logistic Regression Model using the Logit function ?
# Build a logistic regression model
model_logistic = logit(formula='efs ~ C(cyto_score_detail) + C(tce_match) + C(hepatic_severe) + C(prior_tumor)' , data=data_train).fit()

# Print the Model Logistic results
print(model_logistic.summary()) 


# Regression Model using the OLS function ?
model_ols = ols(formula='efs_time ~ C(cyto_score_detail) + C(tce_match) + C(hepatic_severe) + (prior_tumor)' , data=data_train).fit()

# print the model OLS results
print(model_ols.summary())


plt.figure(figsize=(12,8))

# First subplot: Logistic regression plot for 'cyto_score_detail' vs 'efs'
# Creates the first subplot in a 2x2 grid
plt.subplot(2,2,1)  

sns.regplot(x='cyto_score_detail' , y='efs' , data=data_train , logistic=True)  # Logistic regression plot
# Set title and axis labels for this plot
plt.title("Logistic Regression: Effect of Cyto_Score_Detail on EFS")  # Title of the plot
plt.xlabel("Cyto_Score_Detail (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS (Event-Free Survival)")  # Y-axis label

# Second subplot: Logistic regression plot for 'tce_match' vs 'efs'
# Creates the second subplot in the grid
plt.subplot(2,2,2) 

sns.regplot(x='tce_match' , y='efs' , data=data_train , logistic=True)  # Logistic regression plot
# Set title and axis labels for this plot
plt.title("Logistic Regression: Effect of Tce_Match on EFS")  # Title of the plot
plt.xlabel("Tce_Match (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS (Event-Free Survival)")  # Y-axis label

# Third subplot: Logistic regression plot for 'hepatic_severe' vs 'efs'
# Creates the third subplot in the grid
plt.subplot(2,2,3)

sns.regplot(x='hepatic_severe' , y='efs' , data=data_train , logistic=True)  # Logistic regression plot
# Set title and axis labels for this plot
plt.title("Logistic Regression: Effect of Hepatic_Severe on EFS")  # Title of the plot
plt.xlabel("Hepatic_Severe (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS (Event-Free Survival)")  # Y-axis label

# Fourth subplot: Logistic regression plot for 'prior_tumor' vs 'efs'
# Creates the fourth subplot in the grid
plt.subplot(2,2,4)  

sns.regplot(x='prior_tumor' , y='efs' , data=data_train , logistic=True)  # Logistic regression plot
# Set title and axis labels for this plot
plt.title("Logistic Regression: Effect of Prior_Tumor on EFS")  # Title of the plot
plt.xlabel("Prior_Tumor (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS (Event-Free Survival)")  # Y-axis label

# Adjust the layout to prevent overlap and ensure the plots fit well
plt.tight_layout()
# Show the complete plot with all four subplots
plt.show()


plt.figure(figsize=(12,8))

# First subplot: Linear regression plot for 'cyto_score_detail' vs 'efs_time'
# Creates the first subplot in a 2x2 grid
plt.subplot(2,2,1)  

sns.regplot(x='cyto_score_detail' , y='efs_time' , data=data_train , logistic=False)  # Linear regression plot
# Set title and axis labels for this plot
plt.title("Linear Regression: Effect of Cyto_Score_Detail on EFS-Time")  # Title of the plot
plt.xlabel("Cyto_Score_Detail (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS-Time (Event-Free Survival)")  # Y-axis label

# Second subplot: Linear regression plot for 'tce_match' vs 'efs_time'
# Creates the second subplot in the grid
plt.subplot(2,2,2) 

sns.regplot(x='tce_match' , y='efs_time' , data=data_train , logistic=False)  # Linear regression plot
# Set title and axis labels for this plot
plt.title("Linear Regression: Effect of Tce_Match on EFS-Time")  # Title of the plot
plt.xlabel("Tce_Match (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS-Time (Event-Free Survival)")  # Y-axis label

# Third subplot: Logistic regression plot for 'hepatic_severe' vs 'efs_time'
# Creates the third subplot in the grid
plt.subplot(2,2,3)

sns.regplot(x='hepatic_severe' , y='efs_time' , data=data_train , logistic=False)  # Linear regression plot
# Set title and axis labels for this plot
plt.title("Linear Regression: Effect of Hepatic_Severe on EFS-Time")  # Title of the plot
plt.xlabel("Hepatic_Severe (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS-Time (Event-Free Survival)")  # Y-axis label

# Fourth subplot: Logistic regression plot for 'prior_tumor' vs 'efs_time'
# Creates the fourth subplot in the grid
plt.subplot(2,2,4)  

sns.regplot(x='prior_tumor' , y='efs_time' , data=data_train , logistic=False)  # Linear regression plot
# Set title and axis labels for this plot
plt.title("Linear Regression: Effect of Prior_Tumor on EFS-Time")  # Title of the plot
plt.xlabel("Prior_Tumor (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS-Time (Event-Free Survival)")  # Y-axis label

# Adjust the layout to prevent overlap and ensure the plots fit well
plt.tight_layout()
# Show the complete plot with all four subplots
plt.show()



# Test the function with 'cyto_score_detail' as the first variable and 'efs' as the second variable.
# This checks if there's a significant association between these two categorical variables.
chi2_contingency_test('cyto_score_detail', 'efs')


# Test the function with 'tce_match' as the first variable and 'efs' as the second variable.
chi2_contingency_test('tce_match' , 'efs')


# Test the function with 'hepatic_severe' as the first variable and 'efs' as the second variable.
chi2_contingency_test('hepatic_severe' , 'efs')


# Test the function with 'prior_tumor' as the first variable and 'efs' as the second variable.
chi2_contingency_test('prior_tumor' , 'efs')


# Plot Kaplan-Meier curve for 'cyto_score_detail' column
plot_kaplan_meier('cyto_score_detail', 'Kaplan-Meier Survival Curve for Cyto_Score_Detail')



# Plot Kaplan-Meier curve for 'tce_match' column
plot_kaplan_meier('tce_match', 'Kaplan-Meier Survival Curve for Tce_Match')


# plot Kaplan-Meier curve for 'hepatic_severe' column
plot_kaplan_meier('hepatic_severe' , 'Kaplan-Meier Survival Curve for Hepatic_Severe')


# plot Kaplan-Meier curve for 'prior_tumor' column
plot_kaplan_meier('prior_tumor' , 'Kaplan-Meier Survival Curve for Prior_Tumor')


# Set the overall figure size
plt.figure(figsize=(12, 8))

# Plot 1: cyto_score_detail vs EFS
plt.subplot(2, 2, 1)

sns.countplot(x='cyto_score_detail', hue='efs', data=data_train, palette='Set2')  # Create a count plot
plt.title("Distribution of Cyto_Score_Detail vs EFS")  # Add title for the plot
plt.xlabel("Cyto_Score_Detail")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Plot 2: tce_match vs EFS
plt.subplot(2, 2, 2)

sns.countplot(x='tce_match', hue='efs', data=data_train, palette='Set2')  # Create a count plot
plt.title("Distribution of Tce_Match vs EFS")  # Add title for the plot
plt.xlabel("Tce_Match")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Plot 3: hepatic_severe vs EFS
plt.subplot(2, 2, 3)

sns.countplot(x='hepatic_severe', hue='efs', data=data_train, palette='Set2')  # Create a count plot
plt.title("Distribution of Hepatic_Severe vs EFS")  # Add title for the plot
plt.xlabel("Hepatic_Severe")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Plot 4: prior_tumor vs EFS
plt.subplot(2, 2, 4)

sns.countplot(x='prior_tumor', hue='efs', data=data_train, palette='Set2')  # Create a count plot
plt.title("Distribution of Prior_Tumor vs EFS")  # Add title for the plot
plt.xlabel("Prior_Tumor")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Adjust the layout to prevent overlap and ensure everything fits
plt.tight_layout()
# Display the plot
plt.show()


#17- Is there a relationship between the 'donor_age' column and 'age_at_hct' and 'peptic_ulcer' 
# and 'gvhd_proph'  the 'efs' or 'efs_time' columns?

# Logistic Regression Model using the Logit function ?
# Build a logistic regression model
model_logistic = logit(formula='efs ~ + C(peptic_ulcer) + C(gvhd_proph)' , data=data_train).fit()

# Print the Model Logistic results
print(model_logistic.summary()) 


# Regression Model using the OLS function ?
model_ols = ols(formula='efs_time ~ C(peptic_ulcer) + (gvhd_proph)' , data=data_train).fit()

# print the model OLS results
print(model_ols.summary())


plt.figure(figsize=(12,8))

# First subplot: Logistic regression plot for 'donor_age' vs 'efs'
# Creates the first subplot in a 2x2 grid
plt.subplot(2,2,1)  

sns.regplot(x='donor_age' , y='efs' , data=data_train , logistic=True)  # Logistic regression plot
# Set title and axis labels for this plot
plt.title("Logistic Regression: Effect of Donor_Age on EFS")  # Title of the plot
plt.xlabel("Donor_Age (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS (Event-Free Survival)")  # Y-axis label

# Second subplot: Logistic regression plot for 'age_at_hct' vs 'efs'
# Creates the second subplot in the grid
plt.subplot(2,2,2) 

sns.regplot(x='age_at_hct' , y='efs' , data=data_train , logistic=True)  # Logistic regression plot
# Set title and axis labels for this plot
plt.title("Logistic Regression: Effect of Age_At_Hct on EFS")  # Title of the plot
plt.xlabel("Age_At_Hct (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS (Event-Free Survival)")  # Y-axis label

# Third subplot: Logistic regression plot for 'peptic_ulcer' vs 'efs'
# Creates the third subplot in the grid
plt.subplot(2,2,3)

sns.regplot(x='peptic_ulcer' , y='efs' , data=data_train , logistic=True)  # Logistic regression plot
# Set title and axis labels for this plot
plt.title("Logistic Regression: Effect of Peptic_Ulcer on EFS")  # Title of the plot
plt.xlabel("Peptic_Ulcer (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS (Event-Free Survival)")  # Y-axis label

# Fourth subplot: Logistic regression plot for 'gvhd_proph' vs 'efs'
# Creates the fourth subplot in the grid
plt.subplot(2,2,4)  

sns.regplot(x='gvhd_proph' , y='efs' , data=data_train , logistic=True)  # Logistic regression plot
# Set title and axis labels for this plot
plt.title("Logistic Regression: Effect of Gvhd_Proph on EFS")  # Title of the plot
plt.xlabel("Gvhd_Proph (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS (Event-Free Survival)")  # Y-axis label

# Adjust the layout to prevent overlap and ensure the plots fit well
plt.tight_layout()
# Show the complete plot with all four subplots
plt.show()


plt.figure(figsize=(12,8))

# First subplot: Linear regression plot for 'donor_age' vs 'efs_time'
# Creates the first subplot in a 2x2 grid
plt.subplot(2,2,1)  

sns.regplot(x='donor_age' , y='efs_time' , data=data_train , logistic=False)  # Linear regression plot
# Set title and axis labels for this plot
plt.title("Linear Regression: Effect of Donor_Age on EFS-Time")  # Title of the plot
plt.xlabel("Donor_Age (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS-Time (Event-Free Survival)")  # Y-axis label

# Second subplot: Linear regression plot for 'age_at_hct' vs 'efs_time'
# Creates the second subplot in the grid
plt.subplot(2,2,2) 

sns.regplot(x='age_at_hct' , y='efs_time' , data=data_train , logistic=False)  # Linear regression plot
# Set title and axis labels for this plot
plt.title("Linear Regression: Effect of Age_At_Hct on EFS-Time")  # Title of the plot
plt.xlabel("Age_At_Hct (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS-Time (Event-Free Survival)")  # Y-axis label

# Third subplot: Logistic regression plot for 'peptic_ulcer' vs 'efs_time'
# Creates the third subplot in the grid
plt.subplot(2,2,3)

sns.regplot(x='peptic_ulcer' , y='efs_time' , data=data_train , logistic=False)  # Linear regression plot
# Set title and axis labels for this plot
plt.title("Linear Regression: Effect of Peptic_Ulcer on EFS-Time")  # Title of the plot
plt.xlabel("Peptic_Ulcer (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS-Time (Event-Free Survival)")  # Y-axis label

# Fourth subplot: Logistic regression plot for 'gvhd_proph' vs 'efs_time'
# Creates the fourth subplot in the grid
plt.subplot(2,2,4)  

sns.regplot(x='gvhd_proph' , y='efs_time' , data=data_train , logistic=False)  # Linear regression plot
# Set title and axis labels for this plot
plt.title("Linear Regression: Effect of Gvhd_Proph on EFS-Time")  # Title of the plot
plt.xlabel("Gvhd_Proph (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS-Time (Event-Free Survival)")  # Y-axis label

# Adjust the layout to prevent overlap and ensure the plots fit well
plt.tight_layout()
# Show the complete plot with all four subplots
plt.show()


# Test the function with 'peptic_ulcer' as the first variable and 'efs' as the second variable.
chi2_contingency_test('peptic_ulcer' , 'efs')


# Test the function with 'gvhd_proph' as the first variable and 'efs' as the second variable.
chi2_contingency_test('gvhd_proph' , 'efs')


# Create a copy of the data to avoid modifying the original dataset
my_data = data_train.copy()
# Define a function to categorize a numerical column into age groups based on bins
def categorize_age_group(bins , labels , column_name):
    # Use pd.cut to discretize the continuous values into specified age groups
    my_data[column_name] = pd.cut(my_data[column_name] , bins=bins, labels=labels)
    return my_data


# Define the bins (ranges) for the age groups
bins = [0,10,20,30,40,50,60,70] # The start and end points of the age ranges
# Define the labels for each age group
labels = ['0-10','10-20','20-30','30-40','40-50','50-60','60-70'] # Corresponding labels for the bins

# Apply the function to categorize the 'age_at_hct' column into age groups
column_age_at_hct = categorize_age_group(bins , labels , 'age_at_hct')



# Set the overall figure size
plt.figure(figsize=(12, 8))

# plot 1: donogr-age
plt.subplot(2,2,1)
# countplot
sns.countplot(x='age_at_hct', hue='efs',data=my_data, palette='Set2')  # Create a count plot
plt.title("Distribution of Age-at-HCT vs EFS")  # Add title for the plot
plt.xlabel("Age-at-HCT")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis


# Plot 2: age_at_hct vs EFS
plt.subplot(2, 2, 2)

sns.scatterplot(x='age_at_hct', y='efs_time', data=my_data)  # Create a count plot
plt.title("Event-Free Survival Time vs Age at HCT")  # Add title for the plot
plt.xlabel("Age-At-Hct")  # Label for the x-axis
plt.ylabel("Event-Free Survival Time (months)")  # Label for the y-axis

# Plot 3: peptic_ulcer vs EFS
plt.subplot(2, 2, 3)

sns.countplot(x='peptic_ulcer', hue='efs', data=data_train, palette='Set2')  # Create a count plot
plt.title("Distribution of Peptic_Ulcer vs EFS")  # Add title for the plot
plt.xlabel("Peptic_Ulcer")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Plot 4: gvhd_proph vs EFS
plt.subplot(2, 2, 4)

sns.countplot(x='gvhd_proph', hue='efs', data=data_train, palette='Set2')  # Create a count plot
plt.title("Distribution of gvhd_proph vs EFS")  # Add title for the plot
plt.xlabel("Gvhd_Proph")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Adjust the layout to prevent overlap and ensure everything fits
plt.tight_layout()
# Display the plot
plt.show()


# Define age bins and corresponding labels
bins = [18, 30, 40, 50, 60, 70, 85]  # Represents age ranges as boundaries for grouping
labels = ["18-30", "30-40", "40-50", "50-60", "60-70", "70-85"]  # Labels for each age group

# Further categorize or process the 'donor_age' column using the categorize_age_group function
column_donor_age = categorize_age_group(bins, labels, 'donor_age')



# Set the overall figure size
plt.figure(figsize=(12, 8))

# Plot 4: donor-age vs graft_type
plt.subplot(2, 2, 1)

sns.countplot(x='donor_age', hue='graft_type', data=my_data, palette='Set2')  # Create a count plot
plt.title("Distribution of Donor_Age vs Graft-Type")  # Add title for the plot
plt.xlabel("Donor_Age")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Plot 4: age-at-hct vs rituximab
plt.subplot(2, 2, 2)

sns.countplot(x='age_at_hct', hue='rituximab', data=my_data, palette='Set2')  # Create a count plot
plt.title("Distribution of Age-at-HCT vs EFS")  # Add title for the plot
plt.xlabel("Age-at-HCT")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Plot 3: age_at_hct vs conditioning_intensity:
plt.subplot(2, 2, 3)

sns.countplot(x='age_at_hct', hue='conditioning_intensity', data=my_data, palette='Set2')  # Create a count plot
plt.title("Distribution of Age-at-HCT vs Conditioning_Intensity")  # Add title for the plot
plt.xlabel("Age-at-HCT")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Plot 4: age-at-hct vs in_vivo_tcd 
plt.subplot(2, 2, 4)

sns.countplot(x='age_at_hct', hue='in_vivo_tcd', data=my_data, palette='Set2')  # Create a count plot
plt.title("Distribution of Age-at-HCT vs In_vivo_TCD")  # Add title for the plot
plt.xlabel("In_vivo_TCD")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Adjust the layout to prevent overlap and ensure everything fits
plt.tight_layout()
# Display the plot
plt.show()


# Create figure with 2 vertical subplots (2 rows, 1 column)
fig, axes = plt.subplots(2, 1, figsize=(14, 12))  # Increased height for vertical spacing

# -----------------------------------------------------------------
# Plot 1: Top plot (age_at_hct vs gvhd_proph)
# -----------------------------------------------------------------
hue_categories_1 = sorted(my_data['gvhd_proph'].unique())
palette_1 = sns.color_palette("Blues", n_colors=len(hue_categories_1))

sns.histplot(data=my_data,x='age_at_hct',hue='gvhd_proph',multiple='stack',palette=palette_1,ax=axes[0],legend=False)
axes[0].set_title("Stacked Histogram of Age at HCT and GVHD Prophylaxis", pad=20)
axes[0].set_xlabel("Age at HCT")
axes[0].set_ylabel("Count")

# Legend for top plot (right side)
legend_patches_1 = [Patch(color=color, label=label) for color, label in zip(palette_1, hue_categories_1)]
axes[0].legend(handles=legend_patches_1,bbox_to_anchor=(1.05, 1),loc='upper left',title='GVHD Prophylaxis')

# -----------------------------------------------------------------
# Plot 2: Bottom plot (age_at_hct vs tbi_status)
# -----------------------------------------------------------------
hue_categories_2 = sorted(my_data['tbi_status'].unique())
palette_2 = sns.color_palette("Blues", n_colors=len(hue_categories_2))

sns.histplot(data=my_data,x='age_at_hct',hue='tbi_status',multiple='stack',palette=palette_2,ax=axes[1],legend=False)

axes[1].set_title("Stacked Histogram of Age at HCT and Tbi-Status", pad=20)
axes[1].set_xlabel("Age at HCT")
axes[1].set_ylabel("Count")

# Legend for bottom plot (right side)
legend_patches_2 = [Patch(color=color, label=label) for color, label in zip(palette_2, hue_categories_2)]
axes[1].legend(handles=legend_patches_2,bbox_to_anchor=(1.0, 1),loc='upper left',title='TBI Status')

# -----------------------------------------------------------------
# Final layout adjustments
# -----------------------------------------------------------------
plt.tight_layout()
plt.subplots_adjust(hspace=0.3,right=0.85)

plt.show()


# plot Kaplan-Meier curve for 'peptic_ulcer' column
plot_kaplan_meier('peptic_ulcer' , 'Kaplan-Meier Survival Curve for Peptic_Ulcer')


# plot Kaplan-Meier curve for 'gvhd_proph' column
plot_kaplan_meier('gvhd_proph' , 'Kaplan-Meier Survival Curve for Gvhd_Proph')


# 18- Is there a relationship between the 'hla_match_b_high' column and 'hla_low_res_8' 
# and 'hla_low_res_10' and 'hla_match_drb1_high'  the 'efs' or 'efs_time' columns?

# Logistic Regression Model using the Logit function ?
# Build a logistic regression model
model_logistic = logit(formula='efs ~ C(hla_match_b_high) + C(hla_low_res_8) + C(hla_low_res_10) + (hla_match_drb1_high)' , data=data_train).fit()

# Print the Model Logistic results
print(model_logistic.summary()) 


# Regression Model using the OLS function ?
model_ols = ols(formula='efs_time ~ C(hla_match_b_high) + C(hla_low_res_8) + C(hla_low_res_10) + (hla_match_drb1_high)' , data=data_train).fit()

# print the model OLS results
print(model_ols.summary())


plt.figure(figsize=(12,8))

# First subplot: Logistic regression plot for 'hla_match_b_high' vs 'efs'
# Creates the first subplot in a 2x2 grid
plt.subplot(2,2,1)  

sns.regplot(x='hla_match_b_high' , y='efs' , data=data_train , logistic=True)  # Logistic regression plot
# Set title and axis labels for this plot
plt.title("Logistic Regression: Effect of HLA_match_B_high on EFS")  # Title of the plot
plt.xlabel("HLA_match_B_high (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS (Event-Free Survival)")  # Y-axis label

# Second subplot: Logistic regression plot for 'hla_low_res_8' vs 'efs'
# Creates the second subplot in the grid
plt.subplot(2,2,2) 

sns.regplot(x='hla_low_res_8' , y='efs' , data=data_train , logistic=True)  # Logistic regression plot
# Set title and axis labels for this plot
plt.title("Logistic Regression: Effect of HLA_LOW_RES_8 on EFS")  # Title of the plot
plt.xlabel("HLA_LOW_RES_8 (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS (Event-Free Survival)")  # Y-axis label

# Third subplot: Logistic regression plot for 'hla_low_res_10' vs 'efs'
# Creates the third subplot in the grid
plt.subplot(2,2,3)

sns.regplot(x='hla_low_res_10' , y='efs' , data=data_train , logistic=True)  # Logistic regression plot
# Set title and axis labels for this plot
plt.title("Logistic Regression: Effect of HLA_LOW_RES_10 on EFS")  # Title of the plot
plt.xlabel("HLA_LOW_RES_10 (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS (Event-Free Survival)")  # Y-axis label

# Fourth subplot: Logistic regression plot for 'hla_match_drb1_high' vs 'efs'
# Creates the fourth subplot in the grid
plt.subplot(2,2,4)  

sns.regplot(x='hla_match_drb1_high' , y='efs' , data=data_train , logistic=True)  # Logistic regression plot
# Set title and axis labels for this plot
plt.title("Logistic Regression: Effect of HLA_match_drb1_high on EFS")  # Title of the plot
plt.xlabel("HLA_match_drb1_high (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS (Event-Free Survival)")  # Y-axis label

# Adjust the layout to prevent overlap and ensure the plots fit well
plt.tight_layout()
# Show the complete plot with all four subplots
plt.show()


plt.figure(figsize=(12,8))

# First subplot: Linear regression plot for 'hla_match_b_high' vs 'efs_time'
# Creates the first subplot in a 2x2 grid
plt.subplot(2,2,1)  

sns.regplot(x='hla_match_b_high' , y='efs_time' , data=data_train , logistic=False)  # Linear regression plot
# Set title and axis labels for this plot
plt.title("Linear Regression: Effect of hla_match_B_high on EFS-Time")  # Title of the plot
plt.xlabel("hla_match_B_high (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS-Time (Event-Free Survival)")  # Y-axis label

# Second subplot: Linear regression plot for 'hla_low_res_8' vs 'efs_time'
# Creates the second subplot in the grid
plt.subplot(2,2,2) 

sns.regplot(x='hla_low_res_8' , y='efs_time' , data=data_train , logistic=False)  # Linear regression plot
# Set title and axis labels for this plot
plt.title("Linear Regression: Effect of HLA_low_res_8 on EFS-Time")  # Title of the plot
plt.xlabel("HLA_low_res_8 (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS-Time (Event-Free Survival)")  # Y-axis label

# Third subplot: Logistic regression plot for 'hla_low_res_10' vs 'efs_time'
# Creates the third subplot in the grid
plt.subplot(2,2,3)

sns.regplot(x='hla_low_res_10' , y='efs_time' , data=data_train , logistic=False)  # Linear regression plot
# Set title and axis labels for this plot
plt.title("Linear Regression: Effect of HLA_low_res_10 on EFS-Time")  # Title of the plot
plt.xlabel("HLA_low_res_10 (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS-Time (Event-Free Survival)")  # Y-axis label

# Fourth subplot: Logistic regression plot for 'hla_match_drb1_high' vs 'efs_time'
# Creates the fourth subplot in the grid
plt.subplot(2,2,4)  

sns.regplot(x='hla_match_drb1_high' , y='efs_time' , data=data_train , logistic=False)  # Linear regression plot
# Set title and axis labels for this plot
plt.title("Linear Regression: Effect of HLA_match_drb1_high on EFS-Time")  # Title of the plot
plt.xlabel("HLA_match_drb1_high (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS-Time (Event-Free Survival)")  # Y-axis label

# Adjust the layout to prevent overlap and ensure the plots fit well
plt.tight_layout()
# Show the complete plot with all four subplots
plt.show()


# plot Kaplan-Meier curve for 'hla_match_b_high' column
plot_kaplan_meier('hla_match_b_high' , 'Kaplan-Meier Survival Curve for HLA_match_B_high')


# plot Kaplan-Meier curve for 'hla_low_res_8' column
plot_kaplan_meier('hla_low_res_8' , 'Kaplan-Meier Survival Curve for HLA_LOW_RES_8')


# plot Kaplan-Meier curve for 'hla_low_res_10' column
plot_kaplan_meier('hla_low_res_10' , 'Kaplan-Meier Survival Curve for HLA_LOW_RES_10')


# plot Kaplan-Meier curve for 'hla_match_drb1_high' column
plot_kaplan_meier('hla_match_drb1_high' , 'Kaplan-Meier Survival Curve for HLA_match_drb1_high')


# Set the overall figure size
plt.figure(figsize=(12, 8))

# Plot 1: hla_match_b_high vs EFS
plt.subplot(2, 2, 1)

sns.countplot(x='hla_match_b_high', hue='efs', data=data_train, palette='Set2')  # Create a count plot
plt.title("Distribution of HLA_match_B_high vs EFS")  # Add title for the plot
plt.xlabel("HLA_match_B_high")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Plot 2: hla_low_res_8 vs EFS
plt.subplot(2, 2, 2)

sns.countplot(x='hla_low_res_8', hue='efs', data=data_train, palette='Set2')  # Create a count plot
plt.title("Distribution of HLA_low_RES_8 vs EFS")  # Add title for the plot
plt.xlabel("HLA_low_RES_8")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Plot 3: hla_low_res_10 vs EFS
plt.subplot(2, 2, 3)

sns.countplot(x='hla_low_res_10', hue='efs', data=data_train, palette='Set2')  # Create a count plot
plt.title("Distribution of HLA-LOW-RES vs EFS")  # Add title for the plot
plt.xlabel("HLA_LOW_RES_10")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Plot 4: hla_match_drb1_high vs EFS
plt.subplot(2, 2, 4)

sns.countplot(x='hla_match_drb1_high', hue='efs', data=data_train, palette='Set2')  # Create a count plot
plt.title("Distribution of HLA_match_drb1_high vs EFS")  # Add title for the plot
plt.xlabel("HLA_match_drb1_high")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Adjust the layout to prevent overlap and ensure everything fits
plt.tight_layout()
# Display the plot
plt.show()


# Logistic Regression Model using the Logit function ?
# Build a logistic regression model
model_logistic = logit(formula='efs ~ C(rheum_issue) + C(sex_match) + C(tce_div_match) + (race_group)' , data=data_train).fit()

# Print the Model Logistic results
print(model_logistic.summary()) 


# 19- Is there a relationship between the 'rheum_issue' column and 'sex_match' 
# and 'tce_div_match' and 'race_group'  the 'efs' or 'efs_time' columns?


# Logistic Regression Model using the Logit function ?
# Build a logistic regression model
model_logistic = logit(formula='efs ~ C(rheum_issue) + C(sex_match) + C(tce_div_match) + (race_group)' , data=data_train).fit()

# Print the Model Logistic results
print(model_logistic.summary())


# Regression Model using the OLS function ?
model_ols = ols(formula='efs_time ~ C(rheum_issue) + C(sex_match) + C(tce_div_match) + (race_group)' , data=data_train).fit()

# print the model OLS results
print(model_ols.summary())


plt.figure(figsize=(12,8))

# First subplot: Logistic regression plot for 'rheum_issue' vs 'efs'
# Creates the first subplot in a 2x2 grid
plt.subplot(2,2,1)  

sns.regplot(x='rheum_issue' , y='efs' , data=data_train , logistic=True)  # Logistic regression plot
# Set title and axis labels for this plot
plt.title("Logistic Regression: Effect of Rheum_Issue on EFS")  # Title of the plot
plt.xlabel("Rheum_Issue (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS (Event-Free Survival)")  # Y-axis label

# Second subplot: Logistic regression plot for 'sex_match' vs 'efs'
# Creates the second subplot in the grid
plt.subplot(2,2,2) 

sns.regplot(x='sex_match' , y='efs' , data=data_train , logistic=True)  # Logistic regression plot
# Set title and axis labels for this plot
plt.title("Logistic Regression: Effect of Sex_Match on EFS")  # Title of the plot
plt.xlabel("Sex_Match (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS (Event-Free Survival)")  # Y-axis label

# Third subplot: Logistic regression plot for 'tce_div_match' vs 'efs'
# Creates the third subplot in the grid
plt.subplot(2,2,3)

sns.regplot(x='tce_div_match' , y='efs' , data=data_train , logistic=True)  # Logistic regression plot
# Set title and axis labels for this plot
plt.title("Logistic Regression: Effect of TCE_DIV_match on EFS")  # Title of the plot
plt.xlabel("TCE_DIV_match (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS (Event-Free Survival)")  # Y-axis label

# Fourth subplot: Logistic regression plot for 'race_group' vs 'efs'
# Creates the fourth subplot in the grid
plt.subplot(2,2,4)  

sns.regplot(x='race_group' , y='efs' , data=data_train , logistic=True)  # Logistic regression plot
# Set title and axis labels for this plot
plt.title("Logistic Regression: Effect of Race_Group on EFS")  # Title of the plot
plt.xlabel("Race_Group (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS (Event-Free Survival)")  # Y-axis label

# Adjust the layout to prevent overlap and ensure the plots fit well
plt.tight_layout()
# Show the complete plot with all four subplots
plt.show()




plt.figure(figsize=(12,8))

# First subplot: Linear regression plot for 'rheum_issue' vs 'efs_time'
# Creates the first subplot in a 2x2 grid
plt.subplot(2,2,1)  

sns.regplot(x='rheum_issue' , y='efs_time' , data=data_train , logistic=False)  # Linear regression plot
# Set title and axis labels for this plot
plt.title("Linear Regression: Effect of Rheum_Issue on EFS-Time")  # Title of the plot
plt.xlabel("Rheum_Issue (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS-Time (Event-Free Survival)")  # Y-axis label

# Second subplot: Linear regression plot for 'sex_match' vs 'efs_time'
# Creates the second subplot in the grid
plt.subplot(2,2,2) 

sns.regplot(x='sex_match' , y='efs_time' , data=data_train , logistic=False)  # Linear regression plot
# Set title and axis labels for this plot
plt.title("Linear Regression: Effect of Sex_Match on EFS-Time")  # Title of the plot
plt.xlabel("Sex_Match (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS-Time (Event-Free Survival)")  # Y-axis label

# Third subplot: Logistic regression plot for 'tce_div_match' vs 'efs_time'
# Creates the third subplot in the grid
plt.subplot(2,2,3)

sns.regplot(x='tce_div_match' , y='efs_time' , data=data_train , logistic=False)  # Linear regression plot
# Set title and axis labels for this plot
plt.title("Linear Regression: Effect of TCE_DIV_match on EFS-Time")  # Title of the plot
plt.xlabel("TCE_DIV_match (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS-Time (Event-Free Survival)")  # Y-axis label

# Fourth subplot: Logistic regression plot for 'race_group' vs 'efs_time'
# Creates the fourth subplot in the grid
plt.subplot(2,2,4)  

sns.regplot(x='race_group' , y='efs_time' , data=data_train , logistic=False)  # Linear regression plot
# Set title and axis labels for this plot
plt.title("Linear Regression: Effect of Race_Group on EFS-Time")  # Title of the plot
plt.xlabel("Race_Group (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS-Time (Event-Free Survival)")  # Y-axis label

# Adjust the layout to prevent overlap and ensure the plots fit well
plt.tight_layout()
# Show the complete plot with all four subplots
plt.show()




# Test the function with 'rheum_issue' as the first variable and 'efs' as the second variable.
# This checks if there's a significant association between these two categorical variables.
chi2_contingency_test('rheum_issue', 'efs')


# Test the function with 'sex_match' as the first variable and 'efs' as the second variable.
chi2_contingency_test('sex_match' , 'efs')


# Test the function with 'tce_div_match' as the first variable and 'efs' as the second variable.
chi2_contingency_test('tce_div_match' , 'efs')


# Test the function with 'race_group' as the first variable and 'efs' as the second variable.
chi2_contingency_test('race_group' , 'efs')


# plot Kaplan-Meier curve for 'rheum_issue' column
plot_kaplan_meier('rheum_issue' , 'Kaplan-Meier Survival Curve for Rheum_Issue')


# plot Kaplan-Meier curve for 'sex_match' column
plot_kaplan_meier('sex_match' , 'Kaplan-Meier Survival Curve for Sex_Match')


# plot Kaplan-Meier curve for 'tce_div_match' column
plot_kaplan_meier('tce_div_match' , 'Kaplan-Meier Survival Curve for TCE_DIV_match')


# plot Kaplan-Meier curve for 'race_group' column
plot_kaplan_meier('race_group' , 'Kaplan-Meier Survival Curve for Race_Group')


# Set the overall figure size
plt.figure(figsize=(12, 8))

# Plot 1: rheum_issue vs EFS
plt.subplot(2, 2, 1)

sns.countplot(x='rheum_issue', hue='efs', data=data_train, palette='Set2')  # Create a count plot
plt.title("Distribution of Rheum_Issue vs EFS")  # Add title for the plot
plt.xlabel("Rheum_Issue")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Plot 2: sex_match vs EFS
plt.subplot(2, 2, 2)

sns.countplot(x='sex_match', hue='efs', data=data_train, palette='Set2')  # Create a count plot
plt.title("Distribution of Sex_Match vs EFS")  # Add title for the plot
plt.xlabel("Sex_Match")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Plot 3: tce_div_match vs EFS
plt.subplot(2, 2, 3)

sns.countplot(x='tce_div_match', hue='efs', data=data_train, palette='Set2')  # Create a count plot
plt.title("Distribution of TCE_DIV_match vs EFS")  # Add title for the plot
plt.xlabel("TCE_DIV_match")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Plot 4: race_group vs EFS
plt.subplot(2, 2, 4)

sns.countplot(x='race_group', hue='efs', data=data_train, palette='Set2')  # Create a count plot
plt.title("Distribution of Race_Group vs EFS")  # Add title for the plot
plt.xlabel("Race_Group")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Adjust the layout to prevent overlap and ensure everything fits
plt.tight_layout()
# Display the plot
plt.show()


# 20- Is there a relationship between the 'comorbidity_score' column and 
# 'karnofsky_score' and 'hepatic_mild' and 'donor_related'  the 'efs' or 'efs_time' columns?

# Logistic Regression Model using the Logit function ?
# Build a logistic regression model
model_logistic = logit(formula='efs ~ C(comorbidity_score) + C(karnofsky_score) + C(hepatic_mild) + C(donor_related)' , data=data_train).fit()

# Print the Model Logistic results
print(model_logistic.summary()) 


# Regression Model using the OLS function ?
model_ols = ols(formula='efs_time ~ C(comorbidity_score) + C(karnofsky_score) + C(hepatic_mild) + C(donor_related)' , data=data_train).fit()

# print the model OLS results
print(model_ols.summary())


plt.figure(figsize=(12,8))

# First subplot: Logistic regression plot for 'comorbidity_score' vs 'efs'
# Creates the first subplot in a 2x2 grid
plt.subplot(2,2,1)  

sns.regplot(x='comorbidity_score' , y='efs' , data=data_train , logistic=True)  # Logistic regression plot
# Set title and axis labels for this plot
plt.title("Logistic Regression: Effect of Comorbidity_Score on EFS")  # Title of the plot
plt.xlabel("Comorbidity_Score (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS (Event-Free Survival)")  # Y-axis label

# Second subplot: Logistic regression plot for 'karnofsky_score' vs 'efs'
# Creates the second subplot in the grid
plt.subplot(2,2,2) 

sns.regplot(x='karnofsky_score' , y='efs' , data=data_train , logistic=True)  # Logistic regression plot
# Set title and axis labels for this plot
plt.title("Logistic Regression: Effect of Karnofsky_Score on EFS")  # Title of the plot
plt.xlabel("Karnofsky_Score (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS (Event-Free Survival)")  # Y-axis label

# Third subplot: Logistic regression plot for 'hepatic_mild' vs 'efs'
# Creates the third subplot in the grid
plt.subplot(2,2,3)

sns.regplot(x='hepatic_mild' , y='efs' , data=data_train , logistic=True)  # Logistic regression plot
# Set title and axis labels for this plot
plt.title("Logistic Regression: Effect of Hepatic_Mild on EFS")  # Title of the plot
plt.xlabel("Hepatic_Mild (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS (Event-Free Survival)")  # Y-axis label

# Fourth subplot: Logistic regression plot for 'donor_related' vs 'efs'
# Creates the fourth subplot in the grid
plt.subplot(2,2,4)  

sns.regplot(x='donor_related' , y='efs' , data=data_train , logistic=True)  # Logistic regression plot
# Set title and axis labels for this plot
plt.title("Logistic Regression: Effect of Donor_Related on EFS")  # Title of the plot
plt.xlabel("Donor_Related (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS (Event-Free Survival)")  # Y-axis label

# Adjust the layout to prevent overlap and ensure the plots fit well
plt.tight_layout()
# Show the complete plot with all four subplots
plt.show()


plt.figure(figsize=(12,8))
# First subplot: Linear regression plot for 'comorbidity_score' vs 'efs_time'
# Creates the first subplot in a 2x2 grid
plt.subplot(2,2,1)  

sns.regplot(x='comorbidity_score' , y='efs_time' , data=data_train , logistic=False)  # Linear regression plot
# Set title and axis labels for this plot
plt.title("Linear Regression: Effect of Comorbidity_Score on EFS-Time")  # Title of the plot
plt.xlabel("Comorbidity_Score (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS-Time (Event-Free Survival)")  # Y-axis label

# Second subplot: Linear regression plot for 'karnofsky_score' vs 'efs_time'
# Creates the second subplot in the grid
plt.subplot(2,2,2) 

sns.regplot(x='karnofsky_score' , y='efs_time' , data=data_train , logistic=False)  # Linear regression plot
# Set title and axis labels for this plot
plt.title("Linear Regression: Effect of Karnofsky_Score on EFS-Time")  # Title of the plot
plt.xlabel("Karnofsky_Score (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS-Time (Event-Free Survival)")  # Y-axis label

# Third subplot: Logistic regression plot for 'hepatic_mild' vs 'efs_time'
# Creates the third subplot in the grid
plt.subplot(2,2,3)

sns.regplot(x='hepatic_mild' , y='efs_time' , data=data_train , logistic=False)  # Linear regression plot
# Set title and axis labels for this plot
plt.title("Linear Regression: Effect of Hepatic_Mild on EFS-Time")  # Title of the plot
plt.xlabel("Hepatic_Mild (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS-Time (Event-Free Survival)")  # Y-axis label

# Fourth subplot: Logistic regression plot for 'donor_related' vs 'efs_time'
# Creates the fourth subplot in the grid
plt.subplot(2,2,4)  

sns.regplot(x='donor_related' , y='efs_time' , data=data_train , logistic=False)  # Linear regression plot
# Set title and axis labels for this plot
plt.title("Linear Regression: Effect of Donor_Related on EFS-Time")  # Title of the plot
plt.xlabel("Donor_Related (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS-Time (Event-Free Survival)")  # Y-axis label

# Adjust the layout to prevent overlap and ensure the plots fit well
plt.tight_layout()
# Show the complete plot with all four subplots
plt.show()


# plot Kaplan-Meier curve for 'comorbidity_score' column
plot_kaplan_meier('comorbidity_score' , 'Kaplan-Meier Survival Curve for Comorbidity_Score')


# plot Kaplan-Meier curve for 'karnofsky_score' column
plot_kaplan_meier('karnofsky_score' , 'Kaplan-Meier Survival Curve for Karnofsky_Score')


# plot Kaplan-Meier curve for 'hepatic_mild' column
plot_kaplan_meier('hepatic_mild' , 'Kaplan-Meier Survival Curve for Hepatic_Mild')


# plot Kaplan-Meier curve for 'donor_related' column
plot_kaplan_meier('donor_related' , 'Kaplan-Meier Survival Curve for Donor_Related')


# Test the function with 'comorbidity_score' as the first variable and 'efs' as the second variable.
# This checks if there's a significant association between these two categorical variables.
chi2_contingency_test('comorbidity_score', 'efs')


# Test the function with 'karnofsky_score' as the first variable and 'efs' as the second variable.
chi2_contingency_test('karnofsky_score' , 'efs')


# Test the function with 'hepatic_mild' as the first variable and 'efs' as the second variable.
chi2_contingency_test('hepatic_mild' , 'efs')


# Test the function with 'donor_related' as the first variable and 'efs' as the second variable.
chi2_contingency_test('donor_related' , 'efs')


# Set the overall figure size
plt.figure(figsize=(12, 8))

# Plot 1: comorbidity_score vs EFS
plt.subplot(2, 2, 1)

sns.countplot(x='comorbidity_score', hue='efs', data=data_train, palette='Set2')  # Create a count plot
plt.title("Distribution of Comorbidity_Score vs EFS")  # Add title for the plot
plt.xlabel("Comorbidity_Score")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Plot 2: karnofsky_score vs EFS
plt.subplot(2, 2, 2)

sns.countplot(x='karnofsky_score', hue='efs', data=data_train, palette='Set2')  # Create a count plot
plt.title("Distribution of Karnofsky_Score vs EFS")  # Add title for the plot
plt.xlabel("Karnofsky_Score")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Plot 3: hepatic_mild vs EFS
plt.subplot(2, 2, 3)

sns.countplot(x='hepatic_mild', hue='efs', data=data_train, palette='Set2')  # Create a count plot
plt.title("Distribution of Hepatic_Mild vs EFS")  # Add title for the plot
plt.xlabel("Hepatic_Mild")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Plot 4: donor_related vs EFS
plt.subplot(2, 2, 4)

sns.countplot(x='donor_related', hue='efs', data=data_train, palette='Set2')  # Create a count plot
plt.title("Distribution of Donor_Related vs EFS")  # Add title for the plot
plt.xlabel("Donor_Related")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Adjust the layout to prevent overlap and ensure everything fits
plt.tight_layout()
# Display the plot
plt.show()


# 21- Is there a relationship between the 'melphalan_dose' column 
# and 'cardiac' and 'pulm_moderate'  the 'efs' or 'efs_time' columns?


# Logistic Regression Model using the Logit function ?
# Build a logistic regression model
model_logistic = logit(formula='efs ~ C(melphalan_dose) + C(cardiac) + C(pulm_moderate)', data=data_train).fit()

# Print the Model Logistic results
print(model_logistic.summary())


# Regression Model using the OLS function ?
model_ols = ols(formula='efs_time ~ C(melphalan_dose) + C(cardiac) + C(pulm_moderate)' , data=data_train).fit()

# print the model OLS results
print(model_ols.summary())


plt.figure(figsize=(12,8))

# First subplot: Logistic regression plot for 'melphalan_dose' vs 'efs'
# Creates the first subplot in a 2x2 grid
plt.subplot(2,2,1)  

sns.regplot(x='melphalan_dose' , y='efs' , data=data_train , logistic=True)  # Logistic regression plot
# Set title and axis labels for this plot
plt.title("Logistic Regression: Effect of Melphalan_Dose on EFS")  # Title of the plot
plt.xlabel("Melphalan_Dose (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS (Event-Free Survival)")  # Y-axis label

# Second subplot: Logistic regression plot for 'cardiac' vs 'efs'
# Creates the second subplot in the grid
plt.subplot(2,2,2) 

sns.regplot(x='cardiac' , y='efs' , data=data_train , logistic=True)  # Logistic regression plot
# Set title and axis labels for this plot
plt.title("Logistic Regression: Effect of Cardiac on EFS")  # Title of the plot
plt.xlabel("Cardiac (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS (Event-Free Survival)")  # Y-axis label

# Third subplot: Logistic regression plot for 'pulm_moderate' vs 'efs'
# Creates the third subplot in the grid
plt.subplot(2,2,3)

sns.regplot(x='pulm_moderate' , y='efs' , data=data_train , logistic=True)  # Logistic regression plot
# Set title and axis labels for this plot
plt.title("Logistic Regression: Effect of Pulm_Moderate on EFS")  # Title of the plot
plt.xlabel("Pulm_Moderate (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS (Event-Free Survival)")  # Y-axis label

# Adjust the layout to prevent overlap and ensure the plots fit well
plt.tight_layout()
# Show the complete plot with all four subplots
plt.show()


plt.figure(figsize=(12,8))
# First subplot: Linear regression plot for 'melphalan_dose' vs 'efs_time'
# Creates the first subplot in a 2x2 grid
plt.subplot(2,2,1)  

sns.regplot(x='melphalan_dose' , y='efs_time' , data=data_train , logistic=False)  # Linear regression plot
# Set title and axis labels for this plot
plt.title("Linear Regression: Effect of Melphalan_Dose on EFS-Time")  # Title of the plot
plt.xlabel("Melphalan_Dose (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS-Time (Event-Free Survival)")  # Y-axis label

# Second subplot: Linear regression plot for 'cardiac' vs 'efs_time'
# Creates the second subplot in the grid
plt.subplot(2,2,2) 

sns.regplot(x='cardiac' , y='efs_time' , data=data_train , logistic=False)  # Linear regression plot
# Set title and axis labels for this plot
plt.title("Linear Regression: Effect of Cardiac on EFS-Time")  # Title of the plot
plt.xlabel("Cardiac (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS-Time (Event-Free Survival)")  # Y-axis label

# Third subplot: Logistic regression plot for 'pulm_moderate' vs 'efs_time'
# Creates the third subplot in the grid
plt.subplot(2,2,3)

sns.regplot(x='pulm_moderate' , y='efs_time' , data=data_train , logistic=False)  # Linear regression plot
# Set title and axis labels for this plot
plt.title("Linear Regression: Effect of Pulm_Moderate on EFS-Time")  # Title of the plot
plt.xlabel("Pulm_Moderate (Disease Risk Index)")  # X-axis label
plt.ylabel("EFS-Time (Event-Free Survival)")  # Y-axis label

# Adjust the layout to prevent overlap and ensure the plots fit well
plt.tight_layout()
# Show the complete plot with all four subplots
plt.show()


# plot Kaplan-Meier curve for 'melphalan_dose' column
plot_kaplan_meier('melphalan_dose' , 'Kaplan-Meier Survival Curve for Comorbidity_Score')


# plot Kaplan-Meier curve for 'cardiac' column
plot_kaplan_meier('cardiac' , 'Kaplan-Meier Survival Curve for Karnofsky_Score')


# plot Kaplan-Meier curve for 'pulm_moderate' column
plot_kaplan_meier('pulm_moderate' , 'Kaplan-Meier Survival Curve for Hepatic_Mild')


# Test the function with 'melphalan_dose' as the first variable and 'efs' as the second variable.
# This checks if there's a significant association between these two categorical variables.
chi2_contingency_test('melphalan_dose', 'efs')


# Test the function with 'cardiac' as the first variable and 'efs' as the second variable.
chi2_contingency_test('cardiac' , 'efs')


# Test the function with 'pulm_moderate' as the first variable and 'efs' as the second variable.
chi2_contingency_test('pulm_moderate' , 'efs')


# Set the overall figure size
plt.figure(figsize=(12, 8))

# Plot 1: melphalan_dose vs EFS
plt.subplot(2, 2, 1)

sns.countplot(x='melphalan_dose', hue='efs', data=data_train, palette='Set2')  # Create a count plot
plt.title("Distribution of Melphalan_Dose vs EFS")  # Add title for the plot
plt.xlabel("Melphalan_Dose")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Plot 2: cardiac vs EFS
plt.subplot(2, 2, 2)

sns.countplot(x='cardiac', hue='efs', data=data_train, palette='Set2')  # Create a count plot
plt.title("Distribution of Cardiac vs EFS")  # Add title for the plot
plt.xlabel("Cardiac")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Plot 3: pulm_moderate vs EFS
plt.subplot(2, 2, 3)

sns.countplot(x='pulm_moderate', hue='efs', data=data_train, palette='Set2')  # Create a count plot
plt.title("Distribution of Pulm_Moderate vs EFS")  # Add title for the plot
plt.xlabel("Pulm_Moderate")  # Label for the x-axis
plt.ylabel("Count")  # Label for the y-axis

# Adjust the layout to prevent overlap and ensure everything fits
plt.tight_layout()
# Display the plot
plt.show()


# Separates features (X) and target variable (y), with 'efs_combined' as the target.
x = data_train.drop(['efs' , 'efs_time' , 'ID'], axis=1, inplace=False)  # Features (all columns except 'efs' and 'efs_time' and 'ID')
y_target = data_train['efs']

# Step 1: Split data into 70% training and 30% temporary set (which will be further split)
x_train, x_temp, y_train, y_temp = train_test_split(x, y_target, test_size=0.3, stratify=y_target ,random_state=42)

# Step 2: Split the temporary set into 15% validation (dev) and 15% test
x_dev, x_test, y_dev, y_test = train_test_split(x_temp, y_temp, test_size=0.5, stratify=y_temp ,random_state=42)

# Print dataset sizes for verification
print(f"X_train: {x_train.shape},X_dev: {x_dev.shape} ,X_test: {x_test.shape}")
print(f"Y_train: {y_train.shape},Y_dev: {y_dev.shape} ,Y_test: {y_test.shape}")


# Step 3: Apply standard scaling to the features to standardize the data
scaler = StandardScaler()  # Initialize the scaler
x_train_scaled = scaler.fit_transform(x_train)  # Fit the scaler on the training data and transform it
x_dev_scaled = scaler.transform(x_dev)  # Transform the test data based on the scaler fit on the training data
x_test_scaled = scaler.transform(x_test)  # Transform the test data based on the scaler fit on the training data


# Instantiate the XGBClassifier model
xgb_model = XGBClassifier(n_estimators=1000, colsample_bytree=0.7, objective='binary:logistic')

# Define a grid of hyperparameters for tuning the model  
cv_params = {
    'learning_rate': np.linspace(0.01, 0.3, 5,endpoint=True),                # Learning rate
    'max_depth': np.arange(4, 20, 4),                                        # Depth of trees
    'min_child_weight': np.arange(1,6,1),                                    # Minimum sum of instance weights needed in a child node to control model complexity  
}
# Scoring
scoring = {'accuracy', 'precision', 'recall', 'f1'}


# Initialize GridSearchCV
grid_search = GridSearchCV (
    estimator=xgb_model,             # XGBoost model
    param_grid=cv_params,            # Hyperparameter grid
    scoring=scoring,                 # Evaluation metric: accuracy
    cv=5,                            # Number of cross-validation splits
    verbose=1,                       # Print progress
    refit='f1',
    n_jobs=-1,                       # Use all available CPU cores
)


# Run the search for the best hyperparameter combination
grid_search.fit(
    x_train_scaled, y_train,
    eval_set=[(x_dev_scaled, y_dev)],     # Use validation set to monitor performance
    early_stopping_rounds=50,             # Stop if validation loss doesn't improve for 10 rounds
    eval_metric="logloss",                # Metric to monitor
    verbose=True,                        # Print progress
)


# Print the best parameters
print(f"Best hyperparameters: {grid_search.best_params_}")
print(f"Best F1-Score: {grid_search.best_score_}")


# Store the 'ID' column separately before dropping it
id_column = data_test['ID']

# Drop the 'ID' column from the dataset without modifying the original DataFrame
data_test = data_test.drop(['ID'], axis=1, inplace=False)


# Use the best model found by GridSearchCV
best_model = grid_search.best_estimator_


# Compute learning curves for the model
train_sizes, train_scores, test_scores = learning_curve(
    best_model, x_train_scaled, y_train, cv=5, scoring='accuracy')

# Calculate the mean accuracy for training and validation sets
train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)

# Plot the learning curves
plt.plot(train_sizes, train_mean, label='Training Score', marker='o')
plt.plot(train_sizes, test_mean, label='Validation Score', marker='o')

# Add labels and title to the plot
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy Score")
plt.title("Learning Curve")
plt.legend()
plt.show()


# Calculate model accuracy on the training data
train_accuracy = accuracy_score(y_train, best_model.predict(x_train_scaled))

# Calculate model accuracy on the test data
test_accuracy = accuracy_score(y_test, best_model.predict(x_test_scaled))

# Print the training and testing accuracy scores
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)


# Make predictions on the test set
y_pred = best_model.predict(x_test_scaled)

# Predict
print(y_pred)


# Calculate and print the Confusion_matrix (CM) to evaluate model performance.
cm = confusion_matrix(y_test , y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()


# Generate and display the ROC curve using the predicted probabilities
RocCurveDisplay.from_predictions(y_test, y_pred)

# Show the plot
plt.show()


# Generate a classification report for the model's predictions
# The report includes key metrics such as precision, recall, F1-score, and support
# This helps evaluate the model's performance in more detail.
report = classification_report(y_test , y_pred)

# Print the generated classification report
print(f"Report: {report}")


# Make predictions on the data test set
y_pred_data_test = best_model.predict(data_test)

# Predict
print(y_pred_data_test)


# Create a DataFrame for submission with columns 'ID' and 'Prediction'
submission = pd.DataFrame(columns=['ID', 'prediction'])

# Assign the values of id_column to the 'ID' column
submission['ID'] = id_column

# Assign the predicted values (y_pred) to the 'Prediction' column
submission['prediction'] = y_pred_data_test

# Save the submission DataFrame as a CSV file named 'submission.csv' without the index
submission.to_csv('submission.csv', index=False)


# Display the first few rows of the submission DataFrame
submission.head()

