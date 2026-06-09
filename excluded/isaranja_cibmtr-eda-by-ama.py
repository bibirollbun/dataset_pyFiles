# installing  library

#!pip install -q "ray>=2.10.0,<2.40.0"
#!pip install -q autogluon.tabular
#!pip install --no-index --find-links=/kaggle/input/cibmtr-pip-install-autogluon/autogluon "ray>=2.10.0,<2.40.0" autogluon.tabular -q

#!pip install pycaret -q

!pip install -q /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install -q /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install -q /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install -q /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install -q /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl



#importing libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tabulate import tabulate # print data in tabluer format

import seaborn as sns # vizualization library
import matplotlib.pyplot as plt

from scipy.stats import f_oneway, pointbiserialr, pearsonr, spearmanr # statistical analysis

from sklearn.preprocessing import LabelEncoder
import warnings

from lifelines.utils import concordance_index

from sklearn.model_selection import train_test_split


#helper functions

def pearson_correlation(df, feature, target):
    # Calculate Pearson correlation
    correlation, p_value = pearsonr(df[feature].values, df[target])
    
    # Categorize the correlation strength
    if abs(correlation) >= 0.8:
        strength = "high"
    elif abs(correlation) >= 0.5:
        strength = "moderate"
    else:
        strength = "weak"
    
    # Print results
    print(f"Pearson Correlation Coefficient: {correlation:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"The correlation is \033[1m{strength}\033[0m.")
    
    # Check statistical significance
    if p_value < 0.05:
        print("The correlation is statistically \033[1msignificant\033[0m.")
    else:
        print("The correlation is \033[1mNOT\033[0m statistically significant.")

def point_biserial_correlation(df, feature, target):

    dfl = df.copy()
        # Encode the binary string column to 0s and 1s
    le = LabelEncoder()
    dfl.loc[:,'Category Encoded'] = le.fit_transform(dfl[feature])
    # Calculate point-biserial correlation
    correlation, p_value = pointbiserialr(dfl['Category Encoded'], dfl[target])
    
    print(f"\nPoint-Biserial Correlation: {correlation:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print(f"There is a \033[1msignificant correlation\033[0m between the binary {feature} and {target} variables.\n")
    else:
        print("There is \033[1mNO\033[0m significant correlation.\n")

def spearman_correlation(df,feature,target):


    # Calculate Spearman's rank correlation
    correlation, p_value = spearmanr(df[feature].values, df[target])
    
    # Categorize the correlation strength
    if abs(correlation) >= 0.8:
        strength = "strong"
    elif abs(correlation) >= 0.5:
        strength = "moderate"
    else:
        strength = "weak"
    
    # Print results
    print(f"Spearman Correlation Coefficient: {correlation:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"The correlation is \033[1m{strength}\033[0m.")
    
    # Check if correlation is impactful
    if p_value < 0.05:
        print("The correlation is statistically significant (\033[1mimpactful\033[0m).")
    else:
        print("The correlation is \033[1mNOT\033[0m statistically significant (not impactful).")

#anova Test
def anova_test(df, feature, target):

    # Group the continuous values based on the categorical column
    groups = [group[target].values for name, group in df.groupby(feature)]
    
    # Perform ANOVA
    f_stat, p_value = f_oneway(*groups)
    
    print(f"F-statistic: {f_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print(f"There is a \033[1msignificant correlation\033[0m between the {feature} and {target} variables.")
    else:
        print(f"There is \033[1mno significant\033[0m correlation between the {feature} and {target} variables.")


def analyse_categorical(dfl, feature, target='risk_gpt', viz=False):
    dfl = df[df['src']=='trn'].copy()

    dfl[feature] = dfl[feature].fillna('Unknown')
    
    print(tabulate(dfl[feature].value_counts(dropna=False).reset_index(),
               headers='keys',
               tablefmt='simple_grid',
               showindex=False))

    #correlation
    if dfl[feature].nunique() == 2 :
        point_biserial_correlation(dfl, feature, target)
    else:
        anova_test(dfl, feature, target)

    if viz :
        # Create a violing plot for column 'Gender'
        _ = plt.figure(figsize=(20, 5))
        _ = sns.violinplot(x=feature, y=target, data=dfl)
        _ = plt.title(f'{target} vs {feature}')
        
        # Adjust layout to avoid overlap
        plt.tight_layout()
        
        # Show the plot
        plt.show()

def analyse_ordinal(df, feature, target='risk_gpt', viz=False):

    # Null value analysis

    dfl = df[df['src']=='trn'].copy()

    dfl[feature] = dfl[feature].fillna(-1)
    
    print(tabulate(dfl[feature].value_counts(dropna=False).sort_index().reset_index(),
               headers='keys',
               tablefmt='simple_grid',
               showindex=False))
    
    # Correlation 
    # Print the correlation
    spearman_correlation(dfl, feature, target)
    
    if viz:
        # Create a violing plot for column 'Gender'
        _ = plt.figure(figsize=(20, 5))
        _ = sns.violinplot(x=feature, y=target, data=dfl)
        _ = plt.title(f'Premium amount vs {feature}')
        
        # Adjust layout to avoid overlap
        plt.tight_layout()
        
        # Show the plot
        plt.show()

def analyse_continues(df, feature, target='risk_gpt', viz=False):

    dfl = df[df['src']=='trn'].copy()
    data = [
        ["Null count", dfl[feature].isnull().sum()],
        ["Null percentage", round(dfl[feature].isnull().mean()*100,2)],
        ["Min", dfl[feature].min()],
        ["Max", dfl[feature].max()]]
    
    print(tabulate(data, headers=["Attribute", "Value"], tablefmt="simple_grid", numalign="right"))
    
    # Print the correlation
    pearson_correlation(dfl[dfl[feature].notna()], feature, target)


    # plotting 
    fig, ax1 = plt.subplots(figsize=(20, 4))

    bins = 100
    bins = np.linspace(dfl[feature].min(), dfl[feature].max(), bins)
    
    # Create a new column for bin labels (which bin each value of x falls into)
    feature_binned = feature + '_binned'
    
    dfb = dfl.assign(**{feature_binned:pd.cut(dfl[feature], bins)})
    
    # Calculate the mean of 'y' for each bin
    bin_mean = dfb.groupby(feature_binned)[target].mean().reset_index()

    _ = sns.histplot(dfl[feature], bins=bins, kde=False, color='lightgray', edgecolor='black', ax=ax1)
    _ = ax1.set_xlabel(feature_binned)
    _ = ax1.set_ylabel('Frequency')
    _ = ax1.grid(True)
    _ = ax1.set_title(f'Histogram of {feature_binned} with Mean of Premium Amount per Bin')


    _ = ax2 = ax1.twinx() # Create the second y-axis (ax2) that shares the x-axis with ax1

    # Plot the mean of 'y' in each bin as a line plot on the second y-axis
    _ = ax2.plot(bin_mean[feature_binned].apply(lambda x: x.mid), bin_mean[target], color='blue', marker='o', linestyle='-', linewidth=2)
    _ = ax2.set_ylabel(f'Mean of {feature_binned}')
    
    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Show the plot
    plt.show()

# risk_score suggested by gpt
def gpt_risk_score(efs: pd.Series, efs_time: pd.Series, alpha=1, epsilon=1e-6) -> pd.Series:
    """
    Calculate risk scores for survival data.

    Args:
        efs (pd.Series): Event occurrence indicator (1 if event occurred, 0 if censored).
        efs_time (pd.Series): Time to event or censoring.
        alpha (float): Weight for the event indicator in the risk score.
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        pd.Series: Risk scores for each individual.
    """
    # Ensure no zero values in efs_time to avoid division by zero
    efs_time = efs_time + epsilon

    # Calculate risk score
    risk_score = (1 / efs_time) * (efs + alpha)
    return risk_score

# scoring funciton
class ParticipantVisibleError(Exception):
    pass


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    
    del solution[row_id_column_name]
    del submission[row_id_column_name]
    
    event_label = 'efs'
    interval_label = 'efs_time'
    prediction_label = 'prediction'
    for col in submission.columns:
        if not pd.api.types.is_numeric_dtype(submission[col]):
            raise ParticipantVisibleError(f'Submission column {col} must be a number')
    # Merging solution and submission dfs on ID
    merged_df = pd.concat([solution, submission], axis=1)
    merged_df.reset_index(inplace=True)
    merged_df_race_dict = dict(merged_df.groupby(['race_group']).groups)
    metric_list = []
    for race in merged_df_race_dict.keys():
        # Retrieving values from y_test based on index
        indices = sorted(merged_df_race_dict[race])
        merged_df_race = merged_df.iloc[indices]
        # Calculate the concordance index
        c_index_race = concordance_index(
                        merged_df_race[interval_label],
                        -merged_df_race[prediction_label],
                        merged_df_race[event_label])
        metric_list.append(c_index_race)
    return float(np.mean(metric_list)-np.sqrt(np.var(metric_list)))

# only to use in this dataset
def desc(index):
    print(f"measure : {des.loc[index,'variable']}, description : {des.loc[index,'description']}, values : {df[des.loc[index,'variable']].unique()}")
    print("\n",df[des.loc[index,'variable']].value_counts(dropna=False))

# column data type classify
def classify_columns(df):

    classification = {
        'cat': [],
        'ord': [],
        'num': []
    }
    
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            # Categorical columns (string or category types)
            classification['cat'].append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            unique_values = df[col].nunique()
            if unique_values < 5:
                # Ordinal columns (numeric with < 5 unique values)
                classification['ord'].append(col)
            else:
                # Numerical columns
                classification['num'].append(col)
    
    return classification

# clean column values
def clean_categorical_columns(df):

    # Identify categorical columns
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns

    # Apply regex to each categorical column
    for col in categorical_columns:
        df[col] = df[col].str.replace(r'[^A-Za-z0-9_]+', '', regex=True)
    
    return df


#loading data
trn = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
tst = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')
sub = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv')
des = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/data_dictionary.csv')

trn['src']='trn'
tst['src']='tst'

#single target
df = pd.concat([trn, tst], ignore_index=True)
df.replace([float('inf'), -float('inf')], pd.NA, inplace=True)
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# calculating the ris score
df['risk_gpt'] = gpt_risk_score(df.efs, df.efs_time)

#with pd.option_context('display.max_columns', None): # setting the max rows
#    display(df.head())



warnings.filterwarnings('ignore', category=FutureWarning)
print(tabulate(df['efs'].value_counts(dropna=False).reset_index(),
               headers='keys',
               tablefmt='simple_grid',
               showindex=False))

# Create two subplots for two variables
fig, axes = plt.subplots(1, 2, figsize=(16, 4))  # 1 row, 2 columns

# KDE plot for 'sepal_length' on the first subplot
sns.kdeplot(
    data=df,
    x="efs_time",
    hue="efs",  # Differentiates by species
    fill=True,      # Adds shading under the curve
    common_norm=False,  # Prevents normalizing across hues
    alpha=0.6,      # Adjust transparency of the fill
    linewidth=1.5,   # Adjust line thickness
    ax=axes[0]      # Specifies the first subplot
)
axes[0].set_title("KDE Plot of efs_time by efs", fontsize=14)
axes[0].set_xlabel("efs_time", fontsize=12)
axes[0].set_ylabel("Density", fontsize=12)

# KDE plot for 'petal_length' on the second subplot
sns.kdeplot(
    data=df,
    x="risk_gpt",
    hue="efs",  # Differentiates by species
    fill=True,      # Adds shading under the curve
    common_norm=False,  # Prevents normalizing across hues
    alpha=0.6,      # Adjust transparency of the fill
    linewidth=1.5,   # Adjust line thickness
    ax=axes[1]      # Specifies the second subplot
)
axes[1].set_title("KDE Plot of risk_gpt by efs", fontsize=14)
axes[1].set_xlabel("risk_gpt", fontsize=12)
axes[1].set_ylabel("Density", fontsize=12)

# Adjust the layout
plt.tight_layout()
plt.show()



mapping = {
    'Low': 'Low',
    'Intermediate': 'Intermediate',
    'High': 'High',
    'Very high': 'Very High',
    'Intermediate - TED AML case <missing cytogenetics': 'Intermediate TED AML case',
    'High - TED AML case <missing cytogenetics': 'High TED AML case',
    'N/A - non-malignant indication': 'Non Malignant',
    'N/A - pediatric': 'Pediatric',
    'N/A - disease not classifiable': 'Not Classifiable',
    'TBD cytogenetics': 'Unknown',
    'Missing disease status': 'Unknown',
    np.nan: 'Unknown'
}
df['dri_score'] = df['dri_score'].map(mapping)

analyse_categorical(df,'dri_score','risk_gpt')



df['psych_disturb'] = df['psych_disturb'].fillna('Not done')

analyse_categorical(df,'psych_disturb','risk_gpt')



mapping = {
    'Favorable': 'Favorable',
    'Normal': 'Normal',
    'Intermediate': 'Intermediate',
    'Poor': 'Poor',
    'Other': 'Other',
    'TBD': 'Unknown',
    'Not tested': 'Unknown',
    np.nan: 'Unknown'
}
df['cyto_score'] = df['cyto_score'].map(mapping)

analyse_categorical(df,'cyto_score','risk_gpt')



df['diabetes'] = df['diabetes'].fillna('Not done')

analyse_categorical(df,'diabetes','risk_gpt')



df['hla_match_c_high'] = df['hla_match_c_high'].fillna(-1)
analyse_categorical(df,'hla_match_c_high','risk_gpt')



df['hla_high_res_8'] = df['hla_high_res_8'].fillna(-1)
analyse_ordinal(df,'hla_high_res_8','risk_gpt')



mapping = {
    'No TBI': 'No TBI',
    'TBI + Cy +- Other': 'TBI with Cyclophosphamide',
    'TBI +- Other, >cGy': 'TBI High Dose',
    'TBI +- Other, <=cGy': 'TBI Low Dose',
    'TBI +- Other, unknown dose': 'TBI Unknown Dose',
    'TBI +- Other, -cGy, fractionated': 'TBI Fractionated',
    'TBI +- Other, -cGy, single': 'TBI Single',
    'TBI +- Other, -cGy, unknown dose': 'TBI cGy Unknown Dose'
}
df['tbi_status'] = df['tbi_status'].map(mapping)

analyse_categorical(df,'tbi_status','risk_gpt')



df['arrhythmia'] = df['arrhythmia'].fillna('Not done')
analyse_categorical(df, 'arrhythmia', 'risk_gpt')



df['hla_low_res_6'] = df['hla_low_res_6'].fillna(-1)
analyse_categorical(df, 'hla_low_res_6', 'risk_gpt')



analyse_categorical(df,'graft_type','risk_gpt')



df['vent_hist'] = df['vent_hist'].fillna('Unknown')
analyse_categorical(df,'vent_hist','risk_gpt')



df['renal_issue'] = df['renal_issue'].fillna('Not done')
analyse_categorical(df, 'renal_issue', 'risk_gpt')



df['pulm_severe'] = df['pulm_severe'].fillna('Not done')
analyse_categorical(df, 'pulm_severe', 'risk_gpt')



analyse_categorical(df, 'prim_disease_hct', 'risk_gpt')



df['hla_high_res_6'] = df['hla_high_res_6'].fillna(0)
analyse_ordinal(df, 'hla_high_res_6', 'risk_gpt')



mapping = {
    '+/+': 'PP',
    '+/-': 'PN',
    '-/+': 'NP',
    '-/-': 'NN',
    np.nan: 'Unknown'
}
df['cmv_status'] = df['cmv_status'].map(mapping)

analyse_categorical(df, 'cmv_status', 'risk_gpt')



df['hla_high_res_10'] = df['hla_high_res_10'].fillna(0)
analyse_ordinal(df, 'hla_high_res_10', 'risk_gpt')



df['hla_match_dqb1_high'] = df['hla_match_dqb1_high'].fillna(0)
analyse_ordinal(df, 'hla_match_dqb1_high', 'risk_gpt')



df['tce_imm_match'] = df['tce_imm_match'].fillna('Unknown')

analyse_categorical(df, 'tce_imm_match', 'risk_gpt')



df['hla_nmdp_6'] = df['hla_nmdp_6'].fillna(0)
analyse_ordinal(df, 'hla_nmdp_6', 'risk_gpt')



df['hla_match_c_low'] = df['hla_match_c_low'].fillna(0)
analyse_ordinal(df, 'hla_match_c_low', 'risk_gpt')



df['rituximab'] = df['rituximab'].fillna('Unknown')

analyse_categorical(df, 'rituximab', 'risk_gpt')



df['hla_match_drb1_low'] = df['hla_match_drb1_low'].fillna(0)
analyse_ordinal(df, 'hla_match_drb1_low', 'risk_gpt')



df['hla_match_dqb1_low'] = df['hla_match_dqb1_low'].fillna(0)
analyse_ordinal(df, 'hla_match_dqb1_low', 'risk_gpt')



analyse_categorical(df, 'prod_type', 'risk_gpt')



mapping = {
    'Favorable': 'Favorable',
    'Intermediate': 'Intermediate',
    'Poor': 'Poor',
    'TBD': 'Unknown',
    'Not tested': 'Unknown',
    np.nan: 'Unknown'
}
df['cyto_score_detail'] = df['cyto_score_detail'].map(mapping)

analyse_categorical(df,'cyto_score_detail')



mapping = {
    'MAC': 'MAC',
    'RIC': 'TIC',
    'NMA': 'NMA',
    'TBD': 'Unknown',
    'No drugs reported': 'Unknown',
    'N/A, F(pre-TED) not submitted': 'Unknown',
    np.nan: 'Unknown'
}
df['conditioning_intensity'] = df['conditioning_intensity'].map(mapping)

analyse_categorical(df,'conditioning_intensity')



df['ethnicity'] = df['ethnicity'].fillna('Unknown')

analyse_categorical(df,'ethnicity')



analyse_ordinal(df,'year_hct')



df['obesity'] = df['obesity'].fillna('Not done')

analyse_categorical(df,'obesity')



df['mrd_hct'] = df['mrd_hct'].fillna('Unknown')

analyse_categorical(df,'mrd_hct')



df['in_vivo_tcd'] = df['in_vivo_tcd'].fillna('Unknown')

analyse_categorical(df,'in_vivo_tcd')



df['tce_match'] = df['tce_match'].fillna('Unknown')

analyse_categorical(df,'tce_match')



df['hla_match_a_high'] = df['hla_match_a_high'].fillna(0)
analyse_ordinal(df,'hla_match_a_high')



df['hepatic_severe'] = df['hepatic_severe'].fillna('Not done')

analyse_categorical(df,'hepatic_severe')



# Suppress FutureWarnings for this cell
warnings.filterwarnings('ignore', category=FutureWarning)
#df['donor_age'] = df['donor_age'].fillna(df['donor_age'].mean())
df['donor_age'] = df['donor_age'].fillna(0)

analyse_continues(df, 'donor_age')



df['prior_tumor'] = df['prior_tumor'].fillna('Not done')

analyse_categorical(df,'prior_tumor')



df['hla_match_b_low'] = df['hla_match_b_low'].fillna(0)
analyse_categorical(df, 'hla_match_b_low')



df['peptic_ulcer'] = df['peptic_ulcer'].fillna('Not done')

analyse_categorical(df,'peptic_ulcer')



analyse_continues(df,'age_at_hct')



df['hla_match_a_low'] = df['hla_match_a_low'].fillna(0)
analyse_ordinal(df,'hla_match_a_low')



mapping = {
    'FKalone': 'FKalone',
    'Other GVHD Prophylaxis': 'Other GVHD Prophylaxis',
    'Cyclophosphamide alone': 'Cyclophosphamide alone',
    'FK+ MMF +- others': 'FKp MMF pn others',
    'TDEPLETION +- other': 'TDEPLETION pn other',
    'CSA + MMF +- others(not FK)': 'CSA p MMF pn others not FK',
    'CSA + MTX +- others(not MMF,FK)': 'CSA p MTX pn others not MMF FK',
    'FK+ MTX +- others(not MMF)': 'FKp MTX pn others not MMF',
    'Cyclophosphamide +- others': 'Cyclophosphamide pn others',
    'CSA alone': 'CSA alone',
    'TDEPLETION alone': 'TDEPLETION alone',
    'No GvHD Prophylaxis': 'No GvHD Prophylaxis',
    'CDselect alone': 'CDselect alone',
    'CDselect +- other': 'CDselect pn other',
    'Parent Q = yes, but no agent': 'Other',
    'FK+- others(not MMF,MTX)': 'FKpn others not MMF MTX',
    'CSA +- others(not FK,MMF,MTX)': 'CSA pn others not FK MMF MTX',
    np.nan : 'Unknown'
}

df['gvhd_proph'] = df['gvhd_proph'].map(mapping)

analyse_categorical(df, 'gvhd_proph')



df['rheum_issue'] = df['rheum_issue'].fillna('Not done')

analyse_categorical(df,'rheum_issue')



df['sex_match'] = df['sex_match'].fillna('Unknown')

analyse_categorical(df,'sex_match')



df['hla_match_b_high'] = df['hla_match_b_high'].fillna(0)
analyse_categorical(df,'hla_match_b_high')



analyse_categorical(df, 'race_group')



df['comorbidity_score'] = df['comorbidity_score'].fillna(0)
analyse_ordinal(df,'comorbidity_score')



df['karnofsky_score'] = df['karnofsky_score'].fillna(0)
analyse_ordinal(df,'karnofsky_score')



df['hepatic_mild'] = df['hepatic_mild'].fillna('Not done')

analyse_categorical(df,'hepatic_mild')



mapping = {
    'Permissive mismatched': 'Permissive mismatched',
    'GvH non-permissive': 'GvH non permissive',
    'HvG non-permissive': 'HvG non permissive',
    'Bi-directional non-permissive': 'Bi directional nonpermissive',
    np.nan: 'Unknown'
}

df['tce_div_match'] = df['tce_div_match'].map(mapping)

analyse_categorical(df,'tce_div_match')



mapping = {
    'Related': 'Related',
    'Unrelated': 'Unrelated',
    'Multiple donor (non-UCB)': 'Multiple',
    np.nan: 'Unknown'
}
df['donor_related'] = df['donor_related'].map(mapping)

analyse_categorical(df,'donor_related')



mapping = {
    'N/A, Mel not given': 'No Melphalan',
    'MEL': 'Melphalan Given',
    np.nan: 'Unknown'
}
df['melphalan_dose'] = df['melphalan_dose'].map(mapping)

analyse_categorical(df,'melphalan_dose')



df['hla_low_res_8'] = df['hla_low_res_8'].fillna(0)
analyse_ordinal(df,'hla_low_res_8')



df['cardiac'] = df['cardiac'].fillna('Not done')

analyse_categorical(df, 'cardiac')



df['hla_match_drb1_high'] = df['hla_match_drb1_high'].fillna(0)
analyse_categorical(df, 'hla_match_drb1_high')



df['pulm_moderate'] = df['pulm_moderate'].fillna('Not done')

analyse_categorical(df, 'pulm_moderate')



df['hla_low_res_10'] = df['hla_low_res_10'].fillna(0)
analyse_ordinal(df, 'hla_low_res_10')


# Features list
feature_list = [
# 'ID',
 'dri_score',
 'psych_disturb',
 'cyto_score',
 'diabetes',
 'hla_match_c_high',
 'hla_high_res_8',
 'tbi_status',
 'arrhythmia',
 'hla_low_res_6',
 'graft_type',
 'vent_hist',
 'renal_issue',
 'pulm_severe',
 'prim_disease_hct',
 'hla_high_res_6',
 'cmv_status',
 'hla_high_res_10',
 'hla_match_dqb1_high',
 'tce_imm_match',
 'hla_nmdp_6',
 'hla_match_c_low',
 'rituximab',
 'hla_match_drb1_low',
 'hla_match_dqb1_low',
 'prod_type',
 'cyto_score_detail',
 'conditioning_intensity',
 'ethnicity',
 'year_hct',
 'obesity',
 'mrd_hct',
 'in_vivo_tcd',
 'tce_match',
 'hla_match_a_high',
 'hepatic_severe',
 'donor_age',
 'prior_tumor',
 'hla_match_b_low',
 'peptic_ulcer',
 'age_at_hct',
 'hla_match_a_low',
 'gvhd_proph',
 'rheum_issue',
 'sex_match',
 'hla_match_b_high',
 'race_group',
 'comorbidity_score',
 'karnofsky_score',
 'hepatic_mild',
 'tce_div_match',
 'donor_related',
 'melphalan_dose',
 'hla_low_res_8',
 'cardiac',
 'hla_match_drb1_high',
 'pulm_moderate',
 'hla_low_res_10',
# 'efs',
# 'efs_time',
# 'src',
 'risk_gpt'
]


#Feature categorization
cat_col = [
'dri_score',
'psych_disturb',
'cyto_score',
'diabetes',
'tbi_status',
'arrhythmia',
'graft_type',
'vent_hist',
'renal_issue',
'pulm_severe',
'prim_disease_hct',
'cmv_status',
'tce_imm_match',
'rituximab',
'prod_type',
'cyto_score_detail',
'conditioning_intensity',
'ethnicity',
'obesity',
'mrd_hct',
'in_vivo_tcd',
'tce_match',
'hepatic_severe',
'prior_tumor',
'peptic_ulcer',
'gvhd_proph',
'rheum_issue',
'sex_match',
'race_group',
'hepatic_mild',
'tce_div_match',
'donor_related',
'melphalan_dose',
'cardiac',
'pulm_moderate',
]

ord_col= [
'hla_match_c_high',
'hla_match_dqb1_high',
'hla_match_c_low',
'hla_match_drb1_low',
'hla_match_dqb1_low',
'hla_match_a_high',
'hla_match_b_low',
'hla_match_a_low',
'hla_match_b_high',
'hla_match_drb1_high',
]

num_col= [
'hla_high_res_8',
'hla_low_res_6',
'hla_high_res_6',
'hla_high_res_10',
'hla_nmdp_6',
'year_hct',
'donor_age',
'age_at_hct',
'comorbidity_score',
'karnofsky_score',
'hla_low_res_8',
'hla_low_res_10',
'risk_gpt'
]


# Train test split
# Value formatting of categorical columns
dfc = clean_categorical_columns(df)

# Splitting data
X_trn, X_val = train_test_split(dfc[dfc['src']=='trn'], test_size=0.2, random_state=42, stratify=dfc.loc[dfc['src']=='trn','race_group'])


### Training

import h2o
from h2o.automl import H2OAutoML

# Initialize H2O cluster
h2o.init(verbose=False)

# Convert pandas DataFrame to H2OFrame
trn_hf = h2o.H2OFrame(X_trn)
#trn_hf[cat_col] = trn_hf[cat_col].asfactor()

# Define the target and feature columns
y = 'risk_gpt'
X = feature_list

# Split the data into training and testing sets
#trn_hf, val_hf = trn_hf.split_frame(ratios=[.8], seed=42)

# Initialize H2O AutoML model
aml = H2OAutoML(max_models=10, 
                seed=1, 
                max_runtime_secs=60, 
                #include_algos=["DeepLearning"]
                include_algos=["GBM", "XGBoost", "DeepLearning", "GLM", "StackedEnsemble"]
               )

# Train the model
_ = aml.train(x=X, y=y, training_frame=trn_hf)

leaderboard_df = aml.leaderboard.as_data_frame(use_multi_thread=True)
display(leaderboard_df)


# Validation
# Get the leader model (best model from AutoML run)
leader_model = aml.leader

# Make predictions on the validation set
# Convert pandas DataFrame to H2OFrame
val_hf = h2o.H2OFrame(X_val)
#val_hf[cat_col] = val_hf[cat_col].asfactor()
val_hf['prediction'] = leader_model.predict(val_hf)

# Convert the predictions to pandas DataFrame for easier inspection
val_df = val_hf.as_data_frame(use_multi_thread=True)
y_pred = val_df.prediction
y_true = val_df['risk_gpt'].values

# evaluate model performance (e.g., Root Mean Squared Log Error)

sci = score(val_df[['ID','efs','efs_time','race_group']],val_df[['ID','prediction']],'ID')

print('\n')
print(f'Stratified Concordance Index: {sci:.3f}')


### Submission

# Training with full dataset
# Convert pandas DataFrame to H2OFrame
trn_hf = h2o.H2OFrame(df.loc[df['src']=='trn',:].copy())

# Define the target and feature columns
y = 'risk_gpt'
X = feature_list

# Initialize H2O AutoML model
aml = H2OAutoML(max_models=20, 
                seed=1, 
                max_runtime_secs=3600, 
                include_algos=["GBM", "XGBoost", "DeepLearning", "GLM", "StackedEnsemble"]
               )

# Train the model
_ = aml.train(x=X, y=y, training_frame=trn_hf)

leaderboard_df = aml.leaderboard.as_data_frame(use_multi_thread=True)
display(leaderboard_df)

# Prediction
df.loc[df['src']=='tst','risk_gpt'] = 1.0 # null value error handling
tst_hf= h2o.H2OFrame(df.loc[df['src']=='tst',:].copy())
tst_hf['prediction'] = leader_model.predict(tst_hf)

# Submission file generation
tst_df = tst_hf.as_data_frame(use_multi_thread=True)
submission = tst_df[['ID','prediction']]
submission.to_csv('submission.csv')

print("Submission file generated")

