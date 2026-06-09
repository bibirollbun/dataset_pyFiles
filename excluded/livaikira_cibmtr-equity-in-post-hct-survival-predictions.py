import pandas as pd


train_data_dir = '/kaggle/input/equity-post-HCT-survival-predictions/train.csv'
test_data_dir = '/kaggle/input/equity-post-HCT-survival-predictions/test.csv'
dictionary_dir = '/kaggle/input/equity-post-HCT-survival-predictions/data_dictionary.csv'


train_df = pd.read_csv(train_data_dir)
dictionary_df = pd.read_csv(dictionary_dir)


import matplotlib.pyplot as plt


train_df


dictionary_df


train_df.info()


train_df.isnull().sum()



import random

def clean_data(df):
    # Handling missing values in the `dri_score` column by the most frequent value
    df['dri_score'] = df['dri_score'].fillna(df['dri_score'].mode()[0])
    # Replace missing values with the most common value of each column in: 'psych_disturb'
    df = df.fillna({'psych_disturb': df['psych_disturb'].mode()[0]})
    # Handling missing values in the `cyto_score` column by a random value in the column
    df['cyto_score'] = df['cyto_score'].fillna(df['cyto_score'].sample(1).values[0])
    # Handling missing values in the `diabetes` column by the most frequent value
    df['diabetes'] = df['diabetes'].fillna(df['diabetes'].mode()[0])
    # Fill missing values in 'hla_match_c_high' with random values from the same column
    non_nulls = df['hla_match_c_high'].dropna().values
    df['hla_match_c_high'] = df['hla_match_c_high'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `hla_high_res_8` column by random values from the column
    non_nulls = df['hla_high_res_8'].dropna().values
    df['hla_high_res_8'] = df['hla_high_res_8'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `arrhythmia` column by random values from the column
    non_nulls = df['arrhythmia'].dropna().values
    df['arrhythmia'] = df['arrhythmia'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `hla_low_res_6` column by random values from the column
    non_nulls = df['hla_low_res_6'].dropna().values
    df['hla_low_res_6'] = df['hla_low_res_6'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Replace missing values with the most common value of each column in: 'vent_hist'
    df = df.fillna({'vent_hist': df['vent_hist'].mode()[0]})
    # Fill missing values for `renal_issue` column by random values from the column
    non_nulls = df['renal_issue'].dropna().values
    df['renal_issue'] = df['renal_issue'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `pulm_severe` column by random values from the column
    nom_nulls = df['pulm_severe'].dropna().values
    df['pulm_severe'] = df['pulm_severe'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `hla_high_res_6` column by random values from the column
    non_nulls = df['hla_high_res_6'].dropna().values
    df['hla_high_res_6'] = df['hla_high_res_6'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Replace missing values with the most common value of each column in: 'cmv_status'
    df = df.fillna({'cmv_status': df['cmv_status'].mode()[0]})
    # Fill missing values for `tce_imm_match` column by random values from the column
    non_nulls = df['tce_imm_match'].dropna().values
    df['tce_imm_match'] = df['tce_imm_match'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `hla_nmdp_6` column by random values from the column
    non_nulls = df['hla_nmdp_6'].dropna().values
    df['hla_nmdp_6'] = df['hla_nmdp_6'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `hla_match_c_low` column by random values from the column
    non_nulls = df['hla_match_c_low'].dropna().values
    df['hla_match_c_low'] = df['hla_match_c_low'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `rituximab` column by random values from the column
    non_nulls = df['rituximab'].dropna().values
    df['rituximab'] = df['rituximab'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `hla_high_res_8` column by random values from the column
    non_nulls = df['hla_match_drb1_low'].dropna().values
    df['hla_match_drb1_low'] = df['hla_match_drb1_low'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `hla_match_dqb1_low` column by random values from the column
    non_nulls = df['hla_match_dqb1_low'].dropna().values
    df['hla_match_dqb1_low'] = df['hla_match_dqb1_low'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `cyto_score_detail` column by random values from the column
    non_nulls = df['cyto_score_detail'].dropna().values
    df['cyto_score_detail'] = df['cyto_score_detail'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `cyto_score` column by random values from the column
    non_nulls = df['cyto_score'].dropna().values
    df['cyto_score'] = df['cyto_score'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `hla_high_res_10` column by random values from the column
    non_nulls = df['hla_high_res_10'].dropna().values
    df['hla_high_res_10'] = df['hla_high_res_10'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `hla_match_dqb1_high` column by random values from the column
    non_nulls = df['hla_match_dqb1_high'].dropna().values
    df['hla_match_dqb1_high'] = df['hla_match_dqb1_high'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `conditioning_intensity` column by random values from the column
    non_nulls = df['conditioning_intensity'].dropna().values
    df['conditioning_intensity'] = df['conditioning_intensity'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `ethnicity` column by random values from the column
    non_nulls = df['ethnicity'].dropna().values
    df['ethnicity'] = df['ethnicity'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `obesity` column by random values from the column
    non_nulls = df['obesity'].dropna().values
    df['obesity'] = df['obesity'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `mrd_hct` column by random values from the column
    non_nulls = df['mrd_hct'].dropna().values
    df['mrd_hct'] = df['mrd_hct'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Replace missing values with the most common value of each column in: 'in_vivo_tcd'
    df = df.fillna({'in_vivo_tcd': df['in_vivo_tcd'].mode()[0]})
    # Fill missing values for `tce_match` column by random values from the column
    non_nulls = df['tce_match'].dropna().values
    df['tce_match'] = df['tce_match'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `hla_match_a_high` column by random values from the column
    non_nulls = df['hla_match_a_high'].dropna().values
    df['hla_match_a_high'] = df['hla_match_a_high'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `hepatic_severe` column by random values from the column
    non_nulls = df['hepatic_severe'].dropna().values
    df['hepatic_severe'] = df['hepatic_severe'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `donor_age` column by random values from the column
    non_nulls = df['donor_age'].dropna().values
    df['donor_age'] = df['donor_age'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `prior_tumor` column by random values from the column
    non_nulls = df['prior_tumor'].dropna().values
    df['prior_tumor'] = df['prior_tumor'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `hla_match_b_low` column by random values from the column
    non_nulls = df['hla_match_b_low'].dropna().values
    df['hla_match_b_low'] = df['hla_match_b_low'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `peptic_ulcer` column by random values from the column
    non_nulls = df['peptic_ulcer'].dropna().values
    df['peptic_ulcer'] = df['peptic_ulcer'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `hla_match_a_low` column by random values from the column
    non_nulls = df['hla_match_a_low'].dropna().values
    df['hla_match_a_low'] = df['hla_match_a_low'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Replace missing values with the most common value of each column in: 'gvhd_proph'
    df = df.fillna({'gvhd_proph': df['gvhd_proph'].mode()[0]})
    # Fill missing values for `rheum_issue` column by random values from the column
    non_nulls = df['rheum_issue'].dropna().values
    df['rheum_issue'] = df['rheum_issue'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Replace missing values with the most common value of each column in: 'sex_match'
    df = df.fillna({'sex_match': df['sex_match'].mode()[0]})
    # Fill missing values for `hla_match_b_high` column by random values from the column
    non_nulls = df['hla_match_b_high'].dropna().values
    df['hla_match_b_high'] = df['hla_match_b_high'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `comorbidity_score` column by random values from the column
    non_nulls = df['comorbidity_score'].dropna().values
    df['comorbidity_score'] = df['comorbidity_score'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `karnofsky_score` column by random values from the column
    non_nulls = df['karnofsky_score'].dropna().values
    df['karnofsky_score'] = df['karnofsky_score'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `hepatic_mild` column by random values from the column
    non_nulls = df['hepatic_mild'].dropna().values
    df['hepatic_mild'] = df['hepatic_mild'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `tce_div_match` column by random values from the column
    non_nulls = df['tce_div_match'].dropna().values
    df['tce_div_match'] = df['tce_div_match'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Replace missing values with the most common value of each column in: 'donor_related'
    df = df.fillna({'donor_related': df['donor_related'].mode()[0]})
    # Fill missing values for `melphalan_dose` column by random values from the column
    non_nulls = df['melphalan_dose'].dropna().values
    df['melphalan_dose'] = df['melphalan_dose'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `hla_low_res_8` column by random values from the column
    non_nulls = df['hla_low_res_8'].dropna().values
    df['hla_low_res_8'] = df['hla_low_res_8'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `cardiac` column by random values from the column
    non_nulls = df['cardiac'].dropna().values
    df['cardiac'] = df['cardiac'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `hla_match_drb1_high` column by random values from the column
    non_nulls = df['hla_match_drb1_high'].dropna().values
    df['hla_match_drb1_high'] = df['hla_match_drb1_high'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `pulm_moderate` column by random values from the column
    non_nulls = df['pulm_moderate'].dropna().values
    df['pulm_moderate'] = df['pulm_moderate'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `hla_low_res_10` column by random values from the column
    non_nulls = df['hla_low_res_10'].dropna().values
    df['hla_low_res_10'] = df['hla_low_res_10'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    df = df.drop(columns=['efs_time'])
    df = df.drop(columns=['ID'])
    return df

df_clean = clean_data(train_df.copy())
df_clean


def clean_data_test(df):
    # Handling missing values in the `dri_score` column by the most frequent value
    df['dri_score'] = df['dri_score'].fillna(df['dri_score'].mode()[0])
    # Replace missing values with the most common value of each column in: 'psych_disturb'
    df = df.fillna({'psych_disturb': df['psych_disturb'].mode()[0]})
    # Handling missing values in the `cyto_score` column by a random value in the column
    df['cyto_score'] = df['cyto_score'].fillna(df['cyto_score'].sample(1).values[0])
    # Handling missing values in the `diabetes` column by the most frequent value
    df['diabetes'] = df['diabetes'].fillna(df['diabetes'].mode()[0])
    # Fill missing values in 'hla_match_c_high' with random values from the same column
    non_nulls = df['hla_match_c_high'].dropna().values
    df['hla_match_c_high'] = df['hla_match_c_high'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `hla_high_res_8` column by random values from the column
    non_nulls = df['hla_high_res_8'].dropna().values
    df['hla_high_res_8'] = df['hla_high_res_8'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `arrhythmia` column by random values from the column
    non_nulls = df['arrhythmia'].dropna().values
    df['arrhythmia'] = df['arrhythmia'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `hla_low_res_6` column by random values from the column
    non_nulls = df['hla_low_res_6'].dropna().values
    df['hla_low_res_6'] = df['hla_low_res_6'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Replace missing values with the most common value of each column in: 'vent_hist'
    df = df.fillna({'vent_hist': df['vent_hist'].mode()[0]})
    # Fill missing values for `renal_issue` column by random values from the column
    non_nulls = df['renal_issue'].dropna().values
    df['renal_issue'] = df['renal_issue'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `pulm_severe` column by random values from the column
    nom_nulls = df['pulm_severe'].dropna().values
    df['pulm_severe'] = df['pulm_severe'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `hla_high_res_6` column by random values from the column
    non_nulls = df['hla_high_res_6'].dropna().values
    df['hla_high_res_6'] = df['hla_high_res_6'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Replace missing values with the most common value of each column in: 'cmv_status'
    df = df.fillna({'cmv_status': df['cmv_status'].mode()[0]})
    # Fill missing values for `tce_imm_match` column by random values from the column
    non_nulls = df['tce_imm_match'].dropna().values
    df['tce_imm_match'] = df['tce_imm_match'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `hla_nmdp_6` column by random values from the column
    non_nulls = df['hla_nmdp_6'].dropna().values
    df['hla_nmdp_6'] = df['hla_nmdp_6'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `hla_match_c_low` column by random values from the column
    non_nulls = df['hla_match_c_low'].dropna().values
    df['hla_match_c_low'] = df['hla_match_c_low'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `rituximab` column by random values from the column
    non_nulls = df['rituximab'].dropna().values
    df['rituximab'] = df['rituximab'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `hla_high_res_8` column by random values from the column
    non_nulls = df['hla_match_drb1_low'].dropna().values
    df['hla_match_drb1_low'] = df['hla_match_drb1_low'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `hla_match_dqb1_low` column by random values from the column
    non_nulls = df['hla_match_dqb1_low'].dropna().values
    df['hla_match_dqb1_low'] = df['hla_match_dqb1_low'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `cyto_score_detail` column by random values from the column
    non_nulls = df['cyto_score_detail'].dropna().values
    df['cyto_score_detail'] = df['cyto_score_detail'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `cyto_score` column by random values from the column
    non_nulls = df['cyto_score'].dropna().values
    df['cyto_score'] = df['cyto_score'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `hla_high_res_10` column by random values from the column
    non_nulls = df['hla_high_res_10'].dropna().values
    df['hla_high_res_10'] = df['hla_high_res_10'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `hla_match_dqb1_high` column by random values from the column
    non_nulls = df['hla_match_dqb1_high'].dropna().values
    df['hla_match_dqb1_high'] = df['hla_match_dqb1_high'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `conditioning_intensity` column by random values from the column
    non_nulls = df['conditioning_intensity'].dropna().values
    df['conditioning_intensity'] = df['conditioning_intensity'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `ethnicity` column by random values from the column
    non_nulls = df['ethnicity'].dropna().values
    df['ethnicity'] = df['ethnicity'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `obesity` column by random values from the column
    non_nulls = df['obesity'].dropna().values
    df['obesity'] = df['obesity'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `mrd_hct` column by random values from the column
    non_nulls = df['mrd_hct'].dropna().values
    df['mrd_hct'] = df['mrd_hct'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Replace missing values with the most common value of each column in: 'in_vivo_tcd'
    df = df.fillna({'in_vivo_tcd': df['in_vivo_tcd'].mode()[0]})
    # Fill missing values for `tce_match` column by random values from the column
    non_nulls = df['tce_match'].dropna().values
    df['tce_match'] = df['tce_match'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `hla_match_a_high` column by random values from the column
    non_nulls = df['hla_match_a_high'].dropna().values
    df['hla_match_a_high'] = df['hla_match_a_high'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `hepatic_severe` column by random values from the column
    non_nulls = df['hepatic_severe'].dropna().values
    df['hepatic_severe'] = df['hepatic_severe'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `donor_age` column by random values from the column
    non_nulls = df['donor_age'].dropna().values
    df['donor_age'] = df['donor_age'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `prior_tumor` column by random values from the column
    non_nulls = df['prior_tumor'].dropna().values
    df['prior_tumor'] = df['prior_tumor'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `hla_match_b_low` column by random values from the column
    non_nulls = df['hla_match_b_low'].dropna().values
    df['hla_match_b_low'] = df['hla_match_b_low'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `peptic_ulcer` column by random values from the column
    non_nulls = df['peptic_ulcer'].dropna().values
    df['peptic_ulcer'] = df['peptic_ulcer'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `hla_match_a_low` column by random values from the column
    non_nulls = df['hla_match_a_low'].dropna().values
    df['hla_match_a_low'] = df['hla_match_a_low'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Replace missing values with the most common value of each column in: 'gvhd_proph'
    df = df.fillna({'gvhd_proph': df['gvhd_proph'].mode()[0]})
    # Fill missing values for `rheum_issue` column by random values from the column
    non_nulls = df['rheum_issue'].dropna().values
    df['rheum_issue'] = df['rheum_issue'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Replace missing values with the most common value of each column in: 'sex_match'
    df = df.fillna({'sex_match': df['sex_match'].mode()[0]})
    # Fill missing values for `hla_match_b_high` column by random values from the column
    non_nulls = df['hla_match_b_high'].dropna().values
    df['hla_match_b_high'] = df['hla_match_b_high'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `comorbidity_score` column by random values from the column
    non_nulls = df['comorbidity_score'].dropna().values
    df['comorbidity_score'] = df['comorbidity_score'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `karnofsky_score` column by random values from the column
    non_nulls = df['karnofsky_score'].dropna().values
    df['karnofsky_score'] = df['karnofsky_score'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `hepatic_mild` column by random values from the column
    non_nulls = df['hepatic_mild'].dropna().values
    df['hepatic_mild'] = df['hepatic_mild'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `tce_div_match` column by random values from the column
    non_nulls = df['tce_div_match'].dropna().values
    df['tce_div_match'] = df['tce_div_match'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Replace missing values with the most common value of each column in: 'donor_related'
    df = df.fillna({'donor_related': df['donor_related'].mode()[0]})
    # Fill missing values for `melphalan_dose` column by random values from the column
    non_nulls = df['melphalan_dose'].dropna().values
    df['melphalan_dose'] = df['melphalan_dose'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `hla_low_res_8` column by random values from the column
    non_nulls = df['hla_low_res_8'].dropna().values
    df['hla_low_res_8'] = df['hla_low_res_8'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `cardiac` column by random values from the column
    non_nulls = df['cardiac'].dropna().values
    df['cardiac'] = df['cardiac'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `hla_match_drb1_high` column by random values from the column
    non_nulls = df['hla_match_drb1_high'].dropna().values
    df['hla_match_drb1_high'] = df['hla_match_drb1_high'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `pulm_moderate` column by random values from the column
    non_nulls = df['pulm_moderate'].dropna().values
    df['pulm_moderate'] = df['pulm_moderate'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    # Fill missing values for `hla_low_res_10` column by random values from the column
    non_nulls = df['hla_low_res_10'].dropna().values
    df['hla_low_res_10'] = df['hla_low_res_10'].apply(lambda x: random.choice(non_nulls) if pd.isnull(x) else x)
    df = df.drop(columns=['ID'])
    return df


def preprocess_for_model(df: pd.DataFrame, type: str) -> dict:
    """
    Preprocess data for model input
    Args:
        df: Input DataFrame
    Returns:
        dict: Features dictionary for model input
    """
    
    if type == 'train':
        # Clean data first
        df = clean_data(df.copy())
        df = df.drop(columns=['efs'])
    else:
        df = clean_data_test(df.copy())
    
    # # Create features dictionary with correct shapes
    features_dict = {}
    for name, values in df.items():
        if values.dtype == object:
            # Categorical features as strings
            features_dict[name] = values.astype(str).values.reshape(-1, 1)
        else:
            # Numeric features as float32
            features_dict[name] = values.astype('float32').values.reshape(-1, 1)
            
    return features_dict


train_features = df_clean.copy()
train_labels = train_features.pop('efs')


import tensorflow as tf
from tensorflow.keras import layers

import numpy as np


inputs = {}

for name, column in train_features.items():
  dtype = column.dtype
  if dtype == object:
    dtype = tf.string
  else:
    dtype = tf.float32

  inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

inputs






numeric_inputs = {name:input for name,input in inputs.items()
                  if input.dtype==tf.float32}

x = layers.Concatenate()(list(numeric_inputs.values()))
norm = layers.Normalization()
norm.adapt(np.array(df_clean[numeric_inputs.keys()]))
all_numeric_inputs = norm(x)

all_numeric_inputs


preprocessed_inputs = [all_numeric_inputs]


for name, input in inputs.items():
  if input.dtype == tf.float32:
    continue

  lookup = layers.StringLookup(vocabulary=np.unique(train_features[name]))
  one_hot = layers.CategoryEncoding(num_tokens=lookup.vocabulary_size())

  x = lookup(input)
  x = one_hot(x)
  preprocessed_inputs.append(x)


preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)

cibmtr_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)

# tf.keras.utils.plot_model(model = cibmtr_preprocessing , rankdir="LR", dpi=72, show_shapes=True)


cibmtr_features_dict = {name: np.array(value) 
                         for name, value in train_features.items()}


features_dict = {name:values[:1] for name, values in cibmtr_features_dict.items()}
cibmtr_preprocessing(features_dict)


from sklearn import metrics

def cibmtr_model(preprocessing_head, inputs):
  body = tf.keras.Sequential([
    layers.Dense(64, activation='relu'),
    layers.Softmax(),    
    layers.Dense(1)
  ])

  preprocessed_inputs = preprocessing_head(inputs)
  result = body(preprocessed_inputs)
  model = tf.keras.Model(inputs, result)

  model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
  return model

cibmtr_model = cibmtr_model(cibmtr_preprocessing, inputs)


cibmtr_model.fit(x=cibmtr_features_dict, y=train_labels, epochs=100)


cibmtr_model.predict(features_dict)


# cibmtr_model.save("model_2.keras")


# Load the test data
test_df = pd.read_csv(test_data_dir)


ID = test_df['ID']


test_features = preprocess_for_model(test_df, 'test')


cibmtr_model.predict(test_features)


# Save the results to a CSV file

results = cibmtr_model.predict(test_features)

# Bring the results into the right shape and in the right data type (0 or 1)
results = results > 0.5
results = results.astype(int)

results_df = pd.DataFrame({'ID': ID, 'prediction': results.flatten()})
results_df.to_csv('submission.csv', index=False)




