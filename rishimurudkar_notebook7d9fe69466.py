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





import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.util import Surv
from lifelines.utils import concordance_index
from sksurv.metrics import concordance_index_censored
# import joblib


def prepare_data(df, categorical_cols, id_col='ID'):
    # Create a copy of the dataframe
    data = df.copy()
    
    # Ensure efs is integer (event indicator: 0 or 1)
    data['efs'] = data['efs'].astype(int)
    
    # Drop the ID column if it exists
    if id_col in data.columns:
        data = data.drop(columns=[id_col])
        print(f"Dropped column: {id_col}")
    else:
        print(f"No column named '{id_col}' found in the dataset")
    
    # Separate features and target
    X = data.drop(['efs', 'efs_time'], axis=1)
    y = Surv.from_arrays(event=data['efs'], time=data['efs_time'])
    
    # Define preprocessing for categorical and numerical columns
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    numerical_cols = [col for col in X.columns if col not in categorical_cols]
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_cols),
            ('num', numerical_transformer, numerical_cols)
        ])
    
    # Fit and transform the data
    X_preprocessed = preprocessor.fit_transform(X)
    
    # Get feature names after one-hot encoding
    cat_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols)
    feature_names = np.concatenate([cat_feature_names, numerical_cols])
    
    return X_preprocessed, y, feature_names, preprocessor




def split_data_for_hyperopt(df, train_size=0.7, val_size=0.15, test_size=0.15,
                           categorical_cols=[
        'dri_score', 'psych_disturb', 'cyto_score', 'diabetes', 'tbi_status',
        'arrhythmia', 'graft_type', 'vent_hist', 'renal_issue', 'pulm_severe',
        'prim_disease_hct', 'cmv_status', 'tce_imm_match', 'rituximab',
        'prod_type', 'cyto_score_detail', 'conditioning_intensity', 'ethnicity',
        'obesity', 'mrd_hct', 'in_vivo_tcd', 'tce_match', 'hepatic_severe',
        'prior_tumor', 'peptic_ulcer', 'gvhd_proph', 'rheum_issue', 'sex_match',
        'race_group', 'hepatic_mild', 'tce_div_match', 'donor_related',
        'melphalan_dose', 'cardiac', 'pulm_moderate'
    ], id_col='ID'):
    assert train_size + val_size + test_size == 1.0, "Split sizes must sum to 1"
    
    # Prepare data
    X, y, feature_names, preprocessor = prepare_data(df, categorical_cols, id_col)
    
    # Split into train + (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_size + test_size), random_state=42
    )
    
    # Split temp into validation and test
    val_proportion = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_proportion), random_state=42
    )
    
    # Print sizes
    print(f"Training set size: {len(X_train)} ({len(X_train)/len(X):.2%})")
    print(f"Validation set size: {len(X_val)} ({len(X_val)/len(X):.2%})")
    print(f"Test set size: {len(X_test)} ({len(X_test)/len(X):.2%})")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names



def evaluate_model(model, X, y):
    """Evaluate model using concordance index"""
    prediction = model.predict(X)
    
    # Extract event indicators and times
    event = y['event']
    time = y['time']
    
    # Calculate concordance index (without negating prediction)
    c_index, _, _, _, _ = concordance_index_censored(event, time, prediction)
    
    return c_index


categorical_cols = [
        'dri_score', 'psych_disturb', 'cyto_score', 'diabetes', 'tbi_status',
        'arrhythmia', 'graft_type', 'vent_hist', 'renal_issue', 'pulm_severe',
        'prim_disease_hct', 'cmv_status', 'tce_imm_match', 'rituximab',
        'prod_type', 'cyto_score_detail', 'conditioning_intensity', 'ethnicity',
        'obesity', 'mrd_hct', 'in_vivo_tcd', 'tce_match', 'hepatic_severe',
        'prior_tumor', 'peptic_ulcer', 'gvhd_proph', 'rheum_issue', 'sex_match',
        'race_group', 'hepatic_mild', 'tce_div_match', 'donor_related',
        'melphalan_dose', 'cardiac', 'pulm_moderate'
]

import pandas as pd
df = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
X_train, X_val, X_test, y_train, y_val, y_test, feature_names = split_data_for_hyperopt(
        df,
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        categorical_cols=categorical_cols,
        id_col='ID'
)


X_train.shape


model = GradientBoostingSurvivalAnalysis(
    n_estimators=300,
    learning_rate=0.033266652862807916,
    max_depth=10,
    min_samples_split=7,
    subsample=0.9566352087990605,
    max_features='log2',
    n_iter_no_change=8,
    validation_fraction=0.17499766883832907,
    random_state=42
)

 # Train the model
model.fit(X_train, y_train)
        
# joblib.dump(model, '/kaggle/input/gradientboostingsurvivalanalysis-model/scikitlearn/default/1/model.pkl')


import joblib


# joblib.dump(model, '/kaggle/working/model.pkl')


# Evaluate on validation set
c_index = evaluate_model(model, X_val, y_val)


c_index


test_df = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")


model.n_features_in_


def prepare_test_data(df, categorical_cols, column_names):
    """
    Prepare test data ensuring it has all the columns the model was trained on
    
    Args:
        df: Test dataframe
        categorical_cols: List of categorical column names before encoding
        column_names: Complete list of column names after one-hot encoding that model expects
    """
    # Create a copy of the dataframe
    data = df.copy()
    
    # Store IDs for submission
    ids = data['ID'].values if 'ID' in data.columns else np.arange(len(data))
    
    # Define preprocessing for categorical and numerical columns
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    numerical_cols = [col for col in data.columns if col not in categorical_cols and col != 'ID']
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_cols),
            ('num', numerical_transformer, numerical_cols)
        ],
        remainder='drop'  # Drop columns like ID
    )
    
    # Fit and transform the data
    X_preprocessed = preprocessor.fit_transform(data)
    
    # Get feature names after one-hot encoding
    cat_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols)
    feature_names = np.concatenate([cat_feature_names, numerical_cols])
    
    # Check if we have all required columns
    missing_columns = set(column_names) - set(feature_names)
    if missing_columns:
        print(f"Warning: {len(missing_columns)} columns are missing in the test data:")
        print(list(missing_columns)[:10], "..." if len(missing_columns) > 10 else "")
        
        # Create a DataFrame with the correct columns (fill missing with zeros)
        X_final = np.zeros((X_preprocessed.shape[0], len(column_names)))
        
        # Map feature positions
        for i, feature in enumerate(feature_names):
            if feature in column_names:
                col_idx = list(column_names).index(feature)
                X_final[:, col_idx] = X_preprocessed[:, i]
    else:
        # Reorder columns to match training data
        X_final = np.zeros((X_preprocessed.shape[0], len(column_names)))
        for i, feature in enumerate(column_names):
            if feature in feature_names:
                col_idx = list(feature_names).index(feature)
                X_final[:, i] = X_preprocessed[:, col_idx]
    
    return X_final, ids



# Define categorical columns (same as in training)
categorical_cols = [
        'dri_score', 'psych_disturb', 'cyto_score', 'diabetes', 'tbi_status',
        'arrhythmia', 'graft_type', 'vent_hist', 'renal_issue', 'pulm_severe',
        'prim_disease_hct', 'cmv_status', 'tce_imm_match', 'rituximab',
        'prod_type', 'cyto_score_detail', 'conditioning_intensity', 'ethnicity',
        'obesity', 'mrd_hct', 'in_vivo_tcd', 'tce_match', 'hepatic_severe',
        'prior_tumor', 'peptic_ulcer', 'gvhd_proph', 'rheum_issue', 'sex_match',
        'race_group', 'hepatic_mild', 'tce_div_match', 'donor_related',
        'melphalan_dose', 'cardiac', 'pulm_moderate'
]

column_names = [
  "dri_score_High",
  "dri_score_High - TED AML case <missing cytogenetics",
  "dri_score_Intermediate",
  "dri_score_Intermediate - TED AML case <missing cytogenetics",
  "dri_score_Low",
  "dri_score_Missing",
  "dri_score_Missing disease status",
  "dri_score_N/A - disease not classifiable",
  "dri_score_N/A - non-malignant indication",
  "dri_score_N/A - pediatric",
  "dri_score_TBD cytogenetics",
  "dri_score_Very high",
  "psych_disturb_Missing",
  "psych_disturb_No",
  "psych_disturb_Not done",
  "psych_disturb_Yes",
  "cyto_score_Favorable",
  "cyto_score_Intermediate",
  "cyto_score_Missing",
  "cyto_score_Normal",
  "cyto_score_Not tested",
  "cyto_score_Other",
  "cyto_score_Poor",
  "cyto_score_TBD",
  "diabetes_Missing",
  "diabetes_No",
  "diabetes_Not done",
  "diabetes_Yes",
  "tbi_status_No TBI",
  "tbi_status_TBI + Cy +- Other",
  "tbi_status_TBI +- Other, -cGy, fractionated",
  "tbi_status_TBI +- Other, -cGy, single",
  "tbi_status_TBI +- Other, -cGy, unknown dose",
  "tbi_status_TBI +- Other, <=cGy",
  "tbi_status_TBI +- Other, >cGy",
  "tbi_status_TBI +- Other, unknown dose",
  "arrhythmia_Missing",
  "arrhythmia_No",
  "arrhythmia_Not done",
  "arrhythmia_Yes",
  "graft_type_Bone marrow",
  "graft_type_Peripheral blood",
  "vent_hist_Missing",
  "vent_hist_No",
  "vent_hist_Yes",
  "renal_issue_Missing",
  "renal_issue_No",
  "renal_issue_Not done",
  "renal_issue_Yes",
  "pulm_severe_Missing",
  "pulm_severe_No",
  "pulm_severe_Not done",
  "pulm_severe_Yes",
  "prim_disease_hct_AI",
  "prim_disease_hct_ALL",
  "prim_disease_hct_AML",
  "prim_disease_hct_CML",
  "prim_disease_hct_HD",
  "prim_disease_hct_HIS",
  "prim_disease_hct_IEA",
  "prim_disease_hct_IIS",
  "prim_disease_hct_IMD",
  "prim_disease_hct_IPA",
  "prim_disease_hct_MDS",
  "prim_disease_hct_MPN",
  "prim_disease_hct_NHL",
  "prim_disease_hct_Other acute leukemia",
  "prim_disease_hct_Other leukemia",
  "prim_disease_hct_PCD",
  "prim_disease_hct_SAA",
  "prim_disease_hct_Solid tumor",
  "cmv_status_+/+",
  "cmv_status_+/-",
  "cmv_status_-/+",
  "cmv_status_-/-",
  "cmv_status_Missing",
  "tce_imm_match_G/B",
  "tce_imm_match_G/G",
  "tce_imm_match_H/B",
  "tce_imm_match_H/H",
  "tce_imm_match_Missing",
  "tce_imm_match_P/B",
  "tce_imm_match_P/G",
  "tce_imm_match_P/H",
  "tce_imm_match_P/P",
  "rituximab_Missing",
  "rituximab_No",
  "rituximab_Yes",
  "prod_type_BM",
  "prod_type_PB",
  "cyto_score_detail_Favorable",
  "cyto_score_detail_Intermediate",
  "cyto_score_detail_Missing",
  "cyto_score_detail_Not tested",
  "cyto_score_detail_Poor",
  "cyto_score_detail_TBD",
  "conditioning_intensity_MAC",
  "conditioning_intensity_Missing",
  "conditioning_intensity_N/A, F(pre-TED) not submitted",
  "conditioning_intensity_NMA",
  "conditioning_intensity_No drugs reported",
  "conditioning_intensity_RIC",
  "conditioning_intensity_TBD",
  "ethnicity_Hispanic or Latino",
  "ethnicity_Missing",
  "ethnicity_Non-resident of the U.S.",
  "ethnicity_Not Hispanic or Latino",
  "obesity_Missing",
  "obesity_No",
  "obesity_Not done",
  "obesity_Yes",
  "mrd_hct_Missing",
  "mrd_hct_Negative",
  "mrd_hct_Positive",
  "in_vivo_tcd_Missing",
  "in_vivo_tcd_No",
  "in_vivo_tcd_Yes",
  "tce_match_Fully matched",
  "tce_match_GvH non-permissive",
  "tce_match_HvG non-permissive",
  "tce_match_Missing",
  "tce_match_Permissive",
  "hepatic_severe_Missing",
  "hepatic_severe_No",
  "hepatic_severe_Not done",
  "hepatic_severe_Yes",
  "prior_tumor_Missing",
  "prior_tumor_No",
  "prior_tumor_Not done",
  "prior_tumor_Yes",
  "peptic_ulcer_Missing",
  "peptic_ulcer_No",
  "peptic_ulcer_Not done",
  "peptic_ulcer_Yes",
  "gvhd_proph_CDselect +- other",
  "gvhd_proph_CDselect alone",
  "gvhd_proph_CSA + MMF +- others(not FK)",
  "gvhd_proph_CSA + MTX +- others(not MMF,FK)",
  "gvhd_proph_CSA +- others(not FK,MMF,MTX)",
  "gvhd_proph_CSA alone",
  "gvhd_proph_Cyclophosphamide +- others",
  "gvhd_proph_Cyclophosphamide alone",
  "gvhd_proph_FK+ MMF +- others",
  "gvhd_proph_FK+ MTX +- others(not MMF)",
  "gvhd_proph_FK+- others(not MMF,MTX)",
  "gvhd_proph_FKalone",
  "gvhd_proph_Missing",
  "gvhd_proph_No GvHD Prophylaxis",
  "gvhd_proph_Other GVHD Prophylaxis",
  "gvhd_proph_Parent Q = yes, but no agent",
  "gvhd_proph_TDEPLETION +- other",
  "gvhd_proph_TDEPLETION alone",
  "rheum_issue_Missing",
  "rheum_issue_No",
  "rheum_issue_Not done",
  "rheum_issue_Yes",
  "sex_match_F-F",
  "sex_match_F-M",
  "sex_match_M-F",
  "sex_match_M-M",
  "sex_match_Missing",
  "race_group_American Indian or Alaska Native",
  "race_group_Asian",
  "race_group_Black or African-American",
  "race_group_More than one race",
  "race_group_Native Hawaiian or other Pacific Islander",
  "race_group_White",
  "hepatic_mild_Missing",
  "hepatic_mild_No",
  "hepatic_mild_Not done",
  "hepatic_mild_Yes",
  "tce_div_match_Bi-directional non-permissive",
  "tce_div_match_GvH non-permissive",
  "tce_div_match_HvG non-permissive",
  "tce_div_match_Missing",
  "tce_div_match_Permissive mismatched",
  "donor_related_Missing",
  "donor_related_Multiple donor (non-UCB)",
  "donor_related_Related",
  "donor_related_Unrelated",
  "melphalan_dose_MEL",
  "melphalan_dose_Missing",
  "melphalan_dose_N/A, Mel not given",
  "cardiac_Missing",
  "cardiac_No",
  "cardiac_Not done",
  "cardiac_Yes",
  "pulm_moderate_Missing",
  "pulm_moderate_No",
  "pulm_moderate_Not done",
  "pulm_moderate_Yes",
  "hla_match_c_high",
  "hla_high_res_8",
  "hla_low_res_6",
  "hla_high_res_6",
  "hla_high_res_10",
  "hla_match_dqb1_high",
  "hla_nmdp_6",
  "hla_match_c_low",
  "hla_match_drb1_low",
  "hla_match_dqb1_low",
  "year_hct",
  "hla_match_a_high",
  "donor_age",
  "hla_match_b_low",
  "age_at_hct",
  "hla_match_a_low",
  "hla_match_b_high",
  "comorbidity_score",
  "karnofsky_score",
  "hla_low_res_8",
  "hla_match_drb1_high",
  "hla_low_res_10"
]

# Prepare test data
X_test, ids = prepare_test_data(test_df, categorical_cols, column_names)
print(f"Preprocessed test data shape: {X_test.shape}")
    
# Generate predictions
print("Generating predictions...")
predictions = model.predict(X_test)

def create_submission(predictions, ids, output_path):
    """
    Create submission file in the required format
    """
    submission = pd.DataFrame({
        'ID': ids,
        'prediction': predictions
    })
    
    submission.to_csv(output_path, index=False)
    print(f"Submission file created at {output_path}")
    
    return submission
    
# Create submission file
submission = create_submission(predictions, ids, "/kaggle/working/submission.csv")
    
print(f"Prediction range: {predictions.min()} to {predictions.max()}")
print(f"Total predictions: {len(predictions)}")
    


preds = pd.read_csv("/kaggle/working/submission.csv")

preds.head()




