# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# # Input data files are available in the read-only "../input/" directory
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb

from lifelines import KaplanMeierFitter
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline

from collections import Counter


# Ignore Warning
import warnings
import re

warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")
data_dict = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/data_dictionary.csv")
data_dict['allele_binary'] = data_dict['description'].str.contains('allele')
data_dict['antigen_binary'] = data_dict['description'].str.contains('antigen')
sample_submission = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")
combined = pd.concat([train,test],axis=0,ignore_index=True)

# Load suggested targets from discussion
def transform_survival_probability(df, time_col='efs_time', event_col='efs'):
    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col], df[event_col])
    y = kmf.survival_function_at_times(df[time_col]).values
    return y
    
train["y"] = transform_survival_probability(train, time_col='efs_time', event_col='efs')

# consider both efs and efs_time
train['efs_time_new'] = train['efs_time'].copy()
train.loc[train['efs'] == 0, 'efs_time_new'] *= -1


print(train['ID'].nunique(), train.shape)
display(train.head(), data_dict)


receipent_features = []

# data_dict['description'] = data_dict['description'].fillna('NAN')


recipient_exclusive_features = data_dict[data_dict['description'].str.contains('Recipient /', case=True, na=False)]['description']
donor_exclusive_features = data_dict[data_dict['description'].str.contains('Donor',case=True, na=False)]['description']
shared_features = data_dict[data_dict['description'].str.contains('Recipient /|Donor', case=True, na=False)]['description']


# Combine all the features we want to exclude
all_excluded_features = pd.concat([recipient_exclusive_features, donor_exclusive_features, shared_features]).unique()

# Select rows where the description is not in the excluded features
other_features = data_dict[~data_dict['description'].isin(all_excluded_features)]

other_features
# data_dict[data_dict['description'].isin(recipient_exclusive_features)]


# Full Credits to https://www.kaggle.com/code/ambrosm/esp-eda-which-makes-sense
plt.figure(figsize=(12, 3))
plt.subplot(1, 2, 1)
plt.hist(train.donor_age, bins=50, color='skyblue')
plt.title('Donor age histogram')
plt.xlabel('donor_age')
plt.ylabel('count')
plt.subplot(1, 2, 2)
plt.title('Patient age histogram')
plt.hist(train.age_at_hct, bins=50, color='skyblue')
plt.xlabel('age_at_hct')
plt.tight_layout()


# Full credits to https://www.kaggle.com/code/ambrosm/esp-eda-which-makes-sense
race_groups = train['race_group'].unique()
_, axs = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(12, 9))
for race_group, ax in zip(race_groups, axs.ravel()):
    ax.hist(train.age_at_hct[train.race_group == race_group],
            bins=np.linspace(0, 74, 38),
            color='skyblue', alpha=0.5)
    ax.set_title(f'Patient age histogram for {race_group}')
    ax.set_xlabel('age_at_hct')
    ax.set_ylabel('count')
plt.tight_layout()


# human lucocyte antigens; proteins found on outside of cell body used to classify foreign substances
# low resolution means  identifies broader groups of HLA proteins rather than specific alleles so it is a weaker feature compared to high_resolution


# If low res, check for missing numeric values
hla_columns = [col for col in train.columns if col.startswith('hla') and train[col].dtype.kind in 'biufc']

fig, axes = plt.subplots(nrows=(len(hla_columns) + 1) // 2, ncols=2, figsize=(12, 4 * ((len(hla_columns) + 1) // 2)))
fig.suptitle('Distribution of HLA Features', fontsize=16)

for i, col in enumerate(hla_columns):
    ax = axes[i // 2, i % 2]
    train[col].plot(kind='hist', ax=ax, bins=30, edgecolor='black')
    ax.set_title(col)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')

plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.show()


# Not done 
train[train['diabetes'] == 'Not done'][[
    'arrhythmia',
    'diabetes',
    'renal_issue',
    'pulm_severe',
    'rituximab',
    'obesity',
    'mrd_hct'
]]


# TCI stands for Transplant Conditioning Intensity. It is a scoring system developed to measure and define the intensity of conditioning regimens used in allogeneic hematopoietic cell transplantation (HCT)14.
# The TCI score indicates the intensity of the pre-transplantation conditioning regimen24.
# The potential for treatment-related toxicity and anti-leukemic efficacy8.
# The risk of early non-relapse mortality (NRM) and relapse (REL) after HCT25.
# The TCI score is calculated by summing the doses (weight scores) of each component in the pre-HCT regimen3. It categorizes conditioning regimens into three risk groups:
# Low TCI: Scores of [1-2]
# Intermediate TCI: Scores of [2.5-3.5]
# High TCI: Scores of [4-6]45

train['tbi_status'].value_counts(dropna=False)


# cmv_status
# -/- (Recipient negative / Donor negative)
# +/+ (Recipient positive / Donor positive)
# +/- (Recipient positive / Donor negative)
# -/+ (Recipient negative / Donor positive)

train['cmv_status'].value_counts(dropna=False)


train['tce_imm_match'].value_counts(dropna=False)
train[['tce_imm_match', 'tce_match', 'tce_div_match']].drop_duplicates().sort_values('tce_match')


# train[['cyto_score', 'cyto_score_detail']].drop_duplicates()
train['cyto_score_detail'].value_counts(dropna=False)


# train['gvhd_proph'].value_counts(dropna=False)
train['gvhd_proph'].unique()


train['donor_related'].value_counts(dropna=False)


train['melphalan_dose'].value_counts(dropna=False)


train['efs'].value_counts(normalize=True)


# Assuming you have a DataFrame called 'df' with columns 'efs_time' and 'efs'
sns.histplot(data=train, x='efs_time', hue='efs', kde=True, stat='probability')

plt.title('Distribution of EFS Time Colored by EFS; EFS = 1 means patient died (event happened) ')
plt.xlabel('EFS Time')
plt.ylabel('Count')

plt.show()


def feature_engineering(df, data_dict):
    df = df.copy(deep=True)
    
    #TODO: what to do with NaN?
    dri_score_mapping = {
        'N/A - non-malignant indication': 0,
        'N/A - disease not classifiable': 0,
        'N/A - pediatric': 0,
        'TBD cytogenetics': 0,
        'Missing disease status': 0,
        'Low': 1,
        'Intermediate': 2,
        'Intermediate - TED AML case <missing cytogenetics': 2,
        'High': 3,
        'High - TED AML case <missing cytogenetics': 3
    }
    df['dri_score_mapped'] = df['dri_score'].map(dri_score_mapping)

    
    cyto_score_mapping = {
        'Not tested': 0,
        'Other': 0,
        'TBD': 0,
        'Poor': 1,
        'Normal': 2,
        'Intermediate': 3,
        'Favorable': 4
    }
    df['cyto_score_mapped'] = df['cyto_score'].map(cyto_score_mapping)

    # Ranked in ascending order for risk of death; +/+ could be equally bad because you have both different CMV
    cmv_status_mapping = {
        '-/-': 0, #(Recipient negative / Donor negative)
        '+/-': 1, #(Recipient positive / Donor negative) # CMV-seropositive recipients (+/+ and +/-) have a higher risk of CMV reactivation and generally poorer overall survival compared to seronegative recipients15.
        '+/+': 2, #(Recipient positive / Donor positive) # For CMV-seropositive recipients, having a CMV-seropositive donor (+/+) may be slightly beneficial compared to a CMV-seronegative donor (+/-) due to the potential transfer of CMV-specific immunity5.
        '-/+': 3, #(Recipient negative / Donor positive) # The -/+ combination (recipient negative, donor positive) is associated with the highest risk, as it has been reported to lead to poorer clinical outcomes
    }
    df['cmv_status_mapped'] = df['cmv_status'].map(cmv_status_mapping)

    
    tce_imm_mapping = {
        'P/P': 0,
        'G/G': 0,
        'H/H': 0, #P/P, G/G, and H/H represent permissive matches within the same TCE group, which are associated with the lowest risk of complications and mortality
        'P/G': 1,
        'P/H': 1, #P/G and P/H are likely permissive mismatches, as they involve the low immunogenicity group (P) with either intermediate (G) or high (H) immunogenicity groups. These mismatches are generally better tolerated than non-permissive mismatches
        'G/B': 2,
        'H/B': 2, # G/B, H/B, and P/B involve mismatches with the highest immunogenicity group (B), which are considered non-permissive and associated with increased risk of acute GVHD, transplant-related mortality, and overall mortality
        'P/B': 3
    }
    df['tce_imm_match_mapped'] = df['tce_imm_match'].map(tce_imm_mapping)

    donor_related_mapping = {
        'Related': 0,
        'Multiple donor (non-UCB)':1,
        'Unrelated': 2,
    }
    df['donor_related_mapped'] = df['donor_related'].map(donor_related_mapping)

    donor_related_mapping = {  
    }
    gvhd_proph_mapping = {
        "FK+ MTX +- others (not MMF)": 0,
        "CSA + MTX +- others (not MMF, FK)": 1,
        "FK+ MMF +- others": 2,
        "CSA + MMF +- others (not FK)": 3,
        "Cyclophosphamide +- others": 4,
        "TDEPLETION +- other": 5,
        "FK alone": 6,
        "CSA alone": 7,
        "Other GVHD Prophylaxis": 8,
        "Cyclophosphamide alone": 9,
        "TDEPLETION alone": 10,
        "No GvHD Prophylaxis": 11
    }
    df['gvhd_proph_mapped'] = df['gvhd_proph'].map(gvhd_proph_mapping)


    
    ## One Hot Encode: graft_type, vent_hist (nan own category), prim_disease_hct, prod_type, conditioning_intensity, ethnicity, gvhd_proph, sex_match, race_group, tce_match, tce_div_match


    # Not done = Missing not at random; https://www.kaggle.com/competitions/equity-post-HCT-survival-predictions/discussion/556106
    ## Ordinal Encode: diabetes (y/n/not done), arrhythmia (y/n/not done), renal_issue (y/n/not done), pulm_severe(y/n/not done), rituximab (y/n/nan), obesity (y/n/nan), mrd_hct(y/n/nan), in_vivo_tcd(y/n/nan), hepatic_severe(y/n/not done), prior_tumor, rheum_issue, hepatic_mild, cardiac, pulm_moderate
    
    ## Not sure: psych_disturb

    ## No need to encode: year_hct, donor_age, age_at_hct,comorbidity_score, karnofsky_score

    df['tbi_dummy'] = np.where(df['tbi_status'] == 'No TBI', 0, 1)
    df['melphalan_dose_dummy'] = np.where(df['melphalan_dose'] == 'MEL', 0, 1)
    
    ## Hmm this is just a constant feature; need to transform later
    # for _, row in data_dict[['variable', 'antigen_binary']].iterrows():
    #     if row['antigen_binary']:
    #         df[f"{row['variable']}_antigen_true"] = 1
    # for _, row in data_dict[['variable', 'allele_binary']].iterrows():
    #     if row['allele_binary']:
    #         df[f"{row['variable']}_allele_true"] = 1
            
    return df



train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")
data_dict = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/data_dictionary.csv")
data_dict['allele_binary'] = data_dict['description'].str.contains('allele')
data_dict['antigen_binary'] = data_dict['description'].str.contains('antigen')
sample_submission = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")

# Load suggested targets from discussion
def transform_survival_probability(df, time_col='efs_time', event_col='efs'):
    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col], df[event_col])
    y = kmf.survival_function_at_times(df[time_col]).values
    return y
    
train["y"] = transform_survival_probability(train, time_col='efs_time', event_col='efs')
# consider both efs and efs_time
train['efs_time_new'] = train['efs_time'].copy()
train.loc[train['efs'] == 0, 'efs_time_new'] *= -1


IDS = ["ID"]
TARGETS = ["efs","efs_time","y", "efs_time_new"]
RMV = IDS+TARGETS
FEATURES = [c for c in train.columns if not c in RMV]
print(f"There are {len(FEATURES)} FEATURES")

CATS = []
for c in FEATURES:
    if train[c].dtype=="object":
        CATS.append(c)
        train[c] = train[c].fillna("NAN")
        test[c] = test[c].fillna("NAN")
print(f"In these features, there are {len(CATS)} CATEGORICAL FEATURES")

categorical_features_already_encoded = [
    'dri_score',
    'cyto_score',
    'cmv_status',
    'tce_imm_match',
    'gvhd_proph'
]
categorical_features = [
    i for i in data_dict.query("type == 'Categorical'")['variable'].tolist() 
    if i != 'efs' #and i not in categorical_features_already_encoded
]


custom_encoded_numerical_features = [
    'dri_score_mapped',
    'cyto_score_mapped',
    'cmv_status_mapped',
    'tce_imm_match_mapped',
    'gvhd_proph_mapped',
    'tbi_dummy',
    'melphalan_dose_dummy'
]
numerical_features = [i for i in data_dict.query("type == 'Numerical'")['variable'].tolist() if i!='efs_time']
numerical_features = numerical_features + custom_encoded_numerical_features


combined = pd.concat([train,test],axis=0,ignore_index=True)

# Post-processing for efficient compute
combined[categorical_features] = combined[categorical_features].astype('category')
for c in FEATURES:
    if combined[c].dtype=="float64":
        combined[c] = combined[c].astype("float32")
    if combined[c].dtype=="int64":
        combined[c] = combined[c].astype("int32")


#################################### Temp
FEATURES = [
    #categorical
    'race_group',
    #numerical
    'donor_age',
    'age_at_hct',
    # custom encoded
    'dri_score_mapped',
    'cyto_score_mapped',
    'cmv_status_mapped',
    'tce_imm_match_mapped',
    'gvhd_proph_mapped',
    'tbi_dummy',
    'melphalan_dose_dummy',

    ## must have
    # 'efs_time',
]
numerical_features=FEATURES


combined2 = feature_engineering(df=combined, data_dict = data_dict)[FEATURES + TARGETS]




# Full credits for work below to https://www.kaggle.com/code/lucasdataartist/modeling-post-hct-survival-cv-679-lb-684/notebook
numeric_transformer = Pipeline(steps = [
    ('passthrough', 'passthrough')
])

# categorical_transformer_onehot = Pipeline(steps = [
#     ('onehot', OneHotEncoder(sparse_output = False, handle_unknown = 'ignore'))
# ])

preprocessor = ColumnTransformer(
    transformers = [
        ('num', numeric_transformer, numerical_features),
        # ('cat', categorical_transformer_onehot, categorical_features)
    ]
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor)
])
pipeline.fit(combined2[FEATURES])


# Cleaning function to remove spaces and special characters
def clean_column_names(columns):
    cleaned_columns = []
    column_counts = Counter()

    for col in columns:
        # Remove spaces and special characters except underscores
        cleaned_col = re.sub(r'[^\w]', '_', col)  # Replace non-alphanumeric except "_" with "_"
        
        # Ensure no duplicates by appending a suffix if necessary
        while cleaned_col in column_counts:
            column_counts[cleaned_col] += 1
            cleaned_col = f"{cleaned_col}_{column_counts[cleaned_col]}"
        
        column_counts[cleaned_col] += 1
        cleaned_columns.append(cleaned_col)

    return cleaned_columns

def make_df_X(df, preprocess_pipeline):

    df_X = preprocess_pipeline.transform(df[FEATURES])

    numeric_encoded_columns = preprocess_pipeline.named_steps['preprocessor'].named_transformers_['num'].get_feature_names_out(numerical_features)
    #categorical_encoded_columns = preprocess_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
    column_names = list(numeric_encoded_columns) #+ list(categorical_encoded_columns)

    df_X = pd.DataFrame(df_X, columns = column_names)
    df_X.columns = clean_column_names(df_X.columns)
    return df_X
    
combined3= make_df_X(df=combined2, preprocess_pipeline = pipeline)
# combined3 = pd.concat([combined2, combined2[TARGETS]], axis = 1)
train = combined2.iloc[:len(train)].copy()
test = combined2.iloc[len(train):].reset_index(drop=True).copy()


FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)


# Initialize out-of-fold and test predictions
oof_xgb_aft = np.zeros(len(train))
pred_xgb_aft = np.zeros(len(test))

# Prepare lower and upper bounds for survival data
train["lower_bound"] = train["efs_time"]  # Observed survival time
train["upper_bound"] = train["efs_time"]
train.loc[train["efs"] == 0, "upper_bound"] = float("inf")  # Censored data upper bound is infinity

# Convert categorical columns to numeric codes
# for col in FEATURES:
#     if train[col].dtype.name == 'category':
#         train[col] = train[col].cat.codes
#         test[col] = test[col].cat.codes

# Set 1
# RMV = ["ID","efs","efs_time","y","efs_time2", "lower_bound", "upper_bound"]
# FEATURES = [c for c in train.columns if not c in RMV]

kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

for fold, (train_index, test_index) in enumerate(kf.split(train)):
    print("#" * 25)
    print(f"### Fold {fold + 1}")
    print("#" * 25)
    
    # Split data for the current fold
    x_train = train.loc[train_index, FEATURES].copy()
    y_train_lower = train.loc[train_index, "lower_bound"].values
    y_train_upper = train.loc[train_index, "upper_bound"].values
    x_valid = train.loc[test_index, FEATURES].copy()
    y_valid_lower = train.loc[test_index, "lower_bound"].values
    y_valid_upper = train.loc[test_index, "upper_bound"].values
    x_test = test[FEATURES].copy()

    # Create DMatrix for training and validation
    dtrain = xgb.DMatrix(
        x_train,
        label_lower_bound=y_train_lower,
        label_upper_bound=y_train_upper,
        enable_categorical=True
    )
    dvalid = xgb.DMatrix(
        x_valid,
        label_lower_bound=y_valid_lower,
        label_upper_bound=y_valid_upper,
        enable_categorical=True
    )
    dtest = xgb.DMatrix(x_test, enable_categorical=True)

    # Specify parameters
    params = {
        "device": "cuda",  # GPU usage
        "subsample": 0.5,
        "colsample_bytree": 0.5,
        "objective": "survival:aft",
        'eval_metric': 'aft-nloglik',
        "tree_method": "hist",
        "learning_rate": 0.03,
        "aft_loss_distribution": 'normal',
        'aft_loss_distribution_scale': 1.20,
        "max_depth": 10,
        "min_child_weight": 5,
        "colsample_bylevel": 0.50,
        "reg_alpha": 0.005,
        "reg_lambda": 0.005,
        "random_state": 42,
    }

    # Set evaluation sets
    evals = [(dtrain, "train"), (dvalid, "validation")]

    # Train the model using xgboost.train
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=10000,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=100
    )

    # INFER OOF
    oof_xgb_aft[test_index] += model.predict(dvalid)
    # INFER TEST
    pred_xgb_aft += model.predict(dtest) / (FOLDS)

# Evaluate OOF performance
rmse = np.sqrt(np.mean((oof_xgb_aft - train["lower_bound"]) ** 2))


# all features rmse:  3202.292769842967
# custom features: 27.651674318524375

# subset:  26.988192594422113
rmse




