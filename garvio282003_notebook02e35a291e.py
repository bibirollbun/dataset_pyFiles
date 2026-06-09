!pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
train=pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
test=pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')


hla_cols = [col for col in train.columns if col.startswith("hla")]
for col in hla_cols:
    unique_values = train[col].dropna().unique()
    print(f"{col}: {len(unique_values)} unique values → {sorted(unique_values)[:10]}")



categorical_cols = train.select_dtypes(include=["object", "category"]).columns.tolist()
numerical_cols = train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = [col for col in categorical_cols if col not in ["ID", "efs", "efs_time"]]
numerical_cols = [col for col in numerical_cols if col not in ["ID", "efs", "efs_time"]]
hla_cols = [col for col in train.columns if "hla" in col.lower()]


hla_bins = {
    'hla_match_c_high': [0.0, 1.0, 2.0],
    'hla_high_res_8': [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    'hla_low_res_6': [2.0, 3.0, 4.0, 5.0, 6.0],
    'hla_high_res_6': [0.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    'hla_high_res_10': [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    'hla_match_dqb1_high': [0.0, 1.0, 2.0],
    'hla_nmdp_6': [2.0, 3.0, 4.0, 5.0, 6.0],
    'hla_match_c_low': [0.0, 1.0, 2.0],
    'hla_match_drb1_low': [1.0, 2.0],
    'hla_match_dqb1_low': [0.0, 1.0, 2.0],
    'hla_match_a_high': [0.0, 1.0, 2.0],
    'hla_match_b_low': [0.0, 1.0, 2.0],
    'hla_match_a_low': [0.0, 1.0, 2.0],
    'hla_match_b_high': [0.0, 1.0, 2.0],
    'hla_low_res_8': [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    'hla_match_drb1_high': [0.0, 1.0, 2.0],
    'hla_low_res_10': [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
}

for col, bins in hla_bins.items():
    train[col] = pd.cut(train[col], bins=bins, labels=False, include_lowest=True)

train[hla_cols] = train[hla_cols].fillna(-1)

for col, bins in hla_bins.items():
    test[col] = pd.cut(test[col], bins=bins, labels=False, include_lowest=True)

test[hla_cols] = test[hla_cols].fillna(-1)


from sklearn.decomposition import PCA
scaler = StandardScaler()
hla_scaled = scaler.fit_transform(train[hla_cols])
pca = PCA(n_components=len(hla_cols))
train_hla_pca = pca.fit_transform(hla_scaled)

optimal_components = 6
pca_opt = PCA(n_components=optimal_components)
train_hla_pca_opt = pca_opt.fit_transform(hla_scaled)

pca_columns = [f"PCA_{i+1}" for i in range(optimal_components)]
train_pca_df = pd.DataFrame(train_hla_pca_opt, columns=pca_columns)

train = pd.concat([train.drop(columns=hla_cols), train_pca_df], axis=1)


hla_scaled_t = scaler.transform(test[hla_cols])  

test_hla_pca = pca.transform(hla_scaled_t) 

test_hla_pca_opt = pca_opt.transform(hla_scaled_t)  

pca_columns_t = [f"PCA_{i+1}" for i in range(optimal_components)]
test_pca_df = pd.DataFrame(test_hla_pca_opt, columns=pca_columns_t)

test = pd.concat([test.drop(columns=hla_cols), test_pca_df], axis=1)


train["year_hct"] -= 2000


bins_year_hct = [8, 12, 16, 20]
labels_year_hct = ['8-12', '13-16', '17-20']

bins_donor_age = [18, 30, 45, 60, 85]
labels_donor_age = ['18-30', '31-45', '46-60', '61-85']

bins_age_at_hct = [0, 20, 40, 60, float('inf')]
labels_age_at_hct = ['0-20', '21-40', '41-60', '61+']

bins_comorbidity_score = [0, 1, 4, 7, 10]
labels_comorbidity_score = ['0', '1-3', '4-6', '7-10']

bins_karnofsky_score = [40, 60, 80, 100]
labels_karnofsky_score = ['40-60', '61-80', '81-100']

train['year_hct'] = pd.to_numeric(train['year_hct'], errors='coerce')
train['donor_age'] = pd.to_numeric(train['donor_age'], errors='coerce')
train['age_at_hct'] = pd.to_numeric(train['age_at_hct'], errors='coerce')
train['comorbidity_score'] = pd.to_numeric(train['comorbidity_score'], errors='coerce')
train['karnofsky_score'] = pd.to_numeric(train['karnofsky_score'], errors='coerce')

train['year_hct'] = pd.cut(train['year_hct'], bins=bins_year_hct, labels=labels_year_hct, right=True)
train['donor_age'] = pd.cut(train['donor_age'], bins=bins_donor_age, labels=labels_donor_age, right=True)
train['age_at_hct'] = pd.cut(train['age_at_hct'], bins=bins_age_at_hct, labels=labels_age_at_hct, right=True)
train['comorbidity_score'] = pd.cut(train['comorbidity_score'], bins=bins_comorbidity_score, labels=labels_comorbidity_score, right=True)
train['karnofsky_score'] = pd.cut(train['karnofsky_score'], bins=bins_karnofsky_score, labels=labels_karnofsky_score, right=True)

train['year_hct'] = train['year_hct'].astype('category')
train['donor_age'] = train['donor_age'].astype('category')
train['age_at_hct'] = train['age_at_hct'].astype('category')
train['comorbidity_score'] = train['comorbidity_score'].astype('category')
train['karnofsky_score'] = train['karnofsky_score'].astype('category')
#-----------------------------

test['year_hct'] = pd.to_numeric(test['year_hct'], errors='coerce')
test['donor_age'] = pd.to_numeric(test['donor_age'], errors='coerce')
test['age_at_hct'] = pd.to_numeric(test['age_at_hct'], errors='coerce')
test['comorbidity_score'] = pd.to_numeric(test['comorbidity_score'], errors='coerce')
test['karnofsky_score'] = pd.to_numeric(test['karnofsky_score'], errors='coerce')

test['year_hct'] = pd.cut(test['year_hct'], bins=bins_year_hct, labels=labels_year_hct, right=True)
test['donor_age'] = pd.cut(test['donor_age'], bins=bins_donor_age, labels=labels_donor_age, right=True)
test['age_at_hct'] = pd.cut(test['age_at_hct'], bins=bins_age_at_hct, labels=labels_age_at_hct, right=True)
test['comorbidity_score'] = pd.cut(test['comorbidity_score'], bins=bins_comorbidity_score, labels=labels_comorbidity_score, right=True)
test['karnofsky_score'] = pd.cut(test['karnofsky_score'], bins=bins_karnofsky_score, labels=labels_karnofsky_score, right=True)

test['year_hct'] = test['year_hct'].astype('category')
test['donor_age'] = test['donor_age'].astype('category')
test['age_at_hct'] = test['age_at_hct'].astype('category')
test['comorbidity_score'] = test['comorbidity_score'].astype('category')
test['karnofsky_score'] = test['karnofsky_score'].astype('category')





# for col, categories in ordinal_categories.items():
#     train[col] = pd.Categorical(train[col], categories=categories, ordered=True)

# label_encoder = LabelEncoder()
# for col in ordinal_categories.keys():
#     train[col] = label_encoder.fit_transform(train[col])

# for col, categories in ordinal_categories.items():
#     test[col] = pd.Categorical(test[col], categories=categories, ordered=True)

# label_encoder = LabelEncoder()
# for col in ordinal_categories.keys():
#     test[col] = label_encoder.fit_transform(test[col])




ordinal_categories = {
    'year_hct': ['8-10', '11-13', '14-16', '17-18', '19-20'],
    'donor_age': ['18-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75-85'],
    'age_at_hct': ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-74'],
    'comorbidity_score': ['0', '1-3', '4-6', '7-10'],
    'karnofsky_score': ['40-60', '61-80', '81-90', '90-100']
}

label_encoders = {} 

for col in ordinal_categories.keys():
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col]) 
    test[col] = le.transform(test[col]) 
    label_encoders[col] = le  




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


train = pd.get_dummies(train, columns=categorical_cols, drop_first=True)
test = pd.get_dummies(test, columns=categorical_cols, drop_first=True)

train, test = train.align(test, join='left', axis=1, fill_value=0)

train = train.astype({col: 'int' for col in train.select_dtypes(include=['bool']).columns})
test = test.astype({col: 'int' for col in test.select_dtypes(include=['bool']).columns})


from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index
from lifelines import WeibullAFTFitter

X = train.drop(columns=['efs', 'efs_time', 'ID'])
y = train[['efs_time', 'efs']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_data = pd.concat([X_train, y_train], axis=1)
aft = WeibullAFTFitter()
aft.fit(train_data, duration_col="efs_time", event_col="efs")
y_pred = aft.predict_expectation(X_test)
c_index = concordance_index(y_test['efs_time'], y_pred, y_test['efs'])
print(f"C-index on validation set: {c_index:.4f}")


X = train.drop(columns=['efs', 'efs_time', 'ID'])
y = train[['efs_time', 'efs']]

full_train_data = pd.concat([X, y], axis=1)
aft.fit(full_train_data, duration_col="efs_time", event_col="efs")
X_test_actual = test.drop(columns=['ID'])
y_pred_actual = aft.predict_expectation(X_test_actual)
y_pred_actual = -y_pred_actual  

submission_df = pd.DataFrame({
    'ID': test['ID'],
    'prediction': y_pred_actual
})

submission_df.to_csv('submission.csv', index=False)
print("Test predictions saved successfully for AFT model.")

