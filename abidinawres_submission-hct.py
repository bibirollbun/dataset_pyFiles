!pip install torchtuples pycox --quiet
!pip install torchtuples pycox --quiet



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import torch
import torchtuples as tt
from pycox.models import CoxPH
import joblib
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


test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")


train_columns = np.load("/kaggle/input/trainoutput/train_columns (2).npy", allow_pickle=True)

# Find columns in test that are not in train_cols
extra_cols_in_test = [col for col in test.columns if col not in train_columns]

print("Extra columns in test not present in training:")
print(extra_cols_in_test)



columns_to_drop = ['vent_hist', 'rituximab', 'rheum_issue', 'melphalan_dose']
test.drop(columns=columns_to_drop, inplace=True)


categorical_cols = ["pulm_moderate","cardiac","hepatic_mild","tce_div_match","donor_related","race_group","sex_match","gvhd_proph","peptic_ulcer","prior_tumor","ethnicity","conditioning_intensity","cyto_score_detail","prod_type","graft_type","renal_issue","pulm_severe","prim_disease_hct", "dri_score","psych_disturb","cyto_score","diabetes","tbi_status","arrhythmia","cmv_status","tce_imm_match","obesity","mrd_hct","in_vivo_tcd","tce_match","hepatic_severe"]
numerical_cols = ["hla_match_c_high", "hla_high_res_8", "hla_low_res_6", "hla_high_res_6", "hla_high_res_10", "hla_match_dqb1_high", "hla_nmdp_6", "hla_match_c_low", "hla_match_drb1_low", "hla_match_dqb1_low", "hla_match_a_high", "donor_age", "hla_match_b_low", "age_at_hct", "hla_match_a_low", "hla_match_b_high", "comorbidity_score", "karnofsky_score", "hla_low_res_8", "hla_match_drb1_high", "hla_low_res_10"]


#smart imputation code
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier

# Numerical - KNN imputer (after isolating numerical data)
knn_imputer = KNNImputer(n_neighbors=5)
test[numerical_cols] = knn_imputer.fit_transform(test[numerical_cols])


# Mode/RF for categorical
for col in categorical_cols:
    if col not in test.columns:
        continue
    if test[col].isnull().sum() == 0:
        continue
    if test[col].nunique() <= 5 or test[col].isnull().mean() > 0.3:
        test[col] = test[col].fillna(test[col].mode()[0])


outlier_cols = ["comorbidity_score", "karnofsky_score", "hla_match_dqb1_low", "hla_match_c_high",
                "hla_match_c_low", "hla_match_dqb1_high", "hla_match_b_high", "hla_match_drb1_high",
                "hla_match_b_low", "hla_match_a_high", "hla_match_a_low"]


def winsorize_series(series, lower=0.01, upper=0.99):
    return np.clip(series, series.quantile(lower), series.quantile(upper))

for col in outlier_cols:
    test[col] = winsorize_series(test[col])


cols_to_adjust = ['hla_high_res_6', 'hla_high_res_8', 'hla_high_res_10', 'hla_low_res_8']


for col in cols_to_adjust:
    test[col] = winsorize_series(test[col], lower=0.005, upper=0.995)


test['comorbidity_score_log'] = np.log1p(test['comorbidity_score'])


object_cols=['dri_score', 'psych_disturb', 'cyto_score', 'diabetes', 'tbi_status',
       'arrhythmia', 'graft_type', 'renal_issue', 'pulm_severe',
       'prim_disease_hct', 'cmv_status', 'tce_imm_match', 'prod_type',
       'cyto_score_detail', 'conditioning_intensity', 'ethnicity', 'obesity',
       'mrd_hct', 'in_vivo_tcd', 'tce_match', 'hepatic_severe', 'prior_tumor',
       'peptic_ulcer', 'gvhd_proph', 'sex_match', 'race_group', 'hepatic_mild',
       'tce_div_match', 'donor_related', 'cardiac', 'pulm_moderate']


label_encoders = joblib.load('/kaggle/input/trainoutput/label_encoders.pkl')
for col in object_cols:
    if col in test.columns:
        le = label_encoders[col]
        test[col] = le.transform(test[col].astype(str))


df = test.copy()
encoder = joblib.load('/kaggle/input/trainoutput/encoder.pkl')
encoded = encoder.transform(test[categorical_cols])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols), index=test.index)

# Combine with other numerical/log features
test_final = pd.concat([test.drop(columns=categorical_cols + ['ID'], errors='ignore'), encoded_df], axis=1)
# Add missing columns with 0s
for col in train_columns:
    if col not in test_final.columns:
        test_final[col] = 0

# Drop extra columns
test_final = test_final[[col for col in train_columns if col in test_final.columns]]

# Optional: print mismatches for debugging
missing = set(train_columns) - set(test_final.columns)
extra = set(test_final.columns) - set(train_columns)
print("Missing:", missing)
print("Extra:", extra)



# Add missing columns with 0s
for col in train_columns:
    if col not in df.columns:
        df[col] = 0
# Remove extra columns that were not in training
df = df[[col for col in train_columns if col in df.columns]]
df = df.drop(columns=['ID', 'efs', 'efs_time'], errors='ignore')


print("Shape after alignment:", df.shape)  # Should be (3, 206)


scaler = joblib.load('/kaggle/input/trainoutput/scaler (2).pkl')
X_scaled = scaler.transform(df).astype('float32')


# 1. Prepare the test features
X_test_tensor = torch.tensor(X_scaled, dtype=torch.float32)

# 2. Rebuild the same network architecture
# Same model architecture
in_features = X_scaled.shape[1]
net = tt.practical.MLPVanilla(in_features, [64, 32], 1, batch_norm=True, dropout=0.1)
model = CoxPH(net)


# Load model weights
checkpoint = torch.load("/kaggle/input/model/tensorflow2/default/1/deepsurv_model (1).pth", map_location='cpu')
model.net.load_state_dict(checkpoint['model_state_dict'])


# Load baseline hazards computed from training
model.baseline_hazards_ = joblib.load('/kaggle/input/trainoutput/baseline_hazards.pkl')
model.baseline_cumulative_hazards_ = joblib.load('/kaggle/input/trainoutput/baseline_cumulative_hazards.pkl')


# Now you can predict survival curves
surv = model.predict_surv_df(X_test_tensor)
risk_scores = -model.predict(X_test_tensor)


# 6. Prepare submission
submission = pd.DataFrame({
    'ID': test['ID'],
    'prediction': risk_scores.numpy().flatten()
})




# 9. Save to CSV
submission.to_csv("/kaggle/working/submission.csv", index=False)
print("Submission saved to submission.csv")

