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


!pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl


import pandas as pd
df=pd.read_csv("/kaggle/input/cibtr-preprocessed-dataset/preprocessed_hct_dataset2.csv")


import pandas as pd
df_test=pd.read_csv("/kaggle/input/test-dataset-hct/preprocessed_test_dataset.csv")


df_test.info()


import numpy as np, pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

print("Train shape:",df.shape)
df.head()


from lifelines import KaplanMeierFitter
kmf=KaplanMeierFitter()
kmf.fit(durations=df["efs_time"],event_observed=df["efs"])
kmf.plot_survival_function()


kmf.plot_cumulative_density()


df_test.info


df.shape


df_test.shape


diff_columns = set(df_test.columns) ^ set(df)  
print("Different columns:", diff_columns)



df.drop(columns=["age_at_hct_bin","target","transplant_intensity","infection_risk","donor_age_bin"],axis=1,inplace=True)


df_test.drop(columns=["years_since_hct","ID","donor_age_bin"],axis=1,inplace=True)





'''import optuna
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.model_selection import train_test_split


X = df.drop(columns=["efs", "efs_time"]) 
y = df[["efs", "efs_time"]]  
print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_data = X_train.copy()
train_data["efs"] = y_train["efs"]
train_data["efs_time"] = y_train["efs_time"]
test_data = X_test.copy()
test_data["efs"] = y_test["efs"]
test_data["efs_time"] = y_test["efs_time"]
# def objective(trial):
#     l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
#     penalizer = trial.suggest_loguniform("penalizer", 1e-5, 1e1)
#     cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
#     cph.fit(train_data, duration_col="efs_time", event_col="efs")

#     c_index = cph.score(test_data, scoring_method="concordance_index")
#     return -c_index  

# study = optuna.create_study(direction="minimize")
# study.optimize(objective, n_trials=50, n_jobs=-1)

# print("Best Hyperparameters:", study.best_params)
# print("Best Concordance Index:", -study.best_value)

# best_params = study.best_params
final_model = CoxPHFitter(penalizer=0.02239258664668549, l1_ratio=0.06123878966529939)
final_model.fit(train_data, duration_col="efs_time", event_col="efs")


final_c_index = final_model.score(test_data, scoring_method="concordance_index")
print("Final Model Concordance Index:", final_c_index)
'''


import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index

# --- Data Preparation ---
# Assuming 'df' is your training DataFrame containing features and the survival columns:
#   - "efs": event indicator (1 if the event occurred, 0 if censored)
#   - "efs_time": time to event or censoring
X = df.drop(columns=["efs", "efs_time"])
y = df[["efs", "efs_time"]]

print("Feature matrix shape:", X.shape)

# Split into training and test sets (for model evaluation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Training with XGBoost ---
# XGBoost’s 'survival:cox' objective uses the survival time as the label.
# We pass the event indicator as sample weights.
model = xgb.XGBRegressor(
    objective='survival:cox',
    eval_metric='cox-nloglik',  # Negative log partial likelihood for Cox model
    learning_rate=0.1,
    max_depth=3,
    n_estimators=100,
    reg_lambda=1.0,    # L2 regularization term
    reg_alpha=0.0,     # L1 regularization term
    random_state=42
)

# Fit the model: note that the label is the survival time, and sample_weight is the event indicator.
model.fit(
    X_train,
    y_train["efs_time"],
    sample_weight=y_train["efs"]
)

# --- Model Evaluation ---
# Get risk scores for the validation set.
# In survival analysis, the model outputs a risk score (or linear predictor).
risk_scores = model.predict(X_val)

# Compute the concordance index.
# Because a higher risk score implies a higher hazard (i.e. shorter survival),
# we pass -risk_scores so that higher values correspond to longer survival times.
c_index = concordance_index(
    y_val["efs_time"],
    -risk_scores,
    y_val["efs"]
)
print("Final Model Concordance Index on validation set:", c_index)

# --- Creating a Submission File ---
# Assuming 'df_test' is your test DataFrame with the same feature columns as in 'df' (without 'efs' and 'efs_time').
# Make sure df_test is preprocessed in the same way as the training data.
test_risk_scores = model.predict(df_test)

# Read the sample submission file.
sub_file = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")

# Replace the prediction column with your risk scores.
sub_file["prediction"] = test_risk_scores

# Save the submission file.
sub_file.to_csv("submission.csv", index=False)



'''sub_file=pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")
# X_test_features = X_test.copy()
predictions1 = final_model.predict_partial_hazard(df_test)
sub_file["prediction"]=predictions1
sub_file.to_csv("submission.csv",index=False)
'''


sub_file


sub = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv')
sub.head()




