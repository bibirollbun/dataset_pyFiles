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





import optuna
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


sub_file=pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")
# X_test_features = X_test.copy()
predictions1 = final_model.predict_partial_hazard(df_test)
sub_file["prediction"]=predictions1
sub_file.to_csv("submission.csv",index=False)


sub_file

