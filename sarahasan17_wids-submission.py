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


# STEP 1: Import Libraries
# ------------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# Define file paths for Kaggle
main_dir = "/kaggle/input/widsdatathon2025/"

# Load training data
train_connect = pd.read_csv(
    f"{main_dir}TRAIN_NEW/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv",
)
train_quant = pd.read_excel(f"{main_dir}TRAIN_NEW/TRAIN_QUANTITATIVE_METADATA_new.xlsx")
train_cat = pd.read_excel(f"{main_dir}TRAIN_NEW/TRAIN_CATEGORICAL_METADATA_new.xlsx")
train_solution = pd.read_excel(f"{main_dir}TRAIN_NEW/TRAINING_SOLUTIONS.xlsx")

# Load test data
test_connect= pd.read_csv(f"{main_dir}TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv")
test_quant = pd.read_excel(f"{main_dir}TEST/TEST_QUANTITATIVE_METADATA.xlsx")
test_cat = pd.read_excel(f"{main_dir}TEST/TEST_CATEGORICAL.xlsx")


train_df = train_solution.merge(train_quant, on="participant_id", how="left")
train_df = train_df.merge(train_cat, on="participant_id", how="left")
train_df = train_df.merge(train_connect, on="participant_id", how="left")

test_df = test_quant.merge(test_cat, on="participant_id", how="left")
test_df = test_df.merge(test_connect, on="participant_id", how="left")

test_ids = test_df["participant_id"]


# Drop not needed
drop_cols = ["participant_id", "PreInt_Demos_Fam_Child_Ethnicity", "PreInt_Demos_Fam_Child_Race"]
train_df.drop(columns=drop_cols, inplace=True, errors="ignore")
test_df.drop(columns=drop_cols, inplace=True, errors="ignore")

# Fill NA with median
train_df.fillna(train_df.median(numeric_only=True), inplace=True)
test_df.fillna(test_df.median(numeric_only=True), inplace=True)


# STEP 4: Apply PCA to Connectome Data
# ------------------------------------------
connectome_cols = train_connect.columns.drop("participant_id")

# Scale connectome separately before PCA
scaler_conn = StandardScaler()
X_conn_scaled = scaler_conn.fit_transform(train_df[connectome_cols])
X_test_conn_scaled = scaler_conn.transform(test_df[connectome_cols])

# Apply PCA
pca = PCA(n_components=50, random_state=42)
X_conn_pca = pca.fit_transform(X_conn_scaled)
X_test_conn_pca = pca.transform(X_test_conn_scaled)

# Convert to DataFrame and rename columns
pca_columns = [f"pca_conn_{i}" for i in range(X_conn_pca.shape[1])]
train_conn_pca_df = pd.DataFrame(X_conn_pca, columns=pca_columns, index=train_df.index)
test_conn_pca_df = pd.DataFrame(X_test_conn_pca, columns=pca_columns, index=test_df.index)

# Drop original connectome columns and add PCA features
train_df.drop(columns=connectome_cols, inplace=True)
test_df.drop(columns=connectome_cols, inplace=True)

train_df = pd.concat([train_df, train_conn_pca_df], axis=1)
test_df = pd.concat([test_df, test_conn_pca_df], axis=1)

# ------------------------------------------
# STEP 5: Separate Features
# ------------------------------------------
# ADHD uses PCA connectome + quantitative
adhd_features = pca_columns + list(train_quant.columns.drop("participant_id"))

# SEX uses mostly categorical + optional quant
sex_features = list(train_cat.columns.drop("participant_id"))
optional_quant_for_sex = ["PreInt_Demos_Household_Income", "PreInt_Demos_Fam_Parent_Education"]
sex_features += [f for f in optional_quant_for_sex if f in train_df.columns]
sex_features = [f for f in sex_features if f in train_df.columns]

print(f"\nâœ… Total ADHD Features Used ({len(adhd_features)}):")
print(adhd_features)

print(f"\nâœ… Total SEX Features Used ({len(sex_features)}):")
print(sex_features)



# STEP 6: Prepare Train/Test Sets
# ------------------------------------------
X_adhd = train_df[adhd_features]
y_adhd = train_df["ADHD_Outcome"]
X_test_adhd = test_df[adhd_features]

X_sex = train_df[sex_features]
y_sex = train_df["Sex_F"]
X_test_sex = test_df[sex_features]

# Scale
scaler_adhd = StandardScaler()
X_adhd_scaled = scaler_adhd.fit_transform(X_adhd)
X_test_adhd_scaled = scaler_adhd.transform(X_test_adhd)

scaler_sex = StandardScaler()
X_sex_scaled = scaler_sex.fit_transform(X_sex)
X_test_sex_scaled = scaler_sex.transform(X_test_sex)

# Balance ADHD data
smote = SMOTE(random_state=42)
X_adhd_res, y_adhd_res = smote.fit_resample(X_adhd_scaled, y_adhd)
print(y_adhd.value_counts())  # Before SMOTE
print(y_adhd_res.value_counts())  # After SMOTE

print(X_adhd_scaled.shape)  # Original number of samples
print(X_adhd_res.shape)  # Number of samples after SMOTE

# ------------------------------------------
# ------------------------------------------
# STEP 7: Voting Classifier Ensemble
# ------------------------------------------
adhd_model = VotingClassifier(
    estimators=[
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('lr', LogisticRegression(max_iter=1000))
    ],
    voting='soft'
)

sex_model = VotingClassifier(
    estimators=[
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('lr', LogisticRegression(max_iter=1000))
    ],
    voting='soft'
)

adhd_model.fit(X_adhd_res, y_adhd_res)
sex_model.fit(X_sex_scaled, y_sex)

# ------------------------------------------
# STEP 8: Validation
# ------------------------------------------
# Validation split for evaluation
X_train_adhd, X_val_adhd, y_train_adhd, y_val_adhd = train_test_split(X_adhd_scaled, y_adhd, test_size=0.2, random_state=42)
X_train_sex, X_val_sex, y_train_sex, y_val_sex = train_test_split(X_sex_scaled, y_sex, test_size=0.2, random_state=42)

# Evaluate
val_pred_adhd = adhd_model.predict(X_val_adhd)
val_pred_sex = sex_model.predict(X_val_sex)

print(f"\nğŸ�¯ ADHD Validation Accuracy: {accuracy_score(y_val_adhd, val_pred_adhd):.4f}")
print(f"ğŸ�¯ Sex Validation Accuracy: {accuracy_score(y_val_sex, val_pred_sex):.4f}")


# STEP 9: Predict on Test Set
# ------------------------------------------
test_adhd_pred = adhd_model.predict(X_test_adhd_scaled)
test_sex_pred = sex_model.predict(X_test_sex_scaled)

# ------------------------------------------
# STEP 10: Create Submission
# ------------------------------------------
submission = pd.DataFrame({
    "participant_id": test_ids,
    "ADHD_Outcome": test_adhd_pred,
    "Sex_F": test_sex_pred
})

submission.to_csv("submission.csv", index=False)
print("\nğŸ“� Submission saved as 'submission.csv'")




