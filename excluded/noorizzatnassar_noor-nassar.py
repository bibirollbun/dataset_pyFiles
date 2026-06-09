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


df3= pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv')
df3


#! pip install lifelines


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#from lifelines.utils import concordance_index


data = pd.read_csv("/kaggle/input/train1/train.csv")



for column in data.columns:
    plt.figure(figsize=(8, 4))

    if data[column].dtype in ['int64', 'float64']:  # Numerical columns
        sns.histplot(data[column], kde=True, bins=30, color='blue', alpha=0.7)
        plt.title(f'Distribution of {column} (Numerical)')
        plt.xlabel(column)
        plt.ylabel('Frequency')

    else:  # Categorical columns
        sns.countplot(y=data[column], palette='viridis', alpha=0.7)
        plt.title(f'Distribution of {column} (Categorical)')
        plt.xlabel('Count')
        plt.ylabel(column)

    plt.tight_layout()
    plt.show()



missing_values = data.isnull().sum()
missing_percentage = (missing_values / len(data)) * 100

missing_df = pd.DataFrame({
    'Feature': missing_values.index,
    'Missing Count': missing_values.values,
    'Missing %': missing_percentage.values
})
missing_df = missing_df.sort_values(by="Missing %", ascending=False)
print(tabulate(missing_df, headers="keys", tablefmt="fancy_grid"))



nan_threshold = 0.5  # 50% threshold
missing_ratios = data.isnull().sum() / len(data)

cols_to_drop = missing_ratios[missing_ratios > nan_threshold].index.tolist()
print(f"Features to drop due to high NaN values: {cols_to_drop}")
data = data.drop(columns=cols_to_drop)


data_filled = data.fillna({
    'dri_score': 'Missing disease status',  # Fill categorical columns with 'Missing disease status'####
    'psych_disturb':  'Not done',#####################################################################
    'cyto_score': 'Not tested',  # fill missing values with 'Not tested'##########################################################################################################################more than 20%
    'diabetes': 'Not done',         # Binary or categorical columns with 'Yes'/'No' ####################
    'age_at_hct': data['age_at_hct'].mean(),  # For numerical columns, use the mean or median
    'hla_match_c_high': 2.0 ,  # Categorical columns can use 'No' or similar
    'hla_high_res_8': 8.0,##########################################################################################more than 20%
    'tbi_status':'No TBI',# fill 'No TBI'the highest freq
    'arrhythmia': 'Not done',##########################################################################
    'hla_low_res_6': 6.0,
    'vent_hist': 'No', # Fill nan = No
    'renal_issue': 'Not done', ######################################################################
    'pulm_severe': 'Not done', ######################################################################
    'prim_disease_hct': 'ALL', # 'Other acute leukemia', or  'Other leukemia' ?/ THE HIGHEST freq ALL
    'hla_high_res_6': 6.0,
    'cmv_status': '+/+', # fill with '+/+' the highest freq
    'hla_high_res_10': 10.0,##########################################################################################more than 20%
    'hla_match_dqb1_high': 2.0,
    'tce_imm_match': 'P/P', # the highest freq ##########################################################################################more than 20%
    'hla_nmdp_6': 6.0,
    'hla_match_c_low': 2.0,
    'rituximab': 'No',
    'hla_match_drb1_low': 2.0, # fill with 2.0, the highest frequent
    'hla_match_dqb1_low': 2.0 ,# fill with 2.0, the highest frequent
    'cyto_score_detail': 'Not tested', ############################################################# more than 20%
    'conditioning_intensity':  'N/A, F(pre-TED) not submitted', #####################################
    'ethnicity': 'Not Hispanic or Latino',# fill with 'Not Hispanic or Latino' , the highest frequent
    'obesity': 'Not done', ###########################################################################
     # 'mrd_hct': 'Negative', #### drop
    'in_vivo_tcd': 'No', #the highest frequent
     #'tce_match': 'Permissive', ##drop
    'hla_match_a_high': 2.0,
    'hepatic_severe' :'Not done', ################################################################
    'donor_age': data['donor_age'].mean(),
    'prior_tumor': 'Not done', ###################################################################
    'hla_match_b_low': 2.0,
    'peptic_ulcer': 'Not done', ###################################################################
    'age_at_hct': data['age_at_hct'].mean(),
    'hla_match_a_low': 2.0,
    'gvhd_proph': 'FK+ MMF +- others',
    'rheum_issue': 'Not done', #####################################################################
    'sex_match':  'M-M',  #the highest frequent
    'hla_match_b_high': 2.0,
    'race_group': 'More than one race', #the highest frequent
    'comorbidity_score': 0.0,
    'karnofsky_score': 90., #the highest frequent
    'hepatic_mild': 'Not done', #####################################################################
    'tce_div_match': 'Permissive mismatched',##########################more than 20%
    'donor_related' : 'Related', #the highest frequent
    'melphalan_dose' :'N/A, Mel not given',
    'hla_low_res_8': 8.0,
    'cardiac': 'Not done', ##########################################################################
    'hla_match_drb1_high': 2.0,
    'pulm_moderate': 'Not done', ####################################################################
    'hla_low_res_10': 10.0,
})



#data_filled.to_csv('data_filled.csv', index=False)


data_filled[data_filled['ID'].duplicated()]


numeric_cols = data_filled.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = data_filled.select_dtypes(include=['object']).columns
print("numeric_cols: ",numeric_cols)
print("\n categorical_cols",categorical_cols)


def plot_outliers(df, numeric_columns):
    plt.figure(figsize=(15, len(numeric_columns) * 5))
    for i, col in enumerate(numeric_columns, 1):
        plt.subplot(len(numeric_columns), 1, i)
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}", fontsize=14)
        plt.xlabel(col, fontsize=12)
    plt.tight_layout()
    plt.show()
plot_outliers(data_filled, numeric_cols)


def handle_outliers(df, numeric_columns):
    for col in numeric_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df[col] = df[col].apply(lambda x: np.clip(x, lower_bound, upper_bound))
    
    return df

data_new = handle_outliers(data_filled, numeric_cols)




data_new


mapping  = {'Not done': 0, 'No': 0, 'Yes': 1}
mapping2 = {"Favorable": 4,"Intermediate": 3,"Poor": 2, "Not tested": 1,"TBD": 0}
mapping7 = {"Normal": 5, "Favorable": 4, "Intermediate": 3, "Poor": 2,"Not tested": 1,  "TBD": 0, "Other": 0}

mapping3= {"Not Hispanic or Latino":1	,"Hispanic or Latino": 2, "Non-resident of the U.S.": 0}
mapping4= {"Related":1	,"Multiple donor (non-UCB)": 2, "Unrelated": 0}
mapping5= {'N/A, Mel not given':0, 'MEL':1}
mapping6= {'BM':0, 'PB':1}
mapping10={'Bone marrow':0,'Peripheral blood':1}
mapping8= {"Very high": 10,"High": 9,"Intermediate": 8,"Low": 7,"N/A - pediatric": 6,"N/A - non-malignant indication": 5,
           "TBD cytogenetics": 4,    "High - TED AML case <missing cytogenetics": 3, "Intermediate - TED AML case <missing cytogenetics": 2,
           "N/A - disease not classifiable": 1,  "Missing disease status": 0}
mapping9= {'Permissive mismatched':0, 'GvH non-permissive':1, 'HvG non-permissive':2, 'Bi-directional non-permissive':3}

data_new['psych_disturb'] = data_new['psych_disturb'].map(mapping)
data_new['diabetes']      = data_new['diabetes'].map(mapping)
data_new['arrhythmia']    = data_new['arrhythmia'].map(mapping)
data_new['vent_hist']     = data_new['vent_hist'].map(mapping)
data_new['renal_issue']   = data_new['renal_issue'].map(mapping)
data_new['pulm_severe']   = data_new['pulm_severe'].map(mapping)
data_new['rituximab']     = data_new['rituximab'].map(mapping)
data_new['cyto_score_detail']= data_new['cyto_score_detail'].map(mapping2)
data_new['ethnicity']        = data_new['ethnicity'].map(mapping3)
data_new['obesity']          = data_new['obesity'].map(mapping)
data_new['in_vivo_tcd']      = data_new['in_vivo_tcd'].map(mapping)
data_new['hepatic_severe']   = data_new['hepatic_severe'].map(mapping)
data_new['prior_tumor']      = data_new['prior_tumor'].map(mapping)
data_new['peptic_ulcer']     = data_new['peptic_ulcer'].map(mapping)
data_new['rheum_issue']      = data_new['rheum_issue'].map(mapping)
data_new['hepatic_mild']     = data_new['hepatic_mild'].map(mapping)
data_new['donor_related']    = data_new['donor_related'].map(mapping4)
data_new['melphalan_dose']   = data_new['melphalan_dose'].map(mapping5)
data_new['cardiac']          = data_new['cardiac'].map(mapping)
data_new['pulm_moderate']    = data_new['pulm_moderate'].map(mapping)
#data_new['prod_type']        = data_new['prod_type'].map(mapping6)
data_new['cyto_score']       = data_new['cyto_score'].map(mapping7)
data_new['dri_score']        = data_new['dri_score'].map(mapping8)
data_new['tce_div_match']    = data_new['tce_div_match'].map(mapping9)
data_new['graft_type']       = data_new['graft_type'].map(mapping10)

mean_encoded = data_new.groupby('gvhd_proph')['efs'].mean()
data_new['gvhd_proph'] = data_new['gvhd_proph'].map(mean_encoded)

mean_encoded2 = data_new.groupby('sex_match')['efs'].mean()
data_new['sex_match'] = data_new['sex_match'].map(mean_encoded2)

mean_encoded3 = data_new.groupby('race_group')['efs'].mean()
data_new['race_group'] = data_new['race_group'].map(mean_encoded3)

mean_encoded4 = data_new.groupby('cmv_status')['efs'].mean()
data_new['cmv_status'] = data_new['cmv_status'].map(mean_encoded4)

mean_encoded5 = data_new.groupby('prim_disease_hct')['efs'].mean()
data_new['prim_disease_hct'] = data_new['prim_disease_hct'].map(mean_encoded5)

mean_encoded6 = data_new.groupby('tbi_status')['efs'].mean()
data_new['tbi_status'] = data_new['tbi_status'].map(mean_encoded6)

mean_encoded7= data_new.groupby('tce_imm_match')['efs'].mean()
data_new['tce_imm_match'] = data_new['tce_imm_match'].map(mean_encoded7)

mean_encoded8= data_new.groupby('prod_type')['efs'].mean()
data_new['prod_type'] = data_new['prod_type'].map(mean_encoded8)

mean_encoded9= data_new.groupby('conditioning_intensity')['efs'].mean()
data_new['conditioning_intensity'] = data_new['conditioning_intensity'].map(mean_encoded9)


# ######### test
# mapping  = {'Not done': 0, 'No': 0, 'Yes': 1}
# mapping2 = {"Favorable": 4,"Intermediate": 3,"Poor": 2, "Not tested": 1,"TBD": 0}
# mapping7 = {"Normal": 5, "Favorable": 4, "Intermediate": 3, "Poor": 2,"Not tested": 1,  "TBD": 0, "Other": 0}

# mapping3= {"Not Hispanic or Latino":1	,"Hispanic or Latino": 2, "Non-resident of the U.S.": 0}
# mapping4= {"Related":1	,"Multiple donor (non-UCB)": 2, "Unrelated": 0}
# mapping5= {'N/A, Mel not given':0, 'MEL':1}
# mapping6= {'BM':0, 'PB':1}
# mapping10={'Bone marrow':0,'Peripheral blood':1}
# mapping8= {"Very high": 10,"High": 9,"Intermediate": 8,"Low": 7,"N/A - pediatric": 6,"N/A - non-malignant indication": 5,
#            "TBD cytogenetics": 4,    "High - TED AML case <missing cytogenetics": 3, "Intermediate - TED AML case <missing cytogenetics": 2,
#            "N/A - disease not classifiable": 1,  "Missing disease status": 0}
# mapping9= {'Permissive mismatched':0, 'GvH non-permissive':1, 'HvG non-permissive':2, 'Bi-directional non-permissive':3}

# data_new['psych_disturb'] = data_new['psych_disturb'].map(mapping)
# data_new['diabetes']      = data_new['diabetes'].map(mapping)
# data_new['arrhythmia']    = data_new['arrhythmia'].map(mapping)
# data_new['vent_hist']     = data_new['vent_hist'].map(mapping)
# data_new['renal_issue']   = data_new['renal_issue'].map(mapping)
# data_new['pulm_severe']   = data_new['pulm_severe'].map(mapping)
# data_new['rituximab']     = data_new['rituximab'].map(mapping)
# #data_new['cyto_score_detail']= data_new['cyto_score_detail'].map(mapping2)
# #data_new['ethnicity']        = data_new['ethnicity'].map(mapping3)
# data_new['obesity']          = data_new['obesity'].map(mapping)
# data_new['in_vivo_tcd']      = data_new['in_vivo_tcd'].map(mapping)
# data_new['hepatic_severe']   = data_new['hepatic_severe'].map(mapping)
# data_new['prior_tumor']      = data_new['prior_tumor'].map(mapping)
# data_new['peptic_ulcer']     = data_new['peptic_ulcer'].map(mapping)
# data_new['rheum_issue']      = data_new['rheum_issue'].map(mapping)
# data_new['hepatic_mild']     = data_new['hepatic_mild'].map(mapping)
# # data_new['donor_related']    = data_new['donor_related'].map(mapping4)
# # data_new['melphalan_dose']   = data_new['melphalan_dose'].map(mapping5)
# data_new['cardiac']          = data_new['cardiac'].map(mapping)
# data_new['pulm_moderate']    = data_new['pulm_moderate'].map(mapping)
# # data_new['prod_type']        = data_new['prod_type'].map(mapping6)
# # data_new['cyto_score']       = data_new['cyto_score'].map(mapping7)
# # data_new['dri_score']        = data_new['dri_score'].map(mapping8)
# # data_new['tce_div_match']    = data_new['tce_div_match'].map(mapping9)
# # data_new['graft_type']       = data_new['graft_type'].map(mapping10)

# # mean_encoded = data_new.groupby('gvhd_proph')['efs'].mean()
# # data_new['gvhd_proph'] = data_new['gvhd_proph'].map(mean_encoded)

# # mean_encoded2 = data_new.groupby('sex_match')['efs'].mean()
# # data_new['sex_match'] = data_new['sex_match'].map(mean_encoded2)

# # mean_encoded3 = data_new.groupby('race_group')['efs'].mean()
# # data_new['race_group'] = data_new['race_group'].map(mean_encoded3)

# # mean_encoded4 = data_new.groupby('cmv_status')['efs'].mean()
# # data_new['cmv_status'] = data_new['cmv_status'].map(mean_encoded4)

# # mean_encoded5 = data_new.groupby('prim_disease_hct')['efs'].mean()
# # data_new['prim_disease_hct'] = data_new['prim_disease_hct'].map(mean_encoded5)

# # mean_encoded6 = data_new.groupby('tbi_status')['efs'].mean()
# # data_new['tbi_status'] = data_new['tbi_status'].map(mean_encoded6)

# # mean_encoded7= data_new.groupby('tce_imm_match')['efs'].mean()
# # data_new['tce_imm_match'] = data_new['tce_imm_match'].map(mean_encoded7)

# # mean_encoded8= data_new.groupby('prod_type')['efs'].mean()
# # data_new['prod_type'] = data_new['prod_type'].map(mean_encoded8)

# # mean_encoded9= data_new.groupby('conditioning_intensity')['efs'].mean()
# # data_new['conditioning_intensity'] = data_new['conditioning_intensity'].map(mean_encoded9)


data_new.drop(columns=['hla_match_c_high', 'hla_match_dqb1_high' , 'hla_match_dqb1_low' ,'hla_match_a_high', 'hla_match_c_low','tce_div_match'], inplace=True)


# # # encoder = OneHotEncoder(sparse=False)  # Keep all categories
# # # encoded_data = encoder.fit_transform(data[categorical_cols])
# # #encoder = OneHotEncoder()  # Default sparse matrix
# # #encoded_data = encoder.fit_transform(data[categorical_cols]).toarray()  # Convert to dense array

# col_OneHotEncoder=['dri_score','graft_type','gvhd_proph','sex_match','race_group',
#                   'cmv_status','prim_disease_hct','tbi_status','tce_imm_match','prod_type','conditioning_intensity','cyto_score_detail','ethnicity','donor_related','melphalan_dose','cyto_score']

# one_hot_encoders = {}

# for col in col_OneHotEncoder:
#     encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # Fix applied
#     encoded_data = encoder.fit_transform(data_new[col].values.reshape(-1, 1))
    
#     # Create a DataFrame with correct column names
#     encoded_df = pd.DataFrame(encoded_data, columns=[f"{col}_{cat}" for cat in encoder.categories_[0]])
    
#     # Concatenate new one-hot columns and drop original column
#     data_new = pd.concat([data_new, encoded_df], axis=1)
#     data_new.drop(col, axis=1, inplace=True)
    
#     one_hot_encoders[col] = encoder  # Store encoder for later use




data_new


data_new.set_index(['ID'],inplace= True)


# data_new.drop(columns=['ID'], inplace=True)


data_new.drop(columns=[ 'efs_time' ],inplace=True)


df_filtered= data_new.copy()


df_filtered


X = df_filtered.drop(columns=["efs"])
y = df_filtered["efs"]


 df_filtered["efs"].value_counts()




numeric_cols_new = df_filtered.select_dtypes(include=['float64', 'int64']).columns
numeric_cols_new = numeric_cols_new.drop('efs')


# Standardize numeric features
scaler = StandardScaler()
X = scaler.fit_transform(X)#(X[numeric_cols_new])


X_train, X_2, y_train, y_2 = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_2, y_2, test_size=0.5, random_state=42, stratify=y_2)


# from sklearn.decomposition import PCA
# # 
# pca = PCA(n_components=0.95)
# X_train_pca = pca.fit_transform(X_train)
# X_val_pca = pca.transform(X_val)
# X_test_pca = pca.transform(X_test)


# from xgboost import XGBClassifier

# xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
# xgb.fit(X_train, y_train)

# y_pred = xgb.predict(X_val)
# print("Validation Accuracy:", accuracy_score(y_val, y_pred))
# y_test_pred = xgb.predict(X_test_scaled)
# print("Test Accuracy:", accuracy_score(y_test, y_test_pred))


from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Handle class imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Train the optimized model
clf = RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_split=5, random_state=42)
clf.fit(X_train_resampled, y_train_resampled)

# Validate the model
y_val_pred = clf.predict(X_val_scaled)
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("\nValidation Classification Report:\n", classification_report(y_val, y_val_pred))

# Test set evaluation
y_test_pred = clf.predict(X_test_scaled)
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("\nTest Classification Report:\n", classification_report(y_test, y_test_pred))



scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

GradientBoosting = GradientBoostingClassifier(random_state=42)
GradientBoosting.fit(X_train_resampled, y_train_resampled)

y_val_pred = GradientBoosting.predict(X_val_scaled)
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("\nValidation Classification Report:\n", classification_report(y_val, y_val_pred))

y_test_pred = GradientBoosting.predict(X_test_scaled)
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("\nTest Classification Report:\n", classification_report(y_test, y_test_pred))


clf = RandomForestClassifier(max_depth=20, min_samples_split=5, n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

# Validate the model
y_val_pred = clf.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_val))
print("\nValidation Classification Report:\n", classification_report(y_val, y_val))

# Evaluate on the test set
y_test_pred = clf.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("\nTest Classification Report:\n", classification_report(y_test, y_test_pred))


# y_val_pred_proba = clf.predict_proba(X_val)[:, 1]  # Get probabilities for class 1

# c_index = concordance_index(y_val, y_val_pred_proba)
# print(f"Concordance Index (C-Index): {c_index:.4f}")

# y_test_pred_proba = clf.predict_proba(X_test)[:, 1]
# test_c_index = concordance_index(y_test, y_test_pred_proba)
# print(f"Test Concordance Index (C-Index): {test_c_index:.4f}")



# conf_matrix = confusion_matrix(y_test, y_test_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=clf.classes_)
# disp.plot(cmap='Blues')
# plt.title("Confusion Matrix")
# plt.show()


models = {
   # "Logistic Regression": LogisticRegression(random_state=42),
    
  #  "SVM": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    y_val_pred_proba = model.predict_proba(X_val)[:, 1]  # Get probabilities for class 1

    # c_index = concordance_index(y_val, y_val_pred_proba)
    # print(f"Validation Concordance Index (C-Index) for {model_name}: {c_index:.4f}")

    print(f"Validation Accuracy for {model_name}: {accuracy_score(y_val, y_val_pred)}")
    print(f"\nValidation Classification Report for {model_name}:\n", classification_report(y_val, y_val_pred))

    y_test_pred = model.predict(X_test)
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]

    # test_c_index = concordance_index(y_test, y_test_pred_proba)
    # print(f"Test Concordance Index (C-Index) for {model_name}: {test_c_index:.4f}")

    print(f"Test Accuracy for {model_name}: {accuracy_score(y_test, y_test_pred)}")
    print(f"\nTest Classification Report for {model_name}:\n", classification_report(y_test, y_test_pred))
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=model.classes_)
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix for {model_name}")
    plt.show()

    print("-" * 50)



models = {
    "RandomForest Classifier": RandomForestClassifier(random_state=42),
#    "Logistic Regression": LogisticRegression(random_state=42),
    
  #  "SVM": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}
results = []
for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]

    test_accuracy = accuracy_score(y_test, y_test_pred)
    # test_c_index = concordance_index(y_test, y_test_pred_proba)

    results.append({
        "Model": model_name,
        "Test Accuracy": test_accuracy,
      #  "Test C-Index": test_c_index
    })
    print(f"Test Accuracy for {model_name}: {test_accuracy:.4f}")
 #   print(f"Test Concordance Index (C-Index) for {model_name}: {test_c_index:.4f}")
    print("-" * 50)
    
results_df = pd.DataFrame(results)
print("\nSummary of Test Scores")
print(results_df)
results_df.to_csv("results.csv", index=False)





test_df = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')
test_df = test_df.drop(columns=['mrd_hct', 'tce_match'])


test_data = test_df.fillna({
    'dri_score': 'Missing disease status',  # Fill categorical columns with 'Missing disease status'####
    'psych_disturb':  'Not done',#####################################################################
    'cyto_score': 'Not tested',  # fill missing values with 'Not tested'##########################################################################################################################more than 20%
    'diabetes': 'Not done',         # Binary or categorical columns with 'Yes'/'No' ####################
    'age_at_hct': data['age_at_hct'].mean(),  # For numerical columns, use the mean or median
    'hla_match_c_high': 2.0 ,  # Categorical columns can use 'No' or similar
    'hla_high_res_8': 8.0,##########################################################################################more than 20%
    'tbi_status':'No TBI',# fill 'No TBI'the highest freq
    'arrhythmia': 'Not done',##########################################################################
    'hla_low_res_6': 6.0,
    'vent_hist': 'No', # Fill nan = No
    'renal_issue': 'Not done', ######################################################################
    'pulm_severe': 'Not done', ######################################################################
    'prim_disease_hct': 'ALL', # 'Other acute leukemia', or  'Other leukemia' ?/ THE HIGHEST freq ALL
    'hla_high_res_6': 6.0,
    'cmv_status': '+/+', # fill with '+/+' the highest freq
    'hla_high_res_10': 10.0,##########################################################################################more than 20%
    'hla_match_dqb1_high': 2.0,
    'tce_imm_match': 'P/P', # the highest freq ##########################################################################################more than 20%
    'hla_nmdp_6': 6.0,
    'hla_match_c_low': 2.0,
    'rituximab': 'No',
    'hla_match_drb1_low': 2.0, # fill with 2.0, the highest frequent
    'hla_match_dqb1_low': 2.0 ,# fill with 2.0, the highest frequent
    'cyto_score_detail': 'Not tested', ############################################################# more than 20%
    'conditioning_intensity':  'N/A, F(pre-TED) not submitted', #####################################
    'ethnicity': 'Not Hispanic or Latino',# fill with 'Not Hispanic or Latino' , the highest frequent
    'obesity': 'Not done', ###########################################################################
     # 'mrd_hct': 'Negative', #### drop
    'in_vivo_tcd': 'No', #the highest frequent
     #'tce_match': 'Permissive', ##drop
    'hla_match_a_high': 2.0,
    'hepatic_severe' :'Not done', ################################################################
    'donor_age': data['donor_age'].mean(),
    'prior_tumor': 'Not done', ###################################################################
    'hla_match_b_low': 2.0,
    'peptic_ulcer': 'Not done', ###################################################################
    'age_at_hct': data['age_at_hct'].mean(),
    'hla_match_a_low': 2.0,
    'gvhd_proph': 'FK+ MMF +- others',
    'rheum_issue': 'Not done', #####################################################################
    'sex_match':  'M-M',  #the highest frequent
    'hla_match_b_high': 2.0,
    'race_group': 'More than one race', #the highest frequent
    'comorbidity_score': 0.0,
    'karnofsky_score': 90., #the highest frequent
    'hepatic_mild': 'Not done', #####################################################################
    'tce_div_match': 'Permissive mismatched',##########################more than 20%
    'donor_related' : 'Related', #the highest frequent
    'melphalan_dose' :'N/A, Mel not given',
    'hla_low_res_8': 8.0,
    'cardiac': 'Not done', ##########################################################################
    'hla_match_drb1_high': 2.0,
    'pulm_moderate': 'Not done', ####################################################################
    'hla_low_res_10': 10.0,
})



mapping = {'Not done': 0, 'No': 0, 'Yes': 1}
mapping2 ={"Favorable": 4,"Intermediate": 3,"Poor": 2, "Not tested": 1,"TBD": 0}
mapping7 = {"Normal": 5, "Favorable": 4, "Intermediate": 3, "Poor": 2,"Not tested": 1,  "TBD": 0, "Other": 0}

mapping3= {"Not Hispanic or Latino":1	,"Hispanic or Latino": 2, "Non-resident of the U.S.": 0}
mapping4= {"Related":1	,"Multiple donor (non-UCB)": 2, "Unrelated": 0}
mapping5= {'N/A, Mel not given':0, 'MEL':1}
mapping6={'BM':0, 'PB':1}
mapping10={'Bone marrow':0,'Peripheral blood':1}
mapping8= {"Very high": 10,"High": 9,"Intermediate": 8,"Low": 7,"N/A - pediatric": 6,"N/A - non-malignant indication": 5,
           "TBD cytogenetics": 4,    "High - TED AML case <missing cytogenetics": 3, "Intermediate - TED AML case <missing cytogenetics": 2,
           "N/A - disease not classifiable": 1,  "Missing disease status": 0}
mapping9 = {'Permissive mismatched':0, 'GvH non-permissive':1, 'HvG non-permissive':2, 'Bi-directional non-permissive':3}

test_data['psych_disturb'] = test_data['psych_disturb'].map(mapping)
test_data['diabetes']      = test_data['diabetes'].map(mapping)
test_data['arrhythmia']    = test_data['arrhythmia'].map(mapping)
test_data['vent_hist']     = test_data['vent_hist'].map(mapping)
test_data['renal_issue']   = test_data['renal_issue'].map(mapping)
test_data['pulm_severe']   = test_data['pulm_severe'].map(mapping)
test_data['rituximab']   = test_data['rituximab'].map(mapping)
test_data['cyto_score_detail']= test_data['cyto_score_detail'].map(mapping2)
test_data['ethnicity']= test_data['ethnicity'].map(mapping3)
test_data['obesity']= test_data['obesity'].map(mapping)
test_data['in_vivo_tcd']= test_data['in_vivo_tcd'].map(mapping)
test_data['hepatic_severe']= test_data['hepatic_severe'].map(mapping)
test_data['prior_tumor']= test_data['prior_tumor'].map(mapping)
test_data['peptic_ulcer']= test_data['peptic_ulcer'].map(mapping)
test_data['rheum_issue']= test_data['rheum_issue'].map(mapping)
test_data['hepatic_mild']= test_data['hepatic_mild'].map(mapping)
test_data['donor_related']=test_data['donor_related'].map(mapping4)
test_data['melphalan_dose']= test_data['melphalan_dose'].map(mapping5)
test_data['cardiac']= test_data['cardiac'].map(mapping)
test_data['pulm_moderate']= test_data['pulm_moderate'].map(mapping)
#test_data['prod_type']= test_data['prod_type'].map(mapping6)
test_data['cyto_score']= test_data['cyto_score'].map(mapping7)
test_data['dri_score']= test_data['dri_score'].map(mapping8)
test_data['tce_div_match']=test_data['tce_div_match'].map(mapping9)
test_data['graft_type']=test_data['graft_type'].map(mapping10)


test_data['gvhd_proph'] = test_data['gvhd_proph'].map(mean_encoded)
test_data['sex_match'] = test_data['sex_match'].map(mean_encoded2)
test_data['race_group'] = test_data['race_group'].map(mean_encoded3)
test_data['cmv_status'] = test_data['cmv_status'].map(mean_encoded4)
test_data['prim_disease_hct'] = test_data['prim_disease_hct'].map(mean_encoded5)
test_data['tbi_status'] = test_data['tbi_status'].map(mean_encoded6)
test_data['tce_imm_match'] = test_data['tce_imm_match'].map(mean_encoded7)
test_data['prod_type'] = test_data['prod_type'].map(mean_encoded8)
test_data['conditioning_intensity'] = test_data['conditioning_intensity'].map(mean_encoded9)



missing_values = test_data.isnull().sum()
missing_percentage = (missing_values / len(test_data)) * 100

missing_df = pd.DataFrame({
    'Feature': missing_values.index,
    'Missing Count': missing_values.values,
    'Missing %': missing_percentage.values
})
missing_df = missing_df.sort_values(by="Missing %", ascending=False)
print(tabulate(missing_df, headers="keys", tablefmt="fancy_grid"))



test_data.set_index(['ID'], inplace=True)


test_data


data_new


test_data.drop(columns=['hla_match_c_high', 'hla_match_dqb1_high' , 'hla_match_dqb1_low' ,'hla_match_a_high', 'hla_match_c_low','tce_div_match'], inplace=True)


# y_test_pred = xgb.predict_proba(test_data)[:, 1] 
# y_test_pred


y_test_pred = model.predict_proba(test_data)[:, 1] 
y_test_pred


y_test_pred = clf.predict_proba(test_data)[:, 1] 
y_test_pred


y_test_pred = GradientBoosting.predict_proba(test_data)[:, 1] 
y_test_pred


y_test_pred = clf.predict_proba(test_data)[:, 1]  #clf
output = pd.DataFrame({'ID': test_data.index, 'prediction': y_test_pred})
output.to_csv('/kaggle/working/submission.csv', index=True)


output


















