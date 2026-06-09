# Standard libraries
import os
import json
import numpy as np
import pandas as pd

import seaborn as sns
import math

# Model building
import sklearn #has tools for ML and statistical modeling
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score, confusion_matrix, roc_curve
from scipy.stats import zscore, pearsonr, uniform
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, StratifiedKFold, RandomizedSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier #ML Algorithm for classification and regression

from scipy.io import loadmat

# Utility
import matplotlib.pyplot as plt

# Warnings
import warnings
warnings.filterwarnings("ignore")
print("packages imported")


# === Load TRAIN data ===
train_path = "/kaggle/input/widsdatathon2025/TRAIN_NEW"
connectome_train = pd.read_csv(f"{train_path}/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv")
quant_meta_train = pd.read_excel(f"{train_path}/TRAIN_QUANTITATIVE_METADATA_new.xlsx")
cat_meta_train = pd.read_excel(f"{train_path}/TRAIN_CATEGORICAL_METADATA_new.xlsx")
targets_train = pd.read_excel(f"{train_path}/TRAINING_SOLUTIONS.xlsx")

# Check shapes
print("Connectome:", connectome_train.shape)
print("Quantitative metadata:", quant_meta_train.shape)
print("Categorical metadata:", cat_meta_train.shape)
print("Targets:", targets_train.shape)


# === Load TEST data ===
test_path = "/kaggle/input/widsdatathon2025/TEST"
connectome_test = pd.read_csv(f"{test_path}/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv")
quant_meta_test = pd.read_excel(f"{test_path}/TEST_QUANTITATIVE_METADATA.xlsx")
cat_meta_test = pd.read_excel(f"{test_path}/TEST_CATEGORICAL.xlsx")


# Drop participant_id
demo_cols = cat_meta_train.columns.drop('participant_id')

# Define number of plots
n_cols = 3  # Number of columns in subplot grid
n_plots = len(demo_cols)
n_rows = math.ceil(n_plots / n_cols)

# Create subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
axes = axes.flatten()  # Flatten to index easily

# Plot each variable
for i, col in enumerate(demo_cols):
    sns.countplot(x=col, data=cat_meta_train, ax=axes[i])
    axes[i].set_title(f"Distribution of {col}")
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].set_xlabel(col)

# Turn off unused subplots if any
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()


# Drop 'participant_id'
quant_cols = quant_meta_train.columns.drop('participant_id')

# Subplot grid configuration
n_cols = 3
n_plots = len(quant_cols)
n_rows = math.ceil(n_plots / n_cols)

# Create subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
axes = axes.flatten()  # Flatten axes to iterate easily

# Plot each histogram
for i, col in enumerate(quant_cols):
    quant_meta_train[col].plot.hist(bins=20, ax=axes[i])
    axes[i].set_title(f"Distribution of {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Frequency")

# Turn off unused axes (if any)
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.suptitle("Quantitative Variable Distributions", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()


print(targets_train['ADHD_Outcome'].value_counts())
targets_train['ADHD_Outcome'].value_counts().plot(kind='bar', color='blue')
plt.title('ADHD Outcome')
plt.xlabel('Outcome (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

# Gender distribution
print(targets_train['Sex_F'].value_counts())
targets_train['Sex_F'].value_counts().plot(kind='bar', color='blue')
plt.title('Gender Distribution')
plt.xlabel('Gender (0 = Male, 1 = Female)')
plt.ylabel('Count')
plt.show()


for col in cat_meta_train.select_dtypes(include='int').columns:
    cat_meta_train[col] = cat_meta_train[col].astype('category')
    
# Creating a list of all of the columns except the first
columns_to_encode = cat_meta_train.columns[1:].tolist()

# Print the columns to encode
print("Columns to encode:", columns_to_encode)
# encoding categorical data
train_encoded = pd.get_dummies(cat_meta_train[columns_to_encode], drop_first=True)
train_encoded = train_encoded.applymap(lambda x: 1 if x is True else (0 if x is False else x))
# Combine encoded columns with the rest of the DataFrame
cat_train_final = pd.concat([cat_meta_train.drop(columns=columns_to_encode), train_encoded], axis=1)

# Make sure it looks correct
cat_train_final.head()


# convert our int variables to categories
for col in cat_meta_test.select_dtypes(include='int').columns:
    cat_meta_test[col] = cat_meta_test[col].astype('category')

# Encode categorical variables in test
test_encoded = pd.get_dummies(cat_meta_test[columns_to_encode], drop_first=True)
test_encoded = test_encoded.map(lambda x: 1 if x is True else (0 if x is False else x))

# Ensure test_encoded has the same columns as train_encoded
missing_cols = set(train_encoded.columns) - set(test_encoded.columns)
for col in missing_cols:
    test_encoded[col] = 0  # Add missing columns with 0 values

# Ensure test_encoded columns are in the same order as train_encoded
test_encoded = test_encoded.reindex(columns=train_encoded.columns, fill_value=0)

# Combine encoded columns with the rest of the DataFrame
cat_test_final = pd.concat([cat_meta_test.drop(columns=columns_to_encode), test_encoded], axis=1)

cat_test_final.head()


cat_train_final['For']='Training'
cat_test_final['For']='Testing'

combine_cat_df=pd.concat([cat_train_final, cat_test_final], ignore_index=True)
print("number of null values")
print(combine_cat_df.isna().sum())
print(combine_cat_df.isna().sum().sum())
combine_cat_imp_df=combine_cat_df


# Replace null values in all columns with nulls with the mean of the column for training data
for col in combine_cat_imp_df.columns:
    if combine_cat_imp_df[col].isna().sum() > 0:  # Check if the column has NaN values
        combine_cat_imp_df[col] = combine_cat_imp_df[col].fillna(combine_cat_imp_df[col].mode()[0]) 


print(combine_cat_imp_df.isna().sum().sum()) # should now be zero


# Split back into train and test sets based on 'For' column
cat_train_data = combine_cat_imp_df[combine_cat_imp_df['For'] == 'Training'].drop('For', axis=1)
cat_test_data = combine_cat_imp_df[combine_cat_imp_df['For'] == 'Testing'].drop('For', axis=1)

# Verify the split
print("Training set shape:", cat_train_data.shape)
print("Test set shape:", cat_test_data.shape)

# Verify that we have all data
total_rows = len(combine_cat_imp_df)
split_rows = len(cat_train_data) + len(cat_test_data)
print("\nTotal rows in combined data:", total_rows)
print("Total rows after splitting:", split_rows)
print("All rows accounted for:", total_rows == split_rows)


quant_meta_train['For']='Training'
quant_meta_test['For']='Testing'
combine_quant_df=pd.concat([quant_meta_train, quant_meta_test], ignore_index=True)
print("number of null values")
print(combine_quant_df.isna().sum())
print(combine_quant_df.isna().sum().sum())

combine_quant_imp_df=combine_quant_df


# Replace nulls by sampling from a Gaussian distribution (column-wise mean and std)
for col in combine_quant_imp_df.columns:
    if combine_quant_imp_df[col].isna().sum() > 0:
        mean = combine_quant_imp_df[col].mean()
        std = combine_quant_imp_df[col].std()
        n_missing = combine_quant_imp_df[col].isna().sum()
        
        # Sample from Gaussian distribution for missing entries
        sampled_values = np.random.normal(loc=mean, scale=std, size=n_missing)
        
        # Assign sampled values to NaNs
        combine_quant_imp_df.loc[combine_quant_imp_df[col].isna(), col] = sampled_values

# Confirm no NaNs remain
print(combine_quant_imp_df.isna().sum().sum())  # should now be zero


# Split back into train and test sets based on 'For' column
quant_train_data = combine_quant_imp_df[combine_quant_imp_df['For'] == 'Training'].drop('For', axis=1)
quant_test_data = combine_quant_imp_df[combine_quant_imp_df['For'] == 'Testing'].drop('For', axis=1)

# Verify the split
print("Training set shape:", quant_train_data.shape)
print("Test set shape:", quant_test_data.shape)

# Verify that we have all data
total_rows = len(combine_quant_imp_df)
split_rows = len(quant_train_data) + len(quant_test_data)
print("\nTotal rows in combined data:", total_rows)
print("Total rows after splitting:", split_rows)
print("All rows accounted for:", total_rows == split_rows)


train_df = pd.merge(cat_train_data, quant_train_data, on = 'participant_id') #default
train_target_df = pd.merge(train_df, targets_train, on = 'participant_id') #default
train_target_Conn_df= pd.merge(train_target_df,  connectome_train, on = 'participant_id') #default
train_target_Conn_df.head()


test_df = pd.merge(cat_test_data, quant_test_data, on = 'participant_id')
test_Conn_df = pd.merge(test_df, connectome_test, on = 'participant_id')

test_Conn_df.head()


X_train = train_target_Conn_df.drop(columns = ['participant_id', 'ADHD_Outcome', 'Sex_F'])


Y_train = targets_train.drop(columns = ['participant_id'])


X_train2 = train_target_df.drop(columns = ['participant_id', 'ADHD_Outcome', 'Sex_F'])


Y_train2 = targets_train.drop(columns = ['participant_id'])


!pip install iterative-stratification


from sklearn.metrics import f1_score, classification_report
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import joblib

# Use all numeric features
adhd_features2 = X_train2.select_dtypes(include=[np.number]).columns.tolist()
sexf_features2 = adhd_features2.copy()

best_f1_2 = -1
best_adhd_model2 = None
best_sex_model2 = None
f1_scores2 = []

mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(mskf.split(X_train2, Y_train2)):
    print(f"\n--- Fold {fold+1} ---")
    X_tr, X_val = X_train.iloc[train_idx], X_train2.iloc[val_idx]
    Y_tr, Y_val = Y_train.iloc[train_idx], Y_train2.iloc[val_idx]
    
    # Optional: keep only numeric columns in case of mixed dtypes
    X_tr = X_tr[adhd_features2]
    X_val = X_val[adhd_features2]

    spw_adhd = (Y_tr['ADHD_Outcome'] == 0).sum() / (Y_tr['ADHD_Outcome'] == 1).sum()
    spw_sex = (Y_tr['Sex_F'] == 0).sum() / (Y_tr['Sex_F'] == 1).sum()

    xgb_adhd = XGBClassifier(scale_pos_weight=spw_adhd, objective='binary:logistic',
                             n_estimators=100, learning_rate=0.01, max_depth=3,
                             subsample=0.5, use_label_encoder=False, eval_metric='logloss')

    xgb_sex = XGBClassifier(scale_pos_weight=spw_sex, objective='binary:logistic',
                            n_estimators=100, learning_rate=0.01, max_depth=3,
                            subsample=0.5, use_label_encoder=False, eval_metric='logloss')

    xgb_adhd.fit(X_tr, Y_tr['ADHD_Outcome'])
    xgb_sex.fit(X_tr, Y_tr['Sex_F'])

    adhd_pred = xgb_adhd.predict(X_val)
    sex_pred = xgb_sex.predict(X_val)
    Y_pred = np.vstack([adhd_pred, sex_pred]).T

    f1 = f1_score(Y_val, Y_pred, average='macro')
    f1_scores2.append(f1)

    print(f"Macro F1 score: {f1:.4f}")
    print(classification_report(Y_val, Y_pred, target_names=['ADHD_Outcome', 'Sex_F']))

    if f1 > best_f1_2:
        best_f1_2 = f1
        best_adhd_model2 = xgb_adhd
        best_sex_model2 = xgb_sex
        X_val_best2 = X_val.copy()  # <-- Save validation features for SHAP
        Y_val_best2 = Y_val.copy()  # <-- Optional: Save validation labels too


# Save models
#joblib.dump(best_adhd_model2, 'best_adhd_model2.pkl')
#joblib.dump(best_sex_model2, 'best_sex_model2.pkl')

# Save F1 scores
#pd.Series(f1_scores2).to_csv('f1_scores.csv', index_label='Fold', header=['Macro_F1'])

# Feature importance diagnostics
adhd_importances2 = pd.Series(best_adhd_model2.feature_importances_, index=adhd_features2).sort_values(ascending=False)
sexf_importances2 = pd.Series(best_sex_model2.feature_importances_, index=sexf_features2).sort_values(ascending=False)

# Save importance to CSV for further analysis
#adhd_importances2.to_csv('adhd_feature_importances2.csv')
#sexf_importances2.to_csv('sexf_feature_importances2.csv')

print("Mean F1 score across folds:", np.mean(f1_scores2))



import shap

explainer_adhd = shap.TreeExplainer(best_adhd_model2)
shap_values_adhd = explainer_adhd.shap_values(X_val_best2)

explainer_sex = shap.TreeExplainer(best_sex_model2)
shap_values_sex = explainer_sex.shap_values(X_val_best2)

# Convert to DataFrames
shap_df_adhd = pd.DataFrame(shap_values_adhd, columns=adhd_features2)
shap_df_sex = pd.DataFrame(shap_values_sex, columns=sexf_features2)
shap.summary_plot(shap_values_adhd, X_val_best2)


shap_df_adhd.abs().mean().sort_values(ascending=False).head(10)


shap_df_sex.abs().mean().sort_values(ascending=False).head(10)


sexf_low_shap_features = shap_df_sex.abs().mean()[shap_df_sex.abs().mean() < 0.009].index.tolist()

sexf_low_shap_features


adhd_low_shap_features = shap_df_adhd.abs().mean()[shap_df_adhd.abs().mean() < 0.009].index.tolist()

adhd_low_shap_features


adhd_low_imp_features = adhd_importances2[adhd_importances2 < 0.028].index.tolist()
print(adhd_low_imp_features)
sex_low_imp_features = sexf_importances2[sexf_importances2 < 0.025].index.tolist()
print(sex_low_imp_features)


# Remove low-importance features
adhd_features3 = [f for f in X_train2.select_dtypes(include=[np.number]).columns if f not in adhd_low_imp_features]
sexf_features3 = [f for f in X_train2.select_dtypes(include=[np.number]).columns if f not in sex_low_imp_features]
#[f for f in adhd_features3 if f not in sex_low_imp_features]  # ensures both pruned

best_f1_3 = -1
best_adhd_model3 = None
best_sex_model3 = None
f1_scores3 = []

mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(mskf.split(X_train2, Y_train2)):
    print(f"\n--- Fold {fold+1} ---")
    
    X_tr, X_val = X_train2.iloc[train_idx], X_train2.iloc[val_idx]
    Y_tr, Y_val = Y_train2.iloc[train_idx], Y_train2.iloc[val_idx]

    # Apply filtered features
    X_tr_adhd = X_tr[adhd_features3]
    X_val_adhd = X_val[adhd_features3]
    
    X_tr_sex = X_tr[sexf_features3]
    X_val_sex = X_val[sexf_features3]

    spw_adhd = (Y_tr['ADHD_Outcome'] == 0).sum() / (Y_tr['ADHD_Outcome'] == 1).sum()
    spw_sex = (Y_tr['Sex_F'] == 0).sum() / (Y_tr['Sex_F'] == 1).sum()

    xgb_adhd = XGBClassifier(scale_pos_weight=spw_adhd, objective='binary:logistic',
                             n_estimators=100, learning_rate=0.01, max_depth=3,
                             subsample=0.5, use_label_encoder=False, eval_metric='logloss')

    xgb_sex = XGBClassifier(scale_pos_weight=spw_sex, objective='binary:logistic',
                            n_estimators=100, learning_rate=0.01, max_depth=3,
                            subsample=0.5, use_label_encoder=False, eval_metric='logloss')

    xgb_adhd.fit(X_tr_adhd, Y_tr['ADHD_Outcome'])
    xgb_sex.fit(X_tr_sex, Y_tr['Sex_F'])

    adhd_pred = xgb_adhd.predict(X_val_adhd)
    sex_pred = xgb_sex.predict(X_val_sex)
    Y_pred = np.vstack([adhd_pred, sex_pred]).T

    f1 = f1_score(Y_val, Y_pred, average='macro')
    f1_scores3.append(f1)

    print(f"Macro F1 score: {f1:.4f}")
    print(classification_report(Y_val, Y_pred, target_names=['ADHD_Outcome', 'Sex_F']))

    if f1 > best_f1_3:
        best_f1_3 = f1
        best_adhd_model3 = xgb_adhd
        best_sex_model3 = xgb_sex

# Save models
#joblib.dump(best_adhd_model3, 'best_adhd_model3.pkl')
#joblib.dump(best_sex_model3, 'best_sex_model3.pkl')

# Save F1 scores
#pd.Series(f1_scores3).to_csv('f1_scores3.csv', index_label='Fold', header=['Macro_F1'])

# Feature importance diagnostics
adhd_importances3 = pd.Series(best_adhd_model3.feature_importances_, index=adhd_features3).sort_values(ascending=False)
sexf_importances3 = pd.Series(best_sex_model3.feature_importances_, index=sexf_features3).sort_values(ascending=False)

# Save importance to CSV for further analysis
#adhd_importances3.to_csv('adhd_feature_importances3.csv')
#sexf_importances3.to_csv('sexf_feature_importances3.csv')

print("Mean F1 score across folds:", np.mean(f1_scores3))



'''
# Remove low-importance features
adhd_features5 = [f for f in X_train.select_dtypes(include=[np.number]).columns if f not in sexf_low_shap_features]
sexf_features5 = [f for f in X_train.select_dtypes(include=[np.number]).columns if f not in adhd_low_shap_features]

best_f1_5 = -1
best_adhd_model5 = None
best_sex_model5 = None
f1_scores5 = []

mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(mskf.split(X_train, Y_train)):
    print(f"\n--- Fold {fold+1} ---")
    
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    Y_tr, Y_val = Y_train.iloc[train_idx], Y_train.iloc[val_idx]

    # Apply filtered features
    X_tr_adhd = X_tr[adhd_features5]
    X_val_adhd = X_val[adhd_features5]
    
    X_tr_sex = X_tr[sexf_features5]
    X_val_sex = X_val[sexf_features5]

    spw_adhd = (Y_tr['ADHD_Outcome'] == 0).sum() / (Y_tr['ADHD_Outcome'] == 1).sum()
    spw_sex = (Y_tr['Sex_F'] == 0).sum() / (Y_tr['Sex_F'] == 1).sum()

    xgb_adhd = XGBClassifier(scale_pos_weight=spw_adhd, objective='binary:logistic',
                             n_estimators=100, learning_rate=0.01, max_depth=3,
                             subsample=0.5, use_label_encoder=False, eval_metric='logloss')

    xgb_sex = XGBClassifier(scale_pos_weight=spw_sex, objective='binary:logistic',
                            n_estimators=100, learning_rate=0.01, max_depth=3,
                            subsample=0.5, use_label_encoder=False, eval_metric='logloss')

    xgb_adhd.fit(X_tr_adhd, Y_tr['ADHD_Outcome'])
    xgb_sex.fit(X_tr_sex, Y_tr['Sex_F'])

    adhd_pred = xgb_adhd.predict(X_val_adhd)
    sex_pred = xgb_sex.predict(X_val_sex)
    Y_pred = np.vstack([adhd_pred, sex_pred]).T

    f1 = f1_score(Y_val, Y_pred, average='macro')
    f1_scores5.append(f1)

    print(f"Macro F1 score: {f1:.4f}")
    print(classification_report(Y_val, Y_pred, target_names=['ADHD_Outcome', 'Sex_F']))
    if f1 > best_f1_5:
        best_f1_5 = f1
        best_adhd_model5 = xgb_adhd
        best_sex_model5 = xgb_sex
        X_val_best5 = X_val.copy()
        Y_val_best5 = Y_val.copy()

### Save models
#joblib.dump(best_adhd_model5, 'best_adhd_model5.pkl')
#joblib.dump(best_sex_model5, 'best_sex_model5.pkl')

### Save F1 scores
pd.Series(f1_scores5).to_csv('f1_scores5.csv', index_label='Fold', header=['Macro_F1'])

### Feature importance diagnostics
adhd_importances5 = pd.Series(best_adhd_model5.feature_importances_, index=adhd_features5).sort_values(ascending=False)
sexf_importances5 = pd.Series(best_sex_model5.feature_importances_, index=sexf_features5).sort_values(ascending=False)

### Save importance to CSV for further analysis
adhd_importances5.to_csv('adhd_feature_importances5.csv')
sexf_importances5.to_csv('sexf_feature_importances5.csv')

print("Mean F1 score across folds:", np.mean(f1_scores5))

'''



adhd_importances5 = pd.read_csv('/kaggle/input/feature-importance-calculated-before/adhd_feature_importances5.csv', header=None)
adhd_importances5.columns = ['feature', 'importance']

## Filter based on importance threshold
adhd_high_imp_features_all_data = adhd_importances5[adhd_importances5['importance'] > 0.0005]['feature'].tolist()


sexf_importances5=pd.read_csv('/kaggle/input/feature-importance-calculated-before/sexf_feature_importances5.csv', header=None)
sexf_importances5.columns = ['feature', 'importance']
sexf_high_imp_features_all_data = sexf_importances5[sexf_importances5['importance'] > 0.0002]['feature'].tolist()



#adhd_high_imp_features_all_data = adhd_importances5[adhd_importances5 > 0.0005].index.tolist()
#sexf_high_imp_features_all_data = sexf_importances5[sexf_importances5 > 0.0005].index.tolist() #0.0002


#import xgboost as xgb
# Remove low-importance features
#adhd_features6 = [f for f in X_train.select_dtypes(include=[np.number]).columns if f in adhd_high_imp_features_all_data]
#sexf_features6 = [f for f in X_train.select_dtypes(include=[np.number]).columns if f not in sexf_high_imp_features_all_data]

adhd_features6 = [f for f in X_train.select_dtypes(include=[np.number]).columns 
                  if f in adhd_high_imp_features_all_data and f not in sexf_low_shap_features]

sexf_features6 = [f for f in X_train.select_dtypes(include=[np.number]).columns 
                  if f in sexf_high_imp_features_all_data and f not in adhd_low_shap_features]


best_f1_6 = -1
best_adhd_model6 = None
best_sex_model6 = None
f1_scores6 = []



mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(mskf.split(X_train, Y_train)):
    print(f"\n--- Fold {fold+1} ---")
    
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    Y_tr, Y_val = Y_train.iloc[train_idx], Y_train.iloc[val_idx]

    # Apply filtered features
    X_tr_adhd = X_tr[adhd_features6]
    X_val_adhd = X_val[adhd_features6]
    
    X_tr_sex = X_tr[sexf_features6]
    X_val_sex = X_val[sexf_features6]

    spw_adhd = (Y_tr['ADHD_Outcome'] == 0).sum() / (Y_tr['ADHD_Outcome'] == 1).sum()
    spw_sex = 1.2*(Y_tr['Sex_F'] == 0).sum() / (Y_tr['Sex_F'] == 1).sum() #1.2 is better

    #defauls n_estimators=100, learning_rate=0.01, max_depth=3,subsample=0.5,

    xgb_adhd = XGBClassifier(scale_pos_weight=spw_adhd, objective='binary:logistic',
                             n_estimators=800, learning_rate=0.005, max_depth=5,
                             subsample=0.7, use_label_encoder=False, eval_metric='logloss',
                             colsample_bytree=0.7,
                             min_child_weight=2,
                             gamma=0.1
                             #reg_alpha=0.1 #didnt improve scores
                             )

    xgb_sex = XGBClassifier(scale_pos_weight=spw_sex, objective='binary:logistic',
                            n_estimators=400, learning_rate=0.01, max_depth=3,
                            subsample=0.7, use_label_encoder=False, eval_metric='logloss',
                            colsample_bytree=0.5,
                            min_child_weight=5,
                            gamma=0.2,
                            #reg_alpha=0.5 #didnt improve scores
                            #reg_lambda=2   #didnt improve scores
                            )

    xgb_adhd.fit(X_tr_adhd, Y_tr['ADHD_Outcome'])
    xgb_sex.fit(X_tr_sex, Y_tr['Sex_F'])

    adhd_pred = xgb_adhd.predict(X_val_adhd)
    sex_pred = xgb_sex.predict(X_val_sex)
    Y_pred = np.vstack([adhd_pred, sex_pred]).T

    f1 = f1_score(Y_val, Y_pred, average='macro')
    f1_scores6.append(f1)

    print(f"Macro F1 score: {f1:.4f}")
    print(classification_report(Y_val, Y_pred, target_names=['ADHD_Outcome', 'Sex_F']))

    if f1 > best_f1_6:
        best_f1_6 = f1
        best_adhd_model6 = xgb_adhd
        best_sex_model6 = xgb_sex
        X_val_best6 = X_val.copy()
        Y_val_best6 = Y_val.copy()

# Save models
#joblib.dump(best_adhd_model6, 'best_adhd_model6.pkl')
#joblib.dump(best_sex_model6, 'best_sex_model6.pkl')

# Save F1 scores
#pd.Series(f1_scores6).to_csv('f1_scores6.csv', index_label='Fold', header=['Macro_F1'])

# Feature importance diagnostics
adhd_importances6 = pd.Series(best_adhd_model6.feature_importances_, index=adhd_features6).sort_values(ascending=False)
sexf_importances6 = pd.Series(best_sex_model6.feature_importances_, index=sexf_features6).sort_values(ascending=False)

# Save importance to CSV for further analysis
#adhd_importances6.to_csv('adhd_feature_importances6.csv')
#sexf_importances6.to_csv('sexf_feature_importances6.csv')

print("Mean F1 score across folds:", np.mean(f1_scores6))



explainer_adhd6 = shap.TreeExplainer(best_adhd_model6)
shap_values_adhd6 = explainer_adhd6.shap_values(X_val_best6[adhd_features6])

explainer_sex6 = shap.TreeExplainer(best_sex_model6)
shap_values_sex6 = explainer_sex6.shap_values(X_val_best6[sexf_features6])

# Convert to DataFrames
shap_df_adhd6 = pd.DataFrame(shap_values_adhd6, columns=adhd_features6)
shap_df_sex6 = pd.DataFrame(shap_values_sex6, columns=sexf_features6)


# Predict probabilities and classes
sex_probs_val6 = best_sex_model6.predict_proba(X_val_best6[sexf_features6])[:, 1]
sex_preds_val6 = (sex_probs_val6 > 0.5).astype(int)  # Using default 0.5 threshold
# True labels
true_labels = Y_val_best6['Sex_F'].values

# Find indices where model predicted 1 but true label was 0
false_positive_indices = np.where((sex_preds_val6 == 1) & (true_labels == 0))[0]
# SHAP values for just the false positives
shap_fp = shap_df_sex6.iloc[false_positive_indices]

# Original feature values for these FPs (optional)
X_fp = X_val_best6[sexf_features6].iloc[false_positive_indices]
# Mean absolute shap value per feature across false positives
shap_fp_mean = shap_fp.abs().mean().sort_values(ascending=False)
sexf_fp_shap20 =shap_fp_mean.head(20)
#print(sexf_fp_shap20)
sexf_fp_shap20.plot(kind='barh')
plt.title('Top 20 Features Driving False Positives (Sex_F)')
plt.xlabel('Mean |SHAP value|')
plt.gca().invert_yaxis()
plt.show()


# Full training data for ADHD
X_train_adhd_full = X_train[adhd_features6]
y_train_adhd_full = Y_train['ADHD_Outcome']

# Full training data for Sex_F
X_train_sex_full = X_train[sexf_features6]
y_train_sex_full = Y_train['Sex_F']

# Recreate models with same settings
#final_adhd_model6 = XGBClassifier(scale_pos_weight=(y_train_adhd_full == 0).sum() / (y_train_adhd_full == 1).sum(),
#                                  objective='binary:logistic',
#                                  n_estimators=100, learning_rate=0.01, max_depth=3,
#                                  subsample=0.5, use_label_encoder=False, eval_metric='logloss')


#final_sex_model6 = XGBClassifier(scale_pos_weight=1.2 * (y_train_sex_full == 0).sum() / (y_train_sex_full == 1).sum(),
#                                 objective='binary:logistic',
#                                 n_estimators=100, learning_rate=0.01, max_depth=3,
#                                 subsample=0.5, use_label_encoder=False, eval_metric='logloss')

final_adhd_model6 = XGBClassifier(scale_pos_weight=(y_train_adhd_full == 0).sum() / (y_train_adhd_full == 1).sum(),
                                  objective='binary:logistic',
                                  n_estimators=800, learning_rate=0.005, max_depth=5,
                                  subsample=0.7, use_label_encoder=False, eval_metric='logloss',
                                  #colsample_bytree=0.7,
                                  #min_child_weight=2,
                                  #gamma=0.1
                                  )


final_sex_model6 = XGBClassifier(scale_pos_weight=1.2* (y_train_sex_full == 0).sum() / (y_train_sex_full == 1).sum(),
                                 objective='binary:logistic',
                                 n_estimators=400, learning_rate=0.01, max_depth=3,
                                 subsample=0.7, use_label_encoder=False, eval_metric='logloss',
                                 #colsample_bytree=0.5,
                                 #min_child_weight=5,
                                 #gamma=0.2
                                 )

# Train on full training data
final_adhd_model6.fit(X_train_adhd_full, y_train_adhd_full);
final_sex_model6.fit(X_train_sex_full, y_train_sex_full);



X_test= test_Conn_df.drop(columns = ['participant_id'])
# Prepare test set (use same feature subsets!)
X_test_adhd = X_test[adhd_features6]
X_test_sex = X_test[sexf_features6]

# Predict
adhd_test_pred = final_adhd_model6.predict(X_test_adhd)
sex_test_pred = final_sex_model6.predict(X_test_sex)

# Combine into final prediction
final_test_pred = np.vstack([adhd_test_pred, sex_test_pred]).T


participant_id = test_Conn_df['participant_id']
# Convert predictions to a DataFrame
predictions_df = pd.DataFrame(
   final_test_pred,
    columns=['Predicted_ADHD', 'Predicted_Gender']
)
# Combine participant IDs with predictions
result_df = pd.concat([participant_id.reset_index(drop=True), predictions_df], axis=1)
result_df


result_df.to_csv('submission_XGBOOST_GaussianIMP.csv', index=False) 
print("submission has been saved")




