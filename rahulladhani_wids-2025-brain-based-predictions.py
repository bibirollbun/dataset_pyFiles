# Data handling
import pandas as pd
import numpy as np

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# Settings
sns.set(style="whitegrid")
pd.set_option('display.max_columns', 100)  # Show more columns if needed

print("âœ… Imports done.")



# Load training targets
targets = pd.read_excel("/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAINING_SOLUTIONS.xlsx")

# Load functional brain connectome features
connectome = pd.read_csv("/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv")

# Load categorical metadata
cat_meta = pd.read_excel("/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_CATEGORICAL_METADATA_new.xlsx")

# Load quantitative metadata
quant_meta = pd.read_excel("/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_QUANTITATIVE_METADATA_new.xlsx")



# Merge into one DataFrame called 'wd'
wd = targets.copy()
wd = wd.merge(connectome, on='participant_id', how='left')
wd = wd.merge(cat_meta, on='participant_id', how='left')
wd = wd.merge(quant_meta, on='participant_id', how='left')

# Quick shape check
print("âœ… Shape of merged training data:", wd.shape)
wd.head()



# Check shape and column list
print("Shape:", wd.shape)



# Display the first few rows
wd.head()


# Check data types and non-null counts
wd.info()


# Count missing values
missing = wd.isnull().sum()
missing[missing > 0].sort_values(ascending=False)


# ğŸ“¦ Imports (in case not already imported)
import matplotlib.pyplot as plt
import seaborn as sns

# ğŸ�¯ Plotting ADHD Outcome
plt.figure(figsize=(6, 4))
sns.countplot(data=wd, x='ADHD_Outcome')
plt.title('Distribution of ADHD Diagnosis')
plt.xlabel('ADHD_Outcome (0 = No ADHD, 1 = ADHD)')
plt.ylabel('Count')
plt.tight_layout()
plt.show()



# âš¥ Plotting Sex Distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=wd, x='Sex_F')
plt.title('Distribution of Participant Sex')
plt.xlabel('Sex_F (0 = Male, 1 = Female)')
plt.ylabel('Count')
plt.tight_layout()
plt.show()



# ğŸ§© ADHD by Sex
plt.figure(figsize=(6, 4))
sns.countplot(data=wd, x='Sex_F', hue='ADHD_Outcome')
plt.title('ADHD Diagnosis by Sex')
plt.xlabel('Sex_F (0 = Male, 1 = Female)')
plt.ylabel('Count')
plt.legend(title='ADHD_Outcome', labels=['No ADHD (0)', 'ADHD (1)'])
plt.tight_layout()
plt.show()



plt.figure(figsize=(8, 5))
sns.boxplot(data=wd, x='ADHD_Outcome', y='MRI_Track_Age_at_Scan')
plt.title('Age Distribution by ADHD Diagnosis')
plt.xlabel('ADHD_Outcome')
plt.ylabel('Age at Scan')
plt.tight_layout()
plt.show()



# Filter numeric metadata columns with relatively low dimensions
meta_numeric = wd.select_dtypes(include=['float64', 'int64']).copy()

# Drop fMRI features (those with names like '0throw_1thcolumn') to reduce clutter
meta_numeric = meta_numeric.loc[:, ~meta_numeric.columns.str.contains('throw')]

# Drop participant_id
meta_numeric = meta_numeric.drop(columns=['participant_id'], errors='ignore')

# Plot correlation
plt.figure(figsize=(16, 10))
sns.heatmap(meta_numeric.corr(), cmap='coolwarm', annot=False)
plt.title('Correlation Heatmap (Metadata Only)')
plt.tight_layout()
plt.show()



import missingno as msno

# Show where missing values are
msno.matrix(wd)
plt.title('Missing Value Matrix')
plt.show()



# Manually selected categorical columns (based on previous exploration)
manual_cat_cols = [
    'PreInt_Demos_Fam_Child_Race',
    'PreInt_Demos_Fam_Child_Ethnicity',
    'Barratt_Barratt_P1_Edu',
    'Barratt_Barratt_P2_Edu',
    'Barratt_Barratt_P1_Occ',
    'Barratt_Barratt_P2_Occ',
    'MRI_Track_Scan_Location'
]



# One-hot encode selected columns
wd_encoded = pd.get_dummies(wd, columns=manual_cat_cols, drop_first=True)
print("âœ… One-hot encoding complete. New shape:", wd_encoded.shape)



# Manually selected numerical metadata columns
manual_numeric_cols = [
    'MRI_Track_Age_at_Scan',
    'ColorVision_CV_Score',
    'EHQ_EHQ_Total',
    'APQ_P_APQ_P_PM',
    'APQ_P_APQ_P_PP',
    'APQ_P_APQ_P_OPD',
    'APQ_P_APQ_P_INV',
    'APQ_P_APQ_P_ID',
    'APQ_P_APQ_P_CP',
    'SDQ_SDQ_Difficulties_Total',
    'SDQ_SDQ_Emotional_Problems',
    'SDQ_SDQ_Externalizing',
    'SDQ_SDQ_Generating_Impact',
    'SDQ_SDQ_Hyperactivity',
    'SDQ_SDQ_Internalizing',
    'SDQ_SDQ_Peer_Problems',
    'SDQ_SDQ_Prosocial',
    'SDQ_SDQ_Conduct_Problems'
]



from sklearn.preprocessing import StandardScaler

# Create a copy to preserve original
wd_scaled = wd_encoded.copy()

# Apply standard scaling
scaler = StandardScaler()
wd_scaled[manual_numeric_cols] = scaler.fit_transform(wd_scaled[manual_numeric_cols])

print("âœ… Numerical metadata scaling complete.")



# We'll work with the scaled DataFrame
X_final = wd_scaled.copy()

# Fill all NaNs with column-wise means
X_final = X_final.fillna(X_final.mean(numeric_only=True))

print("âœ… Missing values filled with column means.")



# Identify metadata/target columns
meta_target_cols = [
    'participant_id', 'ADHD_Outcome', 'Sex_F'
] + manual_numeric_cols + list(wd_encoded.columns.difference(wd.columns))  # One-hot encoded cols

# fMRI = all columns not in meta_target_cols
fmri_cols = [col for col in wd_encoded.columns if col not in meta_target_cols]

# Extract fMRI features
fmri_data = wd_encoded[fmri_cols]

print("fMRI data shape:", fmri_data.shape)



from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# PCA: retain 95% of variance
pca = PCA(n_components=0.95, random_state=42)
fmri_pca = pca.fit_transform(fmri_data)

# Show result
print("âœ… Reduced fMRI shape:", fmri_pca.shape)

# Plot explained variance
plt.figure(figsize=(10, 4))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.title('Explained Variance by PCA Components')
plt.grid()
plt.tight_layout()
plt.show()



# Convert PCA output into DataFrame with clear names
pca_df = pd.DataFrame(fmri_pca, columns=[f'pca_{i+1}' for i in range(fmri_pca.shape[1])])
pca_df.index = wd_encoded.index  # align index

# Select encoded metadata columns
metadata_cols = [col for col in wd_scaled.columns 
                 if col not in ['participant_id', 'ADHD_Outcome', 'Sex_F'] + fmri_cols]

meta_data = wd_scaled[metadata_cols]

# Combine metadata and PCA components
X_final = pd.concat([meta_data.reset_index(drop=True), pca_df.reset_index(drop=True)], axis=1)
y_final = wd_scaled[['ADHD_Outcome', 'Sex_F']]

print("âœ… Final shape for training:", X_final.shape)

# This step to handle missing values
X_final = X_final.fillna(X_final.mean())
print("âœ… All missing values handled in X_final.")



from sklearn.model_selection import train_test_split

# Stratify by ADHD outcome to preserve class balance
X_train, X_val, y_train, y_val = train_test_split(
    X_final, y_final, test_size=0.2, random_state=42, stratify=y_final['ADHD_Outcome']
)

print("âœ… Train shape:", X_train.shape)
print("âœ… Validation shape:", X_val.shape)



from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score

# New model
hgb_model = HistGradientBoostingClassifier(random_state=42)

# Wrap it for multi-output
multi_model = MultiOutputClassifier(hgb_model)

# Train
multi_model.fit(X_train, y_train)

# Predict
y_pred = multi_model.predict(X_val)


# Evaluate
adhd_f1 = f1_score(y_val['ADHD_Outcome'], y_pred[:, 0])
sex_f1 = f1_score(y_val['Sex_F'], y_pred[:, 1])
avg_f1 = (adhd_f1 + sex_f1) / 2

print(f"ğŸ“Š ADHD F1 Score: {adhd_f1:.4f}")
print(f"ğŸ“Š Sex F1 Score:  {sex_f1:.4f}")
print(f"ğŸ�� Final Avg F1 Score: {avg_f1:.4f}")



from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier

# Base HistGBM Model (same as before)
base_hist = HistGradientBoostingClassifier(random_state=42)

multi_hist = MultiOutputClassifier(base_hist)
multi_hist.fit(X_train, y_train)

# Predict and Evaluate
y_pred_hist = multi_hist.predict(X_val)

f1_adhd_hist = f1_score(y_val['ADHD_Outcome'], y_pred_hist[:, 0])
f1_sex_hist = f1_score(y_val['Sex_F'], y_pred_hist[:, 1])

print(f"ğŸ“Š ADHD F1: {f1_adhd_hist:.4f}, Sex F1: {f1_sex_hist:.4f}, Avg: {(f1_adhd_hist + f1_sex_hist)/2:.4f}")



from sklearn.model_selection import GridSearchCV

# Grid Search
param_grid = {
    'estimator__learning_rate': [0.1, 0.05, 0.01],
    'estimator__max_iter': [100, 300, 500]
}

grid_search = GridSearchCV(
    MultiOutputClassifier(HistGradientBoostingClassifier(random_state=42)),
    param_grid,
    scoring='f1_micro',
    cv=3,
    verbose=3,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("âœ… Best params:", grid_search.best_params_)
print("âœ… Best Score:", grid_search.best_score_)



param_grid2 = {
    'estimator__max_depth': [3, 5, 7, 10],
    'estimator__min_samples_leaf': [10, 20, 30]
}

grid_search2 = GridSearchCV(
    MultiOutputClassifier(
        HistGradientBoostingClassifier(
            random_state=42,
            learning_rate=grid_search.best_params_['estimator__learning_rate'],
            max_iter=grid_search.best_params_['estimator__max_iter']
        )
    ),
    param_grid2,
    scoring='f1_micro',
    cv=3,
    verbose=3,
    n_jobs=-1
)

grid_search2.fit(X_train, y_train)

print("âœ… Best params:", grid_search2.best_params_)
print("âœ… Best Score:", grid_search2.best_score_)



param_grid3 = {
    'estimator__max_leaf_nodes': [20, 31, 50],
    'estimator__l2_regularization': [0.0, 1.0, 5.0],  # <-- added useful parameter
    'estimator__min_samples_leaf': [10, 20, 30]        # <-- optional if you want more tuning
}

grid_search3 = GridSearchCV(
    MultiOutputClassifier(
        HistGradientBoostingClassifier(
            random_state=42,
            learning_rate=grid_search.best_params_['estimator__learning_rate'],
            max_iter=grid_search.best_params_['estimator__max_iter'],
            max_depth=grid_search2.best_params_['estimator__max_depth'],
            min_samples_leaf=grid_search2.best_params_['estimator__min_samples_leaf']
        )
    ),
    param_grid=param_grid3,
    scoring='f1_micro',
    cv=3,
    verbose=3,
    n_jobs=-1
)

grid_search3.fit(X_train, y_train)

print("âœ… Best params:", grid_search3.best_params_)
print("âœ… Best Score:", grid_search3.best_score_)



from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

final_model = MultiOutputClassifier(
    HistGradientBoostingClassifier(
        random_state=42,
        learning_rate=grid_search.best_params_['estimator__learning_rate'],
        max_iter=grid_search.best_params_['estimator__max_iter'],
        max_depth=grid_search2.best_params_['estimator__max_depth'],
        min_samples_leaf=grid_search3.best_params_['estimator__min_samples_leaf'],
        max_leaf_nodes=grid_search3.best_params_['estimator__max_leaf_nodes'],
        l2_regularization=grid_search3.best_params_['estimator__l2_regularization']
    )
)

# Train on full training data
final_model.fit(X_train, y_train)

print("âœ… Final model retrained!")



# 1. Load the test files
test_connectome = pd.read_csv("/kaggle/input/widsdatathon2025/TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv")
test_cat_meta = pd.read_excel("/kaggle/input/widsdatathon2025/TEST/TEST_CATEGORICAL.xlsx")
test_quant_meta = pd.read_excel("/kaggle/input/widsdatathon2025/TEST/TEST_QUANTITATIVE_METADATA.xlsx")
test_ids = pd.read_excel("/kaggle/input/widsdatathon2025/SAMPLE_SUBMISSION.xlsx")

# 2. Merge all test data
test_wd = test_ids[['participant_id']].copy()
test_wd = test_wd.merge(test_connectome, on='participant_id', how='left')
test_wd = test_wd.merge(test_cat_meta, on='participant_id', how='left')
test_wd = test_wd.merge(test_quant_meta, on='participant_id', how='left')

# 3. One-hot encode categorical columns
test_encoded = pd.get_dummies(test_wd, columns=manual_cat_cols, drop_first=True)

# 4. Scale numeric metadata
test_encoded[manual_numeric_cols] = scaler.transform(test_encoded[manual_numeric_cols])

# 5. Fill missing values with mean of training set
test_encoded = test_encoded.fillna(X_final.mean())

# 6. Fix missing fMRI columns
for col in fmri_cols:
    if col not in test_encoded.columns:
        test_encoded[col] = 0
test_fmri = test_encoded[fmri_cols]

# 7. PCA transform
test_pca = pca.transform(test_fmri)
test_pca_df = pd.DataFrame(test_pca, columns=[f'pca_{i+1}' for i in range(test_pca.shape[1])])
test_pca_df.index = test_encoded.index

# 8. Fix missing metadata columns
for col in meta_data.columns:
    if col not in test_encoded.columns:
        test_encoded[col] = 0
test_meta = test_encoded[meta_data.columns]

# 9. Combine metadata + PCA
X_test_final = pd.concat([test_meta.reset_index(drop=True), test_pca_df.reset_index(drop=True)], axis=1)

print("âœ… X_test_final ready:", X_test_final.shape)



# Predict
final_preds = final_model.predict(X_test_final)

# Save submission
submission_final = pd.DataFrame({
    'participant_id': test_encoded['participant_id'],
    'ADHD_Outcome': final_preds[:, 0],
    'Sex_F': final_preds[:, 1]
})
submission_final.to_csv("submission_final_hgb.csv", index=False)
print("âœ… Final submission created!")


