pip install geomstats


import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

import geomstats.backend as gs
import geomstats.datasets.utils as data_utils
import geomstats.backend as gs
from geomstats.geometry.skew_symmetric_matrices import SkewSymmetricMatrices

# Read in the train data
df_soln = pd.read_excel('/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAINING_SOLUTIONS.xlsx').sort_values('participant_id')
df_conn = pd.read_csv('/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv').sort_values('participant_id')

# Create separate dataframes with index set to participant_id
df_soln_idx = df_soln.set_index("participant_id")
df_conn_idx = df_conn.set_index("participant_id")

# Read in the test data
df_conn_test = pd.read_csv('/kaggle/input/widsdatathon2025/TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv')
df_test = df_conn_test.set_index("participant_id")

# Extract the ADHD and sex solutions and sort the data by participant_id
df_soln_adhd = df_soln[['participant_id', 'ADHD_Outcome']].sort_values('participant_id')
df_soln_sex = df_soln[['participant_id', 'Sex_F']].sort_values('participant_id')


# Function to load connectomes as symmetric matrices

def load_connectomes(df_conn, as_vectors=False):
    """
    Load brain connectome data, returning symmetric matrices with ones on the diagonal and patient IDs.
    
    Parameters:
        df_conn (DataFrame): DataFrame with 'participant_id' and flattened connectivity data.
        as_vectors (bool): If True, returns flattened vectors instead of matrices.
    
    Returns:
        matrices (array): Symmetric connectivity matrices or vectors
        patient_id (array): Participant IDs
    """
    
    patient_id = gs.array(df_conn['participant_id'])
    data = gs.array(df_conn.drop('participant_id', axis=1))
    
    if as_vectors:
        return data, patient_id

    mat = SkewSymmetricMatrices(200).matrix_representation(data)
    mat = gs.eye(200) - gs.transpose(gs.tril(mat), (0, 2, 1))
    mat = 1.0 / 2.0 * (mat + gs.transpose(mat, (0, 2, 1)))

    return mat, patient_id



# Load connectome matrices (train)
data, patient_id = load_connectomes(df_conn)

# Load connectome matrices (test)
data_test, patient_id_test = load_connectomes(df_conn_test)

# Load labels separately
labels_adhd = gs.array(df_soln_adhd['ADHD_Outcome'])
labels_sex = gs.array(df_soln_sex['Sex_F'])



from geomstats.geometry.spd_matrices import SPDMatrices

# Define SPD manifold for 200x200 matrices
manifold = SPDMatrices(200, equip=False)

# Check SPD manifold membership for training and test connectomes
print("SPD Manifold Check:")
print("Train data SPD membership:", gs.all(manifold.belongs(data)))
print("Test data SPD membership:", gs.all(manifold.belongs(data_test)))



# Function to correct matrices by adding diagonal offset
def make_spd(matrix, eps=1e-6):
    # Adds a small correction to ensure the matrix is SPD.
    eigenvalues = np.linalg.eigvals(matrix)
    min_eigenvalue = np.min(eigenvalues)

    if min_eigenvalue < 0:
        correction = -min_eigenvalue + eps
        correction_matrix = correction * np.eye(matrix.shape[0])
        return matrix + correction_matrix
    return matrix

# Apply correction to test matrices
print("\nApplying SPD correction to test data...")
data_test_corrected = np.array([make_spd(mat) for mat in data_test])

print("Original test matrix shape:", data_test.shape)
print("Corrected test matrix shape:", data_test_corrected.shape)

# Verify correction
print("\nSPD membership after correction:")
print(gs.all(manifold.belongs(data_test_corrected)))



pip install pyriemann==0.7


# Extract probability features for ADHD and Sex

from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict

def get_spd_probabilities(X_train, y_train, X_test, participant_ids_train, participant_ids_test, task_name):
    """
    Generate cross-validated probabilities for train and predict probabilities for test using SPD -> Tangent Space -> Logistic Regression.
    
    Returns:
        df_train_probs: DataFrame with train probabilities
        df_test_probs: DataFrame with test probabilities
    """
    
    # Define pipeline
    pipeline = Pipeline([
        ('ts', TangentSpace(metric='riemann')),
        ('clf', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
    ])
    
    # Cross-validated predicted probabilities (train)
    stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_probs = cross_val_predict(pipeline, X_train, y_train, cv=stratified_cv, method='predict_proba')[:, 1]

    # Store train probabilities
    df_train_probs = pd.DataFrame({
        'participant_id': participant_ids_train,
        f'{task_name}_spd_prob': train_probs
    }).set_index('participant_id')

    # Fit model on full training data
    pipeline.fit(X_train, y_train)

    # Predict probabilities for test
    #test_probs = pipeline.predict_proba(X_test)[:, 1]

    # Transform test data to tangent space - this and the next 2 steps are needed if running on Kaggle
    X_test_ts = pipeline.named_steps['ts'].transform(X_test)

    # Remove imaginary part if present
    X_test_ts = X_test_ts.real

    # Predict probabilities
    test_probs = pipeline.named_steps['clf'].predict_proba(X_test_ts)[:, 1]
    

    df_test_probs = pd.DataFrame({
        'participant_id': participant_ids_test,
        f'{task_name}_spd_prob': test_probs
    }).set_index('participant_id')

    return df_train_probs, df_test_probs



# ====================
# Run for ADHD
# ====================

df_train_probs_adhd, df_test_probs_adhd = get_spd_probabilities(
    X_train=data, 
    y_train=labels_adhd, 
    X_test=data_test_corrected, 
    participant_ids_train=patient_id,
    participant_ids_test=patient_id_test,
    task_name="adhd"
)

# ====================
# Run for Sex
# ====================

df_train_probs_sex, df_test_probs_sex = get_spd_probabilities(
    X_train=data, 
    y_train=labels_sex, 
    X_test=data_test_corrected, 
    participant_ids_train=patient_id,
    participant_ids_test=patient_id_test,
    task_name="sex"
)

# ====================
# Combine Train and Test
# ====================

# Combine train SPD probabilities
df_train_probs_combined = pd.concat([df_train_probs_adhd, df_train_probs_sex], axis=1)

# Combine test SPD probabilities
df_test_probs_combined = pd.concat([df_test_probs_adhd, df_test_probs_sex], axis=1)



# Final check
print(df_train_probs_combined.shape)
print(df_train_probs_combined.head())


from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import f1_score, roc_auc_score, brier_score_loss
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import scipy
from scipy.stats import kstest, ttest_ind, ks_2samp, mannwhitneyu, mode
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# Load Quantitative and Categorical data
df_train_quant = pd.read_excel("/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_QUANTITATIVE_METADATA_new.xlsx").sort_values('participant_id').set_index("participant_id")
df_train_cat = pd.read_excel("/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_CATEGORICAL_METADATA_new.xlsx").sort_values('participant_id').set_index("participant_id")

df_test_quant = pd.read_excel("/kaggle/input/widsdatathon2025/TEST/TEST_QUANTITATIVE_METADATA.xlsx").set_index("participant_id")
df_test_cat = pd.read_excel("/kaggle/input/widsdatathon2025/TEST/TEST_CATEGORICAL.xlsx").set_index("participant_id")

# Combine Quantitative, Categorical, and SPD Connectome Probabilities
df_train_combined = pd.concat([df_train_quant, df_train_cat, df_train_probs_combined], axis=1)
df_test_combined = pd.concat([df_test_quant, df_test_cat, df_test_probs_combined], axis=1)

# Validate indices match before merging
assert all(df_train_combined.index == df_soln_idx.index), "Train Label IDs don't match train IDs"

# Combine features + targets
df_train_all = pd.concat([df_train_combined, df_soln_idx], axis=1)



# Use SimpleImputer with median strategy for APQ and SDQ
from sklearn.impute import SimpleImputer

# ======== APQ ========
imputer = SimpleImputer(strategy='median')
df_train_all[['APQ_P_APQ_P_CP', 'APQ_P_APQ_P_ID', 'APQ_P_APQ_P_INV', 'APQ_P_APQ_P_OPD', 'APQ_P_APQ_P_PM', 
          'APQ_P_APQ_P_PP']] = imputer.fit_transform(df_train_all[['APQ_P_APQ_P_CP', 'APQ_P_APQ_P_ID', 'APQ_P_APQ_P_INV', 'APQ_P_APQ_P_OPD', 
                                                         'APQ_P_APQ_P_PM', 'APQ_P_APQ_P_PP']])

df_test_combined[['APQ_P_APQ_P_CP', 'APQ_P_APQ_P_ID', 'APQ_P_APQ_P_INV', 'APQ_P_APQ_P_OPD', 'APQ_P_APQ_P_PM', 
          'APQ_P_APQ_P_PP']] = imputer.fit_transform(df_test_combined[['APQ_P_APQ_P_CP', 'APQ_P_APQ_P_ID', 'APQ_P_APQ_P_INV', 'APQ_P_APQ_P_OPD', 
                                                         'APQ_P_APQ_P_PM', 'APQ_P_APQ_P_PP']])

# ======== SDQ ========
df_train_all[['SDQ_SDQ_Conduct_Problems', 'SDQ_SDQ_Difficulties_Total', 'SDQ_SDQ_Emotional_Problems', 'SDQ_SDQ_Externalizing', 
          'SDQ_SDQ_Generating_Impact', 'SDQ_SDQ_Hyperactivity', 'SDQ_SDQ_Internalizing', 'SDQ_SDQ_Peer_Problems', 
          'SDQ_SDQ_Prosocial']] = imputer.fit_transform(df_train_all[['SDQ_SDQ_Conduct_Problems', 'SDQ_SDQ_Difficulties_Total', 
                                                                 'SDQ_SDQ_Emotional_Problems', 'SDQ_SDQ_Externalizing', 
                                                                 'SDQ_SDQ_Generating_Impact', 'SDQ_SDQ_Hyperactivity', 
                                                                 'SDQ_SDQ_Internalizing', 'SDQ_SDQ_Peer_Problems', 'SDQ_SDQ_Prosocial']])

df_test_combined[['SDQ_SDQ_Conduct_Problems', 'SDQ_SDQ_Difficulties_Total', 'SDQ_SDQ_Emotional_Problems', 'SDQ_SDQ_Externalizing', 
          'SDQ_SDQ_Generating_Impact', 'SDQ_SDQ_Hyperactivity', 'SDQ_SDQ_Internalizing', 'SDQ_SDQ_Peer_Problems', 
          'SDQ_SDQ_Prosocial']] = imputer.fit_transform(df_test_combined[['SDQ_SDQ_Conduct_Problems', 'SDQ_SDQ_Difficulties_Total', 
                                                                 'SDQ_SDQ_Emotional_Problems', 'SDQ_SDQ_Externalizing', 
                                                                 'SDQ_SDQ_Generating_Impact', 'SDQ_SDQ_Hyperactivity', 
                                                                 'SDQ_SDQ_Internalizing', 'SDQ_SDQ_Peer_Problems', 'SDQ_SDQ_Prosocial']])

# ======== EHQ and Color Vision ========
# Use simple imputer with 'most frequent' strategy

# First, our EHQ test data has 3 values of -100.05, and 50 values of 100.05. Let's fix those.
df_test_combined['EHQ_EHQ_Total'] = df_test_combined['EHQ_EHQ_Total'].clip(lower=-100, upper=100)

imputer = SimpleImputer(strategy='most_frequent')
df_train_all['EHQ_EHQ_Total'] = imputer.fit_transform(df_train_all[['EHQ_EHQ_Total']])
df_test_combined['EHQ_EHQ_Total'] = imputer.transform(df_test_combined[['EHQ_EHQ_Total']])

df_train_all['ColorVision_CV_Score'] = imputer.fit_transform(df_train_all[['ColorVision_CV_Score']])
df_test_combined['ColorVision_CV_Score'] = imputer.transform(df_test_combined[['ColorVision_CV_Score']])

# ======== Child Ethnicity and Child Race ========
# Let's assign them all to unknown (3 for ethnicity and 10 for race)
df_train_all['PreInt_Demos_Fam_Child_Ethnicity'] = df_train_all['PreInt_Demos_Fam_Child_Ethnicity'].fillna(3)
df_test_combined['PreInt_Demos_Fam_Child_Ethnicity'] = df_test_combined['PreInt_Demos_Fam_Child_Ethnicity'].fillna(3)

df_train_all['PreInt_Demos_Fam_Child_Race'] = df_train_all['PreInt_Demos_Fam_Child_Race'].fillna(10)
df_test_combined['PreInt_Demos_Fam_Child_Race'] = df_test_combined['PreInt_Demos_Fam_Child_Race'].fillna(10)

# ======== MRI track scan location ========
# Since there are only 3 missing values, and only in the train set, let's use the most frequent location
df_train_all['MRI_Track_Scan_Location'] = df_train_all['MRI_Track_Scan_Location'].fillna(df_train_all['MRI_Track_Scan_Location'].mode()[0])

# ======== Barratt Education and Barratt Occupation ========
# Use a KNNImputer to impute the missing values
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
df_train_all[['Barratt_Barratt_P1_Edu', 'Barratt_Barratt_P2_Edu']] = imputer.fit_transform(df_train_all[['Barratt_Barratt_P1_Edu', 
                                                                                                 'Barratt_Barratt_P2_Edu']])
df_test_combined[['Barratt_Barratt_P1_Edu', 'Barratt_Barratt_P2_Edu']] = imputer.transform(df_test_combined[['Barratt_Barratt_P1_Edu', 
                                                                                           'Barratt_Barratt_P2_Edu']])

df_train_all[['Barratt_Barratt_P1_Occ', 'Barratt_Barratt_P2_Occ']] = imputer.fit_transform(df_train_all[['Barratt_Barratt_P1_Occ', 
                                                                                                 'Barratt_Barratt_P2_Occ']])
df_test_combined[['Barratt_Barratt_P1_Occ', 'Barratt_Barratt_P2_Occ']] = imputer.transform(df_test_combined[['Barratt_Barratt_P1_Occ', 
                                                                                           'Barratt_Barratt_P2_Occ']])

# ======== MRI_Track_Age_at_Scan ========
# First, make the zeroes NaN
# Note, test data does not have any missing values for this feature
df_train_all['MRI_Track_Age_at_Scan'] = df_train_all['MRI_Track_Age_at_Scan'].replace(0, np.nan)

from sklearn.neighbors import KNeighborsRegressor

# Split the data into training and testing sets
train_df = df_train_all.dropna(subset=['MRI_Track_Age_at_Scan'])
test_df = df_train_all[df_train_all['MRI_Track_Age_at_Scan'].isna()]

# Create a KNN regressor model
model = KNeighborsRegressor(n_neighbors=5)

# Fit the model to the training data using the highest correlated feature
model.fit(train_df[['APQ_P_APQ_P_PM']], train_df['MRI_Track_Age_at_Scan'])

# Predict the missing values
predicted_ages = model.predict(test_df[['APQ_P_APQ_P_PM']])

#print(predicted_ages)

# Impute the missing values
df_train_all.loc[df_train_all['MRI_Track_Age_at_Scan'].isna(), 'MRI_Track_Age_at_Scan'] = predicted_ages



from sklearn.linear_model import LogisticRegression, RidgeClassifier

# Clean training features - drop targets
X_train_clean = df_train_all.drop(columns=['ADHD_Outcome', 'Sex_F'])

# Targets
y_adhd = df_soln['ADHD_Outcome']
y_sex = df_soln['Sex_F']

# ======= Ridge for ADHD =======

from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import cross_val_predict
from scipy.special import expit

# Cross-validated decision function
ridge = RidgeClassifier(alpha=1.0, random_state=42)
adhd_scores_cv = cross_val_predict(ridge, X_train_clean, y_adhd, cv=5, method='decision_function')

# Convert decision scores to probabilities
adhd_probs_cv = expit(adhd_scores_cv)

# Create a DataFrame
df_train_all_clean = df_train_all.drop(columns=['ADHD_Outcome', 'Sex_F'])
df_train_all_clean['adhd_ridge_prob'] = adhd_probs_cv
#print(df_train_all_clean)

# Ridge Classifier for ADHD
ridge = RidgeClassifier(alpha=1.0, random_state=42)

# Train on full training data
ridge.fit(X_train_clean, y_adhd)

# Get decision_function on test
df_test_all_clean = df_test_combined.copy()
adhd_scores_test = ridge.decision_function(df_test_all_clean)

# Convert decision scores to probabilities
adhd_probs_test = expit(adhd_scores_test)

# Add to test feature dataframe
df_test_all_clean['adhd_ridge_prob'] = adhd_probs_test

# ======= Logistic Regression for Sex =======

logreg = LogisticRegression(max_iter=3000, random_state=42, class_weight='balanced')
sex_probs_cv = cross_val_predict(logreg, X_train_clean, y_sex, cv=5, method='predict_proba')[:, 1]

# Create a DataFrame
df_train_all_clean['sex_lr_prob'] = sex_probs_cv
#print(df_train_all_clean)

# Get predictions on test data
logreg = LogisticRegression(max_iter=3000, random_state=42, class_weight='balanced')

# Train on full training data
logreg.fit(X_train_clean, y_sex)

# Get test probabilities
sex_probs_test = logreg.predict_proba(df_test_combined)[:, 1]

# Add to test feature dataframe
df_test_all_clean['sex_lr_prob'] = sex_probs_test


# Let's use a subset of the features with good correlations

selected_features = [
    # Important SPD and classic model predictions
    'adhd_ridge_prob', 
    'adhd_spd_prob',
    'sex_lr_prob', 
    'sex_spd_prob',
    
    # SDQ highly correlated features
    'SDQ_SDQ_Hyperactivity',
    'SDQ_SDQ_Externalizing',
    'SDQ_SDQ_Difficulties_Total',
    'SDQ_SDQ_Generating_Impact',
    'SDQ_SDQ_Conduct_Problems',
    'SDQ_SDQ_Internalizing',
    'SDQ_SDQ_Prosocial',
    'SDQ_SDQ_Emotional_Problems',

    # ColorVision (mildly correlated with Sex_F)
    'ColorVision_CV_Score',
]

# Subset training and test data
X_train_selected = df_train_all_clean[selected_features]
X_test_selected = df_test_all_clean[selected_features]

# Targets
y_multi = df_soln[['ADHD_Outcome', 'Sex_F']]



from sklearn.multioutput import MultiOutputClassifier

# Define final tuned base model
final_base_lr = LogisticRegression(
    C=1,
    class_weight='balanced',  
    penalty='l2',
    solver='saga',
    max_iter=10000,
    random_state=42
)

# Wrap it in MultiOutputClassifier
multi_output_model = MultiOutputClassifier(final_base_lr)

# Train on all selected features
multi_output_model.fit(X_train_selected, y_multi)

# Predict on test set
multi_preds = multi_output_model.predict(X_test_selected)

# Extract individual columns
adhd_preds_multi = multi_preds[:, 0]
sex_preds_multi = multi_preds[:, 1]

# Build submission dataframe
submission_multiout = pd.DataFrame({
    'participant_id': df_test.index.values,
    'ADHD_Outcome': adhd_preds_multi,
    'Sex_F': sex_preds_multi
})

# Save to CSV
submission_multiout.to_csv('/kaggle/working/submission.csv', index=False)




