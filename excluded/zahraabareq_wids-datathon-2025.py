!pip install -U scikit-learn==1.2.2 imbalanced-learn==0.10.1


# import library
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from nilearn import datasets
from nilearn import image
from nilearn.input_data import NiftiLabelsMasker
from nilearn import plotting
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier


#import data 
Q_df = pd.read_excel('/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_QUANTITATIVE_METADATA_new.xlsx')
Cat_df=pd.read_excel('/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_CATEGORICAL_METADATA_new.xlsx')
conn_mat=pd.read_csv('/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv')
sol=pd.read_excel('/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAINING_SOLUTIONS.xlsx')


Q_df_test = pd.read_excel('/kaggle/input/widsdatathon2025/TEST/TEST_QUANTITATIVE_METADATA.xlsx')
Cat_df_test=pd.read_excel('/kaggle/input/widsdatathon2025/TEST/TEST_CATEGORICAL.xlsx')
conn_mat_test=pd.read_csv('/kaggle/input/widsdatathon2025/TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv')
sub=pd.read_excel('/kaggle/input/widsdatathon2025/SAMPLE_SUBMISSION.xlsx')


q_targets = sol.set_index('participant_id').loc[Q_df['participant_id']][['Sex_F', 'ADHD_Outcome']].reset_index(drop=True)
q_numeric_cols = Q_df.select_dtypes(include='number').columns

q_combined = pd.concat([Q_df[q_numeric_cols].reset_index(drop=True), q_targets], axis=1)

q_corr = q_combined.corr()

q_target_corr = q_corr[['Sex_F', 'ADHD_Outcome']].drop(['Sex_F', 'ADHD_Outcome'])

plt.figure(figsize=(10, 8))
sns.heatmap(q_target_corr, annot=True, cmap='coolwarm', center=0)
plt.title("Correlation of Q_df Numerical Features with Gender and ADHD")
plt.show()



sns.displot(
    data=Q_df.isnull().melt(value_name='missing'),
    y='variable',
    hue='missing',
    multiple='fill',
    height=8,
    aspect=1,
    palette="Set2"
)



cat_targets = sol.set_index('participant_id').loc[Cat_df['participant_id']][['Sex_F', 'ADHD_Outcome']].reset_index(drop=True)
cat_numeric_cols = Cat_df.select_dtypes(include='number').columns

cat_combined = pd.concat([Cat_df[cat_numeric_cols].reset_index(drop=True), cat_targets], axis=1)

cat_corr = cat_combined.corr()

cat_target_corr = cat_corr[['Sex_F', 'ADHD_Outcome']].drop(['Sex_F', 'ADHD_Outcome'])

plt.figure(figsize=(10, 8))
sns.heatmap(cat_target_corr, annot=True, cmap='coolwarm', center=0)
plt.title("Correlation of Cat_df Numerical Features with Gender and ADHD")
plt.show()



sns.displot(
    data=Cat_df.isnull().melt(value_name='missing'),
    y='variable',
    hue='missing',
    multiple='fill',
    height=8,
    aspect=1,
    palette="Set2"
)



#Matrix of patient one (0)
row = conn_mat.iloc[0, 1:]  

n_regions = int((1 + np.sqrt(1 + 8 * len(row))) / 2)
print(n_regions)

matrix = np.zeros((n_regions, n_regions))

triu_indices = np.triu_indices(n_regions, k=1)
matrix[triu_indices] = row.values

matrix = matrix + matrix.T

plt.figure(figsize=(10, 8))
plt.imshow(matrix, cmap='coolwarm')
plt.title('Functional Connectome Matrix (Participant 1)')
plt.colorbar(label='Connectivity Strength')
plt.xlabel('Brain Regions')
plt.ylabel('Brain Regions')
plt.show()



atlas = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7)

atlas_filename = atlas['maps']




img = image.load_img(atlas_filename)
coords = plotting.find_parcellation_cut_coords(labels_img=img)


plotting.view_connectome(matrix, coords,edge_threshold=0.7) 


features = Q_df.drop(columns=['participant_id'])

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

scaled_Q_df = pd.DataFrame(scaled_features, columns=features.columns)
scaled_Q_df['participant_id'] = Q_df['participant_id'].values 


features_test = Q_df_test.drop(columns=['participant_id'])

scaled_features_test = scaler.fit_transform(features_test)

scaled_Q_df_test = pd.DataFrame(scaled_features_test, columns=features_test.columns)
scaled_Q_df_test['participant_id'] = Q_df_test['participant_id'].values 





features = Cat_df.drop(columns=['participant_id'])

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

scaled_Cat_df = pd.DataFrame(scaled_features, columns=features.columns)
scaled_Cat_df['participant_id'] = Cat_df['participant_id'].values 


features_test = Cat_df_test.drop(columns=['participant_id'])

scaled_features_test = scaler.fit_transform(features_test)

scaled_Cat_df_test = pd.DataFrame(scaled_features_test, columns=features_test.columns)
scaled_Cat_df_test['participant_id'] = Cat_df_test['participant_id'].values 


merged_df = scaled_Q_df.merge(scaled_Cat_df, on='participant_id')
merged_df = merged_df.merge(conn_mat, on='participant_id')

numerical_cols = merged_df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = merged_df.select_dtypes(include=['object', 'category']).columns



merged_df_test = scaled_Q_df_test.merge(scaled_Cat_df_test, on='participant_id')
merged_df_test = merged_df_test.merge(conn_mat_test, on='participant_id')



all_df = merged_df.merge(sol, on='participant_id')


median_features = ['MRI_Track_Age_at_Scan', 'EHQ_EHQ_Total']
mode_features = ['PreInt_Demos_Fam_Child_Ethnicity', 'PreInt_Demos_Fam_Child_Race', 'MRI_Track_Scan_Location', 'Barratt_Barratt_P1_Edu', 'Barratt_Barratt_P1_Occ', 'Barratt_Barratt_P2_Edu', 'Barratt_Barratt_P2_Occ', 'ColorVision_CV_Score', 'APQ_P_APQ_P_CP', 'APQ_P_APQ_P_ID', 'APQ_P_APQ_P_INV', 'APQ_P_APQ_P_OPD', 'APQ_P_APQ_P_PM', 'APQ_P_APQ_P_PP', 'SDQ_SDQ_Conduct_Problems', 'SDQ_SDQ_Difficulties_Total', 'SDQ_SDQ_Emotional_Problems', 'SDQ_SDQ_Externalizing', 'SDQ_SDQ_Generating_Impact', 'SDQ_SDQ_Hyperactivity', 'SDQ_SDQ_Internalizing', 'SDQ_SDQ_Peer_Problems', 'SDQ_SDQ_Prosocial']

# Impute missing values in the training data
for col in median_features:
    median_val = all_df[col].median()
    all_df[col] = all_df[col].fillna(median_val)

for col in mode_features:
    mode_val = all_df[col].mode()[0]
    all_df[col] = all_df[col].fillna(mode_val)

# Impute missing values in the test data using values computed from the training set
for col in median_features:
    merged_df_test[col] = merged_df_test[col].fillna(median_val)

for col in mode_features:
    merged_df_test[col] = merged_df_test[col].fillna(mode_val)


X = all_df.drop(columns=['participant_id','ADHD_Outcome','Sex_F'])
y = all_df[['ADHD_Outcome', 'Sex_F']]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


xgb = XGBClassifier(tree_method="hist", objective="binary:logistic",
                    eval_metric="logloss", random_state=42,
                    n_estimators=400, learning_rate=0.05, max_depth=5,
                    subsample=0.8, colsample_bytree=0.8)
lgb = LGBMClassifier(n_estimators=500, learning_rate=0.03,
                     subsample=0.8, colsample_bytree=0.8,
                     objective="binary", random_state=42)
logreg = LogisticRegression(max_iter=200, random_state=42)
voter = VotingClassifier([("xgb",xgb),("lgb",lgb),("lr",logreg)],
                         voting="soft", n_jobs=-1, weights=[2,2,1])


multi_model = MultiOutputClassifier(voter)

multi_model.fit(X_train, y_train)

Y_pred = multi_model.predict(X_test)



print("=== Gender Classification Report ===")
print(classification_report(y_test['Sex_F'], Y_pred[:, 1]))

print("=== ADHD Classification Report ===")
print(classification_report(y_test['ADHD_Outcome'], Y_pred[:, 0]))


X_merged_df_test= merged_df_test.drop(columns=['participant_id'])
y_pred = multi_model.predict(X_merged_df_test)
print("=== Gender Classification Report ===")
print(classification_report(sub_df['Sex_F'], y_pred[:, 1]))

print("=== ADHD Classification Report ===")
print(classification_report(sub_df['ADHD_Outcome'], y_pred[:, 0]))



submission_df = pd.DataFrame({
    'participant_id': sub['participant_id'], 
    'ADHD_Outcome': y_pred[:, 0] ,
    'Sex_F': y_pred[:, 1] 
})

submission_df.to_csv('submission.csv', index=False)




