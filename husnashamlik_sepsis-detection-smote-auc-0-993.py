import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve ,auc,accuracy_score,f1_score


train_data=("/kaggle/input/phems-hackathon-early-sepsis-prediction/training_data")
test_data="/kaggle/input/phems-hackathon-early-sepsis-prediction/testing_data"


sepsis_label=pd.read_csv("/kaggle/input/phems-hackathon-early-sepsis-prediction/training_data/SepsisLabel_train.csv")
devices=pd.read_csv("/kaggle/input/phems-hackathon-early-sepsis-prediction/training_data/devices_train.csv")
drugs_exposure=pd.read_csv("/kaggle/input/phems-hackathon-early-sepsis-prediction/training_data/drugsexposure_train.csv")
measurement_lab=pd.read_csv("/kaggle/input/phems-hackathon-early-sepsis-prediction/training_data/measurement_lab_train.csv")
medications=pd.read_csv("/kaggle/input/phems-hackathon-early-sepsis-prediction/training_data/measurement_meds_train.csv")
observation=pd.read_csv("/kaggle/input/phems-hackathon-early-sepsis-prediction/training_data/observation_train.csv")
demographics=pd.read_csv("/kaggle/input/phems-hackathon-early-sepsis-prediction/training_data/person_demographics_episode_train.csv")
procedures=pd.read_csv("/kaggle/input/phems-hackathon-early-sepsis-prediction/training_data/proceduresoccurrences_train.csv")
measurement_observation=pd.read_csv("/kaggle/input/phems-hackathon-early-sepsis-prediction/training_data/measurement_observation_train.csv")


measurement_observation


#selecting necessary columns
d1=sepsis_label[['person_id','measurement_datetime','SepsisLabel']]
d2=devices[['person_id','device']]
d3=drugs_exposure[['person_id','drug_datetime_hourly','drug_concept_id','route_concept_id']]
d4=measurement_lab[['person_id','Sodium [Moles/volume] in Serum or Plasma','Albumin [Mass/volume] in Serum or Plasma',
        'Glucose [Moles/volume] in Serum or Plasma','White blood cell count', 'Creatinine [Mass/volume] in Blood',
         'Platelet count','Lactate [Moles/volume] in Blood','Hemoglobin [Moles/volume] in Blood','C reactive protein [Mass/volume] in Serum or Plasma']]
d5=medications[['person_id','Systolic blood pressure', 'Diastolic blood pressure',
       'Body temperature', 'Respiratory rate', 'Heart rate',
        'Measurement of oxygen saturation at periphery',
       'Oxygen/Gas total [Pure volume fraction] Inhaled gas']]
d6=observation[['visit_occurrence_id', 'person_id','observation_concept_id',
                 'observation_concept_name', 'valuefilled']]
d7=demographics[['person_id','birth_datetime', 'age_in_months', 'gender']]
d8=procedures[[ 'person_id','procedure']]
d9=measurement_observation[['person_id','Left pupil Diameter Auto', 'Right pupil Diameter Auto',
       'Glasgow coma scale','Right pupil Pupillary response','Left pupil Pupillary response']]



def merge_grouped_datasets(d1, d2, d3, d4, d5, d6,d7,d8,d9):
    d2_grouped = d2.groupby('person_id').first().reset_index()
    d3_grouped = d3.groupby('person_id').first().reset_index()
    d4_grouped = d4.groupby('person_id').mean().reset_index()
    d5_grouped = d5.groupby('person_id').mean().reset_index()
    d6_grouped = d6.groupby('person_id').first().reset_index()
    d7_grouped = d7.groupby('person_id').first().reset_index()
    d8_grouped = d8.groupby('person_id').first().reset_index()
    d9_grouped = d9.groupby('person_id').first().reset_index()
    
    merged_df = d1.copy()
    merged_df = merged_df.merge(d2_grouped, on='person_id', how='left')
    merged_df = merged_df.merge(d3_grouped, on='person_id', how='left')
    merged_df = merged_df.merge(d4_grouped, on='person_id', how='left')
    merged_df = merged_df.merge(d5_grouped, on='person_id', how='left')
    merged_df = merged_df.merge(d6_grouped, on='person_id', how='left')
    merged_df = merged_df.merge(d7_grouped, on='person_id', how='left')
    merged_df = merged_df.merge(d8_grouped, on='person_id', how='left')
    merged_df = merged_df.merge(d9_grouped, on='person_id', how='left')
    return merged_df
merged_df=merge_grouped_datasets(d1, d2, d3, d4, d5, d6,d7,d8,d9)
merged_df.shape


merged_df


merged_df.isnull().sum()


merged_df.columns


target_col='SepsisLabel'
num_col=merged_df.select_dtypes(include=['number']).columns
cat_col=merged_df.select_dtypes(include=['object']).columns
print("Target Columns: ",target_col)
print("\nNumrical Column: ",num_col.tolist())
print("\nCategorical Column: ",cat_col.tolist())


def preprocess_data(df, preprocessor=None, is_train=True):
    target_column = "SepsisLabel" 
    numerical_features = [
    'person_id',  'Sodium [Moles/volume] in Serum or Plasma', 'Albumin [Mass/volume] in Serum or Plasma', 
        'Glucose [Moles/volume] in Serum or Plasma', 'White blood cell count', 'Creatinine [Mass/volume] in Blood',
        'Platelet count', 'Lactate [Moles/volume] in Blood', 'Hemoglobin [Moles/volume] in Blood', 
        'C reactive protein [Mass/volume] in Serum or Plasma', 'Systolic blood pressure', 
        'Diastolic blood pressure', 'Body temperature', 'Respiratory rate', 'Heart rate', 
        'Measurement of oxygen saturation at periphery', 'Oxygen/Gas total [Pure volume fraction] Inhaled gas',
        'visit_occurrence_id', 'observation_concept_id', 'age_in_months', 'Left pupil Diameter Auto', 'Right pupil Diameter Auto', 'Glasgow coma scale']
    
    categorical_features = [ 'device', 'drug_concept_id', 'route_concept_id', 'observation_concept_name', 'valuefilled',
                             'gender', 'procedure', 'Right pupil Pupillary response',
                            'Left pupil Pupillary response']
    
    datetime_features = ['measurement_datetime', 'drug_datetime_hourly', 'birth_datetime']

 # âœ… Drop target variable only if it exists (to avoid KeyError)
    if is_train and target_column in df.columns:
        df = df.drop(columns=[target_column])


    # Function to extract features from datetime columns
    def extract_datetime_features(df, column):
        df[column] = pd.to_datetime(df[column], errors='coerce')
        df[column + '_year'] = df[column].dt.year
        df[column + '_month'] = df[column].dt.month
        df[column + '_day'] = df[column].dt.day
        df[column + '_hour'] = df[column].dt.hour
        df[column + '_weekday'] = df[column].dt.weekday
        return df.drop(columns=[column])

    # Apply DateTime feature extraction
    for dt_col in datetime_features:
        df = extract_datetime_features(df, dt_col)

    

    if is_train:
        # Define transformation pipelines
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer([
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ], remainder='passthrough')

        # Fit and transform training data
        X_processed = preprocessor.fit_transform(df)
        return X_processed, preprocessor  # Return fitted preprocessor for later use
    else:
        # Transform test data using the fitted preprocessor
        X_processed = preprocessor.transform(df)
        return X_processed

# Example Usage:
# Train Data
train_processed, preprocessor = preprocess_data(merged_df, is_train=True)



train_processed[0]


sepsis_test_label=pd.read_csv("/kaggle/input/phems-hackathon-early-sepsis-prediction/testing_data/SepsisLabel_test.csv")
devices_test=pd.read_csv("/kaggle/input/phems-hackathon-early-sepsis-prediction/testing_data/devices_test.csv")
drug_exposure_test=pd.read_csv("/kaggle/input/phems-hackathon-early-sepsis-prediction/testing_data/drugsexposure_test.csv")
measurement_lab_test=pd.read_csv("/kaggle/input/phems-hackathon-early-sepsis-prediction/testing_data/measurement_lab_test.csv")
medications_test=pd.read_csv("/kaggle/input/phems-hackathon-early-sepsis-prediction/testing_data/measurement_meds_test.csv")
measurement_observation_test=pd.read_csv("/kaggle/input/phems-hackathon-early-sepsis-prediction/testing_data/measurement_observation_test.csv")
observation_test=pd.read_csv("/kaggle/input/phems-hackathon-early-sepsis-prediction/testing_data/observation_test.csv")
demographics_test=pd.read_csv("/kaggle/input/phems-hackathon-early-sepsis-prediction/testing_data/person_demographics_episode_test.csv")
procedure_test=pd.read_csv("/kaggle/input/phems-hackathon-early-sepsis-prediction/testing_data/proceduresoccurrences_test.csv")



d10=sepsis_test_label
d11=devices_test[['person_id','device']]
d12=drug_exposure_test[['person_id','drug_datetime_hourly','drug_concept_id','route_concept_id']]
d13=measurement_lab_test[['person_id','Sodium [Moles/volume] in Serum or Plasma','Albumin [Mass/volume] in Serum or Plasma',
        'Glucose [Moles/volume] in Serum or Plasma','White blood cell count', 'Creatinine [Mass/volume] in Blood',
         'Platelet count','Lactate [Moles/volume] in Blood','Hemoglobin [Moles/volume] in Blood','C reactive protein [Mass/volume] in Serum or Plasma']]

d14=medications_test[['person_id','Systolic blood pressure', 'Diastolic blood pressure',
       'Body temperature', 'Respiratory rate', 'Heart rate',
        'Measurement of oxygen saturation at periphery',
       'Oxygen/Gas total [Pure volume fraction] Inhaled gas']]
d15=measurement_observation_test[['person_id','Left pupil Diameter Auto', 'Right pupil Diameter Auto',
       'Glasgow coma scale','Right pupil Pupillary response','Left pupil Pupillary response']]
d16=observation_test[['visit_occurrence_id', 'person_id','observation_concept_id',
                 'observation_concept_name', 'valuefilled']]
d17=demographics_test[['person_id','birth_datetime', 'age_in_months', 'gender']]
d18=procedure_test[['person_id','procedure']]


test_merge=merge_grouped_datasets(d10, d11, d12, d13, d14, d15,d16,d17,d18)


test_merge.shape


test_processed = preprocess_data(test_merge, preprocessor=preprocessor, is_train=False)


print(test_processed.shape)
print(train_processed.shape)


sepsis_count=merged_df['SepsisLabel'].value_counts()
sepsis_count


import matplotlib.pyplot as plt

# Ensure only numeric values are used
sepsis_count = merged_df['SepsisLabel'].value_counts()

plt.figure(figsize=(10, 6))
plt.pie(sepsis_count, labels=sepsis_count.index.astype(str), autopct='%1.1f%%', colors=['skyblue', 'salmon'])
plt.title('Sepsis Label Distribution')
plt.show()



train_processed


sepsis_label


y=sepsis_label['SepsisLabel']
X_train,X_test,y_train,y_test=train_test_split(train_processed,y,test_size=0.20,random_state=42)


# Step 1: Handle missing values
imputer = SimpleImputer(strategy='mean')  # You can use 'median' or 'most_frequent' as well
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)




# Step 2: Apply SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_imputed, y_train)

# Step 3: Verify the new class distribution
print("Class distribution after SMOTE:\n", y_train_resampled.value_counts())


# Step 4: Train XGBoost Classifier
xgb_model = XGBClassifier(
    objective='binary:logistic', 
    eval_metric='aucpr',  # Precision-Recall AUC metric as per competition requirement
    random_state=42
)
xgb_model.fit(X_train_resampled, y_train_resampled)

# Step 5: Model Evaluation
y_scores = xgb_model.predict_proba(X_test_imputed)[:, 1]  # Get probability for class 1
precision, recall, _ = precision_recall_curve(y_test, y_scores)
pr_auc = auc(recall, precision)

print(f"Precision-Recall AUC: {pr_auc:.4f}")


rf_model = RandomForestClassifier(
      n_estimators=100,
    random_state=42
)
rf_model.fit(X_train_resampled, y_train_resampled)

# Step 5: Model Evaluation
y_scores = rf_model.predict_proba(X_test_imputed)[:, 1]  # Get probability for class 1
precision, recall, _ = precision_recall_curve(y_test, y_scores)
pr_auc = auc(recall, precision)

print(f"Precision-Recall AUC: {pr_auc:.4f}")


# Step 5: Model Evaluation
y_pred=rf_model.predict(X_test_imputed)# Get probability for class 1
accuracy = accuracy_score(y_test, y_pred)
f1_scores=f1_score(y_test,y_pred)
print(f"Accuracy Score: {pr_auc:.4f}",'\n')
print(f"F1-Score: {pr_auc:.4f}")


import pandas as pd
from sklearn.impute import SimpleImputer

# Step 1: Generate person_id_datetime in test_df
test_merge['person_id_datetime'] = (
    test_merge['person_id'].astype(str) + '_' + 
    test_merge['measurement_datetime'].astype(str)
)

# Step 2: Handle missing values using SimpleImputer (fill with median)
imputer = SimpleImputer(strategy='median')
test_processed_filled = imputer.fit_transform(test_processed)  # Apply imputer to test_processed

# Step 3: Predict probabilities using the trained model
sepsis_probabilities = rf_model.predict_proba(test_processed_filled)[:, 1]  # Probability for SepsisLabel = 1

# Step 4: Create submission DataFrame
submission = pd.DataFrame({
    'person_id_datetime': test_merge['person_id_datetime'],
    'SepsisLabel': sepsis_probabilities
})

# Step 5: Save the submission file
submission.to_csv("submission.csv", index=False)

print("âœ… Submission file saved successfully!")



submission




