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


# Additional packages
!pip install -q umap-learn
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import umap
print("Setup Complete")


# Read data CSVs
test_matrices_data=pd.read_csv("/kaggle/input/widsdatathon2025/TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv")
train_matrices_data=pd.read_csv("/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv")

train_categorical_data=pd.read_excel("/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_CATEGORICAL_METADATA_new.xlsx")
test_categorical_data=pd.read_excel("/kaggle/input/widsdatathon2025/TEST/TEST_CATEGORICAL.xlsx")

train_quantitative_data=pd.read_excel("/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_QUANTITATIVE_METADATA_new.xlsx")
test_quantitative_data=pd.read_excel("/kaggle/input/widsdatathon2025/TEST/TEST_QUANTITATIVE_METADATA.xlsx")

train_solutions=pd.read_excel('/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAINING_SOLUTIONS.xlsx')
data_dictionary=pd.read_excel('/kaggle/input/widsdatathon2025/Data Dictionary.xlsx')

print('data reading complete')


data_dictionary


#Read Sample Submission data and observe format necessary for submission
sample_sub=pd.read_excel(f"/kaggle/input/widsdatathon2025/SAMPLE_SUBMISSION.xlsx")

sample_sub.head()


# Take a look at test data
test_matrices_data.head()


test_categorical_data.head()


test_quantitative_data.head()


# Take a look at train data
train_matrices_data.head()


train_categorical_data.head()


train_quantitative_data.head()


train_solutions.head()


test_matrices_data.info()


test_categorical_data.info()


test_quantitative_data.info()


train_matrices_data.info()


train_categorical_data.info()


train_quantitative_data.info()


print(train_matrices_data.isnull().sum())


print(train_categorical_data.isnull().sum())


print(train_quantitative_data.isnull().sum())


print(test_quantitative_data.isnull().sum())


#  Impute categorical data
cols_to_fill = [
    'PreInt_Demos_Fam_Child_Ethnicity',    
    'PreInt_Demos_Fam_Child_Race',
    'MRI_Track_Scan_Location',
    'Barratt_Barratt_P1_Edu',
    'Barratt_Barratt_P1_Occ',
    'Barratt_Barratt_P2_Edu',
    'Barratt_Barratt_P2_Occ'
]

for col in cols_to_fill:
    train_categorical_data[col].fillna(train_categorical_data[col].median(), inplace=True)

for col in cols_to_fill:
    test_categorical_data[col].fillna(test_categorical_data[col].median(), inplace=True)


#check imputed data
print(train_categorical_data.isnull().sum())
print(test_categorical_data.isnull().sum())


cols_to_fill_quant = [
    'EHQ_EHQ_Total',
    'ColorVision_CV_Score',           
    'APQ_P_APQ_P_CP',                
    'APQ_P_APQ_P_ID',                
    'APQ_P_APQ_P_INV',               
    'APQ_P_APQ_P_OPD',               
    'APQ_P_APQ_P_PM',                
    'APQ_P_APQ_P_PP',                
    'SDQ_SDQ_Conduct_Problems',      
    'SDQ_SDQ_Difficulties_Total',    
    'SDQ_SDQ_Emotional_Problems',    
    'SDQ_SDQ_Externalizing',         
    'SDQ_SDQ_Generating_Impact',     
    'SDQ_SDQ_Hyperactivity',         
    'SDQ_SDQ_Internalizing',         
    'SDQ_SDQ_Peer_Problems',         
    'SDQ_SDQ_Prosocial',
    'MRI_Track_Age_at_Scan'        
]

for col in cols_to_fill_quant:
    train_quantitative_data[col].fillna(train_quantitative_data[col].median(), inplace=True)

for col in cols_to_fill_quant:
    test_quantitative_data[col].fillna(test_quantitative_data[col].median(), inplace=True)


#check imputed data
print(train_quantitative_data.isnull().sum())
print(test_quantitative_data.isnull().sum())


# ADHD distribution
train_solutions['ADHD_Outcome'].value_counts()


train_solutions['ADHD_Outcome'].value_counts().plot(kind='bar', color='blue')
plt.title('ADHD Outcome')
plt.xlabel('Outcome (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()


#Gender distribution
train_solutions['Sex_F'].value_counts().plot(kind='bar', color='blue')
plt.title('Gender Distribution')
plt.xlabel('Gender (0 = Male, 1 = Female)')
plt.ylabel('Count')
plt.show()


# Distribution of the SDQ_SDQ_Emotional_Problems variable
plt.figure(figsize=(8, 6))
sns.histplot(train_quantitative_data['SDQ_SDQ_Emotional_Problems'], kde=True, color='skyblue')
plt.title('Distribution of SDQ_SDQ_Emotional_Problems')
plt.xlabel('SDQ_SDQ_Emotional_Problems')
plt.ylabel('Frequency')
plt.show()


# Correlation with ADHD outcome
# copying 'ADHD_Outcome' from train_solutions into train_quantitative_metadata
train_quantitative_data_copy = train_quantitative_data.copy()
train_quantitative_data_copy['ADHD_Outcome'] = train_solutions['ADHD_Outcome']

plt.figure(figsize=(8, 6))
sns.boxplot(x='ADHD_Outcome', y='SDQ_SDQ_Emotional_Problems', data=train_quantitative_data_copy)
plt.title('SDQ_SDQ_Emotional_Problems vs ADHD Outcome')
plt.xlabel('ADHD Outcome')
plt.ylabel('SDQ_SDQ_Emotional_Problems')
plt.show()


sns.countplot(data=train_categorical_data, x='Barratt_Barratt_P1_Edu', hue=train_solutions['ADHD_Outcome'])
plt.title('ADHD Prevalence by Parent 1 Education')
plt.show()


sns.countplot(data=train_categorical_data, x='Barratt_Barratt_P2_Edu', hue=train_solutions['ADHD_Outcome'])
plt.title('ADHD Prevalence by Parent 2 Education')
plt.show()


sns.countplot(data=train_categorical_data, x='Barratt_Barratt_P1_Occ', hue=train_solutions['ADHD_Outcome'])
plt.title('ADHD Prevalence by Parent 1 Occupation')
plt.show()


sns.countplot(data=train_categorical_data, x='Barratt_Barratt_P2_Occ', hue=train_solutions['ADHD_Outcome'])
plt.title('ADHD Prevalence by Parent 2 Occupation')
plt.show()


sns.countplot(data=train_categorical_data, x='PreInt_Demos_Fam_Child_Ethnicity', hue=train_solutions['ADHD_Outcome'])
plt.title('ADHD Prevalence by Ethnicity')
plt.show()



sns.countplot(data=train_categorical_data, x='PreInt_Demos_Fam_Child_Race', hue=train_solutions['ADHD_Outcome'])
plt.title('ADHD Prevalence by Race')
plt.show()



train_quantitative_data['MRI_Track_Age_at_Scan'].hist(figsize=(12, 10), bins=20)
plt.suptitle("MRI_Track_Age_at_Scan Distributions")
plt.xlabel('MRI_Track_Age_at_Scan')
plt.ylabel('Frequency Count')
plt.show()


# ADHD distribution
train_solutions['ADHD_Outcome'].value_counts()


missing_test_matrices = test_matrices_data.isnull().sum()
missing_test_matrices


#Get rid of trailing spaces for participant_id to ensure join works
for df in (test_categorical_data, test_quantitative_data, train_categorical_data, train_quantitative_data, test_matrices_data, train_matrices_data):
    # Strip the column(s) you're planning to join with
    df['participant_id'] = df['participant_id'].str.strip()


train_categorical_data.head()


train_quantitative_data.head()


# Join test categorical and quantitative on participant_id
test_cat_quant = pd.merge(test_categorical_data, test_quantitative_data, on='participant_id', how='inner')
test_cat_quant.head()


# Join train categorical and quantitative on participant_id
train_cat_quant = pd.merge(train_categorical_data, train_quantitative_data, on='participant_id', how='inner')
train_cat_quant.head()




import numpy as np

from sklearn.decomposition import PCA


# Scale "train_matrices_data"
# Where each row represents a different brain scan and each column represents a voxel

from sklearn.preprocessing import StandardScaler

# Separating the identifier column
identifier = train_matrices_data.iloc[:, 0]  # Assuming the first column is the identifier
features = train_matrices_data.iloc[:, 1:]  # All other columns

# Scaling the feature columns
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Creating a new DataFrame with the scaled values
scaled_train_matrices_data = pd.DataFrame(scaled_features, columns=features.columns)

# Adding back the identifier column
scaled_train_matrices_data.insert(0, train_matrices_data.columns[0], identifier)


scaled_train_matrices_data.head()


# Separate identifiers
ids = scaled_train_matrices_data['participant_id']

# Select feature columns for PCA
feature_cols = [col for col in scaled_train_matrices_data.columns if col not in ['participant_id']]  # adjust if needed
X_features = scaled_train_matrices_data[feature_cols]

# Apply PCA
pca = PCA(n_components=8)  # or whatever number of components you want
X_pca = pca.fit_transform(X_features)

# Convert PCA result into a DataFrame
pca_cols = [f'PC{i+1}' for i in range(X_pca.shape[1])]
X_pca_df = pd.DataFrame(X_pca, columns=pca_cols)

# Reattach the identifiers
X_pca_df['participant_id'] = ids.values

# Now X_pca_df is ready for modeling
print(X_pca_df.head())


#Join dataset with PCA and targets
train_ready = pd.merge(train_cat_quant, X_pca_df, on='participant_id', how='inner')
train_ready = pd.merge(train_ready, train_solutions, on='participant_id', how='inner')
train_ready.head()


# Scale "test_matrices_data"
# Where each row represents a different brain scan and each column represents a voxel

from sklearn.preprocessing import StandardScaler

# Separating the identifier column
identifier = test_matrices_data.iloc[:, 0]  # Assuming the first column is the identifier
features = test_matrices_data.iloc[:, 1:]  # All other columns

# Scaling the feature columns
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Creating a new DataFrame with the scaled values
scaled_test_matrices_data = pd.DataFrame(scaled_features, columns=features.columns)

# Adding back the identifier column
scaled_test_matrices_data.insert(0, test_matrices_data.columns[0], identifier)


scaled_test_matrices_data.head()


# Separate identifiers
test_ids = scaled_test_matrices_data['participant_id']

# Select feature columns for PCA
test_feature_cols = [col for col in scaled_test_matrices_data.columns if col not in ['participant_id']]  # adjust if needed
Y_features = scaled_test_matrices_data[test_feature_cols]

# Apply PCA
pca = PCA(n_components=8)  # or whatever number of components you want
Y_pca = pca.fit_transform(Y_features)

# Convert PCA result into a DataFrame
test_pca_cols = [f'PC{i+1}' for i in range(Y_pca.shape[1])]
Y_pca_df = pd.DataFrame(Y_pca, columns=test_pca_cols)

# Reattach the identifiers
Y_pca_df['participant_id'] = test_ids.values

# Now X_pca_df is ready for modeling
print(Y_pca_df.head())


#Join dataset with PCA and targets
test_ready = pd.merge(test_cat_quant, Y_pca_df, on='participant_id', how='inner')

test_ready.head()


from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy


X = train_ready.drop(columns=["participant_id", "Sex_F", "ADHD_Outcome"])
y = train_ready[["ADHD_Outcome","Sex_F"]]


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Extract individual targets
y_gender_train = y_train["Sex_F"].values
y_adhd_train = y_train["ADHD_Outcome"].values
y_gender_test = y_test["Sex_F"].values
y_adhd_test = y_test["ADHD_Outcome"].values


# Feature scaling (important for NN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------
# 2. Define the Model
# -------------------------
input_layer = Input(shape=(X_train_scaled.shape[1],))
x = Dense(256, activation='relu')(input_layer)
x = Dense(128, activation='relu')(x)

# Output heads
gender_output = Dense(1, activation='sigmoid', name='Sex_F')(x)
adhd_output = Dense(1, activation='sigmoid', name='ADHD_Outcome')(x)

model = Model(inputs=input_layer, outputs=[gender_output, adhd_output])

model.compile(
    optimizer=Adam(),
    loss={"Sex_F": BinaryCrossentropy(), "ADHD_Outcome": BinaryCrossentropy()},
    metrics={"Sex_F": "accuracy", "ADHD_Outcome": "accuracy"}
)



history = model.fit(
    X_train_scaled,
    {"Sex_F": y_gender_train, "ADHD_Outcome": y_adhd_train},
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    verbose=1
)


eval_results = model.evaluate(X_test_scaled, {"Sex_F": y_gender_test, "ADHD_Outcome": y_adhd_test})
print("\nTest Results:")
for name, val in zip(model.metrics_names, eval_results):
    print(f"{name}: {val:.4f}")


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['Sex_F_loss'], label='Gender Loss')
plt.plot(history.history['val_Sex_F_loss'], label='Val Gender Loss')
plt.title('Gender Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['ADHD_Outcome_loss'], label='ADHD Loss')
plt.plot(history.history['val_ADHD_Outcome_loss'], label='Val ADHD Loss')
plt.title('ADHD Loss')
plt.legend()

plt.tight_layout()
plt.show()


participant_ids = test_ready["participant_id"].values
X = test_ready.drop(columns=["participant_id"])

# Step 2: Preprocess and predict
X_scaled = scaler.transform(X)  # Assuming you've already fit the scaler on train data
preds = model.predict(X_scaled)

# Step 3: Process predictions (e.g., binary outputs from sigmoid)
pred_gender = (preds[0] > 0.5).astype(int).flatten()
pred_adhd = (preds[1] > 0.5).astype(int).flatten()

# Step 4: Reattach participant ID into a new DataFrame
test_predictions = pd.DataFrame({
    "participant_id": participant_ids,
    "ADHD_Outcome": pred_adhd,
    "Sex_F": pred_gender
})

# View results
print(test_predictions.head())

      # Save results to CSV
test_predictions.to_csv("submission.csv", index=False)

print("Predictions saved to submission.csv")

