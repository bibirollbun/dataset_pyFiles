# 1. Imports
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.multioutput import MultiOutputClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


def load_data(file_path, file_type):
  '''Load data from a file.

    Args:
        file_path (str): Path to the file.
        file_type (str): Type of the file ('csv' or 'excel').

    Returns:
        pd.DataFrame: Loaded data.

    Raises:
        ValueError: If the file type is not supported.'''   
  if file_type == 'csv':
    data = pd.read_csv(file_path)
    return data
  elif file_type == 'excel':
    data = pd.read_excel(file_path)
    return data
  else:
    raise ValueError("Unsupported file type. Use 'csv' or 'excel'.")


folder_path = "/kaggle/input/widsdatathon2025"
train_fmri_data = load_data(f"{folder_path}/TRAIN_NEW/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv", "csv")
train_categorical_data = load_data(f"{folder_path}/TRAIN_NEW/TRAIN_CATEGORICAL_METADATA_new.xlsx", "excel")
train_quantitative_data = load_data(f"{folder_path}/TRAIN_NEW/TRAIN_QUANTITATIVE_METADATA_new.xlsx", "excel")
test_fmri_data = load_data(f"{folder_path}/TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv", "csv")
test_categorical_data = load_data(f"{folder_path}/TEST/TEST_CATEGORICAL.xlsx", "excel")
test_quantitative_data = load_data(f"{folder_path}/TEST/TEST_QUANTITATIVE_METADATA.xlsx", "excel")
validation_data = load_data(f"{folder_path}/TRAIN_OLD/TRAINING_SOLUTIONS.xlsx", "excel")
sample_submission = load_data(f"{folder_path}/SAMPLE_SUBMISSION.xlsx", "excel")
data_dictionary = load_data(f"{folder_path}/Data Dictionary.xlsx", "excel")
print("Data load is completed!")


# merge data
df_combined = train_fmri_data.merge(train_quantitative_data, on='participant_id', how='inner')
print(df_combined.shape)
df_combined = df_combined.merge(train_categorical_data, on='participant_id', how='inner')
df_combined.shape


# drop participant_id 
df_combined = df_combined.drop(columns=['participant_id'])

X = df_combined
y = validation_data[['ADHD_Outcome', 'Sex_F']]


categorical_features = [
    'Basic_Demos_Enroll_Year', 'Basic_Demos_Study_Site',
    'PreInt_Demos_Fam_Child_Ethnicity', 'PreInt_Demos_Fam_Child_Race',
    'MRI_Track_Scan_Location',
    'Barratt_Barratt_P1_Occ',
    'Barratt_Barratt_P2_Occ'
]

numeric_features = [col for col in df_combined.columns if col not in categorical_features]

numeric_preprocessor = Pipeline(steps=[
    ("imputation_mean", SimpleImputer(strategy='mean')),
    ("scaler", StandardScaler()),
])

categorical_preprocessor = Pipeline(steps=[
    ("imputation_constant",  SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("categorical", categorical_preprocessor, categorical_features),
    ("numerical", numeric_preprocessor, numeric_features)
])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# Input Layer
input_layer = Input(shape=(X_train_processed.shape[1],))

# Shared Layers
x = Dense(512, activation='relu')(input_layer)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)

x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)

# Two Output Heads
adhd_output = Dense(1, activation='sigmoid', name='adhd_output')(x)
sex_output = Dense(1, activation='sigmoid', name='sex_output')(x)

# Build Model
model = Model(inputs=input_layer, outputs=[adhd_output, sex_output])


# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss={'adhd_output': ['binary_crossentropy'], 'sex_output': ['binary_crossentropy']},
    metrics={'adhd_output': ['accuracy'], 'sex_output': ['accuracy']}
)

# Callbacks for Better Training
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    verbose=1
)


model.summary()


# Train
history = model.fit(
    X_train_processed,
    {'adhd_output': y_train['ADHD_Outcome'], 'sex_output': y_train['Sex_F']},
    validation_split=0.2,
    epochs=100,
    batch_size=64,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)


# Predict
adhd_pred, sex_pred = model.predict(X_test_processed)


# Convert probabilities to 0/1
adhd_pred_labels = (adhd_pred > 0.5).astype(int)
sex_pred_labels = (sex_pred > 0.5).astype(int)


# F1 Score
from sklearn.metrics import f1_score

f1_adhd = f1_score(y_test['ADHD_Outcome'], adhd_pred_labels, average='weighted')
f1_sex = f1_score(y_test['Sex_F'], sex_pred_labels, average='weighted')

print(f"F1 Score (ADHD Outcome): {f1_adhd:.4f}")
print(f"F1 Score (Sex_F): {f1_sex:.4f}")


# merge data
test_df_combined = test_fmri_data.merge(test_quantitative_data, on='participant_id', how='inner')
print(test_df_combined.shape)
test_df_combined = test_df_combined.merge(test_categorical_data, on='participant_id', how='inner')

participant_ids = test_df_combined['participant_id']

test_df_combined = test_df_combined[df_combined.columns]
test_df_combined.shape

test_df_combined.convert_dtypes()


X_test_new_processed = preprocessor.transform(test_df_combined)


adhd_pred, sex_pred = model.predict(X_test_new_processed)
adhd_pred_labels = (adhd_pred > 0.5).astype(int)
sex_pred_labels = (sex_pred > 0.5).astype(int)


results_df = pd.DataFrame({
    'participant_id': participant_ids,
    'ADHD_Outcome': adhd_pred_labels[:,0],
    'Sex_F': sex_pred_labels[:,0]
})

results_df


results_df.to_csv('submission.csv', index=False)

