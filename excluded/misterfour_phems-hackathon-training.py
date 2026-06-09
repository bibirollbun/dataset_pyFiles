%%capture
!pip install imblearn


import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import joblib
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Conv1D, GlobalAveragePooling1D,
    BatchNormalization, Activation, Add, LSTM, MultiHeadAttention,
    LayerNormalization, Bidirectional
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import (
    precision_recall_curve, auc, accuracy_score, f1_score,
    roc_auc_score, precision_score, recall_score
)
pd.set_option('display.max_rows', None)
pd.reset_option('display.max_rows')


# Load test data
train_sepsislabel = pd.read_csv('/kaggle/input/phems-hackathon-early-sepsis-prediction/training_data/SepsisLabel_train.csv')
train_sepsislabel['measurement_datetime'] = pd.to_datetime(train_sepsislabel['measurement_datetime'])

# Convert 'measurement_datetime' to datetime
train_sepsislabel['measurement_datetime'] = pd.to_datetime(train_sepsislabel['measurement_datetime'])

# Calculate duration for each person_id
train_sepsislabel['duration'] = train_sepsislabel.groupby('person_id')['measurement_datetime'].transform(lambda x: (x.max() - x.min()).total_seconds()/3600)

train_devices = pd.read_csv('/kaggle/input/phems-hackathon-early-sepsis-prediction/training_data/devices_train.csv')

train_devices_grouped = train_devices.groupby('person_id')['device'].agg(lambda x: ', '.join(x.unique())).reset_index()
train_devices_grouped = pd.DataFrame(train_devices_grouped)

# Drop specified columns
train_devices_hr = train_devices.copy()
train_devices_hr = train_devices_hr.drop(columns=['visit_occurrence_id', 'device'])

# Convert 'device_datetime_hourly' to datetime objects
train_devices_hr['device_datetime_hourly'] = pd.to_datetime(train_devices_hr['device_datetime_hourly'])

# Extract the hour from the datetime column
train_devices_hr['device_hour'] = train_devices_hr['device_datetime_hourly'].dt.hour

# Group by 'person_id' and calculate the mean hour
mean_hours = train_devices_hr.groupby('person_id')['device_hour'].mean().reset_index()
mean_hours['device_mean_hour'] = mean_hours['device_hour']

# Merge the mean hour back into the original DataFrame
train_devices_hr = pd.merge(train_devices_hr, mean_hours[['person_id', 'device_mean_hour']], on='person_id', how='left')

# Remove duplicate 'person_id' rows (keeping the first occurrence)
train_devices_hr = train_devices_hr.drop_duplicates(subset='person_id')
train_devices_hr = train_devices_hr.drop(columns='device_datetime_hourly')

train_devices_merge = pd.merge(train_devices_grouped, train_devices_hr, on='person_id', how='left')

merged_train = pd.merge(train_sepsislabel, train_devices_merge, on='person_id', how='left')

train_lab=pd.read_csv('/kaggle/input/phems-hackathon-early-sepsis-prediction/training_data/measurement_lab_train.csv')

train_lab = train_lab.groupby(['person_id','measurement_datetime'])[['Base excess in Venous blood by calculation',
       'Base excess in Arterial blood by calculation',
       'Phosphate [Moles/volume] in Serum or Plasma',
       'Potassium [Moles/volume] in Blood',
       'Bilirubin.total [Moles/volume] in Serum or Plasma',
       'Neutrophil Ab [Units/volume] in Serum',
       'Bicarbonate [Moles/volume] in Arterial blood',
       'Hematocrit [Volume Fraction] of Blood',
       'Glucose [Moles/volume] in Serum or Plasma',
       'Calcium [Moles/volume] in Serum or Plasma',
       'Chloride [Moles/volume] in Blood',
       'Sodium [Moles/volume] in Serum or Plasma',
       'C reactive protein [Mass/volume] in Serum or Plasma',
       'Carbon dioxide [Partial pressure] in Venous blood',
       'Oxygen [Partial pressure] in Venous blood',
       'Albumin [Mass/volume] in Serum or Plasma',
       'Bicarbonate [Moles/volume] in Venous blood',
       'Oxygen [Partial pressure] in Arterial blood',
       'Carbon dioxide [Partial pressure] in Arterial blood',
       'Interleukin 6 [Mass/volume] in Body fluid',
       'Magnesium [Moles/volume] in Blood', 'Prothrombin time (PT)',
       'Procalcitonin [Mass/volume] in Serum or Plasma',
       'Lactate [Moles/volume] in Blood', 'Creatinine [Mass/volume] in Blood',
       'Fibrinogen measurement', 'Bilirubin measurement',
       'Partial thromboplastin time', ' activated', 'Total white blood count',
       'Platelet count', 'White blood cell count', 'Blood venous pH',
       'D-dimer level', 'Blood arterial pH',
       'Hemoglobin [Moles/volume] in Blood', 'Ionised calcium measurement']].sum().reset_index()

# Ensure 'measurement_datetime' is converted to datetime64[ns] in all DataFrames
def ensure_datetime(df, datetime_column):
    df[datetime_column] = pd.to_datetime(df[datetime_column], errors='coerce')
    return df

# Convert 'measurement_datetime' to datetime in all relevant DataFrames
train_sepsislabel = ensure_datetime(train_sepsislabel, 'measurement_datetime')
train_lab = ensure_datetime(train_lab, 'measurement_datetime')

# Merge the DataFrames
merged_train = pd.merge(merged_train, train_lab, on=['person_id', 'measurement_datetime'], how='left')

train_meds=pd.read_csv('/kaggle/input/phems-hackathon-early-sepsis-prediction/training_data/measurement_meds_train.csv')

train_meds = train_meds.groupby(['person_id','measurement_datetime'])[['Systolic blood pressure', 'Diastolic blood pressure',
       'Body temperature', 'Respiratory rate', 'Heart rate',
       'Measurement of oxygen saturation at periphery',
       'Oxygen/Gas total [Pure volume fraction] Inhaled gas']].sum().reset_index()

# Ensure 'measurement_datetime' is converted to datetime64[ns] in all DataFrames
def ensure_datetime(df, datetime_column):
    df[datetime_column] = pd.to_datetime(df[datetime_column], errors='coerce')
    return df

# Convert 'measurement_datetime' to datetime in all relevant DataFrames
merged_train = ensure_datetime(merged_train, 'measurement_datetime')
train_meds = ensure_datetime(train_meds, 'measurement_datetime')

# Merge the DataFrames
merged_train = pd.merge(merged_train, train_meds, on=['person_id', 'measurement_datetime'], how='left')

train_obs=pd.read_csv('/kaggle/input/phems-hackathon-early-sepsis-prediction/training_data/measurement_observation_train.csv')

train_obs = train_obs.groupby(['person_id','measurement_datetime'])[['Left pupil Diameter Auto', 'Right pupil Diameter Auto',
       'Glasgow coma scale', 'Capillary refill [Time]', 'Pulse',
       'Arterial pulse pressure', 'Right pupil Pupillary response',
       'Left pupil Pupillary response']].sum().reset_index()

train_obs

# Ensure 'measurement_datetime' is converted to datetime64[ns] in all DataFrames
def ensure_datetime(df, datetime_column):
    df[datetime_column] = pd.to_datetime(df[datetime_column], errors='coerce')
    return df

# Convert 'measurement_datetime' to datetime in all relevant DataFrames
merged_train = ensure_datetime(merged_train, 'measurement_datetime')
train_obs = ensure_datetime(train_obs, 'measurement_datetime')

# Merge the DataFrames
merged_train = pd.merge(merged_train, train_obs, on=['person_id', 'measurement_datetime'], how='left')

# Load the data
train_procedure = pd.read_csv('/kaggle/input/phems-hackathon-early-sepsis-prediction/training_data/proceduresoccurrences_train.csv')

# Convert 'procedure_datetime_hourly' to datetime
train_procedure['procedure_datetime_hourly'] = pd.to_datetime(train_procedure['procedure_datetime_hourly'])

# Calculate duration in hours for each person_id
train_procedure['procedure_duration'] = train_procedure.groupby('person_id')['procedure_datetime_hourly'].transform(lambda x: (x.max() - x.min()).total_seconds() / 3600)
train_procedure_A = train_procedure[['person_id','procedure_duration']]
train_procedure_A = train_procedure_A.drop_duplicates(subset=['person_id'])

# Group by 'person_id' and aggregate 'device' by joining values with a separator (e.g., ', ')
train_procedure_grouped = train_procedure.groupby('person_id')[['procedure']].agg(lambda x: ', '.join(x.unique())).reset_index()

# Merge the DataFrames
merged_train = pd.merge(merged_train, train_procedure_grouped, on='person_id', how='left')

# Merge the DataFrames
merged_train = pd.merge(merged_train, train_procedure_A, on='person_id', how='left')

# Sort value by measurement_datetime column
merged_train = merged_train.sort_values(by='measurement_datetime')

train_drug=pd.read_csv('/kaggle/input/phems-hackathon-early-sepsis-prediction/training_data/drugsexposure_train.csv')

train_drug=train_drug[['person_id','drug_datetime_hourly','drug_concept_id','route_concept_id']]

train_drug['route_concept_id'] = train_drug['route_concept_id'].astype(str)

# Convert 'drug_datetime_hourly' to datetime
train_drug['drug_datetime_hourly'] = pd.to_datetime(train_drug['drug_datetime_hourly'])

# Calculate duration for each person_id
train_drug['drug_duration'] = train_drug.groupby('person_id')['drug_datetime_hourly'].transform(lambda x: (x.max() - x.min()).total_seconds()/3600)
train_drug_A = train_drug[['person_id','drug_duration']]
train_drug_A = train_drug_A.drop_duplicates(subset=['person_id'])

# Group by 'person_id' and aggregate 'device' by joining values with a separator (e.g., ', ')
train_drug_grouped = train_drug.groupby(['person_id'])[['drug_concept_id','route_concept_id']].agg(lambda x: ', '.join(x.unique())).reset_index()
train_drug_grouped = pd.DataFrame(train_drug_grouped)

# Merge the DataFrames on the 'person_id' column
merged_train = pd.merge(merged_train, train_drug_grouped, on='person_id', how='left')

# Merge the DataFrames on the 'person_id' column
merged_train = pd.merge(merged_train, train_drug_A, on='person_id', how='left')

train_observation=pd.read_csv('/kaggle/input/phems-hackathon-early-sepsis-prediction/training_data/observation_train.csv')

# Convert 'drug_datetime_hourly' to datetime
train_observation['observation_datetime'] = pd.to_datetime(train_observation['observation_datetime'])

# Calculate duration for each person_id
train_observation['observation_duration'] = train_observation.groupby('person_id')['observation_datetime'].transform(lambda x: (x.max() - x.min()).total_seconds()/3600)
train_observation_A = train_observation[['person_id','observation_duration']]
train_observation_A = train_observation_A.drop_duplicates(subset=['person_id'])
train_observation = train_observation.groupby('person_id')[['observation_concept_name','valuefilled']].agg(lambda x: ', '.join(x.unique())).reset_index()
merged_train = pd.merge(merged_train, train_observation_A, on='person_id', how='left')
merged_train = pd.merge(merged_train, train_observation, on='person_id', how='left')

# Convert 'measurement_datetime' to datetime (if not already done)
merged_train['measurement_datetime'] = pd.to_datetime(merged_train['measurement_datetime'], errors='coerce')

# Drop rows where 'measurement_datetime' is NaN/NaT
# merged_train = merged_train.dropna(subset=['measurement_datetime'])
# Sort the DataFrame by 'measurement_datetime' from earliest to latest (ascending order)
merged_train = merged_train.sort_values(by='measurement_datetime')

merged_train.head()


merged_train.info()


# Drop the 'SepsisLabel' column for features
merged_train_features = merged_train.drop(columns=['SepsisLabel'])

# Keep the 'SepsisLabel' column for target labels
y = merged_train['SepsisLabel']

# Calculate the split point
split_point = int(len(merged_train_features) * 0.8)

# Split the features into training and testing sets
x_train = merged_train_features.iloc[:split_point]  # First 80% for training
x_test = merged_train_features.iloc[split_point:]   # Remaining 20% for testing

# Split the target labels into training and testing sets
y_train = y.iloc[:split_point]  # First 80% for training
y_test = y.iloc[split_point:]   # Remaining 20% for testing

# Check the shapes of the resulting datasets
print(f"x_train shape: {x_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")


date_time = ['measurement_datetime']
numerical_cols = ['person_id',
 'duration',
 'device_hour',
 'device_mean_hour',
 'Base excess in Venous blood by calculation',
 'Base excess in Arterial blood by calculation',
 'Phosphate [Moles/volume] in Serum or Plasma',
 'Potassium [Moles/volume] in Blood',
 'Bilirubin.total [Moles/volume] in Serum or Plasma',
 'Neutrophil Ab [Units/volume] in Serum',
 'Bicarbonate [Moles/volume] in Arterial blood',
 'Hematocrit [Volume Fraction] of Blood',
 'Glucose [Moles/volume] in Serum or Plasma',
 'Calcium [Moles/volume] in Serum or Plasma',
 'Chloride [Moles/volume] in Blood',
 'Sodium [Moles/volume] in Serum or Plasma',
 'C reactive protein [Mass/volume] in Serum or Plasma',
 'Carbon dioxide [Partial pressure] in Venous blood',
 'Oxygen [Partial pressure] in Venous blood',
 'Albumin [Mass/volume] in Serum or Plasma',
 'Bicarbonate [Moles/volume] in Venous blood',
 'Oxygen [Partial pressure] in Arterial blood',
 'Carbon dioxide [Partial pressure] in Arterial blood',
 'Interleukin 6 [Mass/volume] in Body fluid',
 'Magnesium [Moles/volume] in Blood',
 'Prothrombin time (PT)',
 'Procalcitonin [Mass/volume] in Serum or Plasma',
 'Lactate [Moles/volume] in Blood',
 'Creatinine [Mass/volume] in Blood',
 'Fibrinogen measurement',
 'Bilirubin measurement',
 'Partial thromboplastin time',
 ' activated',
 'Total white blood count',
 'Platelet count',
 'White blood cell count',
 'Blood venous pH',
 'D-dimer level',
 'Blood arterial pH',
 'Hemoglobin [Moles/volume] in Blood',
 'Ionised calcium measurement',
 'Systolic blood pressure',
 'Diastolic blood pressure',
 'Body temperature',
 'Respiratory rate',
 'Heart rate',
 'Measurement of oxygen saturation at periphery',
 'Oxygen/Gas total [Pure volume fraction] Inhaled gas',
 'Left pupil Diameter Auto',
 'Right pupil Diameter Auto',
 'Glasgow coma scale',
 'procedure_duration',
 'drug_duration',
 'observation_duration']
categorical_cols = ['device',
 'Capillary refill [Time]',
 'Pulse',
 'Arterial pulse pressure',
 'Right pupil Pupillary response',
 'Left pupil Pupillary response',
 'procedure',
 'drug_concept_id',
 'route_concept_id',
 'observation_concept_name',
 'valuefilled']

extra_cols = [
 'Right pupil Pupillary response',
 'Left pupil Pupillary response']

# Fill missing values for extra columns using the mode of each column
x_train.loc[:, extra_cols] = x_train[extra_cols].apply(lambda col: col.fillna(col.mode()[0]), axis=0)
x_test.loc[:, extra_cols] = x_test[extra_cols].apply(lambda col: col.fillna(col.mode()[0]), axis=0)

# Fill missing values for categorical columns using forward fill and backward fill
x_train.loc[:, categorical_cols] = x_train[categorical_cols].ffill().bfill()
x_test.loc[:, categorical_cols] = x_test[categorical_cols].ffill().bfill()

# Fill missing values for numerical columns using the mean of each column
x_train.loc[:, numerical_cols] = x_train[numerical_cols].fillna(x_train[numerical_cols].mean())
x_test.loc[:, numerical_cols] = x_test[numerical_cols].fillna(x_test[numerical_cols].mean())

# Fill missing values for datetime columns using the mean of each column
x_train.loc[:, date_time] = x_train[date_time].fillna(x_train[date_time].mean())
x_test.loc[:, date_time] = x_test[date_time].fillna(x_test[date_time].mean())

# Display the DataFrame after filling missing values
print(x_train.isnull().sum())
print(x_test.isnull().sum())


x_train.drop('measurement_datetime', axis=1, inplace=True)
x_test.drop('measurement_datetime', axis=1, inplace=True)


df_tr_corr = pd.concat([x_train, y_train], axis=1).copy()
featurex = df_tr_corr.drop(['SepsisLabel'], axis=1)
featurey = df_tr_corr[['SepsisLabel']]

print("featurex", featurex.shape)
print("featurey", featurey.shape)
print('-------------------------------------------------------------------------')


# Identify categorical columns
categorical_cols = featurex.select_dtypes(include=['object']).columns.to_list()

# Convert all categorical columns to strings to avoid mixed types
for col in categorical_cols:
    featurex[col] = featurex[col].astype(str)

# Apply OrdinalEncoder to categorical columns
ordinal = OrdinalEncoder()
for col in categorical_cols:
    featurex[[col]] = ordinal.fit_transform(featurex[[col]])

# Mutual Information Calculation
from sklearn.feature_selection import mutual_info_regression
from joblib import Parallel, delayed

# Calculate mutual information scores for each feature with respect to each target
X, y = featurex, featurey

def calculate_mi_for_target(target_index):
    return mutual_info_regression(X, y.iloc[:, target_index], random_state=42)

# Run parallelized MI calculations on the original y (multi-dimensional)
mi_scores = Parallel(n_jobs=-1)(delayed(calculate_mi_for_target)(i) for i in range(y.shape[1]))

# Convert the list of scores to a DataFrame
mi_scores_df = pd.DataFrame(mi_scores, columns=featurex.columns, index=featurey.columns)
print("\nMutual Information Scores for each feature with respect to each target:")
mi_scores_df


# Aggregate scores across all targets (e.g., by averaging)
aggregated_mi_scores = mi_scores_df.mean(axis=0).sort_values(ascending=False)
print("\nAggregated Mutual Information Scores:")
aggregated_mi_scores


# Additional statistics
print("\nMutual Information Scores Statistics:")
mi_scores_df.T.describe()


columns_to_drop = [
    'person_id',
    "Calcium [Moles/volume] in Serum or Plasma",
    "Left pupil Pupillary response",
    "Capillary refill [Time]",
    "Arterial pulse pressure",
    "Left pupil Diameter Auto",
    "Right pupil Pupillary response",
    "observation_concept_name",
    "Neutrophil Ab [Units/volume] in Serum",
    "Heart rate",
    "Interleukin 6 [Mass/volume] in Body fluid",
    "Glucose [Moles/volume] in Serum or Plasma",
    "C reactive protein [Mass/volume] in Serum or Plasma",
    "Oxygen [Partial pressure] in Venous blood",
    "Albumin [Mass/volume] in Serum or Plasma",
    "Oxygen [Partial pressure] in Arterial blood",
    "Carbon dioxide [Partial pressure] in Arterial blood",
    "Lactate [Moles/volume] in Blood",
    "D-dimer level",
    "Bilirubin measurement",
    " activated",  # Note: There might be a typo here (extra space at the beginning)
    "Total white blood count",
    "Platelet count",
    "White blood cell count",
    "Bicarbonate [Moles/volume] in Arterial blood",
    "Blood venous pH"
]
# Drop columns from x_train
x_train = x_train.drop(columns=columns_to_drop)

# Drop columns from x_test
x_test = x_test.drop(columns=columns_to_drop)

print("x_train shape after dropping columns:", x_train.shape)
print("x_test shape after dropping columns:", x_test.shape)


# Define text columns
text_columns = ['device', 'Pulse', 'procedure', 'drug_concept_id', 'route_concept_id', 'valuefilled']

# Tokenize text data
def tokenize_text(text):
    if pd.isna(text):
        return []
    return str(text).split()

# Train Word2Vec model
def train_word2vec(corpus, vector_size=100, window=5, min_count=1, workers=4):
    model = Word2Vec(sentences=corpus, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
    return model

# Create a corpus from text columns
corpus = []
for col in text_columns:
    corpus.extend(x_train[col].apply(tokenize_text).tolist())

# Train and save the Word2Vec model
word2vec_model = train_word2vec(corpus, vector_size=100)
word2vec_model.save("word2vec.model")

# Define TextToEmbeddingTransformer
class TextToEmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, word2vec_path, text_columns):
        self.word2vec_path = word2vec_path
        self.text_columns = text_columns
        self.word2vec_model = Word2Vec.load(word2vec_path)

    def transform_text_to_embedding(self, text):
        tokens = tokenize_text(text)
        if not tokens:
            return np.zeros(self.word2vec_model.vector_size)
        vectors = [self.word2vec_model.wv[word] for word in tokens if word in self.word2vec_model.wv]
        if not vectors:
            return np.zeros(self.word2vec_model.vector_size)
        return np.mean(vectors, axis=0)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.text_columns:
            X[col] = X[col].apply(self.transform_text_to_embedding)
        return X

# Apply the transformer
text_to_embedding = TextToEmbeddingTransformer("word2vec.model", text_columns)
x_train_transformed = text_to_embedding.transform(x_train)

# Convert transformed embeddings to structured NumPy array
def flatten_embeddings(df, text_columns):
    return np.array([np.hstack(row[text_columns].values) for _, row in df.iterrows()])

x_train_flattened = flatten_embeddings(x_train_transformed, text_columns)

# Save transformer metadata
joblib.dump(text_columns, "text_to_embedding_transformer.pkl")

# Standardize the data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_flattened)
joblib.dump(scaler, "standard_scaler.joblib")

# Apply PCA
pca = PCA(n_components=20, random_state=42)
x_train_pca = pca.fit_transform(x_train_scaled)
joblib.dump(pca, "pca_model.joblib")

# Apply SMOTE for oversampling
smote = SMOTE(random_state=42)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train_pca, y_train)

# Check class distribution after SMOTE
print("\nClass distribution after SMOTE:")
print(pd.Series(y_train_resampled).value_counts(normalize=True))

print("Shape of x_train after SMOTE:", x_train_resampled.shape)
print("Shape of y_train after SMOTE:", y_train_resampled.shape)

# Save resampled data
#joblib.dump((x_train_resampled, y_train_resampled), "resampled_data.pkl")


# Function to transform dataset
def transform_dataset(x_test):
    # Load text embedding transformer
    text_columns = joblib.load("text_to_embedding_transformer.pkl")
    word2vec_model = Word2Vec.load("word2vec.model")

    # Define Transformer
    class TextToEmbeddingTransformer:
        def __init__(self, word2vec_model, text_columns):
            self.word2vec_model = word2vec_model
            self.text_columns = text_columns

        def transform_text_to_embedding(self, text):
            tokens = tokenize_text(text)
            if not tokens:
                return np.zeros(self.word2vec_model.vector_size)
            vectors = [self.word2vec_model.wv[word] for word in tokens if word in self.word2vec_model.wv]
            if not vectors:
                return np.zeros(self.word2vec_model.vector_size)
            return np.mean(vectors, axis=0)

        def transform(self, X):
            X = X.copy()
            for col in self.text_columns:
                if col in X.columns:
                    X[col] = X[col].apply(self.transform_text_to_embedding)
            return X

    text_to_embedding = TextToEmbeddingTransformer(word2vec_model, text_columns)
    x_test_transformed = text_to_embedding.transform(x_test)

    # Convert to NumPy array
    x_test_embedded = np.array([np.hstack(row[text_columns].values) for _, row in x_test_transformed.iterrows()])

    # Load the scaler
    scaler = joblib.load("standard_scaler.joblib")
    x_test_scaled = scaler.transform(x_test_embedded)

    # Load PCA model
    pca = joblib.load("pca_model.joblib")
    x_test_pca = pca.transform(x_test_scaled)

    print("Transformed x_test shape:", x_test_pca.shape)
    return x_test_pca

# Example transformation
x_test_pca = transform_dataset(x_test)


# Prepare time series dataset
def create_time_series_dataset(data, labels, time_steps):
    Xs, ys = [], []
    # Convert labels to NumPy array to avoid Pandas indexing issues
    labels = labels.to_numpy() if isinstance(labels, pd.Series) else np.array(labels)
    for i in range(len(data) - time_steps):
        Xs.append(data[i:(i + time_steps)])
        ys.append(labels[i + time_steps])
    return np.array(Xs), np.array(ys)

# Create time series dataset
time_steps = 100
x_train_time_series, y_train_time_series = create_time_series_dataset(x_train_resampled, y_train_resampled, time_steps)
x_test_time_series, y_test_time_series = create_time_series_dataset(x_test_pca, y_test, time_steps)

print("Shape of time series X_train:", x_train_time_series.shape)
print("Shape of time series y_train:", y_train_time_series.shape)
print("Shape of time series X_test:", x_test_time_series.shape)
print("Shape of time series y_test:", y_test_time_series.shape)


def TCN_block(x, filters, kernel_size, dilation_rate, dropout_rate=0.3):
    """Enhanced TCN Block with residual connection"""
    x_skip = x  # Save input for residual connection
    
    # First convolution
    x = Conv1D(filters, kernel_size, dilation_rate=dilation_rate, padding='causal', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)
    
    # Second convolution
    x = Conv1D(filters, kernel_size, dilation_rate=dilation_rate, padding='causal', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)
    
    # Residual Connection
    if x_skip.shape[-1] != filters:
        x_skip = Conv1D(filters, 1, padding='same')(x_skip)  # Adjust channels
    x = Add()([x, x_skip])
    return x

# Define an expanded TCN model
def create_TCN_model(input_shape, output_shape):
    inputs = Input(shape=input_shape, name="input")
    x = inputs

    # Deeper TCN Blocks
    for dilation_rate in [1, 2, 4, 8, 16]:  # Added more dilation rates
        x = TCN_block(x, filters=256, kernel_size=3, dilation_rate=dilation_rate)  # Doubled filters
    
    # Global Average Pooling
    x = GlobalAveragePooling1D()(x)
    
    # Larger Dense layers
    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)  # Increased neurons
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)  # Increased Dropout

    x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)  # Increased neurons
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    # Output layer
    outputs = Dense(output_shape, activation='sigmoid', name="output")(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Create and compile model
model = create_TCN_model(input_shape=x_train_time_series.shape[1:], output_shape=1)
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4)  # Slightly reduced LR to stabilize training

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=["accuracy", tf.keras.metrics.AUC(name='pr_auc', curve='PR')]
)

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=8, mode='min', restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, mode='min')
]

# Train the model
history = model.fit(
    x_train_time_series, y_train_time_series,
    validation_split=0.2,  # Use validation split due to need to shuffle validation data to make model capture more pattern, normally it should not use validation split for time-series dataset
    batch_size=4,  #128
    epochs=5,  # Increased epochs for better accuracy, you can set up to 150 epochs with early stop (5 epochs for illustration)
    callbacks=callbacks,
    shuffle=True  # Normally Time-Series use shuffle = False, but in this competition I want model to recognize several pattern in the dataset
)

# Save the model
model.save("enhanced_tcn_model.h5")


# Load the saved model
#model = tf.keras.models.load_model("/kaggle/input/mild_model_v2/tensorflow2/default/1/time_series_classification_model.h5")
#model = tf.keras.models.load_model("optimized_tcn_modelV3.h5")

# Predict on the test dataset
y_pred_prob = model.predict(x_train_time_series)  # Get predicted probabilities
y_pred = np.argmax(y_pred_prob, axis=1)  # Convert probabilities to class labels using np.argmax

# Calculate evaluation metrics
accuracy = accuracy_score(y_train_time_series, y_pred)
f1 = f1_score(y_train_time_series, y_pred, average='weighted')  # Use 'weighted' for multi-class classification
roc_auc = roc_auc_score(y_train_time_series, y_pred_prob)       # multi_class='ovr')  # Use 'ovr' for multi-class classification
precision = precision_score(y_train_time_series, y_pred, average='weighted')  # Use 'weighted' for multi-class classification
recall = recall_score(y_train_time_series, y_pred, average='weighted')  # Use 'weighted' for multi-class classification

# Print the evaluation metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")


# Load the saved model
#model = tf.keras.models.load_model("/kaggle/input/mild_model_v2/tensorflow2/default/1/time_series_classification_model.h5")

# Predict on the test dataset
y_pred_prob = model.predict(x_test_time_series)  # Get predicted probabilities
y_pred = np.argmax(y_pred_prob, axis=1)  # Convert probabilities to class labels using np.argmax

# Calculate evaluation metrics
accuracy = accuracy_score(y_test_time_series, y_pred)
f1 = f1_score(y_test_time_series, y_pred, average='weighted')  # Use 'weighted' for multi-class classification
roc_auc = roc_auc_score(y_test_time_series, y_pred_prob, multi_class='ovr')  # Use 'ovr' for multi-class classification
precision = precision_score(y_test_time_series, y_pred, average='weighted')  # Use 'weighted' for multi-class classification
recall = recall_score(y_test_time_series, y_pred, average='weighted')  # Use 'weighted' for multi-class classification

# Print the evaluation metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

