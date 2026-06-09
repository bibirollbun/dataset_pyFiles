!pip install catboost


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


try:
    df_train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
    print("Dataset 'train.csv' loaded successfully.")
except FileNotFoundError:
    print("Error: File 'train.csv' not found in './dataset/'. Make sure the path is correct or adjust it to the Kaggle path.")
    exit()


try:
    df_submission = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
    print("Dataset 'test.csv' loaded successfully.")
except FileNotFoundError:
    print("Error: File 'test.csv' not found in './dataset/'. Make sure the path is correct or adjust it to the Kaggle path.")
    exit() 


def calculate_ap_at_k(true_labels_for_obs, predicted_ranks_for_obs, k):

    if not true_labels_for_obs:
        return 0.0

    precisions = []
    num_correct = 0
    relevant_labels_found = set() 

    for i in range(min(len(predicted_ranks_for_obs), k)):
        current_prediction = predicted_ranks_for_obs[i]

        if current_prediction in true_labels_for_obs and \
           current_prediction not in relevant_labels_found:
            num_correct += 1
            relevant_labels_found.add(current_prediction)
            precision_at_k = num_correct / (i + 1)
            precisions.append(precision_at_k)

    if not precisions:
        return 0.0
    return sum(precisions) / len(precisions)


def map_at_k(y_true, y_pred_probs, k, label_encoder):
    num_observations = len(y_true)
    total_ap = 0.0

    all_class_labels = label_encoder.classes_

    for i in range(num_observations):
        true_label_encoded = y_true[i]
        true_label_string = label_encoder.inverse_transform([true_label_encoded])[0]

        true_labels_for_obs = [true_label_string]

        sorted_class_indices = np.argsort(y_pred_probs[i])[::-1]
        predicted_ranks_for_obs = [all_class_labels[idx] for idx in sorted_class_indices]

        ap = calculate_ap_at_k(true_labels_for_obs, predicted_ranks_for_obs, k)
        total_ap += ap

    return total_ap / num_observations


class AdvancedFeatureEngineer:
    def __init__(self, df):
        self.df = df.copy()
        self.epsilon = 1e-6 

    def create_soil_crop_combo(self):
        self.df['soil_crop_combo'] = self.df['Soil Type'].astype(str) + '_' + self.df['Crop Type'].astype(str)

    def create_nutrient_ratios(self):
        self.df['nitrogen_to_potassium'] = self.df['Nitrogen'] / (self.df['Potassium'] + self.epsilon)
        self.df['nitrogen_to_phosphorous'] = self.df['Nitrogen'] / (self.df['Phosphorous'] + self.epsilon)
        self.df['potassium_to_phosphorous'] = self.df['Potassium'] / (self.df['Phosphorous'] + self.epsilon)

    def create_dryness_index(self):
        self.df['dryness_index'] = self.df['Temparature'] / (self.df['Moisture'] + self.epsilon)

    def create_deviation_features(self):
        nitrogen_means = self.df.groupby('Soil Type')['Nitrogen'].transform('mean')
        self.df['nitrogen_deviation_from_soil_mean'] = self.df['Nitrogen'] - nitrogen_means

        potassium_means = self.df.groupby('Soil Type')['Potassium'].transform('mean')
        self.df['potassium_deviation_from_soil_mean'] = self.df['Potassium'] - potassium_means

        phosphorous_means = self.df.groupby('Soil Type')['Phosphorous'].transform('mean')
        self.df['phosphorous_deviation_from_soil_mean'] = self.df['Phosphorous'] - phosphorous_means

    # def create_numeric_bins(self, num_bins=5):
    #     numeric_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Phosphorous', 'Potassium']
    #     for col in numeric_features:
    #         bin_col_name = f"{col.lower()}_bin"
    #         self.df[bin_col_name] = pd.qcut(self.df[col], q=num_bins, duplicates='drop')

    def run_all(self):
        self.create_soil_crop_combo()
        self.create_nutrient_ratios()
        self.create_dryness_index()
        self.create_deviation_features()
        # self.create_numeric_bins()
        return self.df


extract_features_train = AdvancedFeatureEngineer(df_train)
extract_features_submission = AdvancedFeatureEngineer(df_submission)

df_train_new = extract_features_train.run_all()
df_submission_new = extract_features_submission.run_all()


df_train_new.info()


df_submission_new.info()


# --- LabelEncoder for object columns ---
label_encoder_fertilizer = LabelEncoder()
label_encoder_soil = LabelEncoder()
label_encoder_crop = LabelEncoder()
label_encoder_combo = LabelEncoder()

df_train_new["Fertilizer Name"] = label_encoder_fertilizer.fit_transform(df_train_new["Fertilizer Name"])
df_train_new["Soil Type"] = label_encoder_soil.fit_transform(df_train_new["Soil Type"])
df_train_new["Crop Type"] = label_encoder_crop.fit_transform(df_train_new["Crop Type"])
df_train_new["soil_crop_combo"] = label_encoder_combo.fit_transform(df_train_new["soil_crop_combo"])

# Ensure columns are int64
df_train_new["Soil Type"] = df_train_new["Soil Type"].astype("int64")
df_train_new["Crop Type"] = df_train_new["Crop Type"].astype("int64")
df_train_new["soil_crop_combo"] = df_train_new["soil_crop_combo"].astype("int64")

# --- LabelEncoder for category columns ---
# bin_columns = [
#     "temparature_bin",
#     "humidity_bin",
#     "moisture_bin",
#     "nitrogen_bin",
#     "phosphorous_bin",
#     "potassium_bin"
# ]

# Create a LabelEncoder for each bin column for potential inverse_transform later
# bin_encoders = {}
# for col in bin_columns:
#     le = LabelEncoder()
#     df_train_new[col] = le.fit_transform(df_train_new[col])
#     bin_encoders[col] = le
#     df_train_new[col] = df_train_new[col].astype("int64")

# --- Split Features and Target ---
X = df_train_new.drop(columns=["id", "Fertilizer Name"])
y = df_train_new["Fertilizer Name"]

print("\n--- Training Data Head ---")
print("Features (X) Head:")
print(X.head())
print("\nTarget (y) Head:")
print(y.head())

# --- Train Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining size: {X_train.shape[0]} samples")
print(f"Testing size: {X_test.shape[0]} samples")

# --- Standardize numerical features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeatures have been standardized.")

# --- Define Base Models ---
xgb_clf = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=len(label_encoder_fertilizer.classes_),
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)

adb_clf = AdaBoostClassifier(
    n_estimators=100,
    random_state=42
)

cat_clf = CatBoostClassifier(
    iterations=200,
    learning_rate=0.1,
    verbose=0,
    random_state=42
)

# --- Create Ensemble Voting Classifier ---
ensemble_model = VotingClassifier(
    estimators=[
        ('xgb', xgb_clf),
        ('ada', adb_clf),
        ('cat', cat_clf)
    ],
    voting='soft'  # Use soft voting to average probabilities
)

# --- Train Ensemble ---
ensemble_model.fit(X_train_scaled, y_train)
print("\nEnsemble model (Voting) training completed.")

# --- Predict Probabilities ---
y_pred_probs_ensemble = ensemble_model.predict_proba(X_test_scaled)

# --- Calculate MAP@5 for Ensemble ---
map_5_score_ensemble = map_at_k(
    y_test.values, y_pred_probs_ensemble, k=5, label_encoder=label_encoder_fertilizer
)
print(f"MAP@5 Score for Ensemble on test data: {map_5_score_ensemble:.4f}")

# --- Predict Labels ---
y_pred_ensemble_labels = ensemble_model.predict(X_test_scaled)

# --- Accuracy & Classification Report ---
print(f"Accuracy Score for Ensemble on test data: {accuracy_score(y_test, y_pred_ensemble_labels):.4f}")
print("\nClassification Report for Ensemble on test data:")
print(classification_report(
    y_test,
    y_pred_ensemble_labels,
    target_names=[str(cls) for cls in label_encoder_fertilizer.classes_]
))


# --- Prepare submission IDs ---
submission_ids = df_submission_new['id']

# --- Encode object columns in submission ---
df_submission_new["Soil Type"] = label_encoder_soil.transform(df_submission_new["Soil Type"])
df_submission_new["Crop Type"] = label_encoder_crop.transform(df_submission_new["Crop Type"])
df_submission_new["soil_crop_combo"] = label_encoder_combo.transform(df_submission_new["soil_crop_combo"])

# --- Select features (drop ID) ---
X_submission = df_submission_new.drop(columns=["id"])

# --- Apply scaler ---
X_submission_scaled = scaler.transform(X_submission)

# --- Predict probabilities using ENSEMBLE ---
y_pred_probs_submission = ensemble_model.predict_proba(X_submission_scaled)

# --- Generate top 3 fertilizers for each row ---
top_3_fertilizers = []
for i in range(len(y_pred_probs_submission)):
    num_classes_to_take = min(3, len(label_encoder_fertilizer.classes_))
    top_indices = np.argsort(y_pred_probs_submission[i])[::-1][:num_classes_to_take]
    top_names = label_encoder_fertilizer.inverse_transform(top_indices)
    top_3_fertilizers.append(" ".join(top_names))

# --- Create submission DataFrame ---
submission_df = pd.DataFrame({
    'ID': submission_ids,
    'Fertilizer Name': top_3_fertilizers
})

# --- Save submission file ---
submission_file_name = '/kaggle/working/submission_ensemble.csv'
submission_df.to_csv(submission_file_name, index=False)

print(f"\nSubmission file '{submission_file_name}' has been created successfully!")
print("Head of submission file:")
print(submission_df.head())

