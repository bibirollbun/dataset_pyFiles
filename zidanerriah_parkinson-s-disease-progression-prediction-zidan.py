import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


data_path = '/kaggle/input/amp-parkinsons-disease-progression-prediction/'


clinical_data = pd.read_csv(data_path + 'train_clinical_data.csv')
peptides_data = pd.read_csv(data_path + 'train_peptides.csv')
proteins_data = pd.read_csv(data_path + 'train_proteins.csv')


# Aggregate peptide data: count and mean of PeptideAbundance per visit
peptides_agg = peptides_data.groupby('visit_id')['PeptideAbundance'].agg(
    pep_count='count',
    pep_mean='mean'
).reset_index()

# Aggregate protein data: count and mean of NPX per visit
proteins_agg = proteins_data.groupby('visit_id')['NPX'].agg(
    prot_count='count',
    prot_mean='mean'
).reset_index()


# Merge the aggregated peptide and protein features with the clinical data
merged_data = clinical_data.merge(peptides_agg, on='visit_id', how='left')
merged_data = merged_data.merge(proteins_agg, on='visit_id', how='left')


# Convert medication status to numerical values
merged_data['upd23b_clinical_state_on_medication'] = merged_data['upd23b_clinical_state_on_medication'].map({'On': 1, 'Off': 0})

# Drop unnecessary columns
merged_data = merged_data.drop(columns=['visit_id', 'patient_id', 'updrs_2', 'updrs_3', 'updrs_4'])

# Handle missing values
merged_data = merged_data.dropna(subset=['updrs_1'])
merged_data = merged_data.fillna(0)


# Define features (X) and target variable (y)
X = merged_data.drop(columns=['updrs_1'])
y = merged_data['updrs_1']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize the KNN regressor with 5 neighbors
knn_model = KNeighborsRegressor(n_neighbors=5)

# Train the model on the training data
knn_model.fit(X_train, y_train)


# Predict UPDRS_1 scores on the test set
y_pred = knn_model.predict(X_test)

# Calculate the Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")


# Plot actual vs. predicted UPDRS_1 scores
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue', edgecolors='k')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual UPDRS_1")
plt.ylabel("Predicted UPDRS_1")
plt.title("KNN Predictions vs. Actual UPDRS_1 Scores")

