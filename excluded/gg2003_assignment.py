import os
os.listdir("/kaggle/input")



import pandas as pd

train = pd.read_csv('/kaggle/input/novozymes-enzyme-stability-prediction/train.csv')
test  = pd.read_csv('/kaggle/input/novozymes-enzyme-stability-prediction/test.csv')
sub   = pd.read_csv('/kaggle/input/novozymes-enzyme-stability-prediction/sample_submission.csv')

train.head()


#To check shape and columns

train.shape, test.shape
train.info()
train.describe()



import matplotlib.pyplot as plt
import seaborn as sns

# Missing values
train.isnull().sum()

# Distribution of target
sns.histplot(train['tm'], kde=True)
plt.title("Distribution of Melting Temperature (tm)")
plt.show()

# Sequence lengths
train['seq_len'] = train['protein_sequence'].apply(len)
sns.histplot(train['seq_len'], kde=False)
plt.title("Protein Sequence Length Distribution")
plt.show()



import numpy as np
import pandas as pd
from collections import Counter

# Take a small sample for quick testing
train_sample = train.sample(1000, random_state=42).reset_index(drop=True)

# Amino acids & properties
amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
hydropathy = {  # Kyte-Doolittle scale
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
}
charge_vals = {
    'K': 1, 'R': 1, 'H': 0.1,  # Positive
    'D': -1, 'E': -1           # Negative
}

# Feature extraction for one sequence
def featurize_sequence(seq):
    length = len(seq)
    counts = Counter(seq)
    aa_freq = [counts.get(aa, 0) / length for aa in amino_acids]
    hydrophobicity_score = np.mean([hydropathy.get(aa, 0) for aa in seq])
    net_charge_score = sum(charge_vals.get(aa, 0) for aa in seq)
    return [length, hydrophobicity_score, net_charge_score] + aa_freq

# Apply to sample
features_array = np.array([featurize_sequence(seq) for seq in train_sample["protein_sequence"]])

# DataFrame
feature_cols = ["seq_len", "hydrophobicity", "net_charge"] + [f"aa_{aa}" for aa in amino_acids]
features_df = pd.DataFrame(features_array, columns=feature_cols)

# Combine with pH
X_sample = pd.concat([train_sample[["pH"]].reset_index(drop=True), features_df], axis=1)
y_sample = train_sample["tm"]

print(X_sample.head())



from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Split the encoded data into a training set and a validation set
X_train, X_val, y_train, y_val = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Initialize the model. We'll keep the hyperparameters at their defaults
# to focus on the pipeline, as instructed in the assignment.
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

# Train the model on the training data
print("Training the Random Forest Regressor model...")
model.fit(X_train, y_train)
print("Training complete.")

# Make predictions on the validation set
y_pred_val = model.predict(X_val)

# Evaluate model performance using Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
print(f"Validation RMSE: {rmse:.4f}")

# The assignment uses Spearman's correlation, so it is a good idea to
# include that as well to show a deeper understanding of the problem.
from scipy.stats import spearmanr
spearman_corr, _ = spearmanr(y_val, y_pred_val)
print(f"Validation Spearman's Correlation: {spearman_corr:.4f}")

