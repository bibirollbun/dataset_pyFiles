# ---------------------- Imports ----------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ---------------------- Hyperparameters ----------------------
WINDOW_SIZE = 31
K_MER_SIZE = 3
K_MER_COUNT = 16
TEST_SIZE = 0.1
RANDOM_STATE = 42

# ---------------------- Load Data ----------------------
labels = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_labels.csv')
sequences = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_sequences.csv').set_index("target_id")["sequence"]

labels[["target_id", "residue_index"]] = labels["ID"].str.rsplit("_", n=1, expand=True)
labels["residue_index"] = labels["residue_index"].astype(int)
labels["sequence"] = labels["target_id"].map(sequences)
labels.dropna(subset=["sequence", "x_1", "y_1", "z_1"], inplace=True)

# ---------------------- Feature Engineering ----------------------
nt_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
pad_char = 'N'

def one_hot_encode(seq, center_idx, window):
    pad = pad_char * window
    padded = pad + seq + pad
    center = center_idx + window
    window_seq = padded[center - window:center + window + 1]
    vec = np.zeros((len(window_seq), 4), dtype=np.float32)
    for i, nt in enumerate(window_seq):
        if nt in nt_map:
            vec[i, nt_map[nt]] = 1.0
    return vec

def get_kmer_features(seq, k):
    kmers = {}
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        kmers[kmer] = kmers.get(kmer, 0) + 1
    return kmers

def create_kmer_vector(seq, top_kmers, k):
    kmer_counts = get_kmer_features(seq, k)
    vector = np.zeros(len(top_kmers), dtype=np.float32)
    for i, kmer in enumerate(top_kmers):
        vector[i] = kmer_counts.get(kmer, 0)
    return vector

# Generate top k-mers
all_train_sequences = sequences.values.tolist()
all_kmers = {}
for seq in all_train_sequences:
    kmers = get_kmer_features(seq, K_MER_SIZE)
    for kmer, count in kmers.items():
        all_kmers[kmer] = all_kmers.get(kmer, 0) + count
top_kmers = sorted(all_kmers, key=all_kmers.get, reverse=True)[:K_MER_COUNT]

# Save top k-mers
top_kmers_path = "/kaggle/working/top_kmers.pkl"
joblib.dump(top_kmers, top_kmers_path)
print(f"Top k-mers saved to {top_kmers_path}")

X_sequence = np.stack([
    one_hot_encode(row.sequence, row.residue_index - 1, WINDOW_SIZE)
    for _, row in labels.iterrows()
])

X_kmer = np.stack([
    create_kmer_vector(row.sequence, top_kmers, K_MER_SIZE)
    for _, row in labels.iterrows()
])

# Combine sequence and k-mer features
X = np.concatenate((X_sequence, np.repeat(X_kmer[:, np.newaxis, :], X_sequence.shape[1], axis=1)), axis=2)

target_columns = ["x_1", "y_1", "z_1"]
Y = labels[target_columns].values.astype(np.float32)

# ---------------------- Train/Validation Split ----------------------
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# ---------------------- Target Scaling ----------------------
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_val_scaled = scaler_y.transform(y_val)

# ---------------------- Train XGBoost Regressor ----------------------
print("Training XGBoost Regressor...")
xgbr = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100, # Tune this
    learning_rate=0.1, # Tune this
    max_depth=5,       # Tune this
    random_state=RANDOM_STATE
)

xgbr.fit(X_train.reshape(X_train.shape[0], -1), y_train_scaled) # Flatten the input features

# ---------------------- Evaluate XGBoost Model ----------------------
y_pred_val_scaled = xgbr.predict(X_val.reshape(X_val.shape[0], -1))
y_pred_val = scaler_y.inverse_transform(y_pred_val_scaled)

rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
print(f"XGBoost Validation RMSE: {rmse:.4f}")

# ---------------------- Visualization of Predictions ----------------------
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(y_val[:, 0], y_pred_val[:, 0], alpha=0.5)
plt.xlabel("Actual x_1")
plt.ylabel("Predicted x_1")
plt.title("Actual vs. Predicted x_1 (XGBoost on Validation Set)")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(y_val[:, 1], y_pred_val[:, 1], alpha=0.5)
plt.xlabel("Actual y_1")
plt.ylabel("Predicted y_1")
plt.title("Actual vs. Predicted y_1 (XGBoost on Validation Set)")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(y_val[:, 2], y_pred_val[:, 2], alpha=0.5)
plt.xlabel("Actual z_1")
plt.ylabel("Predicted z_1")
plt.title("Actual vs. Predicted z_1 (XGBoost on Validation Set)")
plt.grid(True)
plt.show()

# ---------------------- Save Trained XGBoost Model and Scaler ----------------------
model_path = "/kaggle/working/xgboost_rna_folding_model.joblib"
scaler_path = "/kaggle/working/xgboost_scaler_y.joblib"
joblib.dump(xgbr, model_path)
joblib.dump(scaler_y, scaler_path)
print(f"Trained XGBoost model saved to {model_path}")
print(f"Target scaler saved to {scaler_path}")


# ---------------------- Imports ----------------------
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

# ---------------------- Hyperparameters (Must match Notebook 1) ----------------------
WINDOW_SIZE = 31
K_MER_SIZE = 3
K_MER_COUNT = 16

# ---------------------- One-Hot Encoding (Same as Notebook 1) ----------------------
nt_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
pad_char = 'N'

def one_hot_encode(seq, center_idx, window):
    pad = pad_char * window
    padded = pad + seq + pad
    center = center_idx + window
    window_seq = padded[center - window:center + window + 1]
    vec = np.zeros((len(window_seq), 4), dtype=np.float32)
    for i, nt in enumerate(window_seq):
        if nt in nt_map:
            vec[i, nt_map[nt]] = 1.0
    return vec

# ---------------------- K-mer Feature Creation (Same as Notebook 1) ----------------------
def get_kmer_features(seq, k):
    kmers = {}
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        kmers[kmer] = kmers.get(kmer, 0) + 1
    return kmers

def create_kmer_vector(seq, top_kmers, k):
    kmer_counts = get_kmer_features(seq, k)
    vector = np.zeros(len(top_kmers), dtype=np.float32)
    for i, kmer in enumerate(top_kmers):
        vector[i] = kmer_counts.get(kmer, 0)
    return vector

# Load the top k-mers saved from Notebook 1
top_kmers_path = "/kaggle/working/top_kmers.pkl"
try:
    top_kmers = joblib.load(top_kmers_path)
    print(f"Loaded top k-mers from {top_kmers_path}")
except FileNotFoundError:
    print(f"Error: {top_kmers_path} not found. Make sure Notebook 1 was run and the file was saved to /kaggle/working/.")
    raise

# ---------------------- Load Test Data ----------------------
test_sequences = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/test_sequences.csv').set_index("target_id")["sequence"]
sample_sub = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/sample_submission.csv')

sample_sub[["target_id", "residue_index"]] = sample_sub.ID.str.rsplit("_", n=1, expand=True)
sample_sub["residue_index"] = sample_sub["residue_index"].astype(int)
sample_sub["sequence"] = sample_sub["target_id"].map(test_sequences)

# ---------------------- Prepare Test Features ----------------------
X_test_sequence = np.stack([
    one_hot_encode(row.sequence, row.residue_index - 1, WINDOW_SIZE)
    for _, row in sample_sub.iterrows()
])

X_test_kmer = np.stack([
    create_kmer_vector(row.sequence, top_kmers, K_MER_SIZE)
    for _, row in sample_sub.iterrows()
])

X_test = np.concatenate((X_test_sequence, np.repeat(X_test_kmer[:, np.newaxis, :], X_test_sequence.shape[1], axis=1)), axis=2)

# ---------------------- Load Trained XGBoost Model and Scaler ----------------------
model_path = "/kaggle/working/xgboost_rna_folding_model.joblib"
scaler_path = "/kaggle/working/xgboost_scaler_y.joblib"
try:
    xgbr_loaded = joblib.load(model_path)
    scaler_y_loaded = joblib.load(scaler_path)
    print(f"Loaded XGBoost model from {model_path}")
    print(f"Loaded target scaler from {scaler_path}")
except FileNotFoundError:
    print(f"Error: Could not find model or scaler in /kaggle/working/. Make sure Notebook 1 was run successfully.")
    raise

# ---------------------- Predict with XGBoost ----------------------
print("Predicting with XGBoost...")
X_test_flattened = X_test.reshape(X_test.shape[0], -1)
y_pred_test_scaled = xgbr_loaded.predict(X_test_flattened)
y_pred_test = scaler_y_loaded.inverse_transform(y_pred_test_scaled)

# ---------------------- Create Submission ----------------------
predictions_df = pd.DataFrame(y_pred_test, columns=["x_1", "y_1", "z_1"])

for i in range(2, 6):
    predictions_df[[f"x_{i}", f"y_{i}", f"z_{i}"]] = predictions_df[["x_1", "y_1", "z_1"]].values

submission_df = pd.concat([sample_sub[["ID"]], predictions_df], axis=1)

submission_df.to_csv("submission.csv", index=False)
print("submission.csv created with shape:", submission_df.shape)

