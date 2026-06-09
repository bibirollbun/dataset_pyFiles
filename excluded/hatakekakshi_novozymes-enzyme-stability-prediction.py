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


!pip install python-Levenshtein
!pip install -q biopandas
!pip install -q biopython
!pip install -q pandarallel
!pip install -q xgboost
!pip install -q plotly


print("Starting Imports...")


# Basic machine learning & data science libraries
import os, random, gc, sys, time, math, re, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# TensorFlow and TPU utilities
import tensorflow as tf
try:
    import tensorflow_hub as tfhub
    import tensorflow_addons as tfa
    import tensorflow_io as tfio
except ImportError:
    pass  # These may not be needed if not using deep learning models

# Bioinformatics libraries for handling protein structures
import Levenshtein
from biopandas.pdb import PandasPdb

# Utility for parallel processing in pandas
from pandarallel import pandarallel
pandarallel.initialize()

# XGBoost for baseline modeling
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, RepeatedKFold


# For Kaggle dataset access (if running on Kaggle)
try:
    from kaggle_datasets import KaggleDatasets
except ImportError:
    KaggleDatasets = None

# Set random seed for reproducibility
def seed_everything(seed=7):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
seed_everything()

print("Imports complete.")


print("\nSetting up Accelerator...\n")
try:
    TPU = tf.distribute.cluster_resolver.TPUClusterResolver()  
except ValueError:
    TPU = None

if TPU:
    print(f"Running on TPU: {TPU.master()}")
    tf.config.experimental_connect_to_cluster(TPU)
    tf.tpu.experimental.initialize_tpu_system(TPU)
    strategy = tf.distribute.experimental.TPUStrategy(TPU)
else:
    print("Running on CPU/GPU")
    strategy = tf.distribute.get_strategy()
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            
print(f"Number of replicas in sync: {strategy.num_replicas_in_sync}")


print("\nLoading Data...\n")
if TPU and KaggleDatasets:
    DATA_DIR = KaggleDatasets().get_gcs_path('novozymes-enzyme-stability-prediction')
else:
    DATA_DIR = "/kaggle/input/novozymes-enzyme-stability-prediction"  # adjust if running locally

print("Data directory:", DATA_DIR)
print("Files in Data Directory:")
for file in tf.io.gfile.glob(os.path.join(DATA_DIR, "*")):
    print(" -", file)


# Load CSVs
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
ss_df    = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

# Load the wildtype PDB file using BioPandas
pdb = PandasPdb()
pdb_df = pdb.read_pdb(os.path.join(DATA_DIR, "wildtype_structure_prediction_af2.pdb"))



print("\nEngineering new features...\n")

# Define mapping for amino acids (long name, 3-letter, and 1-letter codes)
aa_map = {
    "Alanine": ("Ala", "A"), "Arginine": ("Arg", "R"), "Asparagine": ("Asn", "N"),
    "Aspartic_Acid": ("Asp", "D"), "Cysteine": ("Cys", "C"), "Glutamic_Acid": ("Glu", "E"),
    "Glutamine": ("Gln", "Q"), "Glycine": ("Gly", "G"), "Histidine": ("His", "H"),
    "Isoleucine": ("Ile", "I"), "Leucine": ("Leu", "L"), "Lysine": ("Lys", "K"),
    "Methionine": ("Met", "M"), "Phenylalanine": ("Phe", "F"), "Proline": ("Pro", "P"),
    "Serine": ("Ser", "S"), "Threonine": ("Thr", "T"), "Tryptophan": ("Trp", "W"),
    "Tyrosine": ("Tyr", "Y"), "Valine": ("Val", "V")
}
aa_chars_ordered = sorted([v[1] for v in aa_map.values()])  # sort by one-letter code

# Add sequence length to train and test
train_df["n_AA"] = train_df["protein_sequence"].apply(len)
test_df["n_AA"]  = test_df["protein_sequence"].apply(len)

# For each amino acid, count occurrences and compute fraction
for aa in aa_chars_ordered:
    count_col = f"AA_{aa}__count"
    frac_col  = f"AA_{aa}__fraction"
    train_df[count_col] = train_df["protein_sequence"].apply(lambda x: x.count(aa))
    train_df[frac_col]  = train_df[count_col] / train_df["n_AA"]
    test_df[count_col]  = test_df["protein_sequence"].apply(lambda x: x.count(aa))
    test_df[frac_col]   = test_df[count_col] / test_df["n_AA"]

# Encode data_source into an integer for modeling simplicity
ds_str2int = {k: i for i, k in enumerate(train_df["data_source"].dropna().unique())}
train_df["data_source_enum"] = train_df["data_source"].map(ds_str2int)
test_df["data_source_enum"]  = test_df["data_source"].map(ds_str2int)

# Temporary fix for pH: if pH > 14, swap pH and tm (indicates a data error)
def fix_ph(row):
    if row["pH"] > 14:
        # Log the issue and swap values
        # print(f"Swapping pH and tm at index {row.name}")
        row["pH"], row["tm"] = row["tm"], row["pH"]
    return row

train_df = train_df.apply(fix_ph, axis=1)
test_df  = test_df.apply(fix_ph, axis=1)

print("Data engineering complete.\n")
print("Train shape:", train_df.shape, "Test shape:", test_df.shape)


print("\nPerforming Exploratory Data Analysis (EDA)...\n")

# Plot distribution of melting temperatures (tm)
fig_tm = px.histogram(train_df, x="tm", nbins=50,
                        title="Distribution of Melting Temperature (tm)")
fig_tm.show()



# Plot pH distribution
fig_ph = px.histogram(train_df, x="pH", nbins=50,
                        title="Distribution of pH values in training data")
fig_ph.show()



# Plot distribution of protein sequence lengths
fig_len = px.histogram(train_df, x="n_AA", nbins=50, log_y=True,
                       title="Distribution of Protein Sequence Lengths (log scale)")
fig_len.show()





# Check correlations: for each amino acid, print the Spearman correlation of count and fraction with tm
print("Spearman correlations with tm:")
for aa in aa_chars_ordered:
    count_corr = train_df[f"AA_{aa}__count"].corr(train_df["tm"], method='spearman')
    frac_corr  = train_df[f"AA_{aa}__fraction"].corr(train_df["tm"], method='spearman')
    print(f"AA {aa}: count corr = {count_corr:.3f}, fraction corr = {frac_corr:.3f}")



# Create a heatmap of correlations for engineered features (only a sample of columns for clarity)
features = [col for col in train_df.columns if "__fraction" in col] + ["n_AA", "pH", "tm"]
corr = train_df[features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap of Selected Features")
plt.show()



def get_mutation_info(row, wildtype="VPVNPEPDATSVENVALKTGSGDSQSDPIKADLEVKGQSALPFDVDCWAILCKGAPNVLQRVNEKTKNSNRDRSGANKGPFKDPQKWGIKALPPKNPSWSAQDFKSPEEYAFASSLQGGTNAILAPVNLASQNSQGGVLNGFYSANKVAQFDPSKPQQTKGTWFQITKFTGAAGPYCKALGSNDKSVCDKNKNIAGDWGFDPAKWAYQYDEKNNKFNYVGK"):
    # Map the edit type for readability
    terminology_map = {"replace": "substitution", "insert": "insertion", "delete": "deletion"}
    # Compute edit operations
    edits = Levenshtein.editops(wildtype, row["protein_sequence"])
    row["n_edits"] = len(edits)
    if row["n_edits"] == 0:
        row["edit_type"] = pd.NA
        row["edit_idx"] = pd.NA
        row["wildtype_aa"] = pd.NA
        row["mutant_aa"] = pd.NA
    else:
        # We assume a single mutation (the first edit)
        op, i, j = edits[0]
        row["edit_type"] = terminology_map.get(op, op)
        row["edit_idx"] = i  # zero-indexed position in wildtype
        row["wildtype_aa"] = wildtype[i]
        # For deletion, there is no mutant amino acid
        row["mutant_aa"] = row["protein_sequence"][j] if op != "delete" else pd.NA
    return row


def revert_to_wildtype(protein_sequence, edit_type, edit_idx, wildtype_aa, mutant_aa):
    if pd.isna(edit_type):
        return protein_sequence
    if edit_type != "insertion":
        base = protein_sequence[:edit_idx]
        if edit_type == "deletion":
            new_seq = base + wildtype_aa + protein_sequence[edit_idx:]
        else:  # substitution
            new_seq = base + wildtype_aa + protein_sequence[edit_idx+1:]
    else:
        new_seq = protein_sequence[:edit_idx] + wildtype_aa + protein_sequence[edit_idx:]
    return new_seq



def create_mutation_txt_file(test_df, filename="AF70_mutations.txt", include_deletions=False):
    with open(filename, 'w') as f:
        for _, row in test_df.iterrows():
            if pd.isna(row["edit_type"]) or (row["edit_type"]=="deletion" and not include_deletions):
                continue
            # Mutation notation: wildtype letter + 1-indexed position + mutant letter
            mutation = f'{row["wildtype_aa"]}{row["edit_idx"]+1}{"" if pd.isna(row["mutant_aa"]) else row["mutant_aa"]}'
            f.write(mutation + "\n")



def create_wildtype_fasta_file(wildtype_sequence, filename="wildtype_af70.fasta"):
    with open(filename, 'w') as f:
        f.write(f">af70_wildtype\n{wildtype_sequence}")



# Apply mutation info extraction to test dataframe
test_df = test_df.apply(get_mutation_info, axis=1)
print("Mutation info added to test dataframe. Sample:")
display(test_df[["protein_sequence", "edit_type", "edit_idx", "wildtype_aa", "mutant_aa"]].head(10))

# Optionally create mutation and FASTA files
create_mutation_txt_file(test_df)
wildtype_aa = "VPVNPEPDATSVENVALKTGSGDSQSDPIKADLEVKGQSALPFDVDCWAILCKGAPNVLQRVNEKTKNSNRDRSGANKGPFKDPQKWGIKALPPKNPSWSAQDFKSPEEYAFASSLQGGTNAILAPVNLASQNSQGGVLNGFYSANKVAQFDPSKPQQTKGTWFQITKFTGAAGPYCKALGSNDKSVCDKNKNIAGDWGFDPAKWAYQYDEKNNKFNYVGK"
create_wildtype_fasta_file(wildtype_aa)



print("\nTraining baseline model with XGBoost...\n")
# Select feature columns (engineered features)
feature_cols = [col for col in train_df.columns if "__" in col] + ["n_AA", "pH"]

# Initialize the model
baseline_model = XGBRegressor(n_estimators=1000, learning_rate=0.1, max_depth=7, n_jobs=-1, random_state=7)

# Fit the model on the full training data
baseline_model.fit(train_df[feature_cols].to_numpy(), train_df["tm"].to_numpy(), verbose=True)

# Evaluate with cross-validation
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=7)
scores = cross_val_score(baseline_model, train_df[feature_cols].to_numpy(), train_df["tm"].to_numpy(), 
                         scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
print("CV MAE: %.3f (%.3f)" % (np.abs(scores).mean(), np.abs(scores).std()))

# Generate predictions on the test set and prepare submission file
preds = baseline_model.predict(test_df[feature_cols].to_numpy())
ss_df["tm"] = preds
ss_df.to_csv("submission.csv", index=False)
print("\nSubmission file saved as submission.csv")
display(ss_df.head())


# Compute in-sample predictions on the training set for error analysis
train_preds = baseline_model.predict(train_df[feature_cols].to_numpy())
residuals = train_df["tm"].values - train_preds  # error: actual - predicted

# 1. Scatter Plot: Actual vs Predicted tm
plt.figure(figsize=(8,6))
plt.scatter(train_df["tm"], train_preds, alpha=0.5, label='Data points')
plt.plot([train_df["tm"].min(), train_df["tm"].max()],
         [train_df["tm"].min(), train_df["tm"].max()],
         color='red', linestyle='--', label='Ideal Fit')
plt.xlabel("Actual tm")
plt.ylabel("Predicted tm")
plt.title("Scatter Plot: Actual vs Predicted tm")
plt.legend()
plt.show()

# 2. Histogram of Residuals
plt.figure(figsize=(8,6))
plt.hist(residuals, bins=50, color='skyblue', edgecolor='black')
plt.xlabel("Residual (Actual - Predicted)")
plt.ylabel("Frequency")
plt.title("Histogram of Residuals")
plt.show()

# 3. Scatter Plot: Absolute Error vs Actual tm
abs_error = np.abs(residuals)
plt.figure(figsize=(8,6))
plt.scatter(train_df["tm"], abs_error, alpha=0.5, color='green')
plt.xlabel("Actual tm")
plt.ylabel("Absolute Error")
plt.title("Absolute Error vs Actual tm")
plt.show()

# Print overall error metric for reference
mae = np.mean(abs_error)
print("Mean Absolute Error (Training): {:.3f}".format(mae))





