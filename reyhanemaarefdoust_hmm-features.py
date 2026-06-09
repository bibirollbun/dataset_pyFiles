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


#In this notebook, I use a Hidden Markov Model (HMM) to extract statistical features from RNA sequences. These features — including entropy, log-likelihood, and dominant hidden state — can help improve model understanding of sequence dynamics.


!pip install pgmpy hmmlearn


from hmmlearn import hmm
from scipy.stats import entropy
import numpy as np
import pandas as pd



# Valid bases and encoding
valid_nucleotides = ['A', 'C', 'G', 'U']
nucleotide_to_int = {n: i for i, n in enumerate(valid_nucleotides)}

# Load data (adjust path if running locally or on Kaggle)
df = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_sequences.csv")

# Clean and encode
def clean_and_encode(seq):
    return [nucleotide_to_int[n] for n in str(seq) if n in nucleotide_to_int]

encoded_sequences = [clean_and_encode(seq) for seq in df["sequence"]]



# Filter out short sequences
sequences = [seq for seq in encoded_sequences if len(seq) > 1]
X = np.concatenate([np.array(seq).reshape(-1, 1) for seq in sequences])
lengths = [len(seq) for seq in sequences]

# Train a 4-state Multinomial HMM
model = hmm.MultinomialHMM(n_components=4, n_iter=100, random_state=42)
model.fit(X, lengths)



hmm_features = []

for seq in encoded_sequences:
    if len(seq) < 2:
        hmm_features.append({
            "hmm_entropy": np.nan,
            "hmm_log_prob": np.nan,
            "hmm_main_state": np.nan
        })
        continue

    X_seq = np.array(seq).reshape(-1, 1)
    
    # Decode sequence via Viterbi
    logprob, hidden_states = model.decode(X_seq, algorithm="viterbi")
    state_counts = np.bincount(hidden_states, minlength=model.n_components)
    
    # Compute features
    state_probs = state_counts / state_counts.sum()
    hmm_entropy = entropy(state_probs)
    main_state = np.argmax(state_counts)
    
    hmm_features.append({
        "hmm_entropy": hmm_entropy,
        "hmm_log_prob": logprob,
        "hmm_main_state": main_state
    })



# Create DataFrame and join with original
features_df = pd.DataFrame(hmm_features)
df = pd.concat([df, features_df], axis=1)
# Save new version (if needed locally)
df.to_csv("train_sequences_with_hmm_features.csv", index=False)


